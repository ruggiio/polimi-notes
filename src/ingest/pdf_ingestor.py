"""
pdf_ingestor.py — Ingest handout PDFs into RAG with Marker + Claude Vision figure extraction.

Resume-friendly: l'estrazione figure salta le pagine già processate (PNG già
presente in figures_dir), così rilanciarla riprende da dove si era fermata
senza rifare le chiamate a Claude. Per forzare il refresh: env ANTHROPIC_FORCE=1.
"""

import base64
import contextlib
import io
import json
import os
import re
import time
from pathlib import Path

from rich.console import Console

console = Console()


def _extract_text_with_marker(pdf_path: Path) -> str:
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        console.print(f"[cyan]  Marker: converting {pdf_path.name}...[/cyan]")
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        text = rendered.markdown
        console.print(f"[green]  ✓ Marker: {len(text)} chars extracted[/green]")
        return text
    except (ImportError, AttributeError):
        pass
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        console.print(f"[cyan]  Marker v0.3: converting {pdf_path.name}...[/cyan]")
        models = load_all_models()
        text, _, _ = convert_single_pdf(str(pdf_path), models, langs=["English"])
        console.print(f"[green]  ✓ Marker: {len(text)} chars extracted[/green]")
        return text
    except (ImportError, AttributeError):
        pass
    console.print("[yellow]  ⚠ marker-pdf not installed — falling back to pdfplumber[/yellow]")
    return _pdfplumber_extract(pdf_path)


def _pdfplumber_extract(pdf_path: Path) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)
    except Exception as e:
        console.print(f"[red]  pdfplumber also failed: {e}[/red]")
        return ""


def _render_page_png(pdf_path: Path, page_num: int, dpi: int = 200) -> bytes:
    import fitz
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def _clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def _crop_figure(pdf_path: Path, page_num: int, bbox: dict, out_path: Path, dpi: int = 200):
    """Ritaglia una figura. Robusto a bbox mancanti/degeneri (evita il crash
    'code=4: Invalid bandwriter header' di PyMuPDF su rettangoli vuoti)."""
    import fitz
    if not isinstance(bbox, dict):
        raise ValueError("bbox mancante o non valido")
    x1 = _clamp01(bbox.get("x1"))
    y1 = _clamp01(bbox.get("y1"))
    x2 = _clamp01(bbox.get("x2"))
    y2 = _clamp01(bbox.get("y2"))
    if None in (x1, y1, x2, y2):
        raise ValueError("coordinate bbox incomplete")
    # normalizza ordine e impone una dimensione minima
    if x2 <= x1:
        x1, x2 = min(x1, x2), max(x1, x2)
        x2 = min(1.0, x1 + 0.02)
    if y2 <= y1:
        y1, y2 = min(y1, y2), max(y1, y2)
        y2 = min(1.0, y1 + 0.02)
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    pw, ph = page.rect.width, page.rect.height
    rect = fitz.Rect(x1 * pw, y1 * ph, x2 * pw, y2 * ph)
    if rect.width < 1 or rect.height < 1:
        doc.close()
        raise ValueError("bbox troppo piccolo")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    doc.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Claude Vision (Anthropic) — sostituisce Gemini per figure + handwritten.
#  Riusa ANTHROPIC_API_KEY (la stessa chiave con cui generi le note).
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeQuotaError(RuntimeError):
    """Chiave Anthropic non valida/senza credito, o rate limit persistente."""


def _anthropic_client(api_key: str):
    import anthropic
    # timeout esplicito: evita che una chiamata appesa blocchi tutto il run.
    # max_retries=0: i retry sui 429 li gestiamo noi nel loop sotto.
    return anthropic.Anthropic(api_key=api_key, timeout=120.0, max_retries=0)


def _claude_image_block(page_png: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(page_png).decode("utf-8"),
        },
    }


def _claude_text(response) -> str:
    return "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()


def _retry_wait(exc, attempt: int) -> float:
    """Tempo di attesa per un 429: usa l'header retry-after se presente."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            return float(resp.headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    return min(20 * (attempt + 1), 60)


def _transcribe_page_with_claude(page_png: bytes, api_key: str, model: str) -> str:
    import anthropic
    client = _anthropic_client(api_key)
    prompt = (
        "This is a page from handwritten engineering lecture notes.\n\n"
        "Transcribe ALL content visible on this page:\n"
        "- Written text and explanations\n"
        "- Mathematical formulas and equations (use LaTeX notation: $...$ for inline, $$...$$ for display)\n"
        "- Labels on diagrams or figures\n"
        "- Numbered steps or bullet points\n\n"
        "Return only the transcription. If the page is blank or unreadable, return an empty string."
    )
    content = [_claude_image_block(page_png), {"type": "text", "text": prompt}]
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
            return _claude_text(response)
        except anthropic.RateLimitError as e:
            wait = _retry_wait(e, attempt)
            console.print(f"[yellow]    Rate limit Claude — attendo {wait:.0f}s... (tentativo {attempt + 1}/5)[/yellow]")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code in (401, 403):
                raise ClaudeQuotaError(
                    "Chiave ANTHROPIC non valida o senza credito. Controlla ANTHROPIC_API_KEY."
                )
            console.print(f"[yellow]    Claude transcription error {code}: {e}[/yellow]")
            return ""
        except Exception as e:
            console.print(f"[yellow]    Claude transcription error: {e}[/yellow]")
            return ""
    return ""


def _detect_figures(page_png: bytes, api_key: str, model: str) -> list[dict]:
    import anthropic
    client = _anthropic_client(api_key)
    prompt = (
        "This is a page from an engineering lecture handout (Advanced Dynamics of Mechanical Systems).\n\n"
        "Identify every standalone diagram, plot, or figure — NOT inline equations or text blocks.\n"
        "For each figure found, return a JSON array:\n"
        '[\n'
        '  {\n'
        '    "description": "concise description of what the figure shows",\n'
        '    "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}\n'
        '  }\n'
        ']\n'
        "Bounding box: normalized 0.0-1.0, origin top-left, x1y1=top-left, x2y2=bottom-right.\n"
        "Every figure MUST include a complete bbox with all four keys x1,y1,x2,y2.\n"
        "If there are NO figures (text or equation only page), return exactly: []\n"
        "Return ONLY the JSON array, no markdown, no explanation."
    )
    content = [_claude_image_block(page_png), {"type": "text", "text": prompt}]
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": "["},  # prefill: forza un array JSON
                ],
            )
            raw = ("[" + _claude_text(response)).strip()
            raw = re.sub(r"```\s*$", "", raw).strip()
            try:
                figures = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\[.*\]", raw, re.S)
                figures = json.loads(m.group(0)) if m else []
            return figures if isinstance(figures, list) else []
        except anthropic.RateLimitError as e:
            wait = _retry_wait(e, attempt)
            console.print(f"[yellow]    Rate limit Claude — attendo {wait:.0f}s... (tentativo {attempt + 1}/5)[/yellow]")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code in (401, 403):
                raise ClaudeQuotaError(
                    "Chiave ANTHROPIC non valida o senza credito. Controlla ANTHROPIC_API_KEY."
                )
            console.print(f"[yellow]    Claude API error {code}: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]    Claude error: {e}[/yellow]")
            return []
    raise ClaudeQuotaError(
        "Claude risponde 429 anche dopo 5 tentativi sulla stessa pagina: rate limit persistente."
    )


def _quiet_add_lecture(rag, **kwargs) -> int:
    """Chiama rag.add_lecture silenziando il suo output (una riga per chunk
    intaserebbe Colab e può farlo freezare). Ritorna il n. di chunk."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return rag.add_lecture(**kwargs)


def _process_figures(pdf_path: Path, n_pages: int, figures_dir: Path, rag,
                     course_name: str, api_key: str, model: str, force: bool) -> tuple:
    """Rileva + ritaglia + indicizza le figure di un PDF.
    Resume a livello di pagina: se esiste già {stem}_p{n}_fig1.png salta la
    pagina (a meno di force). Output compatto: una riga per pagina con figure."""
    existing = {p.name for p in figures_dir.glob(f"{pdf_path.stem}_p*_fig*.png")}
    figures_added = 0
    errors = 0
    skipped = 0
    for page_num in range(n_pages):
        page_marker = f"{pdf_path.stem}_p{page_num + 1}_fig1.png"
        if page_marker in existing and not force:
            skipped += 1
            continue
        time.sleep(1)  # gentile col rate limit
        page_png = _render_page_png(pdf_path, page_num)
        detected = _detect_figures(page_png, api_key, model)
        if not detected:
            continue
        saved_here = 0
        for fig_idx, fig in enumerate(detected):
            bbox = fig.get("bbox") if isinstance(fig, dict) else None
            fname = f"{pdf_path.stem}_p{page_num + 1}_fig{fig_idx + 1}.png"
            fig_path = figures_dir / fname
            try:
                _crop_figure(pdf_path, page_num, bbox, fig_path)
                desc = fig.get("description", "") if isinstance(fig, dict) else ""
                _quiet_add_lecture(
                    rag,
                    transcript=f"[FIGURE: output/rag/figures/{fname}] {desc}",
                    course_name=course_name,
                    lecture_date="handout",
                    source=f"figure:{pdf_path.stem}:p{page_num + 1}",
                )
                figures_added += 1
                saved_here += 1
            except Exception as e:
                errors += 1
                console.print(f"[yellow]    ⚠ {fname}: {e}[/yellow]")
        if saved_here:
            console.print(f"[dim]  p{page_num + 1}/{n_pages}: {saved_here} figura/e[/dim]")
    if skipped:
        console.print(f"[dim]  ({skipped} pagine già fatte, saltate)[/dim]")
    return figures_added, errors


def _page_count(pdf_path: Path) -> int:
    import fitz
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def ingest_pdf(
    pdf_path: Path,
    course_name: str,
    rag,
    figures_dir: Path,
    gemini_api_key: str = "",   # legacy: ignorato (la pipeline usa Claude Vision)
    gemini_model: str = "",     # legacy: ignorato
    handwritten: bool = False,
    figures_only: bool = False,
    anthropic_api_key: str = "",
    anthropic_model: str = "claude-sonnet-4-20250514",
) -> dict:
    label = " [handwritten]" if handwritten else " [figures only]" if figures_only else ""
    console.print(f"\n[bold cyan]Ingesting:[/bold cyan] {pdf_path.name}{label}")
    figures_dir.mkdir(parents=True, exist_ok=True)
    anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_model = anthropic_model or os.environ.get("ANTHROPIC_MODEL", "") or "claude-sonnet-4-20250514"
    force = os.environ.get("ANTHROPIC_FORCE", "") == "1"
    text_chunks = 0

    if handwritten:
        if not anthropic_api_key:
            console.print("[red]  --handwritten requires ANTHROPIC_API_KEY[/red]")
            return {"text_chunks": 0, "figures": 0}
        try:
            import fitz  # noqa: F401
        except ImportError:
            console.print("[red]  --handwritten requires PyMuPDF (pip install pymupdf)[/red]")
            return {"text_chunks": 0, "figures": 0}
        n_pages = _page_count(pdf_path)
        pages_text = []
        for page_num in range(n_pages):
            console.print(f"[dim]  Transcribing page {page_num + 1}/{n_pages} (Claude Vision)...[/dim]")
            time.sleep(1)
            page_png = _render_page_png(pdf_path, page_num)
            page_text = _transcribe_page_with_claude(page_png, anthropic_api_key, anthropic_model)
            if page_text:
                pages_text.append(f"<!-- Page {page_num + 1} -->\n{page_text}")
        text = "\n\n".join(pages_text)
        if text.strip():
            console.print(f"[green]  ✓ Transcribed {len(pages_text)}/{n_pages} pages ({len(text)} chars)[/green]")
            text_chunks = _quiet_add_lecture(
                rag,
                transcript=text,
                course_name=course_name,
                lecture_date=pdf_path.stem,
                source=f"handwritten:{pdf_path.stem}",
            )
        return {"text_chunks": text_chunks, "figures": 0}

    if figures_only:
        if not anthropic_api_key:
            console.print("[red]  --figures-only requires ANTHROPIC_API_KEY[/red]")
            return {"text_chunks": 0, "figures": 0}
        try:
            import fitz  # noqa: F401
        except ImportError:
            console.print("[red]  --figures-only requires PyMuPDF (pip install pymupdf)[/red]")
            return {"text_chunks": 0, "figures": 0}
        n_pages = _page_count(pdf_path)
        figures_added, errors = _process_figures(
            pdf_path, n_pages, figures_dir, rag, course_name,
            anthropic_api_key, anthropic_model, force,
        )
        msg = f"[green]  ✓ {figures_added} figure indicizzate[/green]"
        if errors:
            msg += f" [yellow]({errors} scartate)[/yellow]"
        console.print(msg)
        return {"text_chunks": 0, "figures": figures_added}

    # ── branch default: testo (Marker) + figure (Claude) ──
    text = _extract_text_with_marker(pdf_path)
    if text.strip():
        text_chunks = _quiet_add_lecture(
            rag,
            transcript=text,
            course_name=course_name,
            lecture_date="handout",
            source=f"handout:{pdf_path.stem}",
        )
        console.print(f"[green]  ✓ testo: {text_chunks} chunk[/green]")

    if not anthropic_api_key:
        console.print("[dim]  No ANTHROPIC_API_KEY — skipping figure extraction[/dim]")
        return {"text_chunks": text_chunks, "figures": 0}

    try:
        import fitz  # noqa: F401
    except ImportError:
        console.print("[yellow]  ⚠ PyMuPDF not installed — skipping figure extraction[/yellow]")
        return {"text_chunks": text_chunks, "figures": 0}

    n_pages = _page_count(pdf_path)
    figures_added, errors = _process_figures(
        pdf_path, n_pages, figures_dir, rag, course_name,
        anthropic_api_key, anthropic_model, force,
    )
    msg = f"[green]  ✓ {figures_added} figure indicizzate[/green]"
    if errors:
        msg += f" [yellow]({errors} scartate)[/yellow]"
    console.print(msg)
    return {"text_chunks": text_chunks, "figures": figures_added}
