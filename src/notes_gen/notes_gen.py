"""
notes_gen.py — LLM-powered LaTeX notes generator

OCR filtering pipeline:
  1. Extract only lines containing mathematical content from OCR
  2. Deduplicate against transcript (remove OCR text already spoken)
  3. Pass filtered OCR + full transcript to Claude in a single call
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from rich.console import Console

console = Console()

Backend = Literal["claude", "ollama", "openai"]


SYSTEM_PROMPT = """You are an expert academic note-taker and LaTeX typesetter for university-level engineering and science courses. Your task is to convert a raw lecture transcript (and optionally OCR-extracted slide/blackboard text) into complete, publication-quality LaTeX lecture notes.

CONTENT RULES:
1. Cover the ENTIRE lecture from start to finish — do not skip, summarise, or omit any concept, derivation, or example discussed.
2. A student studying ONLY from these notes should be able to fully understand the lecture without watching the video.
3. Preserve all technical terminology, variable names, and notation exactly as used by the professor.
4. Reconstruct all mathematical expressions from the transcript into proper LaTeX, even if only spoken aloud (e.g. "L over one plus L" -> $\\frac{L(s)}{1+L(s)}$).
5. If OCR text is provided, integrate it with the transcript — it contains formulas or equations written on slides or blackboard that were NOT spoken aloud. Do not duplicate content already present in the transcript.
6. Include all examples, exercises, and numerical cases discussed, no matter how briefly mentioned.
7. Preserve the professor's physical intuitions and motivations — for every formula or result, include a sentence explaining WHY it holds or what it means physically or intuitively, not just WHAT it is. Use phrases like "This means that...", "Intuitively...", "The reason is that...".
7b. Pay attention to how much time the professor spends on each topic — if the professor repeats, elaborates, or returns to a concept multiple times, treat it as a key concept and give it proportionally more space, detail, and explanation in the notes. Conversely, topics mentioned only briefly should be covered concisely.

STRUCTURE RULES:
8. Organise content into logical \\section{} and \\subsection{} following the natural flow of the lecture.
9. Open each section with a brief prose paragraph contextualising the topic before any formulas.
10. Use \\begin{definition}, \\begin{theorem}, \\begin{lemma}, \\begin{remark}, \\begin{example} environments for all key mathematical content.
11. Write proofs and derivations as flowing mathematical prose with \\begin{align} or \\begin{equation}, never as bullet points.
12. When the lecture introduces relationships between multiple inputs/outputs or variables, always render them as a complete \\begin{tabular} with ALL entries filled in.
13. Write frequency-domain or piecewise approximations as \\begin{cases} formulas, not bullet points.

FORMATTING RULES:
14. Minimize \\begin{itemize} and \\begin{enumerate} — use them only for genuine lists (e.g. schedules, enumerated steps). Prefer prose or mathematical environments instead.
15. Never leave a table partially filled — if data is missing from the transcript, reconstruct it from context or mark it explicitly with "?".
16. Use \\begin{align} for multi-line equations and derivations, \\begin{equation} for single important results.
17. Bold key terms on first introduction with \\textbf{}.

LATEX OUTPUT RULES:
18. Output ONLY valid LaTeX — no prose explanation, no markdown, no code fences before or after.
19. Begin with \\documentclass{article} and end with \\end{document}.
20. Use only these standard packages: amsmath, amssymb, amsthm, geometry, graphicx, inputenc, enumitem. Never use custom \\usepackage{preamble}.
21. Always include \\usepackage[utf8]{inputenc} and \\usepackage{enumitem} in the preamble.
22. Define \\newtheorem for: theorem, definition, lemma, example, remark.
23. Always wrap \\begin{cases} inside math mode: use \\[ \\begin{cases}...\\end{cases} \\] or $\\begin{cases}...\\end{cases}$ — never use \\begin{cases} outside math mode.
24. Never use Unicode subscripts or superscripts (₁₂₃⁰¹²) — always use LaTeX math notation: $\\text{Ni}_3\\text{Ti}$, $\\text{Fe}_2\\text{Mo}$, $\\text{CO}_2$.
"""


# ── OCR Filtering ─────────────────────────────────────────────────────────────

MATH_PATTERNS = [
    r'[=\+\-\*/\^]',
    r'[α-ωΑ-Ω]',
    r'\d+\s*[/]\s*\d+',
    r'[A-Z]\([a-z]\)',
    r'\b(lim|inf|sup|max|min|sum|prod)\b',
    r'\b(dB|Hz|rad|omega|sigma|delta|mu)\b',
    r'[<>≤≥≈≠∞∂∫∑∏√]',
    r'\b\d+\s*(dB|Hz|rad/s)\b',
    r'[A-Za-z]+\s*[=]\s*[A-Za-z0-9\(\)\+\-\*/\^]+',
]

MATH_REGEX = re.compile('|'.join(MATH_PATTERNS))


def _is_mathematical(text: str) -> bool:
    return bool(MATH_REGEX.search(text))


def _similarity(a: str, b: str) -> float:
    words_a = set(re.findall(r'\b\w+\b', a.lower()))
    words_b = set(re.findall(r'\b\w+\b', b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


def filter_ocr(
    ocr_text: str,
    transcript: str,
    similarity_threshold: float = 0.7,
    math_only: bool = True,
) -> str:
    """
    Filter OCR text before passing to the LLM.

    Args:
        ocr_text:             Raw OCR text from slides/blackboard
        transcript:           Full lecture transcript for deduplication
        similarity_threshold: Lines with similarity > this vs transcript are dropped
        math_only:            If True, keep only mathematical content (formulas, symbols)
                              If False, keep ALL slide text not already in transcript
                              Toggle via config: ocr.math_only_filter
    """
    if not ocr_text.strip():
        return ""

    lines = [l.strip() for l in re.split(r'\n|(?<=\])\s', ocr_text) if l.strip()]

    transcript_words = transcript.split()
    transcript_chunks = []
    window = 50
    for i in range(0, len(transcript_words), window // 2):
        chunk = " ".join(transcript_words[i:i + window])
        transcript_chunks.append(chunk)

    filtered = []
    seen = set()

    for line in lines:
        clean = re.sub(r'^\[\d+min\]\s*', '', line).strip()
        if len(clean) < 3:
            continue

        # Math-only filter: skip non-mathematical lines when enabled
        if math_only and not _is_mathematical(clean):
            continue

        # Deduplication: skip if already present in transcript
        is_duplicate = any(
            _similarity(clean, chunk) > similarity_threshold
            for chunk in transcript_chunks
        )
        if is_duplicate:
            continue

        # Deduplication within OCR itself
        clean_normalized = re.sub(r'\s+', ' ', clean.lower())
        if clean_normalized in seen:
            continue
        seen.add(clean_normalized)
        filtered.append(clean)

    mode = "math only" if math_only else "all text"
    if filtered:
        console.print(
            f"[dim]OCR filter ({mode}): {len(lines)} entries → "
            f"{len(filtered)} unique expressions[/dim]"
        )
        return "\n".join(filtered)
    else:
        console.print(f"[dim]OCR filter ({mode}): no new content found[/dim]")
        return ""


# ── Prompt Builder ────────────────────────────────────────────────────────────

def _build_prompt(
    transcript: str,
    ocr_filtered: str,
    course_name: str,
    lecture_date: str,
    figures: list[dict] = None,
) -> str:
    prompt = f"""Convert the following lecture transcript into complete, comprehensive LaTeX notes.
Cover EVERY topic discussed — do not skip or summarise any part of the lecture.

Course: {course_name}
Date: {lecture_date}

--- FULL TRANSCRIPT ---
{transcript}
"""
    if ocr_filtered.strip():
        prompt += f"""
--- ADDITIONAL MATHEMATICAL CONTENT FROM SLIDES/BLACKBOARD ---
(These are formulas and equations written by the professor that were NOT spoken aloud.
Integrate them naturally into the notes where contextually appropriate.)
{ocr_filtered}
"""
    if figures:
        prompt += """
--- FIGURES TO INCLUDE ---
The following figures have been extracted from the lecture slides/video.
For each figure, insert it at the appropriate point in the notes using this LaTeX template:

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.6\\textwidth]{{LATEX_PATH}}
\\caption{{CAPTION}}
\\end{{figure}}

Replace LATEX_PATH and CAPTION with the values below.
Insert each figure near the section where the corresponding topic is discussed (use the timestamp as a guide).

"""
        for fig in figures:
            mins = int(fig["timestamp"] // 60)
            secs = int(fig["timestamp"] % 60)
            prompt += (
                f"[{mins:02d}:{secs:02d}] latex_path={fig['latex_path']} "
                f"caption={fig['caption']}\n"
            )

    prompt += "\nProduce the complete .tex file now, covering the entire lecture:"
    return prompt


# ── LLM Backends ─────────────────────────────────────────────────────────────

def _generate_claude(prompt: str, model: str, api_key: str, max_tokens: int = 16000) -> str:
    import anthropic
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("No Anthropic API key found.")
    client = anthropic.Anthropic(api_key=key)
    console.print(f"[cyan]Generating notes via Claude ({model}) with streaming...[/cyan]")

    full_response = ""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            full_response += text
            print(text, end="", flush=True)
    print()
    return full_response


def _generate_ollama(prompt: str, model: str = "mistral", host: str = "http://localhost:11434") -> str:
    import ollama
    console.print(f"[cyan]Generating notes via Ollama ({model})...[/cyan]")
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        options={"num_ctx": 8192},
    )
    return response["message"]["content"]


def _generate_openai(prompt: str, model: str = "gpt-4o", api_key: str = "", max_tokens: int = 16000) -> str:
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("No OpenAI API key found.")
    client = OpenAI(api_key=key)
    console.print(f"[cyan]Generating notes via OpenAI ({model})...[/cyan]")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content


def _call_backend(prompt: str, backend: str, cfg: dict) -> str:
    if backend == "claude":
        return _generate_claude(
            prompt,
            model=cfg.get("model", "claude-sonnet-4-20250514"),
            api_key=cfg.get("api_key", ""),
            max_tokens=cfg.get("max_tokens", 16000),
        )
    elif backend == "ollama":
        return _generate_ollama(
            prompt,
            model=cfg.get("model", "mistral"),
            host=cfg.get("host", "http://localhost:11434"),
        )
    elif backend == "openai":
        return _generate_openai(
            prompt,
            model=cfg.get("model", "gpt-4o"),
            api_key=cfg.get("api_key", ""),
            max_tokens=cfg.get("max_tokens", 16000),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _clean_latex(raw: str) -> str:
    raw = re.sub(r"^```(?:latex|tex)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _make_pdf_filename(course_name: str, lecture_date: str, suffix: str = None) -> str:
    """
    Generate a clean PDF filename in European date format.
    e.g. '08-10-2025_Methods and Technologies for Feedback Control Systems.pdf'
    e.g. '08-10-2025_SMART MATERIALS - Structural Steel.pdf' (with suffix)
    """
    try:
        parts = lecture_date.split("-")
        if len(parts) == 3:
            date_eu = f"{parts[2]}-{parts[1]}-{parts[0]}"
        else:
            date_eu = lecture_date
    except Exception:
        date_eu = lecture_date

    safe_course = re.sub(r'[<>:"/\\|?*]', '', course_name).strip()
    filename = f"{date_eu}_{safe_course}"
    if suffix:
        safe_suffix = re.sub(r'[<>:"/\\|?*]', '', suffix).strip()
        filename += f" - {safe_suffix}"
    return f"{filename}.pdf"


def compile_pdf(
    tex_path: Path,
    pdf_output_dir: Path,
    course_name: str,
    lecture_date: str,
    suffix: str = None,
) -> Path | None:
    """
    Compile the .tex file and save the PDF to pdf_output_dir with a descriptive filename.
    The .tex is compiled in its own directory (latex/) then the PDF is copied to notes/.
    """
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["pdflatex", "-interaction=nonstopmode",
           "-output-directory", str(tex_path.parent),
           str(tex_path)]
    try:
        for _ in range(2):  # Run twice for cross-references
            subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        console.print(f"[yellow]⚠ pdflatex not available or failed: {e}[/yellow]")
        console.print("[dim]Install MiKTeX or TeX Live to auto-compile PDFs.[/dim]")
        return None

    compiled_pdf = tex_path.with_suffix(".pdf")
    if not compiled_pdf.exists():
        console.print(f"[yellow]⚠ Compiled PDF not found at {compiled_pdf}[/yellow]")
        return None

    pdf_filename = _make_pdf_filename(course_name, lecture_date, suffix)
    final_pdf = pdf_output_dir / pdf_filename

    import shutil
    shutil.copy2(compiled_pdf, final_pdf)
    compiled_pdf.unlink()

    console.print(f"[green]✓ PDF saved:[/green] {final_pdf}")
    return final_pdf


def _merge_latex_chunks(chunks: list[str]) -> str:
    def extract_body(latex: str) -> str:
        m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", latex, re.DOTALL)
        return m.group(1).strip() if m else latex

    preamble_match = re.match(r"(.*?\\begin\{document\})", chunks[0], re.DOTALL)
    preamble = preamble_match.group(1) if preamble_match else "\\documentclass{article}\n\\begin{document}"
    bodies = [extract_body(c) for c in chunks]
    merged_body = "\n\n% --- continued ---\n\n".join(bodies)
    return f"{preamble}\n\n{merged_body}\n\n\\end{{document}}"


# ── Public API ────────────────────────────────────────────────────────────────

def generate_notes(
    merged_data: list[dict],
    output_dir: Path,
    stem: str,
    course_name: str,
    lecture_date: str,
    backend: Backend = "claude",
    backend_config: dict = None,
    compile_pdf_flag: bool = True,
    transcript_path: Path = None,
    pdf_output_dir: Path = None,
    figures: list[dict] = None,
    suffix: str = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = backend_config or {}

    if pdf_output_dir is None:
        pdf_output_dir = output_dir.parent / "notes"

    console.print(f"\n[bold cyan]── Notes Generation ────────────────────────────[/bold cyan]")
    console.print(f"Backend:   [yellow]{backend}[/yellow]")
    console.print(f"Course:    {course_name}")
    console.print(f"Date:      {lecture_date}")
    if suffix:
        console.print(f"Suffix:    {suffix}")
    if figures:
        console.print(f"Figures:   {len(figures)} to embed")

    # ── Load full transcript ──────────────────────────────────────────────────
    full_transcript = ""
    if transcript_path and transcript_path.exists():
        full_transcript = transcript_path.read_text(encoding="utf-8")
        console.print(f"[dim]Loaded transcript: {len(full_transcript.split())} words[/dim]")
    else:
        speech_parts = [entry.get("speech", "") for entry in merged_data if entry.get("speech")]
        full_transcript = " ".join(speech_parts)
        console.print(f"[dim]Transcript from merged data: {len(full_transcript.split())} words[/dim]")

    # ── Build and filter OCR text ─────────────────────────────────────────────
    ocr_parts = []
    seen = set()
    for entry in merged_data:
        ocr = entry.get("ocr_text", "").strip()
        if ocr and ocr not in seen:
            t = entry.get("timestamp_sec", 0)
            mins = int(t // 60)
            ocr_parts.append(f"[{mins:02d}min] {ocr}")
            seen.add(ocr)
    raw_ocr = "\n".join(ocr_parts)

    filtered_ocr = filter_ocr(
        raw_ocr,
        full_transcript,
        math_only=backend_config.get("math_only_ocr", True) if backend_config else True,
    )

    console.print(f"Data pts:  {len(merged_data)} OCR frames → "
                  f"{len(filtered_ocr.splitlines()) if filtered_ocr else 0} unique math expressions")

    # ── Generate notes ────────────────────────────────────────────────────────
    MAX_WORDS_PER_CHUNK = 18000
    words = full_transcript.split()

    if len(words) <= MAX_WORDS_PER_CHUNK:
        console.print(f"  Chunk 1/1...")
        prompt = _build_prompt(full_transcript, filtered_ocr, course_name, lecture_date, figures)
        raw = _call_backend(prompt, backend, cfg)
        final_latex = _clean_latex(raw)
    else:
        chunks = [words[i:i + MAX_WORDS_PER_CHUNK] for i in range(0, len(words), MAX_WORDS_PER_CHUNK)]
        console.print(f"  Long transcript — splitting into {len(chunks)} chunks...")
        latex_sections = []
        for i, chunk_words in enumerate(chunks):
            console.print(f"  Chunk {i+1}/{len(chunks)}...")
            chunk_text = " ".join(chunk_words)
            # Only pass figures to first chunk
            prompt = _build_prompt(
                chunk_text,
                filtered_ocr if i == 0 else "",
                course_name,
                lecture_date,
                figures if i == 0 else None,
            )
            raw = _call_backend(prompt, backend, cfg)
            latex_sections.append(_clean_latex(raw))
        final_latex = _merge_latex_chunks(latex_sections)

    # Always save .tex to output/latex/ (overwritten each time)
    tex_path = output_dir / "lecture_notes.tex"
    tex_path.write_text(final_latex, encoding="utf-8")
    console.print(f"[green]✓ LaTeX saved:[/green] {tex_path}")

    if compile_pdf_flag:
        compile_pdf(tex_path, pdf_output_dir, course_name, lecture_date, suffix)

    return tex_path