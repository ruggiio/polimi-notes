"""
notes_gen.py — LLM-powered LaTeX notes generator

OCR filtering pipeline:
  1. Extract only lines containing mathematical content from OCR
  2. Deduplicate against transcript (remove OCR text already spoken)
  3. Pass filtered OCR + full transcript to Claude in a single call
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from rich.console import Console

console = Console()

Backend = Literal["claude", "ollama", "openai"]


SYSTEM_PROMPT = """You are an expert academic note-taker and LaTeX typesetter for university-level engineering and science courses. Your task is to convert a raw lecture transcript (and optionally OCR-extracted slide/blackboard text) into complete, publication-quality LaTeX lecture notes. Write in a bookish, academic prose style — clear, well-constructed sentences that flow naturally, as in a graduate-level textbook. The notes should read as polished scholarship, not as a transcript dump.

CONTENT RULES:
1. Cover the ENTIRE lecture from start to finish — do not skip, summarise, or omit any concept, derivation, or example discussed.
2. A student studying ONLY from these notes should be able to fully understand the lecture without watching the video. When the professor's explanation of a concept is incomplete, rushed, or unclear, expand it with a correct and complete treatment using standard academic knowledge — do not limit yourself to only what was said.
3. Preserve all technical terminology, variable names, and notation exactly as used by the professor.
4. Reconstruct all mathematical expressions from the transcript into proper LaTeX, even if only spoken aloud (e.g. "L over one plus L" -> $\\frac{L(s)}{1+L(s)}$).
5. If OCR text is provided, integrate it with the transcript — it contains formulas or equations written on slides or blackboard that were NOT spoken aloud. Do not duplicate content already present in the transcript.
6. Include all examples, exercises, and numerical cases discussed, no matter how briefly mentioned.
7. Preserve the professor's physical intuitions and motivations — for every formula or result, include a sentence explaining WHY it holds or what it means physically or intuitively, not just WHAT it is. Use phrases like "This means that...", "Intuitively...", "The reason is that...".
7b. Pay attention to how much time the professor spends on each topic — if the professor repeats, elaborates, or returns to a concept multiple times, treat it as a key concept and give it proportionally more space, detail, and explanation in the notes. Conversely, topics mentioned only briefly should be covered concisely.
7c. Use smooth prose transitions between sections and subsections. Avoid abrupt jumps between topics — introduce each new concept with a sentence that connects it to what came before, so the notes read as a coherent narrative rather than a sequence of isolated facts.

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
    rag_context: str = None,
) -> str:
    prompt = f"""Convert the following lecture transcript into complete, comprehensive LaTeX notes.
Write with a bookish, refined academic style — not a transcript dump, but polished notes a student would enjoy reading.
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

    if rag_context:
        prompt += f"""
--- CONTEXT FROM PREVIOUS LECTURES IN THIS COURSE ---
(Use this context to maintain consistency with terminology, notation, and concepts
introduced in previous lectures. Reference prior material where appropriate.)
{rag_context}
"""

    prompt += "\nProduce the complete .tex file now, covering the entire lecture:"
    return prompt


# ── Tool Definitions for Claude Tool Use ─────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_transcript",
        "description": (
            "Search the lecture transcript for a specific query. "
            "Returns the top 3 matching passages with their approximate position in the text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find in the transcript.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_ocr_at_timestamp",
        "description": (
            "Get OCR text from frames near a specific timestamp in the lecture. "
            "Returns OCR text from frames within the specified time window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp_seconds": {
                    "type": "number",
                    "description": "The timestamp in seconds to search around.",
                },
                "window_seconds": {
                    "type": "number",
                    "description": "The time window (in seconds) around the timestamp to search.",
                    "default": 60,
                },
            },
            "required": ["timestamp_seconds"],
        },
    },
    {
        "name": "get_figure_at_timestamp",
        "description": (
            "Get the closest figure to a given timestamp. "
            "Returns the latex_path of the figure closest to the specified time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp_seconds": {
                    "type": "number",
                    "description": "The timestamp in seconds to find the nearest figure.",
                },
            },
            "required": ["timestamp_seconds"],
        },
    },
]


def _execute_tool(
    tool_name: str,
    tool_input: dict,
    transcript: str,
    merged_data: list[dict],
    figures: list[dict],
) -> str:
    """Execute a tool call and return the result as a string."""
    if tool_name == "search_transcript":
        query = tool_input.get("query", "").lower()
        words = query.split()
        # Split transcript into overlapping chunks and score them
        transcript_words = transcript.split()
        chunk_size = 100
        step = 50
        scored = []
        for i in range(0, len(transcript_words), step):
            chunk = " ".join(transcript_words[i:i + chunk_size])
            chunk_lower = chunk.lower()
            score = sum(1 for w in words if w in chunk_lower)
            if score > 0:
                position_pct = round(100 * i / max(len(transcript_words), 1))
                scored.append((score, position_pct, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]
        if not top:
            return "No matching passages found for the query."
        results = []
        for score, pos, chunk in top:
            results.append(f"[Position: ~{pos}% through lecture]\n{chunk}")
        return "\n\n---\n\n".join(results)

    elif tool_name == "get_ocr_at_timestamp":
        ts = tool_input.get("timestamp_seconds", 0)
        window = tool_input.get("window_seconds", 60)
        matching = [
            entry for entry in merged_data
            if abs(entry.get("timestamp_sec", 0) - ts) <= window
        ]
        if not matching:
            return f"No OCR data found within {window}s of timestamp {ts}s."
        ocr_texts = []
        for entry in matching:
            ocr = entry.get("ocr_text", "").strip()
            if ocr:
                t = entry.get("timestamp_sec", 0)
                mins = int(t // 60)
                secs = int(t % 60)
                ocr_texts.append(f"[{mins:02d}:{secs:02d}] {ocr}")
        if not ocr_texts:
            return f"Frames found near {ts}s but no OCR text extracted."
        return "\n".join(ocr_texts)

    elif tool_name == "get_figure_at_timestamp":
        ts = tool_input.get("timestamp_seconds", 0)
        if not figures:
            return "No figures available."
        closest = min(figures, key=lambda f: abs(f["timestamp"] - ts))
        dist = abs(closest["timestamp"] - ts)
        return (
            f"Closest figure ({dist:.0f}s away):\n"
            f"  latex_path: {closest['latex_path']}\n"
            f"  caption: {closest['caption']}\n"
            f"  timestamp: {closest['timestamp']:.1f}s"
        )

    return f"Unknown tool: {tool_name}"


# ── LLM Backends ─────────────────────────────────────────────────────────────

def _generate_claude(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 16000,
    use_tools: bool = False,
    transcript: str = "",
    merged_data: list[dict] = None,
    figures: list[dict] = None,
) -> str:
    import anthropic
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("No Anthropic API key found.")
    client = anthropic.Anthropic(api_key=key)

    if use_tools:
        # Non-streaming mode with tool use
        console.print(f"[cyan]Generating notes via Claude ({model}) with tool use...[/cyan]")
        messages = [{"role": "user", "content": prompt}]

        while True:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
            }
            response = client.messages.create(**kwargs)

            # Check if response contains tool use blocks
            has_tool_use = any(
                block.type == "tool_use" for block in response.content
            )

            if not has_tool_use:
                # Final response — extract text
                text_parts = [
                    block.text for block in response.content
                    if block.type == "text"
                ]
                full_response = "".join(text_parts)
                print(full_response[:200] + "..." if len(full_response) > 200 else full_response)
                return full_response

            # Process tool use blocks
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    console.print(f"[dim]  Tool call: {block.name}({json.dumps(block.input)})[/dim]")
                    result = _execute_tool(
                        block.name,
                        block.input,
                        transcript,
                        merged_data or [],
                        figures or [],
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
    else:
        # Streaming mode (original behavior)
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


def _call_backend(
    prompt: str,
    backend: str,
    cfg: dict,
    use_tools: bool = False,
    transcript: str = "",
    merged_data: list[dict] = None,
    figures: list[dict] = None,
) -> str:
    if backend == "claude":
        return _generate_claude(
            prompt,
            model=cfg.get("model", "claude-sonnet-4-20250514"),
            api_key=cfg.get("api_key", ""),
            max_tokens=cfg.get("max_tokens", 16000),
            use_tools=use_tools,
            transcript=transcript,
            merged_data=merged_data,
            figures=figures,
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


def _auto_fix_latex(
    tex_path: Path,
    latex_content: str,
    errors: str,
    backend: str,
    cfg: dict,
) -> str | None:
    """
    Send LaTeX errors to Claude for auto-fix. Returns fixed LaTeX or None.
    """
    fix_prompt = (
        "The following LaTeX errors occurred when compiling these notes. "
        "Fix ONLY the errors listed and return the complete corrected .tex file:\n\n"
        f"ERRORS:\n{errors}\n\n"
        f"LATEX:\n{latex_content}"
    )

    console.print("[cyan]Attempting auto-fix of LaTeX errors...[/cyan]")

    try:
        if backend == "claude":
            import anthropic
            key = cfg.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                return None
            client = anthropic.Anthropic(api_key=key)
            response = client.messages.create(
                model=cfg.get("model", "claude-sonnet-4-20250514"),
                max_tokens=cfg.get("max_tokens", 16000),
                system="You are a LaTeX expert. Fix the compilation errors and return the complete corrected .tex file. Output ONLY the LaTeX code, nothing else.",
                messages=[{"role": "user", "content": fix_prompt}],
            )
            return _clean_latex(response.content[0].text)
        elif backend == "openai":
            from openai import OpenAI
            key = cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                return None
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model=cfg.get("model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "You are a LaTeX expert. Fix the compilation errors and return the complete corrected .tex file. Output ONLY the LaTeX code, nothing else."},
                    {"role": "user", "content": fix_prompt},
                ],
                max_tokens=cfg.get("max_tokens", 16000),
            )
            return _clean_latex(response.choices[0].message.content)
        else:
            # Ollama or unknown backend — skip auto-fix
            return None
    except Exception as e:
        console.print(f"[yellow]⚠ Auto-fix API call failed: {e}[/yellow]")
        return None


def compile_pdf(
    tex_path: Path,
    pdf_output_dir: Path,
    course_name: str,
    lecture_date: str,
    suffix: str = None,
    auto_fix: bool = False,
    backend: str = "claude",
    backend_config: dict = None,
) -> Path | None:
    """
    Compile the .tex file and save the PDF to pdf_output_dir with a descriptive filename.
    The .tex is compiled in its own directory (latex/) then the PDF is copied to notes/.
    If auto_fix is True and compilation fails, attempt to fix errors with an LLM call.
    """
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["pdflatex", "-interaction=nonstopmode",
           "-output-directory", str(tex_path.parent),
           str(tex_path)]

    def _run_pdflatex() -> tuple[bool, str]:
        """Run pdflatex twice. Returns (success, error_text)."""
        try:
            for _ in range(2):
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    # Read the .log file for error details
                    log_path = tex_path.with_suffix(".log")
                    error_lines = []
                    if log_path.exists():
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                        for line in log_text.splitlines():
                            if line.startswith("!"):
                                error_lines.append(line)
                    return False, "\n".join(error_lines) if error_lines else "Unknown compilation error"
            return True, ""
        except FileNotFoundError:
            return False, "pdflatex not found"

    success, errors = _run_pdflatex()

    if not success and auto_fix and errors != "pdflatex not found":
        console.print(f"[yellow]⚠ LaTeX compilation failed. Errors:[/yellow]")
        for line in errors.splitlines()[:10]:
            console.print(f"  [dim]{line}[/dim]")

        latex_content = tex_path.read_text(encoding="utf-8")
        fixed = _auto_fix_latex(tex_path, latex_content, errors, backend, backend_config or {})

        if fixed:
            tex_path.write_text(fixed, encoding="utf-8")
            console.print("[cyan]Retrying compilation with fixed LaTeX...[/cyan]")
            success, retry_errors = _run_pdflatex()
            if not success:
                console.print(f"[yellow]⚠ Auto-fix compilation also failed: {retry_errors[:200]}[/yellow]")
                # Restore original
                tex_path.write_text(latex_content, encoding="utf-8")
                console.print("[dim]Restored original .tex file[/dim]")
        else:
            console.print("[yellow]⚠ Auto-fix could not generate corrected LaTeX[/yellow]")

    if not success and errors == "pdflatex not found":
        console.print(f"[yellow]⚠ pdflatex not available[/yellow]")
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
    rag_context: str = None,
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

    use_tools = cfg.get("use_tools", False) and backend == "claude"
    auto_fix = cfg.get("auto_fix_latex", False)
    if use_tools:
        console.print("[cyan]Tool use: enabled[/cyan]")
    if auto_fix:
        console.print("[cyan]Auto-fix LaTeX: enabled[/cyan]")

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
        prompt = _build_prompt(
            full_transcript, filtered_ocr, course_name, lecture_date,
            figures, rag_context,
        )
        raw = _call_backend(
            prompt, backend, cfg,
            use_tools=use_tools,
            transcript=full_transcript,
            merged_data=merged_data,
            figures=figures,
        )
        final_latex = _clean_latex(raw)
    else:
        chunks = [words[i:i + MAX_WORDS_PER_CHUNK] for i in range(0, len(words), MAX_WORDS_PER_CHUNK)]
        console.print(f"  Long transcript — splitting into {len(chunks)} chunks...")
        latex_sections = []
        for i, chunk_words in enumerate(chunks):
            console.print(f"  Chunk {i+1}/{len(chunks)}...")
            chunk_text = " ".join(chunk_words)
            # Only pass figures and RAG context to first chunk
            prompt = _build_prompt(
                chunk_text,
                filtered_ocr if i == 0 else "",
                course_name,
                lecture_date,
                figures if i == 0 else None,
                rag_context if i == 0 else None,
            )
            raw = _call_backend(
                prompt, backend, cfg,
                use_tools=use_tools and i == 0,
                transcript=full_transcript,
                merged_data=merged_data,
                figures=figures,
            )
            latex_sections.append(_clean_latex(raw))
        final_latex = _merge_latex_chunks(latex_sections)

    # Always save .tex to output/latex/ (overwritten each time)
    tex_path = output_dir / "lecture_notes.tex"
    tex_path.write_text(final_latex, encoding="utf-8")
    console.print(f"[green]✓ LaTeX saved:[/green] {tex_path}")

    if compile_pdf_flag:
        compile_pdf(
            tex_path, pdf_output_dir, course_name, lecture_date, suffix,
            auto_fix=auto_fix, backend=backend, backend_config=cfg,
        )

    return tex_path
