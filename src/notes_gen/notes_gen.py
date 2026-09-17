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
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console

console = Console()

Backend = Literal["claude", "claude-code", "ollama", "openai"]


SYSTEM_PROMPT = """You are an expert academic note-taker and LaTeX typesetter for university-level engineering and science courses. Your task is to convert a raw lecture transcript (and optionally OCR-extracted slide/blackboard text) into complete, beautiful LaTeX lecture notes that read like a well-written textbook chapter: clear discursive prose as the backbone, with a few colored boxes that highlight the truly key items.

VOICE AND LANGUAGE:
1. Write the notes in the language stated in the request ("Language of the notes"); if none is stated, in the language of the lecture.
2. Use simple, discursive language — short, clear sentences, as a brilliant friend explaining the subject. Always explain WHY before the formalism. Keep full mathematical rigor, but never sound bureaucratic or dry.

CONTENT RULES:
3. Cover the ENTIRE lecture from start to finish — do not skip, summarise, or omit any concept, derivation, or example discussed.
4. A student studying ONLY from these notes should be able to fully understand the lecture without watching the video. When the professor's explanation of a concept is incomplete, rushed, or unclear, expand it with a correct and complete treatment using standard academic knowledge — do not limit yourself to only what was said.
5. Preserve all technical terminology, variable names, and notation exactly as used by the professor.
6. Reconstruct all mathematical expressions from the transcript into proper LaTeX, even if only spoken aloud (e.g. "L over one plus L" -> $\\frac{L(s)}{1+L(s)}$).
7. If OCR text is provided, integrate it with the transcript — it contains formulas or equations written on slides or blackboard that were NOT spoken aloud. Do not duplicate content already present in the transcript.
8. Include all examples, exercises, and numerical cases discussed, no matter how briefly mentioned.
9. Preserve the professor's physical intuitions and motivations — for every formula or result, include a sentence explaining WHY it holds or what it means physically or intuitively, not just WHAT it is.
10. Pay attention to how much time the professor spends on each topic — if the professor repeats, elaborates, or returns to a concept multiple times, treat it as a key concept and give it proportionally more space. Topics mentioned only briefly should be covered concisely.
11. Use smooth prose transitions between sections and subsections — introduce each new concept with a sentence connecting it to what came before, so the notes read as a coherent narrative.

PROSE-FIRST RULES (the most important formatting principle):
12. The backbone of the document is continuous prose: AT LEAST 70% of the content must be ordinary paragraphs OUTSIDE any box. Boxes punctuate the discourse; they never carry it.
13. All derivations, proofs, discussions and worked reasoning go in flowing prose with \\begin{equation} and \\begin{align} — NEVER inside boxes, never as bullet points.
14. Never place two boxes back to back: there must always be at least one full paragraph of prose between consecutive boxes.
15. Box budget per \\section: the definitions/theorems/examples genuinely stated in the lecture; AT MOST one intuizione box (only when the professor gave a real intuition or analogy worth preserving); attenzione boxes ONLY for genuine pitfalls, easily-forgotten hypotheses, or explicit exam warnings; EXACTLY one sintesi box at the very end of each \\section with 3-5 short bullet points recapping it.
16. Minimize \\begin{itemize}/\\begin{enumerate} in prose — only for genuine lists. (Inside sintesi, bullets are expected.)
17. When the lecture introduces relationships between multiple variables, render them as a complete \\begin{tabular} with ALL entries filled in; never leave a table partially filled — reconstruct missing data from context or mark it "?". A table whose cells hold sentences must be a \\begin{tabularx}{\\textwidth}{lXX} (X columns wrap; l/c columns never do and run off the page).
18. Bold key terms on first introduction with \\textbf{}.

STRUCTURE:
19. Organise content into \\section{} and \\subsection{} following the natural flow of the lecture; titles must be informative ("La matrice degli snapshot", not "Parte 2").
20. Open each section with a short prose paragraph contextualising the topic before any formulas or boxes.
21. Start the body with \\maketitle (title and date are provided by the system).

DOCUMENT SKELETON: the preamble (packages, colors, box environments, headers, \\title and \\date) is added by the system — do NOT write it. Output ONLY the document body: start with \\begin{document} followed by \\maketitle, and end with \\end{document}. The box environments below are already defined; their displayed titles are set by the system in the lecture language.

ENVIRONMENT USAGE:
22. Numbered boxes take a short title and a unique lowercase label:
\\begin{definizione}{Matrice degli snapshot}{snapshot}...\\end{definizione}
\\begin{teorema}{Ottimalita della POD}{pod-opt}...\\end{teorema}
\\begin{esempio}{Equazione del calore}{calore}...\\end{esempio}
23. Unnumbered boxes take no arguments: \\begin{intuizione}...\\end{intuizione}, \\begin{attenzione}...\\end{attenzione}, \\begin{sintesi}\\begin{itemize}...\\end{itemize}\\end{sintesi}.
24. Use plain \\begin{lemma} and \\begin{corollario} (no box) for secondary results, so the page does not become a wall of boxes.

LATEX OUTPUT RULES:
25. Output ONLY valid LaTeX — no prose explanation, no markdown, no code fences before or after.
26. Begin with \\begin{document} (no preamble) and end with \\end{document}.
27. Always wrap \\begin{cases} inside math mode: \\[ \\begin{cases}...\\end{cases} \\] or $\\begin{cases}...\\end{cases}$ — never outside math mode.
28. Never use Unicode subscripts or superscripts (₁₂₃⁰¹²) — always use LaTeX math notation: $\\text{Ni}_3\\text{Ti}$, $\\text{CO}_2$.
29. Never use % characters in \\section/\\subsection titles or box titles; escape special characters (&, %, #, _) in text.
"""



# ── Preambolo fisso, generato dal codice (il modello scrive solo il corpo) ────
LANG_NAMES = {"it": "Italian", "en": "English", "fr": "French", "de": "German", "es": "Spanish"}

# Titoli dei box nella lingua della lezione (rilevata da Whisper); default italiano.
BOX_TITLES = {
    "it": {"definizione": "Definizione", "teorema": "Teorema", "esempio": "Esempio", "intuizione": "Intuizione",
           "attenzione": "Attenzione", "sintesi": "In sintesi", "lemma": "Lemma", "corollario": "Corollario"},
    "en": {"definizione": "Definition", "teorema": "Theorem", "esempio": "Example", "intuizione": "Intuition",
           "attenzione": "Warning", "sintesi": "In summary", "lemma": "Lemma", "corollario": "Corollary"},
}

# lmodern: font vettoriali. Senza, con [T1]{fontenc} e senza cm-super, pdflatex ripiega sui bitmap
# Type 3 (testo sottile/frastagliato, non copiabile). NB: il template è %-formattato: niente "%" qui.
PREAMBLE_TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs,array,multirow,tabularx}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{titlesec}
\usepackage{fancyhdr}
\definecolor{noteblue}{HTML}{185FA5}
\definecolor{notebluebg}{HTML}{E6F1FB}
\definecolor{noteteal}{HTML}{0F6E56}
\definecolor{notetealbg}{HTML}{E1F5EE}
\definecolor{noteamber}{HTML}{854F0B}
\definecolor{noteamberbg}{HTML}{FAEEDA}
\definecolor{notepurple}{HTML}{534AB7}
\definecolor{notepurplebg}{HTML}{EEEDFE}
\definecolor{notered}{HTML}{A32D2D}
\definecolor{noteredbg}{HTML}{FCEBEB}
\definecolor{notegray}{HTML}{444441}
\definecolor{notegraybg}{HTML}{F1EFE8}
\titleformat{\section}{\Large\bfseries\color{noteblue}}{\thesection}{1em}{}[{\color{noteblue}\titlerule[1.2pt]}]
\titleformat{\subsection}{\large\bfseries\color{noteblue}}{\thesubsection}{1em}{}
\newtcbtheorem[number within=section]{definizione}{%(definizione)s}{enhanced,breakable,colback=notebluebg,colframe=noteblue,colbacktitle=notebluebg,coltitle=noteblue,fonttitle=\bfseries,boxrule=0pt,leftrule=3pt,titlerule=0pt,arc=0pt}{def}
\newtcbtheorem[number within=section]{teorema}{%(teorema)s}{enhanced,breakable,colback=notetealbg,colframe=noteteal,colbacktitle=notetealbg,coltitle=noteteal,fonttitle=\bfseries,boxrule=0pt,leftrule=3pt,titlerule=0pt,arc=0pt}{teo}
\newtcbtheorem[number within=section]{esempio}{%(esempio)s}{enhanced,breakable,colback=noteamberbg,colframe=noteamber,colbacktitle=noteamberbg,coltitle=noteamber,fonttitle=\bfseries,boxrule=0pt,leftrule=3pt,titlerule=0pt,arc=0pt}{ex}
\newtcolorbox{intuizione}{enhanced,breakable,colback=notepurplebg,colframe=notepurple,colbacktitle=notepurplebg,coltitle=notepurple,fonttitle=\bfseries,boxrule=0pt,leftrule=3pt,titlerule=0pt,arc=0pt,title=%(intuizione)s}
\newtcolorbox{attenzione}{enhanced,breakable,colback=noteredbg,colframe=notered,colbacktitle=noteredbg,coltitle=notered,fonttitle=\bfseries,boxrule=0pt,leftrule=3pt,titlerule=0pt,arc=0pt,title=%(attenzione)s}
\newtcolorbox{sintesi}{enhanced,breakable,colback=notegraybg,colframe=notegraybg,colbacktitle=notegraybg,coltitle=notegray,fonttitle=\bfseries,boxrule=0pt,titlerule=0pt,arc=2pt,title=%(sintesi)s}
\theoremstyle{plain}
\newtheorem{lemma}{%(lemma)s}[section]
\newtheorem{corollario}[lemma]{%(corollario)s}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape %(course)s}
\fancyhead[R]{\small\itshape %(date)s}
\fancyfoot[C]{\small--- \thepage\ ---}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\headheight}{14pt}
\title{%(course)s}
\author{}
\date{%(date)s}
"""


def _tex_escape(t: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", t)


def build_preamble(course_name: str, lecture_date: str, language: str | None = None) -> str:
    titles = BOX_TITLES.get((language or "it")[:2].lower(), BOX_TITLES["en"])
    return PREAMBLE_TEMPLATE % {**titles, "course": _tex_escape(course_name), "date": _tex_escape(lecture_date)}


def assemble_document(body: str, course_name: str, lecture_date: str, language: str | None = None) -> str:
    """Corpo prodotto dal modello (con o senza preambolo/\\begin{document}) → documento completo."""
    body = body.strip()
    if "\\begin{document}" in body:
        body = body[body.index("\\begin{document}"):]           # scarta un eventuale preambolo del modello
    else:
        body = "\\begin{document}\n\\maketitle\n" + body
    if "\\end{document}" not in body:
        body += "\n\\end{document}\n"
    if "\\maketitle" not in body[:400]:
        body = body.replace("\\begin{document}", "\\begin{document}\n\\maketitle", 1)
    return build_preamble(course_name, lecture_date, language) + "\n" + body + "\n"


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
    course_profile: str = None,
    slides_text: str = None,
    language: str = None,
) -> str:
    lang_line = f"Language of the notes: {LANG_NAMES.get(language[:2].lower(), language)}\n" if language else ""
    prompt = f"""Convert the following lecture transcript into complete, comprehensive LaTeX notes.
Write with a bookish, refined academic style — not a transcript dump, but polished notes a student would enjoy reading.
Cover EVERY topic discussed — do not skip or summarise any part of the lecture.

Course: {course_name}
Date: {lecture_date}
{lang_line}
--- FULL TRANSCRIPT ---
{transcript}
"""
    if re.search(r"\[\d\d:\d\d\]", transcript):
        prompt += """(The [mm:ss] markers give the elapsed lecture time: use them only to place slides and figures
where they were shown. Never reproduce them in the notes.)
"""
    if ocr_filtered.strip():
        prompt += f"""
--- ADDITIONAL MATHEMATICAL CONTENT FROM SLIDES/BLACKBOARD ---
(These are formulas and equations written by the professor that were NOT spoken aloud.
Integrate them naturally into the notes where contextually appropriate.)
{ocr_filtered}
"""
    if slides_text:
        prompt += f"""
--- LECTURE SLIDES (text extracted page by page, in order) ---
(The transcript is the primary source for WHAT was said; the slides are the authority for HOW it is
written: use them to correct transcription errors in technical terms, proper names, symbols and
formulas, to recover the exact wording of definitions, and to follow the professor's own structure.
The deck may cover more, or other, material than this lecture: use ONLY the parts that correspond to
what was actually said. Never add a section, topic or example that appears only in the slides.
Do not turn the notes into a copy of the slides: keep the explanatory prose of the lecture.)
{slides_text}
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
            if "slide" in fig:
                where = f"[{fig['deck']} · slide {fig['slide']}]" if fig.get("deck") else f"[slide {fig['slide']}]"
                if fig.get("timestamp") is not None:
                    where = where[:-1] + f" · shown at {int(fig['timestamp'] // 60):02d}:{int(fig['timestamp'] % 60):02d}]"
            else:
                mins = int(fig["timestamp"] // 60)
                secs = int(fig["timestamp"] % 60)
                where = f"[{mins:02d}:{secs:02d}]"
            prompt += f"{where} latex_path={fig['latex_path']} caption={fig['caption']}\n"
        prompt += ("\nCRITICAL: Use ONLY the exact latex_path values listed above. Never invent or modify figure filenames. "
                   f"These are CANDIDATES, not a list to include: use at most {max(1, len(figures) // 2)} of them, and a figure only "
                   "where the transcript explicitly discusses what it shows (a diagram, plot, scheme, organism or device the "
                   "professor talked about). Do include the ones that match a discussed topic — the notes should have at "
                   "least one figure whenever a candidate fits — but never justify a figure with a caption. "
                   "Write a caption that explains what the figure shows in the context of the lecture, not the slide title.\n")

    if rag_context:
        prompt += f"""
--- CONTEXT FROM COURSE MATERIAL (handouts and previous lectures) ---
(Use this to maintain consistency with terminology, notation, and concepts.
Reference prior material where appropriate.

IMPORTANT — FIGURES: If the context contains a marker like [FIGURE: output/rag/figures/X.png],
a figure image exists at that path. Include it in the LaTeX notes near the relevant topic:

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.75\\textwidth]{{../rag/figures/X.png}}
\\caption{{Description of what the figure shows}}
\\end{{figure}}

Path conversion: "output/rag/figures/X.png" → "../rag/figures/X.png"
(pdflatex runs from output/latex/, so the relative path climbs one level).)
{rag_context}
"""

    if course_profile:
        prompt += f"""
--- COURSE STYLE GUIDE ---
(Terminology, notation and LaTeX conventions for this course.
Follow them strictly so notation stays consistent across all lectures.)
{course_profile}
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
            import httpx
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "thinking": {"type": "adaptive"},
                "timeout": httpx.Timeout(timeout=900.0, connect=30.0),
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
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                print(text, end="", flush=True)
        print()
        return full_response


# ── Claude Code CLI backend ──────────────────────────────────────────────────
# Runs `claude -p` headless: same SYSTEM_PROMPT and prompt as the API backend,
# but billed to the Claude subscription instead of an API key. No tools are
# exposed, so it is a pure text-generation call.

_CLAUDE_SESSION_VARS = (
    "CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ATTENDED",
)


def _claude_bin() -> str:
    import shutil
    for cand in (os.environ.get("CLAUDE_BIN"), shutil.which("claude"),
                 str(Path.home() / ".local" / "bin" / "claude")):
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError("claude CLI not found (set CLAUDE_BIN)")


LAST_USAGE: dict = {}          # usage dell'ultima chiamata claude -p (token, costo, modello, durata)
USAGE_LOG = Path("output/auto/usage.jsonl")


class ClaudeRateLimited(RuntimeError):
    pass


RETRY_WAITS = (60, 180, 600)   # secondi di attesa tra i tentativi su rate limit / errori API


def run_claude_json(prompt: str, system: str, model: str, timeout: int, tools: str = "",
                    extra_args: list[str] | None = None) -> dict:
    """
    Esegue `claude -p` (JSON) e ritorna il dict di risposta. Riprova su rate limit / errori API
    (429, 5xx, overloaded) con attese crescenti; solleva RuntimeError con il messaggio vero.
    """
    cmd = [_claude_bin(), "-p", "--tools", tools, "--output-format", "json",
           "--no-session-persistence", "--model", model, "--system-prompt", system,
           # niente server MCP: risparmia ~14k token di definizioni per chiamata e non li avvia
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', *(extra_args or [])]
    env = {k: v for k, v in os.environ.items() if k not in _CLAUDE_SESSION_VARS}
    last_err = ""
    for attempt in range(len(RETRY_WAITS) + 1):
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=env)
        data = None
        try:
            data = json.loads(result.stdout) if result.stdout.strip() else None
        except json.JSONDecodeError:
            data = None
        if data is not None and result.returncode == 0 and not data.get("is_error"):
            return data
        msg = (str(data.get("result")) if data else "") or result.stderr.strip() or result.stdout.strip()[:300]
        status = data.get("api_error_status") if data else None
        last_err = f"exit {result.returncode}, api_status={status}: {msg[:400]}"
        retryable = status in (429, 500, 502, 503, 529) or re.search(
            r"rate.?limit|overloaded|usage limit|too many requests|529|503", msg, re.I)
        if attempt < len(RETRY_WAITS) and (retryable or (result.returncode != 0 and not msg)):
            wait = RETRY_WAITS[attempt]
            console.print(f"[yellow]  claude -p: {last_err[:120]} — riprovo tra {wait}s[/yellow]")
            time.sleep(wait)
            continue
        break
    raise (ClaudeRateLimited if re.search(r"rate.?limit|usage limit|429", last_err, re.I) else RuntimeError)(
        f"claude -p: {last_err}")


def _claude_code_call(system: str, prompt: str, model: str, timeout: int, purpose: str = "notes",
                      effort: str | None = None) -> str:
    t0 = time.time()
    data = run_claude_json(prompt, system, model, timeout,
                           extra_args=(["--effort", effort] if effort else None))
    out = (data.get("result") or "").strip()
    if not out:
        raise RuntimeError("claude -p returned no output")

    # contabilità: modello principale (il CLI usa anche haiku per piccole cose interne)
    mu = data.get("modelUsage") or {}
    main = max(mu.items(), key=lambda kv: kv[1].get("costUSD", 0))[0] if mu else model
    u = data.get("usage") or {}
    LAST_USAGE.clear()
    LAST_USAGE.update({
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "purpose": purpose, "model": main,
        "in": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
        "cache_read": u.get("cache_read_input_tokens", 0), "cache_write": u.get("cache_creation_input_tokens", 0),
        "out": u.get("output_tokens", 0),
        "thinking": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        "effort": effort or "default",
        "cost_usd": round(data.get("total_cost_usd") or 0, 4),
        "seconds": round(time.time() - t0), "chars": len(out),
    })
    try:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(LAST_USAGE) + "\n")
    except OSError:
        pass
    return out


def _generate_claude_code(prompt: str, model: str = "sonnet", timeout: int = 1800,
                          effort: str | None = None, system: str | None = None,
                          lecture_minutes: float | None = None) -> str:
    console.print(f"[cyan]Generating notes via Claude Code CLI (model={model}, effort={effort or 'default'})...[/cyan]")
    # NB: effort low/medium azzera il thinking ma condensa le note (-22%/-48% di prosa, tabelle perse):
    # misurato il 2026-09-16, incompatibile con la completezza richiesta. Lasciato come opzione esplicita.
    out = _claude_code_call(system or SYSTEM_PROMPT, prompt, model, timeout, purpose="notes", effort=effort)
    u = LAST_USAGE
    console.print(f"[dim]  {u.get('model')}: {u.get('in')} in (cache {u.get('cache_read')}) / {u.get('out')} out "
                  f"(thinking {u.get('thinking')}) tokens, ${u.get('cost_usd')} eq, {u.get('seconds')}s, {len(out)} chars[/dim]")
    return out


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
            model=cfg.get("model", "claude-sonnet-4-6"),
            api_key=cfg.get("api_key", ""),
            max_tokens=cfg.get("max_tokens", 16000),
            use_tools=use_tools,
            transcript=transcript,
            merged_data=merged_data,
            figures=figures,
        )
    elif backend == "claude-code":
        return _generate_claude_code(
            prompt,
            model=cfg.get("model", "sonnet"),
            timeout=cfg.get("timeout", 1800),
            effort=cfg.get("effort"),
            system=cfg.get("_system_prompt"),
            lecture_minutes=cfg.get("_lecture_minutes"),
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

    # 1) patch minima (poche centinaia di token in uscita invece di riscrivere ~20k token di documento)
    if backend == "claude-code":
        patched = _patch_fix_latex(latex_content, errors, cfg)
        if patched:
            return patched
        console.print("[yellow]  Patch non applicabile: riscrittura completa[/yellow]")

    try:
        if backend == "claude":
            import anthropic
            key = cfg.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                return None
            client = anthropic.Anthropic(api_key=key)
            # Streaming: a full corrected .tex can exceed what a non-streaming
            # request can return before the SDK's HTTP timeout.
            fixed = ""
            with client.messages.stream(
                model=cfg.get("model", "claude-sonnet-4-6"),
                max_tokens=cfg.get("max_tokens", 16000),
                system="You are a LaTeX expert. Fix the compilation errors and return the complete corrected .tex file. Output ONLY the LaTeX code, nothing else.",
                messages=[{"role": "user", "content": fix_prompt}],
            ) as stream:
                for text in stream.text_stream:
                    fixed += text
            return _clean_latex(fixed)
        elif backend == "claude-code":
            fixed = _claude_code_call(
                "You are a LaTeX expert. Fix the compilation errors and return the complete corrected .tex file. Output ONLY the LaTeX code, nothing else.",
                fix_prompt, cfg.get("model", "sonnet"), cfg.get("timeout", 1800), purpose="latex-fix",
            )
            return _clean_latex(fixed)
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


PATCH_SYSTEM = ("You are a LaTeX expert. You receive a document and its pdflatex errors. Reply ONLY with a JSON "
                "array of minimal edits, each {\"find\": <exact substring of the document, 1-3 lines, unique>, "
                "\"replace\": <corrected text>}. Fix only the listed errors; never rewrite or reflow other content; "
                "never change wording, only LaTeX syntax. No prose, no code fences.")


def _patch_fix_latex(latex: str, errors: str, cfg: dict) -> str | None:
    """Chiede a Claude solo le sostituzioni minime e le applica; None se non applicabili."""
    prompt = f"ERRORS:\n{errors}\n\nDOCUMENT:\n{latex}"
    try:
        raw = _claude_code_call(PATCH_SYSTEM, prompt, cfg.get("model", "sonnet"), cfg.get("timeout", 1800),
                                purpose="latex-patch")
        m = re.search(r"\[.*\]", raw, re.S)
        edits = json.loads(m.group(0)) if m else None
    except Exception as e:
        console.print(f"[yellow]  Patch fix failed: {type(e).__name__}: {str(e)[:120]}[/yellow]")
        return None
    if not isinstance(edits, list) or not edits:
        return None
    out = latex
    applied = 0
    for e in edits:
        f, r = e.get("find", ""), e.get("replace", "")
        if not f or out.count(f) != 1:
            continue
        out = out.replace(f, r, 1)
        applied += 1
    if applied == 0:
        return None
    # la patch non deve toccare il contenuto: il testo fuori dal LaTeX deve restare (quasi) identico
    before, after = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", latex), re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", out)
    if abs(len(after) - len(before)) > 0.02 * len(before):
        console.print("[yellow]  Patch scartata: modifica il contenuto oltre la soglia[/yellow]")
        return None
    console.print(f"[cyan]  Applied {applied}/{len(edits)} LaTeX patch edit(s)[/cyan]")
    return out


def _repair_figure_paths(latex: str, output_dir: Path) -> str:
    """Make every ``\\includegraphics{../rag/figures/...}`` reference resolvable.

    Two failure modes are handled, both of which make pdflatex silently fall
    back to its *draft* setting (a boxed path instead of the image) while still
    returning 0, so the auto-fix path never fires and the broken figure ships:

    1. The LLM alters the filename when copying it out of the RAG markers — most
       often collapsing a run of spaces to one — so the named file is missing.
    2. Even with the exact name, TeX's input tokenizer collapses *consecutive*
       spaces to a single space, so a file whose name contains two spaces (e.g.
       a source PDF "... Force Fields  Stability ...") can never be addressed by
       ``\\includegraphics`` at all — TeX asks the OS for the single-space name,
       which does not exist.

    For each ``rag/figures`` reference we locate the real source file (exact
    match, else a whitespace-normalised, case-insensitive match). If that name
    cannot be expressed verbatim in TeX (it contains consecutive spaces) we
    materialise a TeX-safe copy beside it and point the reference there. Single
    spaces survive tokenisation and are left as-is; non-rag paths and
    already-correct references are untouched.
    """
    import shutil

    figures_dir = output_dir / "rag" / "figures"
    if not figures_dir.is_dir():
        return latex

    def _norm(name: str) -> str:
        return re.sub(r"\s+", " ", name).strip().lower()

    def _safe(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    real_by_norm: dict[str, str] = {}
    for p in figures_dir.iterdir():
        if p.is_file():
            real_by_norm.setdefault(_norm(p.name), p.name)

    pattern = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]*)(\})")
    fixed: list[str] = []

    def _fix(m: "re.Match") -> str:
        head, path, tail = m.group(1), m.group(2), m.group(3)
        if "rag/figures/" not in path:
            return m.group(0)
        fname = path.rsplit("/", 1)[-1]
        real = fname if (figures_dir / fname).exists() else real_by_norm.get(_norm(fname))
        if not real:
            return m.group(0)
        if re.search(r"\s{2,}", real):
            target = _safe(real)
            if not (figures_dir / target).exists():
                try:
                    shutil.copyfile(figures_dir / real, figures_dir / target)
                except OSError:
                    return m.group(0)
        else:
            target = real
        if target == fname:
            return m.group(0)  # already correct and TeX-safe
        dirpart = path[: len(path) - len(fname)]
        fixed.append(target)
        return f"{head}{dirpart}{target}{tail}"

    out = pattern.sub(_fix, latex)
    if fixed:
        console.print(
            f"[cyan]  Made {len(fixed)} figure path(s) TeX-safe so they embed correctly[/cyan]"
        )
    return out


# macro non definite → pacchetto che le fornisce (fix senza LLM)
_MACRO_PACKAGES = {
    "booktabs": ("toprule", "midrule", "bottomrule", "cmidrule", "addlinespace"),
    "multirow": ("multirow",),
    "siunitx": ("SI", "si", "num", "qty", "unit"),
    "gensymb": ("degree", "celsius", "ohm", "micro"),
    "textcomp": ("texteuro", "textcelsius", "textdegree"),
    "mathtools": ("coloneqq", "prescript", "DeclarePairedDelimiter"),
    "cancel": ("cancel", "bcancel", "xcancel"),
    "bm": ("bm",),
    "physics": ("dv", "pdv", "qty", "abs", "norm", "grad", "curl", "div"),
    "subcaption": ("subfigure", "subcaption"),
    "float": ("newfloat", "floatstyle"),
}


def _quick_fix_latex(latex: str, errors: str) -> str | None:
    """
    Fix deterministici prima di scomodare l'LLM:
      - "File `X.sty' not found" → toglie X dai \\usepackage (e prova a installarlo con tlmgr per la prossima volta)
      - "Undefined control sequence" di macro note → aggiunge il pacchetto che le fornisce
    None se non applicabile.
    """
    missing = re.findall(r"File `([^']+)\.sty' not found", errors)
    if missing:
        def _strip(m: "re.Match") -> str:
            opts, pkgs = m.group(1) or "", m.group(2)
            keep = [x.strip() for x in pkgs.split(",") if x.strip() and x.strip() not in missing]
            if keep == [x.strip() for x in pkgs.split(",") if x.strip()]:
                return m.group(0)                      # riga senza pacchetti mancanti: intatta
            return f"\\usepackage{opts}{{{','.join(keep)}}}" if keep else ""

        fixed = re.sub(r"\\usepackage(\[[^\]]*\])?\{([^}]*)\}", _strip, latex)
        for pkg in missing:                            # per la prossima volta
            try:
                subprocess.run(["tlmgr", "install", pkg], capture_output=True, timeout=120)
            except Exception:
                pass
        console.print(f"[cyan]  Removed missing package(s) {missing} from the preamble[/cyan]")
        return fixed if fixed != latex else None
    if "Undefined control sequence" not in errors:
        return None
    undefined = set(re.findall(r"\\([A-Za-z]+)", errors))
    needed = [pkg for pkg, macros in _MACRO_PACKAGES.items()
              if any(m in undefined for m in macros) and not re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*\b" + pkg + r"\b", latex)]
    if not needed:
        return None
    line = "\\usepackage{" + ",".join(needed) + "}\n"
    m = re.search(r"\\usepackage\{amsmath[^}]*\}\n", latex)
    if m:
        return latex[:m.end()] + line + latex[m.end():]
    m = re.search(r"\\documentclass[^\n]*\n", latex)
    return latex[:m.end()] + line + latex[m.end():] if m else None


TEXT_COLS = 82          # caratteri per riga a 11pt con margini 2.5 cm su A4 (~455pt): oltre, la tabella sborda


def _fix_wide_tables(latex: str) -> str:
    """
    tabular con celle di prosa → tabularx a \textwidth con colonne X: LaTeX non manda a capo
    le colonne l/c, quindi una tabella "aspetto / opzione A / opzione B" con frasi intere esce
    dalla pagina (successo il 17/09/2026: tre tabelle su tre). Stima della larghezza = somma,
    per colonna, della cella più lunga; sopra TEXT_COLS si convertono in X le colonne la cui
    cella più lunga supera 20 caratteri (le etichette corte restano l).
    """
    def cells_of(body: str) -> list[list[str]]:
        rows = []
        for line in re.split(r"\\\\", body):
            line = re.sub(r"\\(toprule|midrule|bottomrule|hline|cline\{[^}]*\})", "", line).strip()
            if line:
                rows.append([c.strip() for c in line.split("&")])
        return rows

    def repl(m: re.Match) -> str:
        spec, body = m.group(1), m.group(2)
        cols = re.findall(r"[lcr]|p\{[^}]*\}|X", spec.replace("|", ""))
        rows = cells_of(body)
        if not rows or not cols or "X" in cols:
            return m.group(0)
        n = len(cols)
        longest = [max((len(re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^}]*\})?", "", r[i])) for r in rows if i < len(r)), default=0)
                   for i in range(n)]
        if sum(longest) + 3 * n <= TEXT_COLS:
            return m.group(0)
        new_cols = ["X" if (c in ("l", "c", "r") and longest[i] > 20) else c for i, c in enumerate(cols)]
        if "X" not in new_cols:
            return m.group(0)
        return f"\\begin{{tabularx}}{{\\textwidth}}{{{''.join(new_cols)}}}{body}\\end{{tabularx}}"

    return re.sub(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", repl, latex, flags=re.S)


LAST_LAYOUT: dict = {}         # esito dell'ultimo controllo di impaginazione (compile_pdf)


def _layout_check(log_path: Path, min_pt: float = 5.0) -> dict:
    """Conta gli "Overfull hbox" del log di pdflatex sopra min_pt: righe/tabelle che escono dal margine."""
    out = {"overfull": 0, "worst_pt": 0.0}
    if not log_path.exists():
        return out
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide", log_path.read_text(errors="replace")):
        pt = float(m.group(1))
        if pt >= min_pt:
            out["overfull"] += 1
            out["worst_pt"] = max(out["worst_pt"], pt)
    return out


def _drop_missing_figures(latex: str, output_dir: Path) -> str:
    """Rimuove i blocchi figure il cui file non esiste (pdflatex li renderebbe come box vuoti)."""
    pattern = re.compile(r"\\begin\{figure\}.*?\\end\{figure\}", re.S)
    dropped = []

    def _check(m: "re.Match") -> str:
        block = m.group(0)
        for g in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", block):
            path = g.group(1).strip()
            if not (output_dir / path).exists() and not Path(path).exists():
                dropped.append(path)
                return ""
        return block

    out = pattern.sub(_check, latex)
    if dropped:
        console.print(f"[yellow]  Dropped {len(dropped)} figure(s) with missing files: {dropped[:3]}[/yellow]")
    return out


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
                        lines = log_text.splitlines()
                        for i, line in enumerate(lines):
                            if line.startswith("!"):
                                ctx = next((l for l in lines[i + 1:i + 6] if l.startswith("l.")), "")
                                error_lines.append(f"{line}  {ctx}".rstrip())
                    return False, "\n".join(error_lines) if error_lines else "Unknown compilation error"
            return True, ""
        except FileNotFoundError:
            return False, "pdflatex not found"

    success, errors = _run_pdflatex()

    if not success and errors != "pdflatex not found":
        original = tex_path.read_text(encoding="utf-8")
        quick = _quick_fix_latex(original, errors)
        if quick and quick != original:
            tex_path.write_text(quick, encoding="utf-8")
            console.print("[cyan]Retrying compilation after deterministic fix (missing packages)...[/cyan]")
            success, errors = _run_pdflatex()
            if not success:
                tex_path.write_text(original, encoding="utf-8")

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

    LAST_LAYOUT.clear()
    LAST_LAYOUT.update(_layout_check(tex_path.with_suffix(".log")))
    if LAST_LAYOUT["overfull"]:
        console.print(f"[yellow]⚠ Layout: {LAST_LAYOUT['overfull']} overfull box(es), "
                      f"worst {LAST_LAYOUT['worst_pt']:.0f}pt beyond the margin[/yellow]")

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
    transcript_text: str = None,
    pdf_output_dir: Path = None,
    figures: list[dict] = None,
    suffix: str = None,
    rag_context: str = None,
    slides_text: str = None,
    language: str = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = backend_config or {}
    lecture_minutes = None
    if transcript_path is not None:
        seg = transcript_path.with_name(transcript_path.stem + "_segments.json")
        if seg.exists():
            try:
                segd = json.loads(seg.read_text(encoding="utf-8"))
                language = language or segd.get("language")
                if segd.get("segments"):
                    lecture_minutes = segd["segments"][-1]["end"] / 60
            except Exception:
                pass
    if lecture_minutes is None and transcript_path is not None:
        lecture_minutes = len(transcript_path.read_text(encoding="utf-8").split()) / 125   # ~125 parole/min parlate
    cfg = dict(cfg, _lecture_minutes=lecture_minutes)

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
    if transcript_text:
        full_transcript = transcript_text            # es. con marcatori [mm:ss] (tools/nightly.py)
        console.print(f"[dim]Transcript given: {len(full_transcript.split())} words[/dim]")
    elif transcript_path and transcript_path.exists():
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
    # claude-sonnet-4-6 has a 1M-token context window: an entire lecture fits
    # in a single call, so chunking (and the lossy chunk merge) only kicks in
    # for pathologically long transcripts.
    MAX_WORDS_PER_CHUNK = 150000
    words = full_transcript.split()

    # Per-course style guide (terminology, notation, LaTeX conventions)
    try:
        from src.course_profiles import load_profile
        course_profile = load_profile(course_name)
    except Exception:
        course_profile = ""
    if course_profile:
        console.print(f"[dim]Course profile loaded for '{course_name}'[/dim]")

    # claude-code: la scheda corso va nel system prompt (stabile per corso → cache hit tra lezioni
    # consecutive); nel messaggio utente resterebbe dietro la trascrizione, mai riusabile.
    if backend == "claude-code" and course_profile:
        cfg = dict(cfg, _system_prompt=SYSTEM_PROMPT + "\n\n--- COURSE STYLE GUIDE ---\n"
                   "(Terminology, notation and LaTeX conventions for this course. Follow them strictly so "
                   "notation stays consistent across all lectures.)\n" + course_profile + "\n")
        course_profile = None

    if len(words) <= MAX_WORDS_PER_CHUNK:
        console.print(f"  Chunk 1/1...")
        prompt = _build_prompt(
            full_transcript, filtered_ocr, course_name, lecture_date,
            figures, rag_context, course_profile, slides_text, language,
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
            # Only pass figures and RAG context to first chunk; the course
            # profile goes to every chunk to keep notation consistent.
            prompt = _build_prompt(
                chunk_text,
                filtered_ocr if i == 0 else "",
                course_name,
                lecture_date,
                figures if i == 0 else None,
                rag_context if i == 0 else None,
                course_profile,
                slides_text if i == 0 else None,
                language,
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

    # La risposta grezza va su disco PRIMA di qualsiasi post-elaborazione: un bug dopo la
    # chiamata (è successo) non deve costare una seconda chiamata al modello
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lecture_notes_raw.tex").write_text(final_latex, encoding="utf-8")

    # Il preambolo lo mettiamo noi: meno token in uscita e nessun errore di preambolo
    final_latex = assemble_document(final_latex, course_name, lecture_date, language)

    # Repair figure paths that the LLM may have mangled (whitespace, etc.)
    # so pdflatex actually embeds them instead of silently using draft mode.
    final_latex = _repair_figure_paths(final_latex, output_dir)
    final_latex = _drop_missing_figures(final_latex, output_dir)
    final_latex = _fix_wide_tables(final_latex)

    # Always save .tex to output/latex/ (overwritten each time)
    tex_path = output_dir / "lecture_notes.tex"
    tex_path.write_text(final_latex, encoding="utf-8")
    console.print(f"[green]✓ LaTeX saved:[/green] {tex_path}")

    # Archive a per-lecture copy so course_builder can later assemble the
    # whole course into a single cohesive PDF.
    try:
        from src.course_profiles import _slugify
        course_dir = Path("output/course") / _slugify(course_name)
        course_dir.mkdir(parents=True, exist_ok=True)
        archive_name = f"{stem}.tex" if stem.startswith(lecture_date) else f"{lecture_date}_{stem}.tex"
        archive_path = course_dir / archive_name
        archive_path.write_text(final_latex, encoding="utf-8")
        console.print(f"[dim]Archived for course build: {archive_path}[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ Course archive failed: {e}[/yellow]")

    if compile_pdf_flag:
        compile_pdf(
            tex_path, pdf_output_dir, course_name, lecture_date, suffix,
            auto_fix=auto_fix, backend=backend, backend_config=cfg,
        )

    return tex_path
