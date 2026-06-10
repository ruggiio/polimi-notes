"""
course_builder.py — Assemble an entire course into one cohesive PDF

During the semester every lecture's .tex is archived in
output/course/<course-slug>/<date>_<stem>.tex (done automatically by
notes_gen). When the course is over, run:

    python -m src.course_builder "Model Order"

This collects all archived lectures in chronological order and produces a
single book-like document (report class: title page, table of contents,
one chapter per lecture) reusing the same colored preamble as the
per-lecture notes.

By default each chapter goes through a Claude "cohesion pass" that turns
the standalone lecture into a chapter of one coherent book: a transition
from the previous chapters, lecture-speak removed ("nella lezione di
oggi..."), notation unified with the course profile. Use --no-polish to
skip it and get a purely mechanical merge (free, instant).

Options:
    --no-polish   skip the Claude cohesion pass
    --no-pdf      write the .tex but do not compile it
    --config      path to config.yaml (default config/config.yaml)
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console

from src.course_profiles import _slugify, load_profile
from src.notes_gen.notes_gen import SYSTEM_PROMPT, _repair_figure_paths, compile_pdf

console = Console()

COURSE_DIR = Path("output/course")
LATEX_DIR = Path("output/latex")

COHESION_SYSTEM = """You are editing one chapter of a unified LaTeX book assembled from a semester of lecture notes. You receive the LaTeX body of one lecture, plus an outline of the chapters that precede it (and optionally a course style guide).

Rewrite the body so it reads as a chapter of ONE coherent book:
1. The very first line of your output must be a comment: % CHAPTER: <short descriptive chapter title in the language of the notes>
2. Open the chapter with a short transition paragraph connecting it to the preceding chapters (when an outline is provided). If the lecture starts by recapping the previous one, fold that recap into this transition instead of repeating it.
3. Remove every reference to the lecture format: "in today's lecture", "nella lezione di oggi", "la volta scorsa", "come visto la settimana scorsa" -> rephrase in book style ("nel capitolo precedente", "come visto nel Capitolo ...").
4. Unify notation and terminology with the style guide and the preceding chapters' outline.
5. Keep ALL technical content: do not shorten, summarise, or drop anything. Keep every definizione/teorema/esempio/intuizione/attenzione/sintesi box, every equation, every figure (\\includegraphics paths must stay exactly as they are).
6. Do NOT output a preamble, \\documentclass, \\begin{document}, \\end{document}, \\maketitle, or \\chapter — only the chapter body (starting with the % CHAPTER comment, then \\section content).
7. Output ONLY LaTeX, no markdown fences, no commentary."""


def _parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _lecture_files(course_name: str) -> list[Path]:
    """Archived lecture .tex files, sorted chronologically by date prefix."""
    course_dir = COURSE_DIR / _slugify(course_name)
    files = sorted(course_dir.glob("*.tex"))
    def key(p: Path):
        date_part = p.stem.split("_", 1)[0]
        return _parse_date(date_part) or datetime.max
    return sorted(files, key=key)


def _extract_body(tex: str) -> str:
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex, re.DOTALL)
    body = m.group(1) if m else tex
    body = re.sub(r"\\maketitle\s*", "", body)
    body = re.sub(r"^\\(title|date|author)\{.*?\}\s*$", "", body, flags=re.MULTILINE)
    return body.strip()


def _course_preamble(course_name: str, date_range: str) -> str:
    """The per-lecture fixed preamble, promoted to report class with chapters."""
    m = re.search(
        r"(\\documentclass.*?\\setlength\{\\headheight\}\{14pt\})",
        SYSTEM_PROMPT, re.DOTALL,
    )
    if not m:
        raise RuntimeError("fixed preamble not found in SYSTEM_PROMPT")
    preamble = m.group(1)
    preamble = preamble.replace(
        "\\documentclass[11pt,a4paper]{article}",
        "\\documentclass[11pt,a4paper]{report}",
    )
    preamble = preamble.replace("COURSENAME", course_name)
    preamble = preamble.replace("LECTUREDATE", "Appunti del corso")
    # Chapter styling (titlesec is already loaded by the preamble)
    preamble += (
        "\n\\titleformat{\\chapter}[hang]"
        "{\\Huge\\bfseries\\color{noteblue}}{\\thechapter}{1em}{}"
        "\n\\titlespacing*{\\chapter}{0pt}{0pt}{30pt}"
    )
    return preamble


def _title_page(course_name: str, date_range: str) -> str:
    return (
        "\\begin{titlepage}\n\\centering\n\\vspace*{3cm}\n"
        f"{{\\color{{noteblue}}\\Huge\\bfseries {course_name}\\par}}\n"
        "\\vspace{1cm}\n{\\Large Appunti del corso\\par}\n"
        f"\\vspace{{0.5cm}}\n{{\\large {date_range}\\par}}\n"
        "\\vfill\n{\\small Generati automaticamente dalle registrazioni delle lezioni\\par}\n"
        "\\end{titlepage}\n\\tableofcontents\n\\clearpage\n"
    )


def _outline_of(body: str, max_chars: int = 2500) -> str:
    """Compact outline of a chapter: section titles + sintesi bullets."""
    parts = []
    for m in re.finditer(r"\\section\{([^}]*)\}", body):
        parts.append(f"- {m.group(1)}")
    m = re.search(r"\\begin\{sintesi\}(.*?)\\end\{sintesi\}", body, re.DOTALL)
    if m:
        items = re.findall(r"\\item\s+(.+)", m.group(1))
        parts += [f"  * {i.strip()}" for i in items]
    return "\n".join(parts)[:max_chars]


def _claude_stream(system: str, prompt: str, cfg: dict) -> str:
    import anthropic
    key = cfg.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("No Anthropic API key found.")
    client = anthropic.Anthropic(api_key=key)
    out = ""
    with client.messages.stream(
        model=cfg.get("model", "claude-sonnet-4-6"),
        max_tokens=cfg.get("max_tokens", 64000),
        system=system,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            out += text
    return out.strip()


def _clean(raw: str) -> str:
    raw = re.sub(r"^```(?:latex|tex)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _polish_chapter(
    body: str, outlines: list[str], course_profile: str, cfg: dict, n: int
) -> tuple[str, str]:
    """Return (chapter_title, polished_body)."""
    prompt_parts = []
    if course_profile:
        prompt_parts.append(f"--- COURSE STYLE GUIDE ---\n{course_profile}")
    if outlines:
        prompt_parts.append(
            "--- OUTLINE OF PRECEDING CHAPTERS ---\n"
            + "\n\n".join(
                f"Capitolo {i + 1}:\n{o}" for i, o in enumerate(outlines)
            )
        )
    prompt_parts.append(f"--- LECTURE BODY (becomes Chapter {n}) ---\n{body}")
    prompt_parts.append("Rewrite it as instructed. Remember: first line must be the % CHAPTER comment.")
    polished = _clean(_claude_stream(COHESION_SYSTEM, "\n\n".join(prompt_parts), cfg))

    title = ""
    m = re.match(r"%\s*CHAPTER:\s*(.+)", polished)
    if m:
        title = m.group(1).strip()
        polished = polished[m.end():].strip()
    return title, polished


def build_course(
    course_name: str,
    config_path: str = "config/config.yaml",
    polish: bool = True,
    make_pdf: bool = True,
) -> Path:
    import yaml
    cfg_all = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    claude_cfg = cfg_all.get("notes", {}).get("claude", {})

    files = _lecture_files(course_name)
    if not files:
        raise SystemExit(
            f"No archived lectures in {COURSE_DIR / _slugify(course_name)}/ — "
            "run some lectures first (they are archived automatically), or copy "
            "old lecture_notes.tex files there as <YYYY-MM-DD>_<name>.tex."
        )
    console.print(f"[bold cyan]Course build:[/bold cyan] {course_name} — {len(files)} lectures")
    for f in files:
        console.print(f"  - {f.name}")

    dates = [d for d in (_parse_date(f.stem.split('_', 1)[0]) for f in files) if d]
    date_range = (
        f"{dates[0].strftime('%d/%m/%Y')} -- {dates[-1].strftime('%d/%m/%Y')}"
        if dates else ""
    )

    course_profile = load_profile(course_name)
    chapters, outlines = [], []
    for n, f in enumerate(files, start=1):
        body = _extract_body(f.read_text(encoding="utf-8"))
        date_str = f.stem.split("_", 1)[0]
        title = f"Lezione {n} ({date_str})"
        if polish:
            console.print(f"[cyan]Cohesion pass {n}/{len(files)}...[/cyan]")
            try:
                new_title, body = _polish_chapter(
                    body, outlines, course_profile, claude_cfg, n
                )
                if new_title:
                    title = new_title
            except Exception as e:
                console.print(f"[yellow]⚠ Cohesion pass failed for {f.name}: {e} — using raw body[/yellow]")
        chapters.append(f"\\chapter{{{title}}}\n\n{body}")
        outlines.append(_outline_of(body))

    doc = "\n".join([
        _course_preamble(course_name, date_range),
        "\\begin{document}",
        _title_page(course_name, date_range),
        "\n\n\\clearpage\n\n".join(chapters),
        "\\end{document}",
    ])

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    doc = _repair_figure_paths(doc, LATEX_DIR)
    tex_path = LATEX_DIR / f"course_{_slugify(course_name)}.tex"
    tex_path.write_text(doc, encoding="utf-8")
    console.print(f"[green]✓ Course LaTeX saved:[/green] {tex_path}")

    if make_pdf:
        compile_pdf(
            tex_path,
            Path(cfg_all.get("notes", {}).get("latex", {}).get("pdf_output_dir", "output/notes")),
            course_name,
            datetime.now().strftime("%Y-%m-%d"),
            suffix="Corso completo",
            auto_fix=cfg_all.get("notes", {}).get("auto_fix_latex", True),
            backend=cfg_all.get("notes", {}).get("backend", "claude"),
            backend_config=claude_cfg,
        )
    return tex_path


def main():
    parser = argparse.ArgumentParser(
        description="Assemble all archived lectures of a course into one cohesive PDF."
    )
    parser.add_argument("course", help="Course name, e.g. 'Model Order'")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-polish", action="store_true",
                        help="Skip the Claude cohesion pass (mechanical merge)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Write the .tex but do not compile")
    args = parser.parse_args()
    build_course(
        args.course,
        config_path=args.config,
        polish=not args.no_polish,
        make_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
