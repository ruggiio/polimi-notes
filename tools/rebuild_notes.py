#!/usr/bin/env python3
"""
rebuild_notes.py — Ricostruisce il PDF di una lezione dalla risposta grezza del modello
(output/latex/lecture_notes_raw.tex, o un .tex/.md a scelta) senza richiamare Claude:
stesso post-processing di generate_notes (preambolo, figure, tabelle larghe) + pdflatex.
Serve quando cambia il post-processing o il preambolo (font, pacchetti, fix tabelle).

  .venv/bin/python tools/rebuild_notes.py --course "VISION BASED 3D MEASUREMENTS" \\
      --date 2026-09-16 --suffix "Introduction to the course" [--raw output/latex/lecture_notes_raw.tex] [--language en]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.notes_gen.notes_gen import (LAST_LAYOUT, _drop_missing_figures, _fix_wide_tables,  # noqa: E402
                                     _repair_figure_paths, assemble_document, compile_pdf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--suffix", default=None, help="argomento (parte del nome del PDF)")
    ap.add_argument("--raw", default="output/latex/lecture_notes_raw.tex")
    ap.add_argument("--language", default=None, help="lingua dei titoli dei box (default: notes.language)")
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    ncfg = cfg["notes"]
    latex_dir = Path(ncfg["latex"]["output_dir"])
    pdf_dir = Path(ncfg["latex"].get("pdf_output_dir", "output/notes"))
    lang = a.language or (ncfg.get("language") if ncfg.get("language") not in (None, "lecture") else None)

    body = Path(a.raw).read_text(encoding="utf-8")
    latex = assemble_document(body, a.course, a.date, lang)
    latex = _repair_figure_paths(latex, latex_dir)
    latex = _drop_missing_figures(latex, latex_dir)
    latex = _fix_wide_tables(latex)
    tex_path = latex_dir / "lecture_notes.tex"
    tex_path.write_text(latex, encoding="utf-8")

    backend = ncfg["backend"]
    pdf = compile_pdf(tex_path, pdf_dir, a.course, a.date, a.suffix,
                      auto_fix=ncfg.get("auto_fix_latex", False), backend=backend,
                      backend_config=dict(ncfg.get(backend, {})))
    if not pdf:
        return 1
    from src.course_profiles import _slugify
    course_dir = Path("output/course") / _slugify(a.course)
    course_dir.mkdir(parents=True, exist_ok=True)
    for old in course_dir.glob(f"{a.date}_*.tex"):
        old.write_text(latex, encoding="utf-8")       # aggiorna la copia archiviata per course_builder
    print(f"layout: {LAST_LAYOUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
