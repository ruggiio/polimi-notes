#!/usr/bin/env python3
"""
compare_models.py — Genera gli stessi appunti con più modelli (via claude -p) e confronta
costo, tempo e metriche strutturali. Output in output/probe/compare/<model>.{tex,pdf,json}
e una tabella riassuntiva in summary.md.

  .venv/bin/python tools/compare_models.py output/transcripts/LEZIONE.txt --course "BIOINSPIRED ROBOTICS" \
      --date 2025-09-29 --topic "Moving in water" --slides output/slides/<slug> --models sonnet opus haiku
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.notes_gen import notes_gen  # noqa: E402
from src.notes_gen.notes_gen import LAST_USAGE, generate_notes  # noqa: E402
from src.slides.slides import SlideDeck  # noqa: E402
from src.course_profiles import _slugify  # noqa: E402


def metrics(tex: str, pdf: Path | None) -> dict:
    m = {
        "pages": None, "chars": len(tex), "words": len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", tex).split()),
        "sections": len(re.findall(r"\\section\{", tex)), "subsections": len(re.findall(r"\\subsection\{", tex)),
        "equations": len(re.findall(r"\\begin\{(equation|align)", tex)), "inline_math": len(re.findall(r"\$[^$]+\$", tex)),
        "tables": len(re.findall(r"\\begin\{tabular\}", tex)), "figures": len(re.findall(r"\\includegraphics", tex)),
    }
    for box in ("definizione", "teorema", "esempio", "intuizione", "attenzione", "sintesi"):
        m[box] = len(re.findall(r"\\begin\{" + box + r"\}", tex))
    if pdf and pdf.exists():
        try:
            out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
            m["pages"] = int(re.search(r"Pages:\s+(\d+)", out).group(1))
        except Exception:
            pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--course", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--topic", default="")
    ap.add_argument("--slides", default=None, help="cartella output/slides/<slug> già estratta")
    ap.add_argument("--models", nargs="+", default=["sonnet", "opus", "haiku"])
    ap.add_argument("--effort", nargs="*", default=[None], help="livelli di effort da provare (es. low medium)")
    ap.add_argument("--out", default="output/probe/compare")
    a = ap.parse_args()

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir = Path("output/latex")
    slides_text, figures = None, None
    if a.slides:
        deck = SlideDeck.load(Path(a.slides))
        if deck:
            text = Path(a.transcript).read_text(encoding="utf-8")
            slides_text = deck.prompt_text(transcript=text)
            figures = [{"slide": f["slide"], "caption": f["hint"],
                        "latex_path": os.path.relpath(f["path"], latex_dir)} for f in deck.figure_list(text, 8)]
            print(f"slides: {len(deck.pages)} pagine, {len(figures)} figure")

    rows = []
    for model, effort in [(m, e) for m in a.models for e in a.effort]:
        tag = f"{model}_{effort}" if effort else model
        print(f"\n════ {tag} ════", flush=True)
        # PDF di questo run isolato in una cartella propria
        pdf_dir = out_dir / f"pdf_{tag}"
        shutil.rmtree(pdf_dir, ignore_errors=True)
        t0 = time.time()
        err = None
        try:
            generate_notes(
                merged_data=[], output_dir=latex_dir, stem=Path(a.transcript).stem,
                course_name=a.course, lecture_date=a.date, backend="claude-code",
                backend_config={"model": model, "timeout": 2400, "auto_fix_latex": True, "effort": effort},
                compile_pdf_flag=True, transcript_path=Path(a.transcript),
                pdf_output_dir=pdf_dir, suffix=a.topic or None, figures=figures, slides_text=slides_text,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print("ERRORE:", err)
        elapsed = round(time.time() - t0)
        pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        pdf = out_dir / f"{tag}.pdf"
        if pdfs:
            shutil.copy2(pdfs[0], pdf)
        tex = latex_dir / "lecture_notes.tex"
        if tex.exists():
            shutil.copy2(tex, out_dir / f"{tag}.tex")
        archived = Path("output/course") / _slugify(a.course) / f"{Path(a.transcript).stem}.tex"
        if archived.exists():
            shutil.copy2(archived, out_dir / f"{tag}_prefix.tex")   # output originale, prima di auto-fix
        usage = dict(LAST_USAGE) if LAST_USAGE.get("purpose") == "notes" else {}
        # tutte le chiamate di questo run (note + eventuali fix) dal log usage
        calls = []
        if notes_gen.USAGE_LOG.exists():
            for line in notes_gen.USAGE_LOG.read_text().splitlines():
                d = json.loads(line)
                if d["at"] >= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)):
                    calls.append(d)
        row = {"model": tag, "seconds": elapsed, "error": err,
               "thinking": sum(c.get("thinking", 0) for c in calls if c["purpose"] == "notes"),
               "calls": len(calls), "tokens_in": sum(c["in"] for c in calls), "tokens_out": sum(c["out"] for c in calls),
               "cost_usd": round(sum(c["cost_usd"] for c in calls), 3),
               "model_id": next((c["model"] for c in calls if c["purpose"] == "notes"), None),
               "pdf": str(pdf) if pdf.exists() else None,
               **(metrics((out_dir / f"{tag}.tex").read_text(), pdf) if (out_dir / f"{tag}.tex").exists() else {})}
        (out_dir / f"{tag}.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(json.dumps(row, indent=1))

    keys = ["model", "model_id", "seconds", "calls", "tokens_in", "tokens_out", "thinking", "cost_usd", "pages", "words",
            "sections", "subsections", "equations", "inline_math", "tables", "figures",
            "definizione", "teorema", "esempio", "intuizione", "attenzione", "sintesi", "error"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
