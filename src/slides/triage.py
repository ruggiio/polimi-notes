"""
triage.py — Selezione delle figure estratte dalle slide con un provino + Haiku (vision).

Le immagini estratte da un pptx sono per metà spazzatura (sfondi, loghi, ritratti dei
docenti, screenshot). Il modello che scrive gli appunti vede solo testo: qui, prima,
un provino numerato (≤ 20 riquadri per foglio) viene mostrato a Haiku via `claude -p`
con il tool Read, che risponde keep/drop + didascalia breve per ogni riquadro.
Costo tipico: ~$0.04 per foglio.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.notes_gen.notes_gen import ClaudeRateLimited, run_claude_json

TILE, COLS, ROWS = 220, 5, 4          # 20 riquadri per foglio

SYSTEM = "You are a terse assistant that triages figures for university lecture notes."
PROMPT = """Use the Read tool to look at the image {sheet}. It is a contact sheet of {n} numbered tiles
(a white badge with a black number at the top-left of each tile; tiles are numbered 1..{n} row by row).

For each tile decide whether it is worth embedding as a figure in lecture notes.
KEEP: diagrams, schemes, plots, tables, drawings, and photos that illustrate a technical subject
(a robot, an animal or organism, a mechanism, an experiment, a prototype, a simulation).
DROP: decorative backgrounds, logos, portraits or group photos of people, screenshots of websites,
text-only or blank tiles, near-duplicates of a kept tile.

Reply ONLY with a JSON array, one object per tile, in tile order:
[{{"n": 1, "keep": true, "caption": "five-to-ten word description"}}, ...]
Give a caption for every kept tile (what the figure shows, not what it means)."""


def build_sheets(figures: list[Path], out_dir: Path) -> list[tuple[Path, list[Path]]]:
    import pymupdf as fitz
    fitz.TOOLS.mupdf_display_errors(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    per = COLS * ROWS
    for k in range(0, len(figures), per):
        batch = figures[k:k + per]
        rows = (len(batch) + COLS - 1) // COLS
        doc = fitz.open()
        page = doc.new_page(width=COLS * TILE, height=rows * TILE)
        for i, f in enumerate(batch):
            x0, y0 = (i % COLS) * TILE, (i // COLS) * TILE
            r = fitz.Rect(x0 + 6, y0 + 6, x0 + TILE - 6, y0 + TILE - 6)
            page.draw_rect(r, color=(0.75, 0.75, 0.75), width=0.8)
            try:
                page.insert_image(r, filename=str(f), keep_proportion=True)
            except Exception:
                pass
            badge = fitz.Rect(x0 + 6, y0 + 6, x0 + 44, y0 + 34)
            page.draw_rect(badge, color=(0, 0, 0), fill=(1, 1, 1), width=1)
            page.insert_text((x0 + 11, y0 + 28), str(i + 1), fontsize=18, fontname="hebo", color=(0, 0, 0))
        sheet = out_dir / f"sheet_{k // per + 1}.png"
        page.get_pixmap(dpi=96).save(sheet)
        doc.close()
        sheets.append((sheet, batch))
    return sheets


def _ask(sheet: Path, n: int, model: str, timeout: int = 300) -> tuple[list[dict], float]:
    data = run_claude_json(PROMPT.format(sheet=sheet.resolve(), n=n), SYSTEM, model, timeout,
                           tools="Read", extra_args=["--add-dir", str(sheet.parent.resolve())])
    text = (data.get("result") or "").strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise RuntimeError(f"triage: risposta non JSON: {text[:200]!r}")
    return json.loads(m.group(0)), data.get("total_cost_usd", 0)


def triage_figures(figures: list[Path], work_dir: Path, model: str = "haiku",
                   log=print) -> tuple[dict[str, str | None], bool]:
    """
    ({nome file: didascalia} per le figure da tenere, {nome: None} per quelle da scartare,
    completo?). In caso di errore su un foglio le sue figure vengono tenute (senza didascalia)
    e il risultato è marcato incompleto, cosi' chi lo mette in cache sa di doverlo rifare.
    Il limite d'uso dell'abbonamento non è un errore del foglio: si propaga e si riprova dopo.
    """
    result: dict[str, str | None] = {}
    total_cost = 0.0
    complete = True
    for sheet, batch in build_sheets(figures, work_dir / "_triage"):
        try:
            verdicts, cost = _ask(sheet, len(batch), model)
            total_cost += cost or 0
            by_n = {int(v.get("n", 0)): v for v in verdicts if isinstance(v, dict)}
            for i, f in enumerate(batch):
                v = by_n.get(i + 1)
                if v is None:
                    result[f.name] = ""
                elif v.get("keep"):
                    result[f.name] = (v.get("caption") or "").strip()
                else:
                    result[f.name] = None
        except ClaudeRateLimited:
            raise
        except Exception as e:
            log(f"triage: errore su {sheet.name} ({type(e).__name__}: {str(e)[:120]}) — tengo tutte")
            complete = False
            for f in batch:
                result[f.name] = ""
    kept = sum(1 for v in result.values() if v is not None)
    log(f"triage: {kept}/{len(figures)} figure tenute (${total_cost:.3f}, {model})"
        f"{'' if complete else ' — incompleto, da rifare'}")
    return result, complete
