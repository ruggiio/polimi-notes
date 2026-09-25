#!/usr/bin/env python3
"""
validate_notes.py — Validazione deterministica (gratis) degli appunti rispetto alla trascrizione,
e confronto tra due versioni (pre/post modifica).

  .venv/bin/python tools/validate_notes.py NOTES.tex --transcript T.txt [--segments T_segments.json]
  .venv/bin/python tools/validate_notes.py NEW.tex --transcript T.txt --baseline OLD.tex

Metriche:
  coverage   per blocchi di ~3 min: frazione dei termini specifici del blocco (≥6 lettere, presenti
             in ≤ 25% dei blocchi) ritrovati nel testo delle note (match per prefisso di 6 lettere).
             Segnala i blocchi sotto soglia → parti di lezione potenzialmente omesse.
  truncated  frasi che finiscono senza punteggiatura prima di un cambio di paragrafo/ambiente.
  structure  sezioni, box, equazioni, tabelle, figure, parole, rapporto prosa/box.
  latex      \\end{document}, ambienti bilanciati.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.slides.slides import STOP, _norm  # noqa: E402

CHUNK_SEC = 180
BOXES = ("definizione", "teorema", "esempio", "intuizione", "attenzione", "sintesi")


def strip_latex(tex: str) -> str:
    body = tex[tex.find("\\begin{document}"):] if "\\begin{document}" in tex else tex
    body = re.sub(r"%.*", "", body)
    # formule e figure diventano un segnaposto inline: "è definito come EQN dove u è…" resta una frase
    body = re.sub(r"\\begin\{(equation|align|figure|tabular|center)\*?\}.*?\\end\{\1\*?\}", " EQN ", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", " X ", body)
    body = re.sub(r"\\\[.*?\\\]", " EQN ", body, flags=re.S)
    body = re.sub(r"\\(?:begin|end)\{[^}]*\}", "\n\n", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
    body = re.sub(r"[{}]", " ", body)
    return body


def chunks_from_transcript(txt: str, segments: Path | None) -> list[dict]:
    if segments and segments.exists():
        segs = json.loads(segments.read_text())["segments"]
        out, cur, start = [], [], 0.0
        for s in segs:
            if cur and s["start"] - start >= CHUNK_SEC:
                out.append({"t0": start, "t1": cur[-1]["end"], "text": " ".join(c["text"] for c in cur)})
                cur, start = [], s["start"]
            if not cur:
                start = s["start"]
            cur.append(s)
        if cur:
            out.append({"t0": start, "t1": cur[-1]["end"], "text": " ".join(c["text"] for c in cur)})
        return out
    words = txt.split()
    n = 400
    return [{"t0": i / 2.5, "t1": (i + n) / 2.5, "text": " ".join(words[i:i + n])} for i in range(0, len(words), n)]


def terms(text: str) -> set[str]:
    return {t for t in _norm(text).split() if len(t) >= 6 and t not in STOP and not t.isdigit()}


def coverage(chunks: list[dict], notes_text: str) -> list[dict]:
    notes_pref = {t[:6] for t in _norm(notes_text).split() if len(t) >= 6}
    df: Counter = Counter()
    per = [terms(c["text"]) for c in chunks]
    for ts in per:
        df.update(set(ts))
    n = max(len(chunks), 1)
    out = []
    for c, ts in zip(chunks, per):
        specific = sorted(t for t in ts if df[t] <= max(1, n // 4))
        if not specific:
            out.append({**c, "coverage": None, "missing": []})
            continue
        found = [t for t in specific if t[:6] in notes_pref]
        missing = [t for t in specific if t[:6] not in notes_pref]
        out.append({**c, "coverage": round(len(found) / len(specific), 2), "missing": missing[:8], "n_terms": len(specific)})
    return out


def _prose_only(tex: str) -> str:
    """Solo i paragrafi di prosa: via titoli, argomenti dei box, item, caption, tabelle, formule."""
    body = tex[tex.find("\\begin{document}"):] if "\\begin{document}" in tex else tex
    body = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}", "\n\n", body)
    body = re.sub(r"\\(?:title|date|author|caption|label)\{[^}]*\}", "\n\n", body)
    body = re.sub(r"\\begin\{(" + "|".join(BOXES) + r")\}(\{[^}]*\}){0,2}", "\n\n", body)
    body = re.sub(r"\\item\b[^\n]*", "\n\n", body)
    body = re.sub(r"\\(?:maketitle|centering|hline|toprule|midrule|bottomrule)", "\n\n", body)
    return strip_latex(body)


def truncated_sentences(tex: str) -> list[str]:
    text = _prose_only(tex)
    paras = [p.strip() for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
    bad = []
    for p in paras:
        p = re.sub(r"\s+", " ", p)
        if len(p.split()) >= 8 and not re.search(r"[.!?:;)\]\"'»”]$", p):
            bad.append(p[-90:])
    return bad


def structure(tex: str) -> dict:
    body = tex[tex.find("\\begin{document}"):] if "\\begin{document}" in tex else tex
    text = strip_latex(tex)
    words = len(text.split())
    box_words = 0
    for b in BOXES:
        for m in re.finditer(r"\\begin\{" + b + r"\}.*?\\end\{" + b + r"\}", body, re.S):
            box_words += len(strip_latex(m.group(0)).split())
    d = {"words": words, "prose_ratio": round(1 - box_words / max(words, 1), 2),
         "sections": len(re.findall(r"\\section\{", body)), "subsections": len(re.findall(r"\\subsection\{", body)),
         "equations": len(re.findall(r"\\begin\{(equation|align)", body)) + len(re.findall(r"\\\[", body)),
         "inline_math": len(re.findall(r"\$[^$]+\$", body)), "tables": len(re.findall(r"\\begin\{tabular\}", body)),
         "figures": len(re.findall(r"\\includegraphics", body))}
    for b in BOXES:
        d[b] = len(re.findall(r"\\begin\{" + b + r"\}", body))
    return d


def latex_sanity(tex: str) -> list[str]:
    issues = []
    if "\\end{document}" not in tex:
        issues.append("manca \\end{document}")
    for env in set(re.findall(r"\\begin\{([a-zA-Z*]+)\}", tex)):
        nb, ne = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", tex)), len(re.findall(r"\\end\{" + re.escape(env) + r"\}", tex))
        if nb != ne:
            issues.append(f"ambiente {env}: {nb} begin / {ne} end")
    return issues


def report(tex_path: Path, transcript: Path, segments: Path | None) -> dict:
    tex = tex_path.read_text(encoding="utf-8")
    txt = transcript.read_text(encoding="utf-8")
    cov = coverage(chunks_from_transcript(txt, segments), strip_latex(tex))
    vals = [c["coverage"] for c in cov if c["coverage"] is not None]
    low = [c for c in cov if c["coverage"] is not None and c["coverage"] < 0.4]
    return {
        "file": str(tex_path), "chars": len(tex),
        "coverage_mean": round(sum(vals) / max(len(vals), 1), 3), "coverage_min": min(vals) if vals else None,
        "chunks": len(cov), "low_chunks": [{"t": f"{int(c['t0'] // 60):02d}:{int(c['t0'] % 60):02d}-{int(c['t1'] // 60):02d}:{int(c['t1'] % 60):02d}",
                                            "coverage": c["coverage"], "missing": c["missing"]} for c in low],
        "coverage_by_chunk": [c["coverage"] for c in cov],
        "truncated": truncated_sentences(tex), "latex_issues": latex_sanity(tex), **structure(tex),
    }


def show(r: dict):
    print(f"── {Path(r['file']).name}")
    print(f"   copertura media {r['coverage_mean']:.2f} (min {r['coverage_min']}) su {r['chunks']} blocchi; "
          f"blocchi < 0.4: {len(r['low_chunks'])}")
    for c in r["low_chunks"][:6]:
        print(f"      {c['t']}  {c['coverage']:.2f}  mancano: {', '.join(c['missing'][:6])}")
    print(f"   parole {r['words']}, prosa {int(r['prose_ratio'] * 100)}%, sezioni {r['sections']}/{r['subsections']}, "
          f"eq {r['equations']}, inline {r['inline_math']}, tabelle {r['tables']}, figure {r['figures']}, "
          f"box " + "/".join(str(r[b]) for b in BOXES))
    print(f"   frasi troncate: {len(r['truncated'])}" + (f"  es. …{r['truncated'][0]}" if r["truncated"] else ""))
    print(f"   latex: {'ok' if not r['latex_issues'] else r['latex_issues']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--baseline", default=None, help="tex della versione precedente per il confronto")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seg = Path(a.segments) if a.segments else Path(a.transcript).with_name(Path(a.transcript).stem + "_segments.json")
    new = report(Path(a.tex), Path(a.transcript), seg)
    show(new)
    verdict = {"ok": not new["latex_issues"] and not new["truncated"] and not new["low_chunks"]}
    if a.baseline:
        old = report(Path(a.baseline), Path(a.transcript), seg)
        show(old)
        print("── confronto (nuovo vs baseline)")
        worse = []
        for i, (cn, co) in enumerate(zip(new["coverage_by_chunk"], old["coverage_by_chunk"])):
            if cn is not None and co is not None and co - cn > 0.15:
                worse.append((i, co, cn))
        print(f"   copertura media {old['coverage_mean']:.2f} → {new['coverage_mean']:.2f}; blocchi peggiorati (>0.15): {len(worse)} {worse[:5]}")
        print(f"   parole {old['words']} → {new['words']} ({100 * (new['words'] / max(old['words'], 1) - 1):+.0f}%), "
              f"prosa {old['prose_ratio']} → {new['prose_ratio']}, sezioni {old['sections']} → {new['sections']}, "
              f"box {sum(old[b] for b in BOXES)} → {sum(new[b] for b in BOXES)}, eq {old['equations']} → {new['equations']}")
        verdict.update({"coverage_ok": new["coverage_mean"] >= old["coverage_mean"] - 0.03 and not worse,
                        "length_ok": new["words"] >= 0.85 * old["words"]})
        verdict["ok"] = verdict["ok"] and verdict["coverage_ok"] and verdict["length_ok"]
    print("VERDETTO:", "PASS" if verdict["ok"] else "FAIL", {k: v for k, v in verdict.items() if k != "ok"})
    if a.json:
        Path(a.json).write_text(json.dumps({"new": new, "verdict": verdict}, indent=1, ensure_ascii=False))
    sys.exit(0 if verdict["ok"] else 1)


if __name__ == "__main__":
    main()
