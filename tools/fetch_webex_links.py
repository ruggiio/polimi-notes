#!/usr/bin/env python3
"""
fetch_webex_links.py — Scarica registrazioni da link Webex diretti (…/ldr.php?RCID=…) elencati dal
docente in un .docx su WeBeep, per i corsi che non passano dall'Archivio registrazioni
(es. Smart Materials 25/26: lezioni nella Personal Room del prof).

  .venv/bin/python tools/fetch_webex_links.py --docx "<file>.docx" \\
      --course "059616 - SMART MATERIALS (VEDANI MAURIZIO)" --first 6 [--from 3] [--list]

Il .docx ha, per ogni lezione, tre paragrafi:
  "Monday Feb. 23, 2026 – <argomento>"
  "Webex meeting recording: <nome>-20260223 0855-1"      ← la data vera è qui
  "Recording link: https://…/ldr.php?RCID=…"
Scrive <output/videos>/<data>_<CORSO>_<argomento>.mp4 + sidecar .json (come fetch_lecture.py):
da lì il nightly trascrive e fa gli appunti. Esce con codice 3 se serve un login manuale.
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.downloader.archive import (LoginRequired, download_media, open_context,  # noqa: E402
                                    open_recording_url, safe_name)


def parse_docx(path: Path) -> list[dict]:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    paras = [re.sub(r"\s+", " ", "".join(re.findall(r"<w:t[^>]*>([^<]*)", p))).strip()
             for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
    out, title, rec = [], None, None
    for t in paras:
        if (m := re.search(r"https?://\S+ldr\.php\?RCID=\w+", t)) and title and rec:
            out.append({"url": m.group(0), "date_iso": f"{rec[0]}-{rec[1]}-{rec[2]}",
                        "time": f"{rec[3][:2]}:{rec[3][2:]}",
                        "topic": re.split(r"\s[–-]\s?", title, maxsplit=1)[-1].strip()})
            title = rec = None
        elif (m := re.search(r"recording:.*-(\d{4})(\d{2})(\d{2}) (\d{4})-\d+", t)):
            rec = m.groups()
        elif re.search(r"\b20\d\d\s*[–-]", t):
            title = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True, type=Path)
    ap.add_argument("--course", required=True, help='campo Corso come in archivio: "059616 - NOME (DOCENTE)"')
    ap.add_argument("--first", type=int, default=None, help="solo le prime N lezioni dell'elenco")
    ap.add_argument("--from", dest="start", type=int, default=1, help="a partire dalla lezione N (1 = la prima)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "videos"))
    ap.add_argument("--profile", default=str(ROOT / "config" / "chrome_profile"))
    a = ap.parse_args()

    recs = parse_docx(a.docx)[a.start - 1:a.first]
    course_name = re.sub(r"^\d+\s*-\s*", "", a.course).split(" (")[0].strip()
    for i, r in enumerate(recs, a.start):
        print(f"  {i:2d}  {r['date_iso']} {r['time']}  {r['topic'][:60]}")
    if a.list:
        return
    email = os.environ.get("POLIMI_EMAIL", "")
    if not email:
        sys.exit("POLIMI_EMAIL mancante in .env")

    with sync_playwright() as pw:
        ctx = open_context(pw, Path(a.profile), headless=True)
        try:
            for r in recs:
                out = Path(a.out) / f"{r['date_iso']}_{safe_name(course_name)}_{safe_name(r['topic'], 40)}.mp4"
                if out.exists():
                    print(f"= {out.name} (già scaricato)")
                    continue
                t0 = time.time()
                info = open_recording_url(ctx, r["url"], email)
                print(f"→ {out.name}: {info.title}, {info.size / 1e6 if info.size else 0:.0f} MB")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.with_suffix(".json").write_text(json.dumps({
                    "transfer_id": "", "date": f"{r['date_iso']} {r['time']}", "date_iso": r["date_iso"],
                    "course": a.course, "kind": "Lezione", "topic": r["topic"], "duration": "",
                    "size": "", "href": r["url"], "webex_title": info.title,
                    "playback_url": info.playback_url}, indent=2, ensure_ascii=False))
                download_media(info, out)
                print(f"✓ {out.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s")
        except LoginRequired as e:
            print(f"\n✗ LOGIN RICHIESTO: {e}\n  → esegui: .venv/bin/python tools/sso_login.py", file=sys.stderr)
            sys.exit(3)
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
