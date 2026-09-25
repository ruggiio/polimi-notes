#!/usr/bin/env python3
"""
fetch_lecture.py — Elenca e scarica registrazioni dall'Archivio PoliMi senza interazione.

  .venv/bin/python tools/fetch_lecture.py --aa 2025 --course BIOINSPIRED --list
  .venv/bin/python tools/fetch_lecture.py --aa 2025 --course BIOINSPIRED --pick oldest
  .venv/bin/python tools/fetch_lecture.py --transfer-id 131271 --aa 2025 --course BIOINSPIRED

Esce con codice 3 se serve un login manuale (→ tools/sso_login.py).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.downloader.archive import (LoginRequired, download_media, goto_archive,  # noqa: E402
                                    open_context, open_recording, safe_name, search_archive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, help="anno di inizio A.A., es. 2025")
    ap.add_argument("--course", help="sottostringa del campo Corso (nome o codice)")
    ap.add_argument("--kind", choices=["LE", "LA", "ES", "OT"], help="forma didattica")
    ap.add_argument("--list", action="store_true", help="solo elenco, niente download")
    ap.add_argument("--pick", choices=["oldest", "newest"], default=None)
    ap.add_argument("--transfer-id", help="scarica questa registrazione")
    ap.add_argument("--out", default=str(ROOT / "output" / "videos"))
    ap.add_argument("--profile", default=str(ROOT / "config" / "chrome_profile"))
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()

    email = os.environ.get("POLIMI_EMAIL", "")
    if not email:
        sys.exit("POLIMI_EMAIL mancante in .env")

    t0 = time.time()
    with sync_playwright() as pw:
        ctx = open_context(pw, Path(a.profile), headless=not a.headed)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            goto_archive(page)
            recs = search_archive(page, aa=a.aa, course=a.course, kind=a.kind)
            print(f"{len(recs)} registrazioni  ({time.time() - t0:.1f}s)")
            for r in recs:
                print(f"  {r.transfer_id:>7}  {r.date:16s} {r.duration:>8}  {r.course_name[:34]:34s} | {r.topic[:50]}")
            if a.list or not (a.pick or a.transfer_id):
                return
            if a.transfer_id:
                target = next((r for r in recs if r.transfer_id == a.transfer_id), None)
                if not target:
                    sys.exit(f"transfer_id {a.transfer_id} non nell'elenco")
            else:
                target = recs[0] if a.pick == "oldest" else recs[-1]

            out = Path(a.out) / f"{target.date_iso}_{safe_name(target.course_name)}_{safe_name(target.topic, 40)}.mp4"
            print(f"\n→ apro {target.transfer_id}: {target.date} — {target.topic}")
            info = open_recording(ctx, page, target, email)
            print(f"  Webex: {info.title}")
            print(f"  mp4: {info.size / 1e6 if info.size else 0:.0f} MB  {info.media_url[:80]}…")
            (out.with_suffix(".json")).parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".json").write_text(json.dumps({**target.to_dict(), "webex_title": info.title,
                                                           "playback_url": info.playback_url}, indent=2, ensure_ascii=False))
        except LoginRequired as e:
            print(f"\n✗ LOGIN RICHIESTO: {e}\n  → esegui: .venv/bin/python tools/sso_login.py", file=sys.stderr)
            sys.exit(3)
        finally:
            ctx.close()

    print(f"  scarico in {out.name} …")
    t1 = time.time()
    last = [0.0]
    def prog(done, total):
        if time.time() - last[0] > 5 or done == total:
            last[0] = time.time()
            print(f"    {done / 1e6:7.0f} / {total / 1e6:.0f} MB  ({100 * done / max(total, 1):.0f}%)", flush=True)
    path = download_media(info, out, progress=prog)
    mb = path.stat().st_size / 1e6
    print(f"✓ {path}  ({mb:.0f} MB, {time.time() - t1:.0f}s download, {time.time() - t0:.0f}s totali)")


if __name__ == "__main__":
    main()
