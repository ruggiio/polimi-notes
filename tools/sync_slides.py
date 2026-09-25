#!/usr/bin/env python3
"""
sync_slides.py — Scarica in locale le slide linkate su WeBeep (cartelle OneDrive/SharePoint).

  .venv/bin/python tools/sync_slides.py "https://webeep.polimi.it/mod/url/view.php?id=488848" --course "BIOINSPIRED ROBOTICS"
  .venv/bin/python tools/sync_slides.py --all          # tutti i corsi con `slides_links` in config.yaml

Destinazione: <auto.slides_dir>/<COURSE>/_links/<nome cartella>/…  (accanto ai file di WeBeep Sync)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.downloader.archive import LoginRequired, open_context  # noqa: E402
from src.slides.onedrive import sync_shared_folder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("link", nargs="?", help="modulo url WeBeep o link OneDrive/SharePoint")
    ap.add_argument("--course", help="nome corso (cartella di destinazione)")
    ap.add_argument("--all", action="store_true", help="tutti i corsi con slides_links in config")
    ap.add_argument("--max-size-mb", type=int, default=None)
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    root = Path(os.path.expanduser(cfg["auto"].get("slides_dir", "~/Scrivania/POLI")))
    jobs = []
    if a.all:
        for c in cfg["auto"].get("courses", []):
            for link in c.get("slides_links", []) or []:
                jobs.append((c["name"], link))
    elif a.link and a.course:
        jobs.append((a.course, a.link))
    else:
        sys.exit("uso: sync_slides.py LINK --course NOME   |   sync_slides.py --all")

    last = [0.0]
    def prog(done, total):
        if time.time() - last[0] > 5 or done == total:
            last[0] = time.time()
            print(f"      {done / 1e6:7.0f} / {total / 1e6:.0f} MB ({100 * done / max(total, 1):.0f}%)", flush=True)

    rc = 0
    with sync_playwright() as pw:
        ctx = open_context(pw, Path(cfg["download"].get("profile_dir", "config/chrome_profile")), headless=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for course, link in jobs:
                dest = root / course / "_links"
                print(f"══ {course} ← {link}")
                got = sync_shared_folder(ctx, page, link, dest, progress=prog, max_size_mb=a.max_size_mb)
                print(f"   {len(got)} file nuovi/aggiornati in {dest}")
        except LoginRequired as e:
            print(f"✗ LOGIN RICHIESTO: {e}", file=sys.stderr); rc = 3
        finally:
            ctx.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
