#!/usr/bin/env python3
"""
sso_login.py — Login manuale (password + 2FA CIE) nel profilo Chromium persistente.

Apre il portale Servizi Online in una finestra; tu completi il login (spunta "Resta
connesso" se proposto). Lo script attende di vedere il portale autenticato e chiude.
Da rifare quando fetch_lecture.py esce con codice 3 (sessione ~10 giorni).
Con la sessione ancora valida il portale si apre già autenticato e lo script esce
senza rinnovare nulla: per rinnovarla in anticipo usa --force, che cancella prima i
cookie *.polimi.it del profilo e costringe a rifare il login.
"""

import argparse
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.downloader.archive import ARCHIVE_SERVICE_ID, PORTALE_URL, goto_robust, open_context  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?", type=Path, default=ROOT / "config" / "chrome_profile")
    ap.add_argument("--force", action="store_true", help="rinnova anche se la sessione è ancora valida")
    args = ap.parse_args()
    profile = args.profile
    with sync_playwright() as pw:
        ctx = open_context(pw, profile, headless=False)
        if args.force:
            ctx.clear_cookies(domain=re.compile(r"(^|\.)polimi\.it$"))
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        goto_robust(page, PORTALE_URL)
        print("Completa il login nella finestra (password + CIE). Attendo il portale…")
        t0 = time.time()
        while time.time() - t0 < 600:
            page.wait_for_timeout(1000)
            try:
                if "portaleservizi" in page.url and \
                        page.locator(f'a[href*="idServizio={ARCHIVE_SERVICE_ID}"]').count() > 0:
                    break
            except Exception:
                pass
        else:
            sys.exit("timeout: login non completato")
        page.wait_for_timeout(1500)
        exp = max((c.get("expires") or 0) for c in ctx.cookies() if c["domain"] == "aunicalogin.polimi.it")
        ctx.close()
    print(f"✓ sessione salvata in {profile}; scade il {time.strftime('%Y-%m-%d %H:%M', time.localtime(exp))}")


if __name__ == "__main__":
    main()
