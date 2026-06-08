"""
downloader.py — Authenticated Webex lecture downloader for PoliMi SSO
"""

import json
import os
import re
import time
import requests
from pathlib import Path

import yt_dlp
from playwright.sync_api import sync_playwright, Page
from rich.console import Console

console = Console()


# ── Date Extraction ───────────────────────────────────────────────────────────

def extract_date_from_title(title: str):
    match = re.search(r'(\d{4})(\d{2})(\d{2})', title)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        date_str = f"{year}-{month}-{day}"
        console.print(f"[green]✓ Date extracted from title:[/green] {date_str} (from: '{title}')")
        return date_str
    console.print(f"[yellow]⚠ Could not extract date from title: '{title}'[/yellow]")
    return None


def get_recording_date(page: Page, playback_url: str):
    try:
        title = page.title()
        if title:
            return extract_date_from_title(title)
    except Exception as e:
        console.print(f"[yellow]⚠ Could not read page title: {e}[/yellow]")
    return None


# ── SSO Login ────────────────────────────────────────────────────────────────

def _do_polimi_sso(page: Page, email: str, codice: str, password: str) -> bool:
    """
    Login manuale: si apre il browser, l'utente completa il login a mano.
    """
    console.print("\n[bold yellow]══════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]  Chromium è aperto. Fai il login manualmente:[/bold yellow]")
    console.print("[yellow]  1. Inserisci email Webex → credenziali PoliMi[/yellow]")
    console.print("[yellow]  2. Completa eventuali pagine intermedie (es. 'Continua')[/yellow]")
    console.print("[yellow]  3. Aspetta di vedere la pagina della lezione con il VIDEO[/yellow]")
    console.print("[yellow]  4. Torna qui e premi INVIO[/yellow]")
    console.print("[bold yellow]══════════════════════════════════════════════════[/bold yellow]\n")
    input("  >>> Premi INVIO quando sei sulla pagina Webex della lezione: ")
    console.print("[green]✓ Login completato[/green]")
    return True


# ── Download ──────────────────────────────────────────────────────────────────

def _download_with_ytdlp(url: str, output_path: Path, headers: dict, cookies: list = None):
    output_path.mkdir(parents=True, exist_ok=True)

    existing = output_path / "lecture.mp4"
    if existing.exists():
        existing.unlink()
        console.print("[dim]Removed existing video for fresh download[/dim]")

    cookie_file = None
    if cookies:
        cookie_file = output_path / "_cookies.txt"
        lines = ["# Netscape HTTP Cookie File", ""]
        for c in cookies:
            domain = c.get("domain", "")
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = int(c.get("expires", 0)) if c.get("expires") and c.get("expires") > 0 else 0
            lines.append(
                f"{domain}\t{flag}\t{c.get('path', '/')}\t{secure}\t{expires}"
                f"\t{c.get('name', '')}\t{c.get('value', '')}"
            )
        cookie_file.write_text("\n".join(lines))

    ydl_opts = {
        "outtmpl": str(output_path / "lecture.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "http_headers": headers,
        "nocheckcertificate": True,
    }
    if cookie_file:
        ydl_opts["cookiefile"] = str(cookie_file)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    if cookie_file:
        cookie_file.unlink(missing_ok=True)

    out = Path(filename)
    if not out.exists():
        mp4 = output_path / "lecture.mp4"
        if mp4.exists():
            out = mp4
    return out


# ── Cookie helpers ────────────────────────────────────────────────────────────

def _save_cookies(page: Page, path: Path):
    cookies = page.context.cookies()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cookies, f, indent=2)
    console.print(f"[dim]Cookies saved to {path}[/dim]")


def _load_cookies(context, path: Path) -> bool:
    if not path.exists():
        return False
    with open(path) as f:
        cookies = json.load(f)
    if not cookies:
        return False
    context.add_cookies(cookies)
    console.print(f"[dim]Loaded saved cookies from {path}[/dim]")
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def download_lecture(
    webex_url: str,
    username: str,
    password: str,
    output_dir: Path,
    cookies_file: Path,
    headless: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    email = os.environ.get("POLIMI_EMAIL", username)
    codice = username.split("@")[0]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        already_authed = _load_cookies(context, cookies_file)

        # Intercetta l'URL del video E i veri URL dei segmenti media
        found_url = None
        found_headers = {}
        real_media_url = None  # URL diretto del file media (m3u8 o mp4)

        def on_request(request):
            nonlocal found_url, found_headers, real_media_url
            url = request.url

            # URL diretto del media (priorità massima)
            if real_media_url is None and any(
                kw in url.lower() for kw in ["nln1.wbx.com", "nln2.wbx.com", ".mp4", ".m3u8"]
            ):
                real_media_url = url
                found_headers = dict(request.headers)
                console.print(f"[green]✓ Media URL diretto:[/green] {url[:120]}...")

            # URL playback come fallback
            is_playback = any(kw in url.lower() for kw in [
                "recording/play", "streamurl", "ciscospark"
            ])
            if is_playback and found_url is None:
                found_url = url
                if not real_media_url:
                    found_headers = dict(request.headers)
                console.print(f"[green]✓ Playback URL:[/green] {url[:120]}...")

        page = context.new_page()
        page.on("request", on_request)

        console.print(f"[cyan]Navigating to:[/cyan] {webex_url}")
        try:
            page.goto(webex_url, wait_until="commit", timeout=60_000)
        except Exception:
            pass

        needs_login = (
            not already_authed
            or "idbroker" in page.url
            or "auth.polimi" in page.url
            or "aunicalogin" in page.url
            or "shibidp" in page.url
        )
        if needs_login:
            _do_polimi_sso(page, email=email, codice=codice, password=password)
            _save_cookies(page, cookies_file)

        time.sleep(2)

        # Estrai data
        recording_date = get_recording_date(page, webex_url)
        if recording_date:
            date_file = output_dir / "lecture_date.txt"
            date_file.write_text(recording_date)

        # Aspetta URL media (max 60s), prova a cliccare play
        console.print("[cyan]In attesa dell'URL del video...[/cyan]")
        for i in range(60):
            if real_media_url or found_url:
                break
            time.sleep(1)
            if i == 3:
                try:
                    play_btn = page.query_selector(
                        'button[aria-label*="play" i], button[title*="play" i], '
                        '.play-button, #play-btn, [class*="play"]'
                    )
                    if play_btn:
                        play_btn.click()
                        console.print("[dim]Clicked play button[/dim]")
                except Exception:
                    pass

        if not real_media_url and not found_url:
            console.print("[yellow]Clicca play nel browser, poi premi INVIO...[/yellow]")
            input("  Premi INVIO dopo aver avviato il video... ")
            time.sleep(5)

        # Usa l'URL diretto se disponibile, altrimenti il playback URL
        download_url = real_media_url or found_url
        cookies = context.cookies()

        # FIX PRINCIPALE: scarica PRIMA di chiudere il browser,
        # tenendo la sessione attiva
        console.print("[cyan]Avvio download (browser ancora aperto)...[/cyan]")
        result_path = _download_with_ytdlp(download_url, output_dir, found_headers, cookies)

        browser.close()

    return result_path, recording_date
