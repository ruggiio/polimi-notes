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

def extract_date_from_title(title: str) -> str | None:
    """
    Extract date from Webex recording title.
    Handles format: "Andrea Maria Zanchettin's Personal Room-20251008 0805-1"
    Returns date in YYYY-MM-DD format, e.g. "2025-10-08"
    """
    match = re.search(r'(\d{4})(\d{2})(\d{2})', title)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        date_str = f"{year}-{month}-{day}"
        console.print(f"[green]✓ Date extracted from title:[/green] {date_str} (from: '{title}')")
        return date_str
    console.print(f"[yellow]⚠ Could not extract date from title: '{title}'[/yellow]")
    return None


def get_recording_date(page: Page, playback_url: str) -> str | None:
    """Extract the recording date from the Webex playback page title."""
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
    Two-step login:
      1. Webex asks for EMAIL
      2. PoliMi asks for CODICE PERSONA + PASSWORD
    """
    try:
        page.wait_for_url(
            re.compile(r"idbroker.*webex\.com|auth\.polimi\.it|login\.polimi|webex\.com/idb"),
            timeout=15_000
        )
    except Exception:
        if "recordingservice" in page.url or "playback" in page.url:
            console.print("[green]✓ Already authenticated via saved session[/green]")
            return True
        raise RuntimeError(f"Unexpected URL: {page.url}")

    # Step 1: Webex email page
    if "idbroker" in page.url or "webex.com/idb" in page.url:
        console.print(f"[yellow]→ Webex login (step 1/2): inserting email {email}...[/yellow]")
        try:
            page.wait_for_selector(
                'input[type="email"], input[name="email"], input[placeholder*="mail"]',
                timeout=8_000
            )
            email_field = page.query_selector(
                'input[type="email"], input[name="email"], input[placeholder*="mail"]'
            )
            email_field.click()
            email_field.fill(email)
            email_field.press("Enter")
        except Exception as e:
            raise RuntimeError(f"Could not fill Webex email field: {e}")

        try:
            page.wait_for_url(
                re.compile(r"auth\.polimi\.it|login\.polimi|sso\.polimi|aunicalogin"),
                timeout=15_000
            )
            console.print("[green]✓ Email accepted, redirected to PoliMi SSO[/green]")
        except Exception:
            raise RuntimeError(f"Webex did not redirect to PoliMi. URL: {page.url}")

    # Step 2: PoliMi credentials page
    console.print(f"[yellow]→ PoliMi SSO (step 2/2): inserting Codice Persona {codice}...[/yellow]")
    try:
        codice_field = page.wait_for_selector(
            'input[placeholder="Codice Persona"], input[name="j_username"], '
            'input[id="j_username"], input[autocomplete="username"]',
            timeout=8_000
        )
        codice_field.click()
        time.sleep(0.2)
        codice_field.fill(codice)
        time.sleep(0.2)

        pwd_field = page.wait_for_selector('input[type="password"]', timeout=5_000)
        pwd_field.click()
        time.sleep(0.2)
        pwd_field.fill(password)
        time.sleep(0.2)

        accedi_btn = page.wait_for_selector(
            'input[value="Accedi"], button:has-text("Accedi")',
            timeout=5_000
        )
        accedi_btn.click()

    except Exception as e:
        raise RuntimeError(f"Could not fill PoliMi credentials: {e}")

    # Step 2b: Handle "Continua" intermediate page
    try:
        page.wait_for_selector(
            'button:has-text("Continua"), input[value="Continua"]',
            timeout=8_000
        )
        console.print("[bold yellow]⚠ Clicca 'Continua' nel browser, poi premi INVIO qui[/bold yellow]")
        input("  Premi INVIO dopo aver cliccato 'Continua'...")
    except Exception:
        pass

    # Step 3: Wait for redirect back to Webex
    try:
        page.wait_for_url(re.compile(r"webex\.com"), timeout=45_000)
        console.print("[green]✓ PoliMi SSO authentication successful[/green]")
        return True
    except Exception:
        if "webex.com" in page.url:
            console.print("[green]✓ Already on Webex[/green]")
            return True
        time.sleep(10)
        if "webex.com" in page.url:
            console.print("[green]✓ Redirected to Webex (delayed)[/green]")
            return True
        raise RuntimeError(f"Did not redirect back to Webex. URL: {page.url}")


# ── Download ──────────────────────────────────────────────────────────────────

def _download_with_ytdlp(url: str, output_path: Path, headers: dict, cookies: list[dict] = None) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)

    # Always remove existing file to force a fresh download
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
) -> tuple[Path, str | None]:
    """
    Authenticate via PoliMi SSO, download the Webex recording,
    and extract the recording date from the page title.

    Returns:
        (video_path, recording_date)
        recording_date is in YYYY-MM-DD format, or None if not found
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    email = os.environ.get("POLIMI_EMAIL", username)
    codice = username.split("@")[0]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        already_authed = _load_cookies(context, cookies_file)

        found_url = None
        found_headers = {}

        def on_request(request):
            nonlocal found_url, found_headers
            url = request.url
            if any(kw in url.lower() for kw in ["record", "media", "stream", "video", "content", "play", "wbx", "nln"]):
                console.print(f"[dim]Request: {url[:120]}[/dim]")
            is_video = (
                any(ext in url for ext in [".mp4", ".m3u8", ".ts"]) or
                any(kw in url.lower() for kw in ["nln1.wbx.com", "ciscospark", "recording/play", "streamurl"])
            )
            if is_video and found_url is None:
                found_url = url
                found_headers = dict(request.headers)
                console.print(f"[green]✓ Video URL intercepted:[/green] {url[:120]}...")

        page = context.new_page()
        page.on("request", on_request)

        console.print(f"[cyan]Navigating to:[/cyan] {webex_url}")
        page.goto(webex_url, wait_until="domcontentloaded", timeout=30_000)

        needs_login = (
            not already_authed
            or "idbroker" in page.url
            or "auth.polimi" in page.url
            or "aunicalogin" in page.url
        )
        if needs_login:
            _do_polimi_sso(page, email=email, codice=codice, password=password)
            _save_cookies(page, cookies_file)

        # Extract date from page title
        recording_date = get_recording_date(page, webex_url)

        # Save date to disk for reuse in subsequent --no-download runs
        if recording_date:
            date_file = output_dir / "lecture_date.txt"
            date_file.write_text(recording_date)
            console.print(f"[dim]Date saved to {date_file}[/dim]")

        console.print("[cyan]Waiting for video player to load...[/cyan]")
        for i in range(30):
            if found_url:
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

        if not found_url:
            console.print("[bold yellow]⚠ Video URL not found automatically.[/bold yellow]")
            console.print("[yellow]Please click play in the browser, then press INVIO here...[/yellow]")
            input("  Premi INVIO dopo aver avviato il video...")
            time.sleep(5)

        cookies = context.cookies()
        browser.close()

    if not found_url:
        raise RuntimeError("Could not intercept video stream URL.")

    return _download_with_ytdlp(found_url, output_dir, found_headers, cookies), recording_date