"""
downloader.py — SharePoint/OneDrive lecture downloader for UniPR Microsoft SSO

Pipeline:
  1. Open SharePoint URL in Playwright browser
  2. Handle Microsoft SSO login (with 2FA support)
  3. Intercept the direct video stream URL
  4. Download via yt-dlp with session cookies
  5. Extract date from filename or page title
"""

import json
import os
import re
import time
from pathlib import Path

import yt_dlp
from playwright.sync_api import sync_playwright, Page
from rich.console import Console

console = Console()


# ── Date Extraction ───────────────────────────────────────────────────────────

def extract_date_from_filename(url: str) -> str | None:
    """
    Try to extract date from SharePoint filename.
    Handles formats like 'Lezione 01 - 2026-03-05.mp4' or '20260305_lecture.mp4'
    Returns date in YYYY-MM-DD format.
    """
    # Try YYYY-MM-DD format
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', url)
    if match:
        date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        console.print(f"[green]✓ Date extracted from URL:[/green] {date_str}")
        return date_str

    # Try YYYYMMDD format
    match = re.search(r'(\d{4})(\d{2})(\d{2})', url)
    if match:
        date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        console.print(f"[green]✓ Date extracted from URL:[/green] {date_str}")
        return date_str

    console.print("[yellow]⚠ Could not extract date from URL — use --date to set manually[/yellow]")
    return None


def extract_date_from_page(page: Page) -> str | None:
    """Try to extract date from the SharePoint page title or metadata."""
    try:
        title = page.title()
        if title:
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', title)
            if match:
                date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                console.print(f"[green]✓ Date extracted from page title:[/green] {date_str}")
                return date_str
            match = re.search(r'(\d{4})(\d{2})(\d{2})', title)
            if match:
                date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                console.print(f"[green]✓ Date extracted from page title:[/green] {date_str}")
                return date_str
    except Exception as e:
        console.print(f"[yellow]⚠ Could not read page title: {e}[/yellow]")
    return None


# ── Microsoft SSO Login ───────────────────────────────────────────────────────

def _do_microsoft_sso(page: Page, email: str, password: str) -> bool:
    """
    Microsoft SSO login flow for university SharePoint/OneDrive.
    Handles standard Microsoft login AND university ADFS (e.g. adfs.unipr.it).
    Steps:
      1. Email entry
      2. Password entry (ADFS uses "Avanti" button, standard uses "Sign in")
      3. 2FA (Microsoft Authenticator) — waits for user to approve
      4. "Stay signed in?" prompt
    """
    try:
        page.wait_for_url(
            re.compile(r"login\.microsoftonline\.com|login\.microsoft\.com|adfs\."),
            timeout=15_000
        )
    except Exception:
        if "sharepoint.com" in page.url or "onedrive" in page.url.lower():
            console.print("[green]✓ Already authenticated via saved session[/green]")
            return True
        console.print(f"[yellow]Current URL: {page.url}[/yellow]")

    # Step 1: Enter email
    console.print(f"[yellow]→ Microsoft login (step 1/3): inserting email {email}...[/yellow]")
    try:
        email_field = page.wait_for_selector(
            'input[type="email"], input[name="loginfmt"], input[placeholder*="email" i]',
            timeout=10_000
        )
        email_field.click()
        email_field.fill(email)

        next_btn = page.wait_for_selector(
            'input[type="submit"], button:has-text("Next"), button:has-text("Avanti")',
            timeout=5_000,
            state="visible",
        )
        next_btn.click()
        console.print("[green]✓ Email submitted[/green]")
    except Exception as e:
        raise RuntimeError(f"Could not fill Microsoft email: {e}")

    # Step 2: Enter password
    # Note: UniPR uses ADFS (adfs.unipr.it) — button is hidden via CSS until JS runs
    console.print("[yellow]→ Microsoft login (step 2/3): inserting password...[/yellow]")

    # Wait for ADFS page to fully load
    time.sleep(2)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=10_000)
    except Exception:
        pass

    try:
        pwd_field = page.wait_for_selector(
            'input[type="password"], input[name="passwd"]',
            timeout=10_000
        )
        time.sleep(0.5)
        pwd_field.click()
        pwd_field.fill(password)
        time.sleep(0.5)

        # Try JS click first (bypasses hidden CSS), fallback to Enter key
        try:
            page.evaluate("var btn = document.getElementById('idSIButton9'); if(btn) btn.click();")
            console.print("[green]✓ Password submitted via JS click[/green]")
        except Exception:
            pwd_field.press("Enter")
            console.print("[green]✓ Password submitted via Enter[/green]")

    except Exception as e:
        raise RuntimeError(f"Could not fill Microsoft password: {e}")

    # Step 3: Handle 2FA (Microsoft Authenticator)
    # After password, Microsoft may show 2FA prompt OR redirect directly
    # Wait up to 60 seconds for either 2FA prompt or successful redirect
    console.print("[yellow]→ Checking for 2FA...[/yellow]")
    try:
        page.wait_for_selector(
            'div[data-viewid="11"], div:has-text("Approve sign in request"), '
            'div:has-text("Approva la richiesta"), #idDiv_SAOTCAS_Title, '
            'div:has-text("Authenticator"), div:has-text("number"), '
            'div:has-text("approva"), div:has-text("notifica")',
            timeout=15_000
        )
        console.print("")
        console.print("[bold yellow]╔══════════════════════════════════════════════════╗[/bold yellow]")
        console.print("[bold yellow]║  2FA RICHIESTO                                  ║[/bold yellow]")
        console.print("[bold yellow]║  Apri Microsoft Authenticator sul telefono      ║[/bold yellow]")
        console.print("[bold yellow]║  e approva la notifica di accesso.              ║[/bold yellow]")
        console.print("[bold yellow]║                                                  ║[/bold yellow]")
        console.print("[bold yellow]║  Dopo aver approvato, premi INVIO qui.          ║[/bold yellow]")
        console.print("[bold yellow]╚══════════════════════════════════════════════════╝[/bold yellow]")
        console.print("")
        input("  >>> Premi INVIO dopo aver approvato su Microsoft Authenticator: ")
        console.print("[green]✓ 2FA approved[/green]")
        time.sleep(2)  # give the page time to process
    except Exception:
        # No 2FA detected — check if we're already redirected
        console.print("[dim]No 2FA prompt detected — checking if already authenticated...[/dim]")

    # Step 4: Handle "Stay signed in?" prompt
    try:
        page.wait_for_selector(
            'button:has-text("Yes"), button:has-text("Sì"), '
            'input[value="Yes"], #idSIButton9',
            timeout=8_000
        )
        yes_btn = page.query_selector(
            'button:has-text("Yes"), button:has-text("Sì"), '
            'input[value="Yes"], #idSIButton9'
        )
        if yes_btn:
            yes_btn.click()
            console.print("[green]✓ Stay signed in confirmed[/green]")
    except Exception:
        pass

    # Wait for redirect back to SharePoint
    try:
        page.wait_for_url(
            re.compile(r"sharepoint\.com|onedrive\.live\.com"),
            timeout=30_000
        )
        console.print("[green]✓ Microsoft SSO authentication successful[/green]")
        return True
    except Exception:
        if "sharepoint.com" in page.url:
            return True
        time.sleep(5)
        if "sharepoint.com" in page.url:
            return True
        raise RuntimeError(f"Did not redirect back to SharePoint. URL: {page.url}")


# ── Download ──────────────────────────────────────────────────────────────────

def _date_from_ytdlp_info(info: dict) -> str | None:
    """
    Extract recording date from yt-dlp info dict. Returns YYYY-MM-DD or None.

    Priority:
    1. upload_date / release_date (YYYYMMDD string — yt-dlp standard field)
    2. timestamp (Unix float → local date)
    3. Regex scan of title, description, webpage_url, filename — handles SharePoint
       recording names like "Recording 2026-03-05" or "Lezione_20260305"
    """
    from datetime import datetime

    # Debug: log available metadata fields so the user can diagnose date issues
    date_fields = {
        k: info.get(k) for k in
        ("upload_date", "release_date", "timestamp", "title", "fulltitle",
         "description", "webpage_url", "filename", "_filename")
        if info.get(k)
    }
    if date_fields:
        console.print(f"[dim]yt-dlp metadata for date extraction: {date_fields}[/dim]")
    else:
        console.print("[yellow]⚠ yt-dlp returned no usable metadata fields for date extraction[/yellow]")

    # Standard yt-dlp date fields (YYYYMMDD)
    for key in ("upload_date", "release_date"):
        raw = info.get(key)
        if raw:
            s = str(raw).strip()
            if len(s) == 8 and s.isdigit():
                return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

    # Unix timestamp
    ts = info.get("timestamp")
    if ts and isinstance(ts, (int, float)) and ts > 0:
        try:
            return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
        except Exception:
            pass

    # Scan text fields — covers SharePoint/Teams recording title formats
    # Also scan filename and _filename (the yt-dlp output path) for dates
    for key in ("title", "fulltitle", "description", "webpage_url", "filename", "_filename"):
        text = str(info.get(key) or "")
        if not text:
            continue
        # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        m = re.search(r'(\d{4})[-./](\d{2})[-./](\d{2})', text)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                return f"{y}-{mo}-{d}"
        # DD-MM-YYYY or DD/MM/YYYY
        m = re.search(r'(\d{2})[-./](\d{2})[-./](\d{4})', text)
        if m:
            d, mo, y = m.group(1), m.group(2), m.group(3)
            if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                return f"{y}-{mo}-{d}"
        # YYYYMMDD compact (e.g. Recording_20260305)
        m = re.search(r'(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)', text)
        if m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                return f"{y}-{mo}-{d}"

    return None


def _download_with_ytdlp_cookies_browser(
    url: str,
    output_path: Path,
    browser: str = "chrome",
) -> tuple[Path, str | None]:
    """
    Download SharePoint video using browser cookies directly.
    Returns (video_path, recording_date) where recording_date may be None.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    existing = output_path / "lecture.mp4"
    if existing.exists():
        existing.unlink()
        console.print("[dim]Removed existing video for fresh download[/dim]")

    ydl_opts = {
        "outtmpl": str(output_path / "lecture.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "cookiesfrombrowser": (browser, None, None, None),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    metadata_date = _date_from_ytdlp_info(info)
    if metadata_date:
        console.print(f"[green]✓ Date from video metadata:[/green] {metadata_date}")

    out = Path(filename)
    if not out.exists():
        mp4 = output_path / "lecture.mp4"
        if mp4.exists():
            out = mp4
    return out, metadata_date


def _download_with_ytdlp_playwright(
    url: str,
    output_path: Path,
    cookies: list[dict],
) -> tuple[Path, str | None]:
    """
    Download SharePoint video using cookies captured by Playwright.
    Fallback if browser cookie method fails.
    Returns (video_path, recording_date) where recording_date may be None.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    existing = output_path / "lecture.mp4"
    if existing.exists():
        existing.unlink()
        console.print("[dim]Removed existing video for fresh download[/dim]")

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
        "cookiefile": str(cookie_file),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    cookie_file.unlink(missing_ok=True)

    metadata_date = _date_from_ytdlp_info(info)
    if metadata_date:
        console.print(f"[green]✓ Date from video metadata:[/green] {metadata_date}")

    out = Path(filename)
    if not out.exists():
        mp4 = output_path / "lecture.mp4"
        if mp4.exists():
            out = mp4
    return out, metadata_date


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
    sharepoint_url: str,
    username: str,
    password: str,
    output_dir: Path,
    cookies_file: Path,
    headless: bool = True,
) -> tuple[Path, str | None]:
    """
    Authenticate via Microsoft SSO, download the SharePoint lecture video,
    and extract the recording date from the URL or page title.

    Returns:
        (video_path, recording_date)
        recording_date is in YYYY-MM-DD format, or None if not found
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    email = os.environ.get("MICROSOFT_EMAIL", username)
    pwd = os.environ.get("MICROSOFT_PASS", password)

    # Try date extraction from URL first (works for well-named files)
    recording_date = extract_date_from_filename(sharepoint_url)

    # Strategy 1: Try yt-dlp with browser cookies (simplest, no Playwright needed)
    console.print("[cyan]Attempting download with browser cookies...[/cyan]")
    for browser in ["chrome", "edge", "firefox"]:
        try:
            video_path, metadata_date = _download_with_ytdlp_cookies_browser(
                sharepoint_url, output_dir, browser
            )
            console.print(f"[green]✓ Downloaded using {browser} cookies[/green]")

            # Prefer URL date, fall back to metadata date
            final_date = recording_date or metadata_date
            if final_date:
                date_file = output_dir / "lecture_date.txt"
                date_file.write_text(final_date)
                if not recording_date and metadata_date:
                    console.print(f"[green]✓ Using metadata date:[/green] {final_date}")

            return video_path, final_date
        except Exception as e:
            console.print(f"[dim]{browser} cookies failed: {e}[/dim]")
            continue

    # Strategy 2: Playwright login + cookie extraction
    console.print("[cyan]Browser cookies failed — using Playwright SSO login...[/cyan]")

    with sync_playwright() as pw:
        browser_instance = pw.chromium.launch(headless=headless)
        context = browser_instance.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        already_authed = _load_cookies(context, cookies_file)

        page = context.new_page()
        console.print(f"[cyan]Navigating to:[/cyan] {sharepoint_url[:80]}...")
        page.goto(sharepoint_url, wait_until="domcontentloaded", timeout=30_000)

        needs_login = (
            not already_authed
            or "login.microsoftonline" in page.url
            or "login.microsoft.com" in page.url
        )

        if needs_login:
            _do_microsoft_sso(page, email=email, password=pwd)
            _save_cookies(page, cookies_file)

        # Wait for SharePoint page to fully load after login
        console.print("[bold yellow]⚠ Attendi che la pagina SharePoint sia completamente caricata[/bold yellow]")
        console.print("[bold yellow]  poi premi INVIO per continuare con il download...[/bold yellow]")
        input("  >>> Premi INVIO quando la pagina è caricata: ")

        # Try to extract date from page if not found in URL
        if not recording_date:
            recording_date = extract_date_from_page(page)

        cookies = context.cookies()
        browser_instance.close()

    # Download with Playwright cookies
    video_path, metadata_date = _download_with_ytdlp_playwright(sharepoint_url, output_dir, cookies)

    # Prefer URL/page date, fall back to metadata date
    final_date = recording_date or metadata_date
    if final_date:
        date_file = output_dir / "lecture_date.txt"
        date_file.write_text(final_date)
        console.print(f"[dim]Date saved to {date_file}[/dim]")

    return video_path, final_date