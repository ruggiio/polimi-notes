"""
batch_scraper.py — Scrape PoliMi registrazioni page to collect all Webex lecture URLs.

Flow:
  1. Open browser (non-headless so user can login manually if needed)
  2. Navigate to the registrazioni page URL
  3. Set "tutte" to show all records at once
  4. Extract each row: date, topic, play button URL
  5. For play buttons that navigate to a new page, capture the destination URL
  6. Return list of {"date", "topic", "webex_url"}
"""

import time
from pathlib import Path

from rich.console import Console

console = Console()


def scrape_registrazioni(
    registrazioni_url: str,
    cookies_file: Path,
) -> list[dict]:
    """
    Navigate the PoliMi registrazioni page and collect all lecture entries.
    Returns list of {"date": str, "topic": str, "webex_url": str}.
    Login is manual: user completes SSO in the opened browser, then presses ENTER.
    """
    from playwright.sync_api import sync_playwright
    from src.downloader.downloader import _load_cookies, _save_cookies

    entries = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )

        _load_cookies(context, cookies_file)
        page = context.new_page()

        console.print(f"[cyan]Navigating to registrazioni page...[/cyan]")
        try:
            page.goto(registrazioni_url, wait_until="networkidle", timeout=30_000)
        except Exception:
            pass

        needs_login = any(kw in page.url for kw in [
            "idbroker", "auth.polimi", "aunicalogin", "shibidp", "login"
        ])
        if needs_login:
            console.print("\n[bold yellow]══════════════════════════════════════════════════[/bold yellow]")
            console.print("[bold yellow]  Fai il login PoliMi nel browser aperto.[/bold yellow]")
            console.print("[yellow]  Quando sei sulla pagina delle registrazioni, premi INVIO.[/yellow]")
            console.print("[bold yellow]══════════════════════════════════════════════════[/bold yellow]\n")
            input("  >>> Premi INVIO quando sei sulla pagina registrazioni: ")
            _save_cookies(page, cookies_file)

        # Show all records at once — click "tutte"
        try:
            tutte = page.locator('a:text-is("tutte"), a:text-is("Tutte")')
            if tutte.count() > 0:
                tutte.first.click()
                page.wait_for_load_state("networkidle", timeout=15_000)
                console.print("[dim]Set to show all records[/dim]")
        except Exception:
            console.print("[dim]Could not click 'tutte' — using current page[/dim]")

        time.sleep(1)

        # Parse table rows
        # Columns: Riproduci(0) | Data Registrazione(1) | Corso(2) | Forma didattica(3) | Argomento(4) | Ospiti(5) | Durata(6) | Dimensione(7)
        rows = page.locator("table tr").all()
        console.print(f"[cyan]Parsing {len(rows)} rows...[/cyan]")

        for row in rows:
            cells = row.locator("td").all()
            if len(cells) < 5:
                continue

            # Date: "25/05/2026 10:31" → "2026-05-25"
            date_raw = cells[1].inner_text().strip().split("\n")[0].strip()
            date_part = date_raw.split(" ")[0] if date_raw else ""
            if "/" in date_part:
                d, m, y = date_part.split("/")
                date_iso = f"{y}-{m}-{d}"
            else:
                date_iso = date_part

            # Topic
            topic = cells[4].inner_text().strip().replace("\n", " ")

            if not date_iso or not topic:
                continue

            # Play URL — try href first, then click-and-capture
            webex_url = None
            play_cell = cells[0]

            try:
                href = play_cell.locator("a").first.get_attribute("href")
                if href:
                    webex_url = href if href.startswith("http") else _resolve_url(registrazioni_url, href)
            except Exception:
                pass

            if not webex_url:
                # Click play and capture new tab / navigation
                try:
                    with context.expect_page(timeout=6_000) as new_page_info:
                        play_cell.locator("a, button").first.click()
                    new_pg = new_page_info.value
                    new_pg.wait_for_load_state("commit", timeout=10_000)
                    webex_url = new_pg.url
                    new_pg.close()
                except Exception:
                    try:
                        play_cell.locator("a, button").first.click()
                        page.wait_for_load_state("networkidle", timeout=8_000)
                        if page.url != registrazioni_url:
                            webex_url = page.url
                            page.go_back(wait_until="networkidle", timeout=8_000)
                    except Exception:
                        pass

            if webex_url:
                entries.append({"date": date_iso, "topic": topic, "webex_url": webex_url})
                console.print(f"  [green]✓[/green] {date_iso} — {topic[:65]}")
            else:
                console.print(f"  [yellow]⚠ Could not get URL for:[/yellow] {date_iso} — {topic[:50]}")

        browser.close()

    console.print(f"\n[bold green]✓ Scraped {len(entries)} lectures[/bold green]")
    return entries


def _resolve_url(base: str, href: str) -> str:
    from urllib.parse import urljoin
    return urljoin(base, href)
