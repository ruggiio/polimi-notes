"""
archive.py — Accesso NON interattivo all'Archivio registrazioni didattica PoliMi.

Si appoggia a un profilo Chromium persistente (config/chrome_profile) in cui vive la
sessione "resta connesso" di aunicalogin (≈10 giorni). Con quella sessione:
  - il portale Servizi Online e l'archivio si aprono senza credenziali
  - Webex chiede solo l'email; l'IdP PoliMi (shibidp) rilascia l'asserzione SAML
    senza password né 2FA
Quando la sessione è scaduta viene sollevata LoginRequired: il chiamante deve far
rifare il login a mano (tools/sso_login.py) — la 2FA CIE non è automatizzabile.

Flusso verificato (2026-09-15):
  Portale.do → [Sessione terminata → "Continua"] → Servizi.do?idServizio=2314
  → ArchivioListActivity.do (form: aa, tipologia, contesto, EVN_SEARCH)
  → a[href*=evn_preview_link&transfer_id=…] (nuova scheda)
  → politecnicomilano.webex.com/login → idbroker (email) → aunicalogin → shibidp
  → …/recording/<id>/playback → richiesta media nfg1wss.webex.com/nbr/…
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, sync_playwright

PORTALE_URL = ("https://servizionline.polimi.it/portaleservizi/portaleservizi/"
               "controller/Portale.do?jaf_currentWFID=main&EVN_SHOW_PORTALE=evento")
ARCHIVE_SERVICE_ID = "2314"
MEDIA_KW = ("nln1.wbx.com", "nln2.wbx.com", "/nbr/", ".m3u8", ".mp4")
LOGIN_KW = ("servizicie.interno.gov.it", "cie.polimi.it")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

JS_CLICK_BY_TEXT = """(txt) => {
  const el = [...document.querySelectorAll('button, input[type=submit], input[type=button], a')]
    .find(e => (e.innerText || e.value || '').trim().toLowerCase() === txt);
  if (!el) return false; el.click(); return true; }"""


class LoginRequired(RuntimeError):
    """La sessione PoliMi è scaduta: serve un login manuale (password + 2FA)."""


class PortalUnavailable(RuntimeError):
    """Il portale PoliMi risponde con un errore temporaneo (POLIJ_xxxxx / pagina troncata)."""


@dataclass
class Recording:
    transfer_id: str
    date: str            # "29/09/2025 08:34"
    date_iso: str        # "2025-09-29"
    course: str          # "062322 - BIOINSPIRED ROBOTICS (CINQUEMANI SIMONE)"
    kind: str            # Lezione / Laboratorio / ...
    topic: str
    duration: str
    size: str
    href: str

    @property
    def course_code(self) -> str:
        return self.course.split(" - ")[0].strip()

    @property
    def course_name(self) -> str:
        return re.sub(r"^\d+\s*-\s*", "", self.course).split(" (")[0].strip()

    def to_dict(self) -> dict:
        return asdict(self)


# ── sessione ─────────────────────────────────────────────────────────────────

def open_context(pw, profile_dir: Path, headless: bool = True) -> BrowserContext:
    profile_dir.mkdir(parents=True, exist_ok=True)
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=headless,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
        user_agent=UA,
        viewport={"width": 1280, "height": 800},
    )
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
    return ctx


def _is_login_page(page: Page) -> bool:
    url = page.url
    if any(k in url for k in LOGIN_KW):
        return True
    if "aunicalogin" in url and "aunicalogin.jsp" in url:
        try:
            return page.locator('input[type="password"]').count() > 0
        except Exception:
            return False
    return False


def ensure_loaded(page: Page, tries: int = 3, min_bytes: int = 1000) -> None:
    """Ricarica se il server ha risposto con un documento troncato (capita a PoliMi nelle ore serali)."""
    for i in range(tries):
        try:
            if len(page.content()) >= min_bytes:
                return
        except Exception:
            pass
        page.wait_for_timeout(2000 * (i + 1))
        page.reload(wait_until="load", timeout=60_000)


def goto_robust(page: Page, url: str, tries: int = 4, min_bytes: int = 1000) -> None:
    """goto + ricarica se il server risponde con un documento troncato."""
    page.goto(url, wait_until="load", timeout=60_000)
    ensure_loaded(page, tries, min_bytes)


def _pass_continua(page: Page, timeout_s: float = 45) -> None:
    """Pagina 'Sessione terminata' → bottone Continua (cliccato via JS: è coperto da un overlay).
    Il server può metterci parecchi secondi a renderla: si aspetta l'elemento, non un tempo fisso."""
    if "SessioneTerminata" not in page.url:
        return
    try:
        page.locator('a:has-text("Continua"), button:has-text("Continua"), input[value="Continua"]').first \
            .wait_for(state="attached", timeout=timeout_s * 1000)
    except Exception:
        raise RuntimeError(f"bottone 'Continua' non comparso entro {timeout_s}s @ {page.url}")
    deadline = time.time() + timeout_s
    while time.time() < deadline and "SessioneTerminata" in page.url:
        if page.evaluate(JS_CLICK_BY_TEXT, "continua"):
            try:
                page.wait_for_url(lambda u: "SessioneTerminata" not in u, timeout=timeout_s * 1000)
                page.wait_for_load_state("networkidle", timeout=timeout_s * 1000)
            except Exception:
                pass
            ensure_loaded(page)
            return
        page.wait_for_timeout(300)
    raise RuntimeError(f"bottone 'Continua' non cliccabile @ {page.url}")


def _dump_page(page: Page, tag: str) -> str:
    """Salva HTML e testo della pagina corrente per diagnosi; ritorna il path."""
    try:
        d = Path("output/probe"); d.mkdir(parents=True, exist_ok=True)
        base = d / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
        base.with_suffix(".html").write_text(page.content(), encoding="utf-8")
        base.with_suffix(".txt").write_text(page.locator("body").inner_text(timeout=3000), encoding="utf-8")
        return str(base.with_suffix(".html"))
    except Exception as e:
        return f"(dump fallito: {e})"


def _recover_internal_error(page: Page) -> None:
    """
    Il portale a volte risponde 500 con "Errore interno, fai click per effettuare il logout e
    ricominciare": si segue quel link (resetta la sessione dell'applicazione, non il 'resta
    connesso' di aunicalogin) e si supera di nuovo la pagina 'Continua'.
    """
    try:
        body = page.content()
    except Exception:
        return
    if "POLIJ_" in body or "Server error" in body:
        # errore applicativo del portale ("temporaneamente non disponibile"): torna al safe point e riprova
        m = re.search(r"\(POLIJ_\d+\)", body)
        try:
            page.locator('a[href*="evn_gotosafepoint"]').first.click(timeout=5_000)
            page.wait_for_load_state("load", timeout=30_000)
        except Exception:
            pass
        raise PortalUnavailable(f"portale PoliMi in errore {m.group(0) if m else ''} (temporaneo)")
    if "rrore interno" not in body:
        return
    m = re.search(r"""href=['"]([^'"]+)['"]""", body)
    if not m:
        raise RuntimeError("errore interno del portale senza link di ripristino")
    page.goto(m.group(1), wait_until="load", timeout=60_000)
    _pass_continua(page, timeout_s=20)
    if not any(c["name"] == "RESTA_CONNESSO" for c in page.context.cookies() if c["domain"] == "aunicalogin.polimi.it"):
        raise LoginRequired("il ripristino del portale ha invalidato la sessione 'resta connesso'")


RETRY_WAITS_MS = (15_000, 30_000, 60_000, 120_000)


def goto_archive(page: Page, attempts: int = 5) -> None:
    """
    Portale → Archivio registrazioni didattica. Solleva LoginRequired se la sessione è scaduta.
    Il portale a volte risponde 500 / pagina troncata (intermittente, lato server): l'intera
    sequenza viene ripetuta con pause crescenti prima di arrendersi.
    """
    last = ""
    for i in range(attempts):
        try:
            goto_robust(page, PORTALE_URL)
            _pass_continua(page, timeout_s=20)
            ensure_loaded(page)
            if _is_login_page(page):
                raise LoginRequired(f"login richiesto @ {page.url}")
            _recover_internal_error(page)
            link = page.locator(f'a[href*="Servizi.do"][href*="idServizio={ARCHIVE_SERVICE_ID}"]').first
            try:
                link.wait_for(state="attached", timeout=20_000)
            except Exception:
                if _is_login_page(page):
                    raise LoginRequired(f"login richiesto @ {page.url}")
                dump = _dump_page(page, "portal_nolink")
                raise RuntimeError(f"link archivio non trovato @ {page.url} (dump: {dump})")
            link.click()
            page.wait_for_load_state("networkidle", timeout=30_000)
            if _is_login_page(page):
                raise LoginRequired(f"login richiesto @ {page.url}")
            if "ArchivioListActivity" not in page.url:
                raise RuntimeError(f"archivio non raggiunto @ {page.url}")
            return
        except LoginRequired:
            raise
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:160]}"
            if i < attempts - 1:
                page.wait_for_timeout(RETRY_WAITS_MS[min(i, len(RETRY_WAITS_MS) - 1)])
    # dopo N tentativi: se siamo comunque davanti a una pagina di login è la sessione, altrimenti il server
    if _is_login_page(page):
        raise LoginRequired(f"login richiesto @ {page.url}")
    raise PortalUnavailable(f"portale non raggiungibile dopo {attempts} tentativi — {last}")


# ── ricerca ──────────────────────────────────────────────────────────────────

def _to_iso(date_raw: str) -> str:
    try:
        d, m, y = date_raw.split(" ")[0].split("/")
        return f"{y}-{m}-{d}"
    except Exception:
        return ""


def search_archive(page: Page, aa: int | None = None, course: str | None = None,
                   kind: str | None = None) -> list[Recording]:
    """
    Applica i filtri dell'archivio e restituisce tutte le righe (pagina 'tutte').
    aa: anno di inizio A.A. (2025 → "2025 / 26"); course: sottostringa del campo Corso
    (es. "BIOINSPIRED" o il codice "062322"); kind: LE/LA/ES/OT.
    """
    if aa is not None:
        page.select_option('select[name="aa"]', str(aa))
    if course:
        page.fill('input[name="contesto"]', course)
    if kind:
        page.select_option('select[name="tipologia"]', kind)
    page.locator('button[name="EVN_SEARCH"]:has-text("Cerca")').first.click()
    page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_timeout(500)

    tutte = page.locator('a:text-is("tutte")')
    if tutte.count():
        tutte.first.click()
        page.wait_for_load_state("networkidle", timeout=30_000)
        page.wait_for_timeout(500)

    rows = page.evaluate("""() => [...document.querySelectorAll('a[href*="evn_preview_link"]')].map(a => {
        const td = [...a.closest('tr').querySelectorAll('td')].map(x => x.innerText.trim().replace(/\\s+/g, ' '));
        return {href: a.getAttribute('href'), td}; })""")
    out = []
    for r in rows:
        td = r["td"]
        if len(td) < 9:
            continue
        m = re.search(r"transfer_id=(\d+)", r["href"])
        if not m:
            continue
        out.append(Recording(
            transfer_id=m.group(1), date=td[2], date_iso=_to_iso(td[2]), course=td[3],
            kind=td[4], topic=td[5], duration=td[7], size=td[8], href=r["href"],
        ))
    out.sort(key=lambda x: (x.date_iso, x.date))
    return out


# ── apertura registrazione su Webex ──────────────────────────────────────────

@dataclass
class MediaInfo:
    media_url: str          # URL diretto del file mp4 (byte-range capable)
    hls_url: str | None     # playlist HLS (fallback per yt-dlp)
    headers: dict           # Referer / User-Agent usati dal player
    cookies: list           # cookie del contesto browser
    title: str
    playback_url: str
    size: int | None        # Content-Length totale dichiarato dal server


def open_recording(ctx: BrowserContext, page: Page, rec: Recording, email: str,
                   timeout_s: int = 120) -> MediaInfo:
    """
    Clicca "Riproduci" (nuova scheda), supera il login Webex con la sola email, avvia il
    player e intercetta la risposta video/mp4. Solleva LoginRequired se compare password/2FA.
    """
    found: dict = {}
    with ctx.expect_page(timeout=15_000) as npi:
        page.locator(f'a[href*="transfer_id={rec.transfer_id}"]').first.click()
    p = npi.value

    def on_response(r):
        if "/nbr/" not in r.url:
            return
        ct = r.headers.get("content-type", "")
        if "video/mp4" in ct and "mp4" not in found:
            found["mp4"] = r.url
            found["headers"] = {k: v for k, v in r.request.headers.items()
                                if k.lower() in ("referer", "user-agent", "origin", "accept")}
            cr = r.headers.get("content-range", "")
            found["size"] = int(cr.rsplit("/", 1)[-1]) if "/" in cr else None
        elif "mpegurl" in ct and "hls" not in found:
            found["hls"] = r.url
    p.on("response", on_response)

    email_done = played = False
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p.wait_for_timeout(500)
        if "mp4" in found:
            break
        u = p.url
        if any(k in u for k in LOGIN_KW):
            raise LoginRequired(f"2FA richiesta @ {u}")
        try:
            if not email_done and ("idbroker" in u or "/idb/" in u):
                f = p.locator('input[type="email"], input[name="email"], #IDToken1, input[placeholder*="mail" i]').first
                if f.count() and f.is_visible():
                    f.fill(email)
                    f.press("Enter")
                    email_done = True
                    continue
            pwd = p.locator('input[type="password"]')
            if pwd.count() and pwd.first.is_visible():
                raise LoginRequired(f"password richiesta @ {u}")
            if "recordingservice" in u and time.time() - t0 > 5:
                # il player carica il manifest da solo; il video parte solo con play()
                p.evaluate("() => { const v = document.querySelector('video'); if (v) { v.muted = true; v.play(); } }")
                played = True
        except LoginRequired:
            raise
        except Exception:
            pass

    if "mp4" not in found:
        body = ""
        try:
            body = p.locator("body").inner_text(timeout=2000)[:300]
        except Exception:
            pass
        url = p.url
        p.close()
        raise RuntimeError(f"URL mp4 non trovato entro {timeout_s}s @ {url} (hls={'hls' in found}) — {body!r}")

    info = MediaInfo(media_url=found["mp4"], hls_url=found.get("hls"), headers=found["headers"],
                     cookies=ctx.cookies(), title=p.title(), playback_url=p.url, size=found.get("size"))
    p.close()
    return info


# ── download ─────────────────────────────────────────────────────────────────

def download_media(info: MediaInfo, out_path: Path, chunk: int = 1 << 20,
                   progress=None) -> Path:
    """
    GET diretto del file mp4 (byte-range) con Referer/UA del player e i cookie del browser.
    Riprende da <out>.part se esiste. Verifica la dimensione finale contro Content-Length.
    """
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")
    sess = requests.Session()
    for c in info.cookies:
        if "webex.com" in c["domain"]:
            sess.cookies.set(c["name"], c["value"], domain=c["domain"], path=c.get("path", "/"))
    headers = {"User-Agent": UA, "Referer": "https://politecnicomilano.webex.com/", **info.headers}
    headers.pop("accept", None)

    done = part.stat().st_size if part.exists() else 0
    if done:
        headers["Range"] = f"bytes={done}-"
    with sess.get(info.media_url, headers=headers, stream=True, timeout=(15, 120)) as r:
        if done and r.status_code != 206:
            done = 0  # il server non ha accettato il resume: ricomincia
        r.raise_for_status()
        total = done + int(r.headers.get("content-length", 0))
        mode = "ab" if done else "wb"
        with open(part, mode) as f:
            for blk in r.iter_content(chunk_size=chunk):
                f.write(blk)
                done += len(blk)
                if progress:
                    progress(done, total)

    size = part.stat().st_size
    if info.size and size != info.size:
        raise RuntimeError(f"download incompleto: {size} / {info.size} byte (riprovare per riprendere)")
    part.rename(out_path)
    return out_path


def session_expiry(profile_dir: Path):
    """
    Scadenza (datetime locale) del cookie RESTA_CONNESSO/SSO_LOGIN di aunicalogin letta dal
    DB cookie di Chromium nel profilo persistente; None se assente. Da usare a browser chiuso.
    """
    import shutil
    import sqlite3
    import tempfile
    from datetime import datetime, timedelta
    db = profile_dir / "Default" / "Cookies"
    if not db.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "Cookies"
        shutil.copy2(db, tmp)
        con = sqlite3.connect(tmp)
        try:
            rows = con.execute(
                "SELECT name, expires_utc FROM cookies WHERE host_key='aunicalogin.polimi.it' "
                "AND name IN ('RESTA_CONNESSO','SSO_LOGIN','polij_user') AND expires_utc > 0").fetchall()
        finally:
            con.close()
    if not rows:
        return None
    exp = max(r[1] for r in rows)                     # microsecondi dal 1601-01-01 (epoch Chromium)
    return datetime(1601, 1, 1) + timedelta(microseconds=exp) + (datetime.now() - datetime.utcnow())


def safe_name(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^\w\s\-.]", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:maxlen].strip()
