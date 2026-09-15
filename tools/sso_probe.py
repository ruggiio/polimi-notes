#!/usr/bin/env python3
"""
sso_probe.py — Diagnostica del login PoliMi/Webex per capire come renderlo non interattivo.

Cosa fa:
  - apre Chromium con un PROFILO PERSISTENTE (user_data_dir), così cookie/localStorage/
    session-storage sopravvivono tra un avvio e l'altro (a differenza del JSON di cookie).
  - naviga all'URL dato e registra ogni navigazione del frame principale (catena redirect).
  - classifica la pagina corrente (webex_email / polimi_login / polimi_2fa / continua /
    webex_playback / unknown) e, con --auto, prova a compilare i campi da .env.
  - la pagina "Continua" viene cliccata via JS (bypassa l'overlay CSS).
  - intercetta l'URL del media (nln*.wbx.com / .m3u8 / .mp4) come prova di "arrivato al video".
  - alla fine scrive un report JSON con: catena redirect, stati visti, cookie per dominio con
    scadenza leggibile, se è servito login, se la 2FA è comparsa.

Uso:
  .venv/bin/python tools/sso_probe.py URL                   # headed, login manuale (osserva e basta)
  .venv/bin/python tools/sso_probe.py URL --auto            # headed, prova login automatico da .env
  .venv/bin/python tools/sso_probe.py URL --auto --headless # seconda corsa: la sessione persiste?

.env: POLIMI_USER (codice persona), POLIMI_EMAIL, POLIMI_PASS
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MEDIA_KW = ["nln1.wbx.com", "nln2.wbx.com", ".m3u8", ".mp4"]
PLAYBACK_KW = ["recording/play", "streamurl", "ciscospark"]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def redact(url: str) -> str:
    """Toglie query string lunghe (SAMLRequest, token...) ma tiene host+path."""
    return re.sub(r"\?.*$", lambda m: "?" + m.group(0)[1:41] + ("…" if len(m.group(0)) > 41 else ""), url)


class Probe:
    def __init__(self, url: str, profile: Path, headless: bool, auto: bool, timeout: int,
                 match: str | None = None, ay: int | None = None):
        self.url = url
        self.match = match
        self.ay = ay
        self.profile = profile
        self.headless = headless
        self.auto = auto
        self.timeout = timeout
        self.log_lines: list[str] = []
        self.nav_chain: list[dict] = []
        self.states_seen: list[dict] = []
        self.media_url: str | None = None
        self.playback_url: str | None = None
        self.login_required = False
        self.twofa_seen = False
        self.continua_seen = False
        self.continua_clicked = False
        self.page_title: str | None = None
        self.twofa_snippet: str | None = None
        self.registrazioni_followed = False
        self.portale_clicked = False
        self.registrazioni_entries: list[dict] = []

    # ── logging ──────────────────────────────────────────────────────────────
    def log(self, msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        self.log_lines.append(line)

    # ── page classification ─────────────────────────────────────────────────
    def classify(self, page) -> str:
        url = page.url.lower()
        try:
            if any(k in url for k in ["recordingservice", "/playback", "/recording/"]) and "webex.com" in url:
                return "webex_playback"
            if "idbroker" in url or "webex.com/idb" in url or "/idb/" in url:
                return "webex_email"
            if "servizicie.interno.gov.it" in url or "cie.polimi.it" in url:
                return "cie_2fa"
            if any(k in url for k in ["aunicalogin", "auth.polimi", "login.polimi", "sso.polimi", "shibidp"]):
                # dentro il dominio PoliMi: distingui login / 2fa / continua
                if page.locator('button:has-text("Continua"), input[value="Continua"], a:has-text("Continua")').count() > 0 \
                        and page.locator('input[type="password"]').count() == 0:
                    return "continua"
                if page.locator('input[type="password"]').count() > 0:
                    return "polimi_login"
                body = ""
                try:
                    body = page.locator("body").inner_text(timeout=1500).lower()
                except Exception:
                    pass
                if page.locator('input[autocomplete="one-time-code"], input[name*="otp" i], input[id*="otp" i], input[name*="token" i]').count() > 0 \
                        or any(k in body for k in ["due fattori", "two-factor", "codice di verifica", "one-time", "authenticator", "otp"]):
                    if not self.twofa_snippet:
                        self.twofa_snippet = body[:600]
                    return "polimi_2fa"
                return "polimi_other"
            if "login.microsoftonline" in url:
                return "microsoft_login"
            if "polimi.it" in url:
                if "portaleservizi/controller/Portale.do" in page.url:
                    return "portale"
                try:
                    txt = page.locator("body").inner_text(timeout=1500).lower()
                    if "data registrazione" in txt or "riproduci" in txt:
                        return "registrazioni"
                except Exception:
                    pass
                return "polimi_other"
            return "unknown"
        except Exception as e:
            return f"error:{type(e).__name__}"

    # ── automated actions ───────────────────────────────────────────────────
    def act(self, page, state: str):
        if not self.auto:
            return
        email = os.environ.get("POLIMI_EMAIL", "")
        codice = os.environ.get("POLIMI_USER", "")
        password = os.environ.get("POLIMI_PASS", "")

        if state == "webex_email":
            if not email:
                self.log("  ! POLIMI_EMAIL mancante in .env — compila a mano")
                return
            f = page.locator('input[type="email"], input[name="email"], input[placeholder*="mail" i]').first
            if f.count() and f.input_value() == "":
                f.fill(email)
                f.press("Enter")
                self.log(f"  → inserita email Webex ({email})")
                time.sleep(2)

        elif state == "polimi_login":
            if not (codice and password):
                self.log("  ! POLIMI_USER / POLIMI_PASS mancanti in .env — compila a mano")
                return
            u = page.locator('input[placeholder="Codice Persona"], input[name="j_username"], #j_username, '
                             'input[autocomplete="username"], input[name="username"], input[type="text"]').first
            p = page.locator('input[type="password"]').first
            if u.count() and u.input_value() == "":
                u.fill(codice)
                p.fill(password)
                self.log(f"  → inserite credenziali PoliMi (codice {codice})")
                before = page.url
                p.press("Enter")
                page.wait_for_timeout(3000)
                if page.url == before and page.locator('input[type="password"]').count() > 0:
                    clicked = page.evaluate("""() => {
                        const el = [...document.querySelectorAll('button, input[type=submit], input[type=button], a')]
                          .find(e => (e.innerText || e.value || '').trim().toLowerCase() === 'accedi');
                        if (!el) return null; el.click(); return el.outerHTML.slice(0, 160);
                    }""")
                    self.log(f"  → Enter non ha inviato, click JS su Accedi: {clicked}")
                    page.wait_for_timeout(3000)

        elif state == "continua" and not self.continua_clicked:
            # click via JS: ignora overlay/visibilità
            clicked = page.evaluate("""() => {
                const el = [...document.querySelectorAll('button, input[type=submit], input[type=button], a')]
                  .find(e => (e.innerText || e.value || '').trim().toLowerCase() === 'continua');
                if (!el) return null;
                el.click();
                return el.outerHTML.slice(0, 200);
            }""")
            if clicked:
                self.continua_clicked = True
                self.log(f"  → cliccato 'Continua' via JS: {clicked}")
                time.sleep(2)
            else:
                self.log("  ! 'Continua' presente ma non trovato via JS")

        elif state == "portale" and not self.portale_clicked:
            self.portale_clicked = True
            link = page.locator('a[href*="Servizi.do"][href*="idServizio=2314"]').first
            if link.count():
                self.log("  → clic su 'Archivio registrazioni didattica' (idServizio=2314)")
                link.click()
                page.wait_for_load_state("networkidle", timeout=30_000)
            else:
                self.log("  ! link idServizio=2314 non trovato nel portale")

        elif state == "registrazioni" and not self.registrazioni_followed:
            self.follow_registrazioni(page)

        elif state in ("polimi_2fa", "cie_2fa"):
            self.log("  ! 2FA richiesta — nessuna automazione. Completa a mano nel browser (60s)…")

        elif state == "webex_playback" and not (self.media_url or self.playback_url):
            try:
                btn = page.locator('button[aria-label*="play" i], button[title*="play" i], .play-button, #play-btn, [class*="play"]').first
                if btn.count():
                    btn.click(timeout=2000)
                    self.log("  → cliccato play")
            except Exception:
                pass

    # ── registrazioni page ───────────────────────────────────────────────────
    def follow_registrazioni(self, page):
        """Elenca le righe della tabella registrazioni e apre la più recente."""
        self.registrazioni_followed = True
        try:
            tutte = page.locator('a:text-is("tutte"), a:text-is("Tutte")')
            if tutte.count() > 0:
                tutte.first.click()
                page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        rows = page.locator("table tr").all()
        entries = []
        for row in rows:
            cells = row.locator("td").all()
            if len(cells) < 5:
                continue
            date_raw = cells[1].inner_text().strip().split("\n")[0].strip()
            topic = cells[4].inner_text().strip().replace("\n", " ")
            course = cells[2].inner_text().strip().replace("\n", " ")
            href = None
            try:
                href = cells[0].locator("a").first.get_attribute("href", timeout=1000)
            except Exception:
                pass
            entries.append({"date": date_raw, "course": course, "topic": topic, "href": href, "row": row})
        self.registrazioni_entries = [{k: v for k, v in e.items() if k != "row"} for e in entries]
        self.log(f"REGISTRAZIONI: {len(entries)} righe")
        for e in entries[:60]:
            self.log(f"   {e['date']:18s} {e['course'][:28]:28s} {e['topic'][:40]:40s} href={'sì' if e['href'] else 'no'}")
        if not entries:
            self.log("  ! nessuna riga trovata: dump primi 600 char del body")
            try:
                self.log("   " + page.locator("body").inner_text(timeout=2000)[:600].replace("\n", " | "))
            except Exception:
                pass
            return
        # filtro per corso / anno accademico e scelta della più vecchia (= prima lezione)
        def parse_date(d):
            try:
                dd, mm, yy = d.split(" ")[0].split("/")
                return (int(yy), int(mm), int(dd))
            except Exception:
                return None
        cand = entries
        if self.match:
            m = self.match.lower()
            cand = [e for e in cand if m in e["course"].lower() or m in e["topic"].lower()]
            self.log(f"  filtro corso '{self.match}': {len(cand)} righe")
        if self.ay:
            lo, hi = (self.ay, 9, 1), (self.ay + 1, 8, 31)
            cand = [e for e in cand if (pd := parse_date(e["date"])) and lo <= pd <= hi]
            self.log(f"  filtro A.A. {self.ay}/{self.ay + 1 - 2000}: {len(cand)} righe")
        if not cand:
            self.log("  ! nessuna riga combacia. Dropdown/select presenti nella pagina:")
            try:
                sels = page.evaluate("""() => [...document.querySelectorAll('select')].map(s => ({
                    name: s.name || s.id, options: [...s.options].slice(0, 12).map(o => o.text.trim()) }))""")
                for sel in sels:
                    self.log(f"   select {sel['name']}: {sel['options']}")
                links = page.evaluate("""() => [...document.querySelectorAll('a')].map(a => a.innerText.trim())
                    .filter(t => /20[0-9][0-9]/.test(t)).slice(0, 20)""")
                self.log(f"   link con anno: {links}")
            except Exception as e:
                self.log(f"   (errore dump: {e})")
            return
        cand.sort(key=lambda e: parse_date(e["date"]) or (0, 0, 0))
        for e in cand[:5]:
            self.log(f"   candidata {e['date']:18s} {e['topic'][:60]}")
        target = cand[0]
        self.log(f"  → seguo la PRIMA lezione: {target['date']} — {target['topic'][:50]}")
        link = target["row"].locator("td").first.locator("a, button").first
        href = target["href"]
        if href and href.startswith("http"):
            page.goto(href, wait_until="commit", timeout=60_000)
            return
        try:
            with page.context.expect_page(timeout=6_000) as npi:
                link.click()
            newp = npi.value
            newp.wait_for_load_state("commit", timeout=15_000)
            self.log(f"  → aperta nuova scheda: {redact(newp.url)[:110]}")
            page.goto(newp.url, wait_until="commit", timeout=60_000)
            newp.close()
        except Exception:
            link.click()
            self.log("  → click nella stessa scheda")

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self) -> dict:
        self.profile.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                user_data_dir=str(self.profile),
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
                viewport={"width": 1280, "height": 800},
            )
            ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            def on_nav(frame):
                if frame == page.main_frame:
                    self.nav_chain.append({"t": ts(), "url": frame.url})
                    self.log(f"NAV  {redact(frame.url)}")

            def on_request(req):
                u = req.url
                ul = u.lower()
                if self.media_url is None and any(k in ul for k in MEDIA_KW):
                    self.media_url = u
                    self.log(f"MEDIA {redact(u)[:140]}")
                if self.playback_url is None and any(k in ul for k in PLAYBACK_KW):
                    self.playback_url = u
                    self.log(f"PLAYBACK {redact(u)[:140]}")

            ctx.on("page", lambda np: self.log(f"POPUP nuova scheda: {redact(np.url)[:120]}"))
            page.on("framenavigated", on_nav)
            page.on("request", on_request)

            self.log(f"GOTO {self.url}  (profile={self.profile}, headless={self.headless}, auto={self.auto})")
            try:
                page.goto(self.url, wait_until="commit", timeout=60_000)
            except PWTimeout:
                self.log("  goto timeout (continuo comunque)")

            deadline = time.time() + self.timeout
            last_state = None
            last_change = time.time()
            while time.time() < deadline:
                state = self.classify(page)
                if state != last_state:
                    self.log(f"STATE {state:16s} @ {redact(page.url)[:110]}")
                    self.states_seen.append({"t": ts(), "state": state, "url": page.url})
                    if state in ("webex_email", "polimi_login"):
                        self.login_required = True
                    if state in ("polimi_2fa", "cie_2fa"):
                        self.twofa_seen = True
                    if state == "continua":
                        self.continua_seen = True
                    last_state = state
                    last_change = time.time()
                try:
                    self.act(page, state)
                except Exception as e:
                    self.log(f"  act() errore su {state}: {type(e).__name__}: {str(e)[:120]}")
                if self.media_url:
                    self.log("✓ URL media intercettato → login + playback OK")
                    break
                if state == "webex_playback" and self.playback_url and time.time() - last_change > 20:
                    self.log("✓ playback URL trovato (nessun media diretto entro 20s)")
                    break
                try:
                    page.wait_for_timeout(500)
                except Exception:
                    time.sleep(0.5)
            else:
                self.log(f"✗ timeout {self.timeout}s: stato finale {last_state}")

            try:
                self.page_title = page.title()
            except Exception:
                pass

            cookies = ctx.cookies()
            ctx.close()

        return self.report(cookies)

    # ── report ───────────────────────────────────────────────────────────────
    def report(self, cookies: list[dict]) -> dict:
        now = time.time()
        by_domain: dict[str, list[dict]] = {}
        for c in cookies:
            exp = c.get("expires", -1)
            if exp and exp > 0:
                exp_iso = datetime.fromtimestamp(exp, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
                ttl_h = round((exp - now) / 3600, 1)
            else:
                exp_iso, ttl_h = "session", None
            by_domain.setdefault(c["domain"], []).append({
                "name": c["name"], "expires": exp_iso, "ttl_hours": ttl_h,
                "httpOnly": c.get("httpOnly"), "secure": c.get("secure"), "len": len(c.get("value", "")),
            })
        for d in by_domain:
            by_domain[d].sort(key=lambda x: (x["ttl_hours"] is None, -(x["ttl_hours"] or 0)))

        rep = {
            "url": self.url, "auto": self.auto, "headless": self.headless,
            "result": "media_found" if self.media_url else ("playback_found" if self.playback_url else "failed"),
            "login_required": self.login_required, "twofa_seen": self.twofa_seen,
            "twofa_snippet": self.twofa_snippet,
            "continua_seen": self.continua_seen, "continua_clicked": self.continua_clicked,
            "page_title": self.page_title,
            "media_url": self.media_url, "playback_url": self.playback_url,
            "registrazioni_entries": self.registrazioni_entries,
            "states": self.states_seen, "nav_chain": self.nav_chain,
            "cookies_by_domain": by_domain,
        }
        return rep


def print_summary(rep: dict):
    print("\n" + "═" * 70)
    print(f"RISULTATO: {rep['result']}   login richiesto: {rep['login_required']}   "
          f"2FA: {rep['twofa_seen']}   Continua: visto={rep['continua_seen']} cliccato={rep['continua_clicked']}")
    print(f"Titolo pagina: {rep['page_title']}")
    print("\nStati attraversati:")
    for s in rep["states"]:
        print(f"  {s['t']}  {s['state']:16s} {redact(s['url'])[:90]}")
    print("\nCookie più longevi per dominio (ttl in ore):")
    for d, cs in sorted(rep["cookies_by_domain"].items()):
        top = cs[0]
        n_sess = sum(1 for c in cs if c["ttl_hours"] is None)
        print(f"  {d:40s} n={len(cs):2d} (session={n_sess})  max: {top['name'][:28]:28s} → {top['expires']} ({top['ttl_hours']}h)")
    if rep.get("twofa_snippet"):
        print("\nPagina 2FA (estratto):\n  " + rep["twofa_snippet"][:400].replace("\n", " | "))
    print("═" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--profile", default=str(ROOT / "config" / "chrome_profile"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--auto", action="store_true", help="compila email/credenziali da .env")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--report", default=None)
    ap.add_argument("--match", default=None, help="sottostringa del corso/argomento (case-insensitive)")
    ap.add_argument("--ay", type=int, default=None, help="anno accademico di inizio, es. 2025 per il 25/26")
    a = ap.parse_args()

    probe = Probe(a.url, Path(a.profile), a.headless, a.auto, a.timeout, a.match, a.ay)
    rep = probe.run()
    print_summary(rep)

    out = Path(a.report) if a.report else ROOT / "output" / "probe" / f"probe_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    (out.with_suffix(".log")).write_text("\n".join(probe.log_lines))
    print(f"\nReport: {out}")
    sys.exit(0 if rep["result"] != "failed" else 1)


if __name__ == "__main__":
    main()
