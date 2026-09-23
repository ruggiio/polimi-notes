"""
onedrive.py — Sincronizza in locale le cartelle OneDrive/SharePoint linkate dai corsi WeBeep.

Molti docenti pubblicano su WeBeep solo un modulo `url` che punta a una cartella condivisa
di OneDrive (polimi365-my.sharepoint.com). WeBeep Sync ignora i moduli `url`; qui:
  1. si apre il link WeBeep con il profilo Chromium persistente (SSO PoliMi già valido:
     WeBeep passa da aunicalogin → shibidp senza password né 2FA),
  2. si estrae il link esterno alla cartella condivisa,
  3. si apre la cartella (le condivisioni "chiunque con il link" non chiedono login) e si
     elencano i file con la REST API di SharePoint usando i cookie del browser,
  4. si scaricano (streaming, con resume) i file nuovi o modificati in <dest>/<nome cartella>/.

Lo stato (dimensione + data modifica per file) è in <dest>/.onedrive_sync.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from playwright.sync_api import BrowserContext, Page

from src.downloader.archive import JS_CLICK_BY_TEXT, LoginRequired, UA, goto_robust

SHARE_HOSTS = ("sharepoint.com", "1drv.ms", "onedrive.live.com")


# ── WeBeep ───────────────────────────────────────────────────────────────────

def _pass_webeep_login(page: Page, timeout_s: int = 60) -> None:
    """Se WeBeep mostra la pagina di login, clicca 'Polimi login' e lascia fare all'SSO."""
    t0 = time.time()
    clicked = False
    while time.time() - t0 < timeout_s:
        u = page.url
        if "SessioneTerminata" in u:
            page.evaluate(JS_CLICK_BY_TEXT, "continua")
        elif "webeep.polimi.it/login" in u and not clicked:
            page.evaluate("""() => { const a = [...document.querySelectorAll('a, button')]
                .find(e => /polimi\\s*login/i.test(e.innerText || '')); if (a) a.click(); }""")
            clicked = True
        elif "servizicie" in u or ("aunicalogin.jsp" in u and page.locator('input[type="password"]').count()):
            raise LoginRequired(f"login richiesto @ {u}")
        elif "webeep.polimi.it" in u and "/login" not in u:
            return
        page.wait_for_timeout(700)
    raise RuntimeError(f"WeBeep: login non completato entro {timeout_s}s @ {page.url}")


def resolve_webeep_link(page: Page, url: str) -> str:
    """Da un modulo url di WeBeep (mod/url/view.php?id=…) al link esterno che contiene."""
    if any(h in url for h in SHARE_HOSTS):
        return url
    goto_robust(page, url)
    _pass_webeep_login(page)
    if any(h in page.url for h in SHARE_HOSTS):        # redirect automatico
        return page.url
    page.wait_for_load_state("networkidle", timeout=30_000)
    links = page.evaluate("""() => [...document.querySelectorAll('a')].map(a => a.href)""")
    ext = [l for l in links if any(h in l for h in SHARE_HOSTS)]
    if not ext:
        raise RuntimeError(f"nessun link OneDrive/SharePoint nella pagina {url}")
    return ext[0]


# ── SharePoint ───────────────────────────────────────────────────────────────

def open_shared_folder(page: Page, share_url: str, timeout_s: int = 60) -> tuple[str, str]:
    """Apre il link condiviso; ritorna (base del sito personale, path server-relative della cartella)."""
    page.goto(share_url, wait_until="commit", timeout=60_000)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        page.wait_for_timeout(800)
        u = page.url
        if "login.microsoftonline.com" in u or "login.live.com" in u:
            raise LoginRequired(f"la condivisione richiede il login Microsoft @ {u}")
        q = parse_qs(urlparse(u).query)
        if "sharepoint.com" in u and "id" in q:
            folder = unquote(q["id"][0])
            m = re.match(r"(https://[^/]+/personal/[^/]+)", u) or re.match(r"(https://[^/]+/sites/[^/]+)", u)
            if not m:
                raise RuntimeError(f"URL SharePoint non riconosciuto: {u}")
            page.wait_for_timeout(1500)
            return m.group(1), folder
    raise RuntimeError(f"cartella condivisa non aperta entro {timeout_s}s @ {page.url}")


def _odata_path(path: str) -> str:
    """Path per una stringa letterale OData: l'apostrofo va raddoppiato, poi URL-encoded."""
    return quote(path.replace("'", "''"), safe="/")


def list_folder(page: Page, site: str, folder: str) -> tuple[list[dict], list[str]]:
    """(file, sottocartelle) di una cartella via REST, con i cookie del browser."""
    enc = _odata_path(folder)
    hdr = {"Accept": "application/json;odata=verbose"}
    r = page.request.get(f"{site}/_api/web/GetFolderByServerRelativeUrl('{enc}')/Files"
                         "?$select=Name,Length,TimeLastModified,ServerRelativeUrl", headers=hdr)
    if not r.ok:
        raise RuntimeError(f"REST {r.status} su {folder}: {r.text()[:200]}")
    files = [{"name": f["Name"], "size": int(f["Length"]), "modified": f["TimeLastModified"],
              "url": f["ServerRelativeUrl"]} for f in r.json()["d"]["results"]]
    r = page.request.get(f"{site}/_api/web/GetFolderByServerRelativeUrl('{enc}')/Folders?$select=Name", headers=hdr)
    subs = [f["Name"] for f in r.json()["d"]["results"] if r.ok and not f["Name"].startswith("Forms")] if r.ok else []
    return files, subs


def _download(ctx: BrowserContext, site: str, f: dict, dest: Path, progress=None) -> Path:
    import requests
    host = urlparse(site).netloc
    sess = requests.Session()
    for c in ctx.cookies():
        if host.endswith(c["domain"].lstrip(".")) or c["domain"].lstrip(".") in host:
            sess.cookies.set(c["name"], c["value"], domain=c["domain"], path=c.get("path", "/"))
    url = f"{site}/_api/web/GetFileByServerRelativeUrl('{_odata_path(f['url'])}')/$value"
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": UA}
    if done:
        headers["Range"] = f"bytes={done}-"
    with sess.get(url, headers=headers, stream=True, timeout=(15, 120)) as r:
        if done and r.status_code != 206:
            done = 0
        r.raise_for_status()
        with open(part, "ab" if done else "wb") as fh:
            for blk in r.iter_content(chunk_size=1 << 20):
                fh.write(blk)
                done += len(blk)
                if progress:
                    progress(done, f["size"])
    if part.stat().st_size != f["size"]:
        raise RuntimeError(f"{f['name']}: scaricati {part.stat().st_size} / {f['size']} byte")
    part.replace(dest)
    return dest


def sync_shared_folder(ctx: BrowserContext, page: Page, link: str, dest_root: Path,
                       exts: tuple[str, ...] = (".pptx", ".ppt", ".pdf", ".odp"),
                       log=print, progress=None, max_size_mb: int | None = None) -> list[Path]:
    """
    Sincronizza la cartella OneDrive raggiunta da `link` (modulo url WeBeep o link diretto)
    in dest_root/<nome cartella>/ (ricorsivo). Ritorna i file scaricati in questo giro.
    """
    share = resolve_webeep_link(page, link)
    site, folder = open_shared_folder(page, share)
    state_path = dest_root / ".onedrive_sync.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    downloaded: list[Path] = []

    def walk(fld: str, rel: Path):
        files, subs = list_folder(page, site, fld)
        for f in files:
            if not f["name"].lower().endswith(exts):
                continue
            key = f"{rel / f['name']}"
            dest = dest_root / rel / f["name"]
            sig = {"size": f["size"], "modified": f["modified"]}
            if dest.exists() and state.get(key) == sig:
                continue
            if max_size_mb and f["size"] > max_size_mb * 1e6:
                log(f"onedrive: salto {key} ({f['size'] / 1e6:.0f} MB > {max_size_mb} MB)")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            log(f"onedrive: ↓ {key} ({f['size'] / 1e6:.0f} MB, modificato {f['modified'][:10]})")
            t0 = time.time()
            _download(ctx, site, f, dest, progress)
            state[key] = sig
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))
            downloaded.append(dest)
            log(f"onedrive: ✓ {dest.name} ({time.time() - t0:.0f}s)")
        for sname in subs:
            walk(f"{fld}/{sname}", rel / sname)

    walk(folder, Path(Path(folder).name))
    return downloaded
