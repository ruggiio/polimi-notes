"""
video_match.py — Quali slide sono state mostrate nel video, e quando.

Le registrazioni Webex di PoliMi sono lo schermo condiviso (slide a tutto schermo, bande nere
ai lati, webcam in un riquadro in alto a destra). Un frame ogni STEP secondi, ritagliato al
riquadro del contenuto e ridotto a miniatura, viene confrontato con le pagine di tutti i deck
del corso: correlazione normalizzata sia dell'intensità sia del gradiente (la struttura: per
le slide di solo testo l'intensità da sola è un grigio uniforme). Slide vere: 0.85-0.95,
false: ≤ 0.6 (misurato il 16/09/2026 su una lezione vera).

Risultato: Timeline = intervalli (t0, t1, deck, pagina, punteggio) + secondi senza match
(lavagna, deck non ancora caricato dal docente, camera sola). Costo: ~1 min di CPU per
un'ora di video 1080p (decodifica ffmpeg), niente GPU, niente OCR.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

STEP = 5                       # secondi tra un frame e l'altro
LW, LH = 320, 180              # decodifica a bassa risoluzione (per il riquadro del contenuto)
W, H = 128, 72                 # miniatura confrontata
MIN_SCORE, MIN_MARGIN = 0.50, 0.06   # sul residuo (template del deck tolto), non sulla correlazione grezza
                               # (calibrato 23/09/2026: copertura 89%, ρ tempo/pagina +0.999)
NEAR_PAGES = 2                 # pagine adiacenti dello stesso deck: stessa risposta, non ambiguità
WEBCAM = (0.22, 0.85)          # riquadro webcam: in alto (22% delle righe) a destra (dal 85% delle colonne)


@dataclass
class Span:
    t0: float
    t1: float
    deck: str                  # path del deck
    page: int                  # 1-based
    score: float


@dataclass
class Timeline:
    spans: list[Span]
    duration: float            # secondi coperti dai frame
    unmatched: float           # secondi senza slide riconosciuta
    decks: list[str] = None    # nomi dei deck confrontati (se cambiano, la timeline va rifatta)

    def by_deck(self) -> dict[str, dict]:
        """{deck: {"seconds": s, "pages": {page: seconds}}} in ordine di tempo mostrato."""
        out: dict[str, dict] = {}
        for s in self.spans:
            d = out.setdefault(s.deck, {"seconds": 0.0, "pages": {}})
            d["seconds"] += s.t1 - s.t0
            d["pages"][s.page] = d["pages"].get(s.page, 0.0) + s.t1 - s.t0
        return dict(sorted(out.items(), key=lambda x: -x[1]["seconds"]))

    def page_spans(self, deck: str, page: int) -> list[tuple[float, float]]:
        return [(s.t0, s.t1) for s in self.spans if s.deck == deck and s.page == page]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"duration": self.duration, "unmatched": self.unmatched,
                                    "decks": self.decks or [], "spans": [asdict(s) for s in self.spans]}, indent=1))

    @staticmethod
    def load(path: Path) -> "Timeline | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return Timeline([Span(**s) for s in d["spans"]], d["duration"], d["unmatched"], d.get("decks") or [])
        except Exception:
            return None


# ── miniature ────────────────────────────────────────────────────────────────

def _mask() -> np.ndarray:
    m = np.ones((H, W), np.float32)
    m[: int(H * WEBCAM[0]), int(W * WEBCAM[1]):] = 0
    return m


MASK = _mask()


def _content_box(lo: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, x1, y0, y1) del contenuto non nero in un frame LWxLH: per le colonne guarda solo la
    metà inferiore (la webcam sta in alto a destra, anche sopra la banda nera), per le righe la
    parte centrale."""
    cols = np.where(lo[LH // 2:, :].mean(0) > 8)[0]
    rows = np.where(lo[:, int(LW * 0.2): int(LW * 0.8)].mean(1) > 8)[0]
    if len(cols) < LW * 0.3 or len(rows) < LH * 0.3:
        return 0, LW, 0, LH
    return int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1


def frame_thumbs(video: Path, step: int = STEP) -> np.ndarray:
    """Miniature (n, H, W) uint8 dei frame ogni `step` secondi, ritagliate al contenuto."""
    from PIL import Image
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "0", "-i", str(video), "-vf",
                          f"fps=1/{step},scale={LW}:{LH}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    lo = np.frombuffer(raw, np.uint8).reshape(-1, LH, LW)
    # riquadro: mediana sui frame (stabile), applicata a tutti: un deck 16:9 in mezzo a deck 4:3
    # resta un caso raro e comunque con punteggio basso
    boxes = np.array([_content_box(f) for f in lo[:: max(1, len(lo) // 40)]])
    x0, x1, y0, y1 = (int(np.median(boxes[:, i])) for i in range(4))
    out = np.empty((len(lo), H, W), np.uint8)
    for i, f in enumerate(lo):
        out[i] = np.asarray(Image.fromarray(f[y0:y1, x0:x1]).resize((W, H), Image.BILINEAR))
    return out


def page_thumbs(deck: Path) -> np.ndarray:
    """Miniature (pagine, H, W) uint8 delle pagine del deck; cache in <deck_cache_dir>/thumbs.npy."""
    from PIL import Image
    from src.slides.slides import as_pdf, deck_cache_dir
    cache = deck_cache_dir(deck) / "thumbs.npy"
    if cache.exists():
        try:
            return np.load(cache)
        except Exception:
            pass
    import pymupdf as fitz
    fitz.TOOLS.mupdf_display_errors(False)
    pdf = as_pdf(deck)
    if not pdf:
        return np.empty((0, H, W), np.uint8)
    thumbs = []
    with fitz.open(pdf) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(0.4, 0.4), colorspace=fitz.csGRAY, alpha=False)
            im = Image.frombytes("L", (pix.width, pix.height), pix.samples).resize((W, H), Image.LANCZOS)
            thumbs.append(np.asarray(im))
    arr = np.stack(thumbs) if thumbs else np.empty((0, H, W), np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


# ── confronto ────────────────────────────────────────────────────────────────

def _features(thumbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(gradiente, intensità) normalizzati e mascherati, (n, H*W) ciascuno."""
    from PIL import Image, ImageFilter
    n = len(thumbs)
    grad = np.empty((n, H * W), np.float32)
    inten = np.empty((n, H * W), np.float32)
    sel = MASK > 0
    for i, a in enumerate(thumbs):
        g = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.0)), np.float32)
        gx = np.zeros_like(g); gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1, :] = g[2:, :] - g[:-2, :]
        for src, dst in ((np.hypot(gx, gy), grad), (a.astype(np.float32), inten)):
            m = src * MASK
            m -= m[sel].mean()
            m *= MASK
            nrm = np.linalg.norm(m)
            dst[i] = (m / nrm if nrm > 1e-6 else m).ravel()
    return grad, inten


def _renorm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-6)


def _resid_corr(F: np.ndarray, Pd: np.ndarray) -> np.ndarray:
    """Correlazione frame×pagine dopo aver tolto da entrambi la pagina media del deck.

    Il template (sfondo, intestazione, impaginazione) e' identico in tutto il deck e domina la
    correlazione: senza toglierlo ogni frame somiglia a ogni pagina ~0.95 e l'argmax e' rumore.
    Sul residuo resta il solo contenuto della slide.
    """
    mu = Pd.mean(0)
    return _renorm(F - mu) @ _renorm(Pd - mu).T


def match_video(video: Path, decks: list[Path], step: int = STEP, log=print) -> Timeline | None:
    """Timeline delle slide mostrate nel video, oppure None se il video non è leggibile."""
    try:
        frames = frame_thumbs(video, step)
    except Exception as e:
        log(f"video: frame non estratti ({type(e).__name__}: {str(e)[:100]})")
        return None
    if not len(frames) or not decks:
        return None
    labels: list[tuple[str, int]] = []
    pages = []
    for d in decks:
        th = page_thumbs(d)
        pages.append(th)
        labels += [(str(d), i + 1) for i in range(len(th))]
    if not labels:
        return None
    P = np.concatenate(pages)
    fg, fi = _features(frames)
    pg, pi = _features(P)
    C = np.empty((len(frames), len(labels)), np.float32)  # (frame, pagine)
    off = 0
    for th in pages:                                      # un deck alla volta: il template e' suo
        sl = slice(off, off + len(th))
        C[:, sl] = 0.5 * (_resid_corr(fg, pg[sl]) + _resid_corr(fi, pi[sl]))
        off += len(th)
    best = C.argmax(1)
    score = C.max(1)
    # "ambiguo" = un'altra parte del materiale spiegherebbe il frame altrettanto bene. La pagina
    # accanto dello stesso deck no: e' la stessa risposta a meno di un'animazione o di un build.
    deck_id = np.concatenate([np.full(len(th), i) for i, th in enumerate(pages)])
    page_no = np.concatenate([np.arange(1, len(th) + 1) for th in pages])
    near = (deck_id[None, :] == deck_id[best][:, None]) & \
           (np.abs(page_no[None, :] - page_no[best][:, None]) <= NEAR_PAGES)
    second = np.where(near, -1.0, C).max(1) if C.shape[1] > 1 else np.full(len(C), -1.0)

    spans: list[Span] = []
    unmatched = 0.0
    for k in range(len(frames)):
        t = k * step
        ok = score[k] >= MIN_SCORE and score[k] - second[k] >= MIN_MARGIN
        if not ok:
            unmatched += step
            continue
        deck, page = labels[best[k]]
        if spans and spans[-1].deck == deck and spans[-1].page == page and spans[-1].t1 == t:
            spans[-1].t1 = t + step
            spans[-1].score = max(spans[-1].score, float(score[k]))
        else:
            spans.append(Span(float(t), float(t + step), deck, page, float(score[k])))
    # un frame isolato (5 s) è più spesso un cambio slide colto a metà che una slide mostrata
    spans = [s for s in spans if s.t1 - s.t0 >= 2 * step or
             any(o is not s and o.deck == s.deck and o.page == s.page for o in spans)]
    return Timeline(spans, float(len(frames) * step), unmatched, [d.name for d in decks])


def mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"
