"""
video_match.py — Quali slide sono state mostrate nel video, e quando.

Le registrazioni Webex di PoliMi sono lo schermo condiviso (slide a tutto schermo, bande nere
ai lati, webcam in un riquadro in alto a destra) oppure la camera d'aula sulla lavagna. Un frame
ogni STEP secondi, ritagliato al riquadro del contenuto, viene confrontato con le pagine di tutti
i deck del corso con EmbeddingGemma 2 (src/slides/eg2.py): coseno tra l'embedding del frame e
quello della pagina renderizzata.

Regola (tarata il 07/10/2026 su VB3DM "Image formation", MOR Lecture 5/6/8, Lab 3/5 su Colab):
slide se il coseno è ≥ MIN_SCORE e supera di MIN_MARGIN il migliore candidato fuori dalle
±NEAR_PAGES pagine dello stesso deck, oppure se è ≥ SURE_SCORE con margine ≥ SURE_MARGIN. Slide vere: coseno 0.86-0.99, margine ~0.10; lavagna: coseno fino a 0.82 ma margine
~0.005 (somiglia a tutte le pagine allo stesso modo). Le pagine titolo di deck con lo stesso
template restano a pari merito e vengono scartate, non assegnate al deck sbagliato.
Rispetto al vecchio confronto a pixel riconosce anche le slide con numerazione diversa, le build
Beamer parziali, il PDF viewer a tutto schermo e la vista di PowerPoint in modifica.

Risultato: Timeline = intervalli (t0, t1, deck, pagina, punteggio) + secondi senza match
(lavagna, deck non ancora caricato dal docente, camera sola). Costo: ~3 min di GPU (A2000) per
90 min di video + ~10 s per deck la prima volta (cache in <deck_cache_dir>/eg2_pages.npy).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

STEP = 5                       # secondi tra un frame e l'altro
LW, LH = 320, 180              # decodifica a bassa risoluzione (per il riquadro del contenuto)
FW, FH = 960, 540              # decodifica dei frame confrontati
MIN_SCORE, MIN_MARGIN = 0.80, 0.04   # coseno EG2 e distacco dal 2° candidato non vicino (07/10/2026:
                               # lavagna/Colab fino a 0.023 di margine, slide vere ~0.10)
SURE_SCORE, SURE_MARGIN = 0.90, 0.02 # coseno ≥ 0.90 la lavagna/Colab non lo raggiungono mai (max 0.818):
                               # basta un margine minore (pagine titolo con lo stesso template: ~0.03)
NEAR_PAGES = 2                 # pagine adiacenti dello stesso deck: stessa risposta, non ambiguità
BATCH = 16                     # frame per chiamata al modello
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


# ── frame ────────────────────────────────────────────────────────────────────

def _content_box(lo: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, x1, y0, y1) del contenuto non nero in un frame LWxLH: per le colonne guarda solo la
    metà inferiore (la webcam sta in alto a destra, anche sopra la banda nera), per le righe la
    parte centrale."""
    cols = np.where(lo[LH // 2:, :].mean(0) > 8)[0]
    rows = np.where(lo[:, int(LW * 0.2): int(LW * 0.8)].mean(1) > 8)[0]
    if len(cols) < LW * 0.3 or len(rows) < LH * 0.3:
        return 0, LW, 0, LH
    return int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1


def frame_boxes(video: Path, step: int = STEP) -> list[tuple[int, int, int, int] | None]:
    """Per ogni frame il riquadro del contenuto (coordinate LWxLH), None = a tutto schermo.

    Riquadro: mediana dei frame con bande nere (schermo condiviso), stabile, applicata a quei frame;
    i frame a tutto schermo (camera sulla lavagna, PDF viewer massimizzato) restano interi. Con una
    mediana unica su tutti, in una lezione metà lavagna il riquadro diventava "tutto schermo"."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "0", "-i", str(video), "-vf",
                          f"fps=1/{step},scale={LW}:{LH}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    lo = np.frombuffer(raw, np.uint8).reshape(-1, LH, LW)
    boxes = np.array([_content_box(f) for f in lo])
    full = (boxes[:, 1] - boxes[:, 0] >= LW - 4) & (boxes[:, 3] - boxes[:, 2] >= LH - 4)
    if not (~full).any():
        return [None] * len(lo)
    med = tuple(int(np.median(boxes[~full][:, i])) for i in range(4))
    return [None if f else med for f in full]


def frame_embeddings(video: Path, step: int = STEP) -> np.ndarray:
    """(frame, 768) EG2 dei frame ogni `step` secondi, ritagliati al contenuto, webcam annerita.
    I frame sono letti in streaming da ffmpeg: un'ora di video a 960x540 non sta in RAM."""
    from PIL import Image
    from src.slides import eg2
    boxes = frame_boxes(video, step)
    sx, sy = FW / LW, FH / LH
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "0", "-i", str(video), "-vf",
                             f"fps=1/{step},scale={FW}:{FH}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                            stdout=subprocess.PIPE)
    size = FW * FH * 3
    out, batch, k = [], [], 0
    try:
        while k < len(boxes):
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            f = np.frombuffer(buf, np.uint8).reshape(FH, FW, 3)
            b = boxes[k]
            if b is not None:
                x0, x1, y0, y1 = int(b[0] * sx), int(b[1] * sx), int(b[2] * sy), int(b[3] * sy)
                f = f[y0:y1, x0:x1].copy()
                h, w = f.shape[:2]
                f[: int(h * WEBCAM[0]), int(w * WEBCAM[1]):] = 0
            batch.append(Image.fromarray(f))
            k += 1
            if len(batch) == BATCH:
                out.append(eg2.encode_images(batch)); batch = []
        if batch:
            out.append(eg2.encode_images(batch))
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    return np.concatenate(out) if out else np.empty((0, eg2.DIM), np.float32)


# ── confronto ────────────────────────────────────────────────────────────────

def match_video(video: Path, decks: list[Path], step: int = STEP, log=print) -> Timeline | None:
    """Timeline delle slide mostrate nel video, oppure None se il video non è leggibile o EG2
    non è installato (allora la scelta dei deck passa alla trascrizione)."""
    from src.slides import eg2
    if not eg2.available():
        log("video: torch/sentence-transformers non installati, niente riconoscimento slide")
        return None
    try:
        labels: list[tuple[str, int]] = []
        pages = []
        for d in decks:
            e = eg2.page_embeddings(d)
            pages.append(e)
            labels += [(str(d), i + 1) for i in range(len(e))]
        if not labels:
            return None
        Q = frame_embeddings(video, step)
    except subprocess.CalledProcessError as e:
        log(f"video: frame non estratti ({str(e)[:100]})")
        return None
    except Exception as e:
        log(f"video: EG2 fallito ({type(e).__name__}: {str(e)[:100]})")
        return None
    finally:
        eg2.release()                                     # VRAM libera per Whisper
    if not len(Q):
        return None
    C = Q @ np.concatenate(pages).T                       # (frame, pagine), coseno
    best = C.argmax(1)
    score = C.max(1)
    # "ambiguo" = un'altra parte del materiale spiegherebbe il frame altrettanto bene. La pagina
    # accanto dello stesso deck no: e' la stessa risposta a meno di un'animazione o di un build.
    deck_id = np.concatenate([np.full(len(e), i) for i, e in enumerate(pages)])
    page_no = np.concatenate([np.arange(1, len(e) + 1) for e in pages])
    near = (deck_id[None, :] == deck_id[best][:, None]) & \
           (np.abs(page_no[None, :] - page_no[best][:, None]) <= NEAR_PAGES)
    second = np.where(near, -1.0, C).max(1) if C.shape[1] > 1 else np.full(len(C), -1.0)

    spans: list[Span] = []
    unmatched = 0.0
    for k in range(len(Q)):
        t = k * step
        m = score[k] - second[k]
        ok = (score[k] >= MIN_SCORE and m >= MIN_MARGIN) or (score[k] >= SURE_SCORE and m >= SURE_MARGIN)
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
    return Timeline(spans, float(len(Q) * step), unmatched, [d.name for d in decks])


def mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"
