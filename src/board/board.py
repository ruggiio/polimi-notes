"""
board.py — Lettura della lavagna dalle registrazioni in aula (telecamera fissa sulla lavagna).

Router: ogni 2 s il frame è "lavagna" se la fascia centrale è per lo più del colore di una lavagna
(verde/ardesia poco saturo); lo schermo condiviso (slide, PC) non lo è mai. Nei tratti alla lavagna:

  1. stati completi: mediana di 30 s (toglie il docente che si muove) e densità di gesso in NS strisce
     verticali; una cancellatura è un calo netto e stabile su ≥ 2 strisce. Il keyframe è la finestra
     più piena prima del calo, ritagliata alle strisce cancellate: ogni pezzo di gesso viene letto una
     volta sola, subito prima di sparire. La fine di ogni tratto alla lavagna vale come cancellatura totale.
  2. lettura con fail-safe: due letture indipendenti (modelli diversi, via `claude -p` + Read) in JSON;
     le formule vengono allineate e confrontate dopo una normalizzazione del LaTeX. Uguali → verificate.
     Diverse, o viste da un solo lettore → un arbitro, con l'immagine e le due versioni, sceglie A, B,
     "equivalenti" o nessuna. Maggioranza 2 su 3 → verificata; altrimenti incerta (con la lettura migliore)
     o illeggibile. Nessuna lettura riceve la trascrizione del parlato: il contesto spinge il modello a
     scrivere la formula che si aspetta invece di quella scritta (errore "da prior").
  3. negli appunti: le verificate si copiano, le incerte escono con \\boardcheck{mm:ss} e la foto;
     `check_notes` verifica a valle che le formule verificate siano arrivate intatte nel .tex.

Solo numpy + PIL + ffmpeg (niente opencv). Risultati in cache in <out_dir>/board.json.
"""

from __future__ import annotations

import difflib
import functools
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

STEP_S = 2                 # un frame ogni 2 s
WIN = 15                   # finestra della mediana: 15 frame = 30 s
SW, SH = 384, 216          # risoluzione di analisi
NS = 6                     # strisce verticali per la densità di gesso
BOARD_TOL = 40             # distanza massima (per canale) dal colore della lavagna del video
OB = 6                     # lato dei blocchi della maschera del docente (a 384×216)
BLOCK_BOARD = 0.35         # blocco con meno di questa frazione di pixel "tipo lavagna" = docente
DARK, BLOCK_DARK = 40, 0.3 # blocco con più del 30% di pixel più scuri di così = docente (abiti scuri)
CAM_FLAT, CAM_TEXTURE = 0.76, 0.5   # telecamera contro schermo scuro (vedi _camera)
BOARD_MIN = 0.40           # frazione "colore lavagna" oltre cui il frame è lavagna
ERASE_DROP = 0.80          # dopo/prima < questo = candidata cancellatura
SAFETY_W = 16              # rete di sicurezza: rilettura delle strisce cambiate ogni 16 finestre (8 min)
SURVIVE = 0.85             # gesso ancora lì dopo il calo ≥ questo = docente davanti, non cancellatura
ALIGN_MIN = 0.7             # somiglianza minima per considerare due letture la stessa formula
MIN_SEGMENT_S = 60         # tratti alla lavagna più corti: ignorati
VERSION = 6                # cambia → la cache board.json si rifà


# ── analisi del video ────────────────────────────────────────────────────────

def _frames(video: Path, w: int = SW, h: int = SH, fps: str = f"1/{STEP_S}", start: float | None = None,
            dur: float | None = None, keyframes_only: bool = False):
    cmd = ["ffmpeg", "-v", "error"] + (["-skip_frame", "nokey"] if keyframes_only else [])
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", str(video)]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def _board_mask(f: np.ndarray, color: np.ndarray | None = None) -> np.ndarray:
    """Pixel "lavagna": scuri e poco saturi (verde, oliva, ardesia: ogni aula ha la sua, misurato su
    tre aule PoliMi: 0.89-0.92 della fascia centrale contro 0.03-0.05 dello schermo condiviso).
    Con `color` (la lavagna di questo video) anche vicini a quel colore: il docente, chiaro o scuro, no."""
    f = f.astype(np.int16)
    mx, mn = f.max(-1), f.min(-1)
    m = (mx > 35) & (mx < 170) & (mx - mn < 45)
    if color is not None:
        m &= np.abs(f - color).max(-1) < BOARD_TOL
    return m


def _box_blur(a: np.ndarray, k: int = 9) -> np.ndarray:
    p = k // 2
    c = np.cumsum(np.cumsum(np.pad(a, ((0, 0), (p + 1, p), (p + 1, p)), mode="edge"), 1), 2)
    return (c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]) / (k * k)


@dataclass
class Shot:
    t0: float                  # inizio della finestra (s)
    t1: float
    strips: list[int]          # strisce ritagliate
    x0: float = 0.0            # ritaglio orizzontale, frazioni della larghezza della lavagna
    x1: float = 1.0
    image: str = ""            # png del ritaglio a piena risoluzione
    items: list[dict] = field(default_factory=list)   # letture riconciliate
    cost: float = 0.0

    @property
    def mmss(self) -> str:
        t = int(self.t1)
        return f"{t // 60:02d}:{t % 60:02d}"


def _camera(band: np.ndarray, m: np.ndarray) -> bool:
    """La regione "tipo lavagna" è ripresa da una telecamera (rumore, sbavature) e non uno schermo
    condiviso a tema scuro (fondi perfettamente piatti)? Misurato su 3 aule e un editor scuro
    (MOR Lab 10): pixel con gradiente orizzontale < 0.5 = 0.40-0.68 sulle lavagne, 0.80-0.89
    sull'editor; gradiente medio (bordi forti esclusi) 0.62-1.36 contro 0.27-0.47."""
    g = band.astype(np.int16).mean(-1)
    dx = np.abs(np.diff(g, axis=1))
    v = dx[m[:, 1:] & m[:, :-1]]
    v = v[v < 8]
    if v.size < 100:
        return False
    return float((v < 0.5).mean()) < CAM_FLAT and float(v.mean()) > CAM_TEXTURE


@functools.lru_cache(maxsize=8)
def board_color(video: Path) -> np.ndarray | None:
    """Controllo rapido (un frame al minuto, solo keyframe: ~10 s): colore della lavagna di questo
    video, o None se non c'è almeno MIN_SEGMENT_S di lavagna (lezione a schermo condiviso)."""
    cols = []
    for f in _frames(video, fps="1/60", keyframes_only=True):
        band = f[SH // 5: SH * 4 // 5, SW // 20: SW * 19 // 20]
        m = _board_mask(band)
        if m.mean() > BOARD_MIN and _camera(band, m):
            cols.append(np.median(band[m], 0))
    if len(cols) * 60 < MIN_SEGMENT_S:
        return None
    return np.median(np.stack(cols), 0).astype(np.int16)


def has_board(video: Path) -> bool:
    return board_color(video) is not None


def analyse(video: Path, log=print, color: np.ndarray | None = None
            ) -> tuple[list[tuple[float, float]], list[Shot], tuple]:
    """Tratti alla lavagna [(t0, t1)], keyframe (senza immagine) e riquadro della lavagna (frazioni)."""
    if color is None:
        color = board_color(video)
    if color is None:
        log("board: nessuna lavagna nel video (controllo rapido)")
        return [], [], ()
    frac, grays, colp, rowp, blocks = [], [], [], [], []
    for f in _frames(video):
        m = _board_mask(f, color)
        # per la maschera del docente: blocchi per lo più non-lavagna (camicia chiara, pelle) o
        # per lo più molto scuri (abiti scuri, capelli)
        gen = _board_mask(f)[:SH // OB * OB, :SW // OB * OB]
        dark = (f.max(-1) < DARK)[:SH // OB * OB, :SW // OB * OB]
        blk = lambda x: x.reshape(SH // OB, OB, SW // OB, OB).mean((1, 3))
        blocks.append((blk(gen) < BLOCK_BOARD) | (blk(dark) > BLOCK_DARK))
        band = f[SH // 5: SH * 4 // 5, SW // 20: SW * 19 // 20]
        mb = m[SH // 5: SH * 4 // 5, SW // 20: SW * 19 // 20]
        # un editor di codice a tema scuro è "scuro e poco saturo" come una lavagna: lo separa la texture
        frac.append(mb.mean() if mb.mean() <= BOARD_MIN or _camera(band, mb) else 0.0)
        grays.append(f.mean(-1).astype(np.uint8))
        colp.append(m[SH // 5: SH * 4 // 5].mean(0))
        rowp.append(m[:, SW // 5: SW * 4 // 5].mean(1))
    frac = np.array(frac)
    isb = frac > BOARD_MIN
    # tratti contigui (buchi ≤ 10 s chiusi: il docente che passa davanti alla telecamera)
    segs, i = [], 0
    while i < len(isb):
        if not isb[i]:
            i += 1
            continue
        j = i
        while j < len(isb) and (isb[j] or isb[j:j + 5].any()):
            j += 1
        if (j - i) * STEP_S >= MIN_SEGMENT_S:
            segs.append((i, j))
        i = j
    if not segs:
        log(f"board: nessun tratto alla lavagna (max frazione {frac.max() if len(frac) else 0:.2f})")
        return [], [], ()
    # riquadro della lavagna: righe/colonne che sono lavagna in almeno il 20% dei frame (il docente ne
    # copre sempre una parte; la media sottostimerebbe il lato dove sta di più), con la fascia sotto la
    # lampada — sovraesposta, fuori dal colore "lavagna" — recuperata da un margine
    idx = [k for a, b in segs for k in range(a, b)]
    colf = np.percentile([colp[k] for k in idx], 80, axis=0)
    rowf = np.percentile([rowp[k] for k in idx], 80, axis=0)
    # dal centro verso l'esterno finché righe/colonne restano lavagna: ci si ferma alla mensola del
    # gesso o alla cornice (chiare) invece di inglobare bancone e pareti grigie oltre
    ry0, ry1 = _grow(rowf > 0.5, SH // 2)
    cx0, cx1 = _grow(colf > 0.5, SW // 2)
    my, mx_ = SH // 25, SW // 80
    # in alto un margine più largo: la fascia sotto la lampada è sovraesposta (fuori dal "tipo
    # lavagna") e lì stanno spesso i titoli
    y0, y1 = max(0, ry0 - 2 * my), min(SH, ry1 + my)
    x0, x1 = max(0, cx0 - mx_), min(SW, cx1 + mx_)
    box = (y0 / SH, y1 / SH, x0 / SW, x1 / SW)

    shots: list[Shot] = []
    for a, b in segs:
        meds = [np.median(np.stack([g[y0:y1, x0:x1] for g in grays[w:w + WIN]]), 0)
                for w in range(a, b - WIN + 1, WIN)]
        if not meds:
            continue
        M = np.stack(meds).astype(np.float32)
        hp = M - _box_blur(M)
        chalk, strong = hp > 12, hp > 15
        occl = _occluded(blocks, a, len(M), (y0, y1, x0, x1))
        if os.environ.get("BOARD_DEBUG"):
            np.savez(os.environ["BOARD_DEBUG"], M=M, occl=occl, box=np.array([y0, y1, x0, x1]))
        occl_wide = _occluded(blocks, a, len(M), (y0, y1, x0, x1), grow=3)   # sopravvivenza: niente bordi
        W = M.shape[2]
        # densità di gesso per striscia misurata sulla sola parte visibile; una striscia coperta
        # per più di metà eredita il valore precedente (il docente davanti non è una cancellatura)
        D = np.zeros((len(M), NS))
        for s in range(NS):
            sl = slice(s * W // NS, (s + 1) * W // NS)
            vis = ~occl[:, :, sl]
            nvis = vis.sum((1, 2))
            dens = (strong[:, :, sl] & vis).sum((1, 2)) / np.maximum(1, nvis)
            for w in range(len(M)):
                D[w, s] = dens[w] if nvis[w] > 0.5 * vis[w].size or w == 0 else D[w - 1, s]
        T = len(D)
        # "c'è gesso": sopra il fondo della striscia (lavagna pulita = 10° percentile nella lezione:
        # sbavature e texture cambiano con aula, luce e riquadro, una soglia assoluta no)
        clean = np.percentile(D, 10, axis=0)
        full = np.maximum(0.008, 1.8 * clean)
        # per NON leggere una striscia basta molto meno: è vuota solo se è davvero pulita (una frase
        # sparsa — "DATA: m snapshots…" — sta appena sopra il fondo, e perderla costa più di un doppione)
        empty = 1.25 * clean + 0.002
        events: dict[int, set] = {}          # finestra del keyframe → strisce cancellate subito dopo
        fl = np.zeros(NS, int)               # per striscia: le finestre prima di qui non fanno da picco
        for t in range(1, T - 1):
            # calo: il minimo del minuto dopo (il docente spesso riscrive subito) sotto il picco recente;
            # sensibile di proposito, i falsi positivi li toglie il controllo di sopravvivenza
            after = D[t + 1:t + 3].min(0)
            er = [s for s in range(NS) if t >= fl[s] and
                  (pk := D[max(fl[s], t - 3):t + 1, s].max()) > full[s] and after[s] < ERASE_DROP * pk]
            if not er:
                continue
            # keyframe: la finestra meno coperta (a pari copertura la più piena) prima del calo
            cols = np.concatenate([np.arange(s * W // NS, (s + 1) * W // NS) for s in er])
            best = min(range(max(max(fl[s] for s in er), t - 4), t + 1),
                       key=lambda k: (round(float(occl[k][:, cols].mean()), 2), -D[k, er].sum()))
            # vera cancellatura o docente davanti? Un pixel di gesso "sopravvive" se è gesso in almeno
            # metà delle finestre dei 2 minuti dopo il calo IN CUI È VISIBILE (mai visibile = sopravvive)
            fut_vis = ~occl[t + 1:t + 5]
            seen = fut_vis.sum(0)
            fut = ((strong[t + 1:t + 5] & fut_vis).sum(0) >= 0.5 * seen) | (seen == 0)
            for s in er:
                sl = slice(s * W // NS, (s + 1) * W // NS)
                now = strong[best, :, sl] & ~occl_wide[best, :, sl]
                if now.sum() > 20 and (now & fut[:, sl]).sum() < SURVIVE * now.sum():
                    events.setdefault(best, set()).add(s)
                    fl[s] = t + 2             # fino alla cancellatura: non più picco per questa striscia
        events.setdefault(T - 1, set()).update(range(NS))    # fine del tratto: tutto quello che resta
        # rete di sicurezza per le cancellature che il rilevatore non vede: una striscia con gesso non
        # letta da SAFETY_W finestre si rilegge (nella finestra meno coperta). Un'aggiunta di poche righe
        # non cambia abbastanza la densità da farsene accorgere: meglio un doppione che un buco
        last = {}
        for w in range(T):
            for s in events.get(w, ()):
                last[s] = w
            if w % SAFETY_W or w == 0 or w >= T - 1:
                continue
            due = [s for s in range(NS) if D[w, s] > empty[s] and w - last.get(s, -SAFETY_W) >= SAFETY_W]
            if due:
                cols = np.concatenate([np.arange(s * W // NS, (s + 1) * W // NS) for s in due])
                best = min(range(max(0, w - 3), w + 1), key=lambda k: round(float(occl[k][:, cols].mean()), 2))
                events.setdefault(best, set()).update(due)
                for s in due:
                    last[s] = best
        for w, er in sorted(events.items()):
            keep = sorted(s for s in er if D[w, s] > empty[s])  # strisce pulite: niente da leggere
            # gruppi contigui; il taglio cade nella colonna più vuota vicino al confine tra strisce
            groups: list[list[int]] = []
            for s in keep:
                if groups and s == groups[-1][-1] + 1:
                    groups[-1].append(s)
                else:
                    groups.append([s])
            col = chalk[w].sum(0)
            t0 = (a + w * WIN) * STEP_S
            for g in groups:
                lo, hi = _cut(col, g[0] * W // NS, W, -1), _cut(col, (g[-1] + 1) * W // NS, W, +1)
                shots.append(Shot(t0=t0, t1=t0 + WIN * STEP_S, strips=g, x0=lo / W, x1=hi / W))
    log(f"board: {len(segs)} tratti alla lavagna ("
        + ", ".join(f"{a * STEP_S // 60}′–{b * STEP_S // 60}′" for a, b in segs) + f"), {len(shots)} keyframe")
    return [(a * STEP_S, b * STEP_S) for a, b in segs], shots, box


def _grow(ok: np.ndarray, c: int, gap: int = 2) -> tuple[int, int]:
    """Intervallo [lo, hi) di `ok` attorno a c (la cella vera più vicina a c), tollerando buchi ≤ gap."""
    idx = np.where(ok)[0]
    if not len(idx):
        return 0, len(ok)
    c = int(idx[np.argmin(np.abs(idx - c))])
    lo = hi = c
    while lo > 0 and ok[max(0, lo - 1 - gap):lo].any():
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1:hi + 2 + gap].any():
        hi += 1
    return lo, hi + 1


def _occluded(blocks: list, a: int, T: int, box: tuple, grow: int = 1) -> np.ndarray:
    """Pixel coperti dal docente nelle mediane delle finestre: blocchi OB×OB "docente" (per lo più
    fuori dal tipo lavagna, o per lo più molto scuri) in almeno metà dei frame della finestra — la
    mediana mostra il docente solo allora —, allargati di `grow` blocchi (braccia, bordi). Il gesso è
    troppo sottile per riempire un blocco. (Una sottrazione dello sfondo nel tempo scambiava per
    docente i blocchi appena cancellati, e il controllo di sopravvivenza perdeva le cancellature.)"""
    y0, y1, x0, x1 = box
    out = np.zeros((T, y1 - y0, x1 - x0), bool)
    for w in range(T):
        d = np.stack(blocks[a + w * WIN: a + (w + 1) * WIN]).mean(0) >= 0.5
        g2 = d.copy()
        for _ in range(grow):
            g = g2.copy()
            g[1:] |= g2[:-1]; g[:-1] |= g2[1:]
            g2 = g.copy()
            g2[:, 1:] |= g[:, :-1]; g2[:, :-1] |= g[:, 1:]
        full = np.zeros((SH, SW), bool)
        full[:g2.shape[0] * OB, :g2.shape[1] * OB] = np.repeat(np.repeat(g2, OB, 0), OB, 1)
        out[w] = full[y0:y1, x0:x1]
    return out


def _cut(col: np.ndarray, x: int, W: int, side: int) -> int:
    """Colonna con meno gesso entro mezza striscia dal confine x, verso l'esterno del ritaglio."""
    if x <= 0 or x >= W:
        return max(0, min(W, x))
    half = W // NS // 2
    lo, hi = (max(0, x - half), x + 1) if side < 0 else (x, min(W, x + half + 1))
    win = np.convolve(col[lo:hi], np.ones(5), mode="same")
    return lo + int(np.argmin(win))


def render(video: Path, shot: Shot, box: tuple, out: Path, color: np.ndarray | None = None) -> Path:
    """Stato della lavagna a fine finestra senza il docente, a piena risoluzione, ritagliato alla lavagna
    e a shot.x0..x1: per ogni pixel la mediana delle RENDER_K osservazioni più recenti (negli ultimi
    RENDER_S secondi) in cui il punto non è coperto dal docente (stessa maschera dell'analisi: blocchi
    fuori dal "tipo lavagna" o molto scuri). La mediana semplice dei 30 s lasciava il docente quando
    stava fermo davanti per più di metà del tempo (MOR Lecture 9: formule coperte)."""
    from PIL import Image
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    y0, y1, x0, x1 = int(box[0] * h), int(box[1] * h), int(box[2] * w), int(box[3] * w)
    bw = x1 - x0
    cx0, cx1 = x0 + int(shot.x0 * bw), x0 + int(shot.x1 * bw)
    if color is None:
        color = board_color(video)
    start = max(0.0, shot.t1 - RENDER_S)
    fr = np.stack(list(_frames(video, w, h, start=start, dur=shot.t1 - start)))[:, y0:y1, cx0:cx1]
    n, H, W = fr.shape[:3]
    # maschera del docente per frame: a bassa risoluzione, a blocchi, allargata di un blocco
    sh, sw = max(1, H * SH // h), max(1, W * SW // w)
    covered = np.zeros((n, H, W), bool)
    for i in range(n):
        small = np.asarray(Image.fromarray(fr[i]).resize((sw, sh), Image.BILINEAR))
        # col colore della lavagna di questo video: una camicia grigio chiaro passa il criterio
        # generico "scuro e poco saturo" ma non questo (MOR Lecture 5)
        m = ~_board_mask(small, color) | (small.max(-1) < DARK)
        hb, wb = max(1, sh // OB), max(1, sw // OB)
        blk = m[:hb * OB, :wb * OB].reshape(hb, OB, wb, OB).mean((1, 3)) > 0.5
        g = blk.copy()
        g[1:] |= blk[:-1]; g[:-1] |= blk[1:]
        g2 = g.copy()
        g2[:, 1:] |= g[:, :-1]; g2[:, :-1] |= g[:, 1:]
        covered[i] = np.asarray(Image.fromarray(g2.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 0
    ok = ~covered
    # rango dall'ultimo frame: le RENDER_K osservazioni libere più recenti di ogni pixel
    rank = np.cumsum(ok[::-1], 0)[::-1]
    use = ok & (rank <= RENDER_K)
    out_img = np.empty((H, W, 3), np.uint8)
    for r0 in range(0, H, 64):                       # a strisce di righe: memoria contenuta
        r1 = min(H, r0 + 64)
        blockf = fr[:, r0:r1].astype(np.float32)
        blockf[~use[:, r0:r1]] = np.nan
        med = np.nanmedian(blockf, 0)
        plain = np.median(fr[-WIN:, r0:r1], 0)       # mai libero: mediana semplice della finestra
        out_img[r0:r1] = np.where(np.isnan(med), plain, med).astype(np.uint8)
    Image.fromarray(out_img).save(out)
    return out


# ── lettura con fail-safe ────────────────────────────────────────────────────

READ_SYSTEM = "You transcribe chalkboard photos from university lectures into LaTeX, faithfully."
READ_PROMPT = """Use the Read tool to look at the image {img}. It is a sheet of {n} numbered panels (white badge
with a black number at the top-left of each panel). Each panel is a crop of a university chalkboard at some
moment of the lecture, with the lecturer digitally removed (faint ghosts or smudges may remain); a panel may be
cut at its left/right edge, and different panels may show overlapping content.

Transcribe EVERYTHING written in EACH panel separately, in reading order within the panel (column by column,
top to bottom), as one JSON object:
{{"items": [{{"panel": 1, "kind": "text", "text": "..."}}, {{"panel": 1, "kind": "formula", "latex": "..."}}, ...]}}
- one "formula" item per equation or standalone mathematical expression (LaTeX without $ delimiters);
  a line of text containing a small inline symbol is a "text" item with the symbol in $...$;
- arrows and annotations become short "text" items ("arrow from X to Y: ...");
- boxed formulas: add "boxed": true.
Transcribe what is WRITTEN, not what you expect: never fix, complete, simplify or normalise the notation,
even if it looks wrong or unusual (a vector arrow, a hat, a dot, an index, a transpose are all significant).
Use [?] for any symbol you cannot read with confidence. Reply with the JSON only."""

JUDGE_SYSTEM = "You are a meticulous proofreader of chalkboard transcriptions. You judge only what is visibly written."
JUDGE_PROMPT = """Use the Read tool to look at the image {img}: numbered panels (white badge, top-left), each a
crop of a chalkboard with the lecturer digitally removed.
Two people transcribed it independently and disagree on the formulas below. For each one, find it in the given
panel and compare every symbol (letters, indices, exponents, arrows, hats, dots, signs, transposes, brackets).

{disputes}

For each #k answer with:
- "A" or "B" if that version matches the board exactly (notation-only differences like \\dfrac vs \\frac do not matter),
- "both" if both are faithful and differ only in LaTeX spelling,
- "neither" if both are wrong: then give your own faithful "latex", or null if the board is unreadable there,
- "absent" if the formula is not on the board at all.
Reply ONLY with JSON: [{{"k": 1, "verdict": "A", "latex": null}}, ...]"""


def normalize(tex: str) -> str:
    """Forma canonica per confrontare due letture: toglie solo differenze di scrittura LaTeX."""
    t = tex.strip().strip("$").strip()
    t = re.sub(r"\\(left|right|big|Big|bigg|Bigg)(?![a-zA-Z])", "", t)
    t = re.sub(r"\\(displaystyle|limits|nolimits|,|;|!|:|quad|qquad)(?![a-zA-Z])", "", t)
    t = re.sub(r"\\[dt]frac", r"\\frac", t)
    t = re.sub(r"\\wide(tilde|hat)(?![a-zA-Z])", r"\\\1", t)
    t = re.sub(r"\\overrightarrow(?![a-zA-Z])", r"\\vec", t)
    for long_, short in (("leq", "le"), ("geq", "ge"), ("neq", "ne"), ("rightarrow", "to"),
                         ("longrightarrow", "to"), ("longmapsto", "mapsto"), ("lbrace", "{"), ("rbrace", "}")):
        t = re.sub(rf"\\{long_}(?![a-zA-Z])", lambda _m, s=short: "\\" + s if s.isalpha() else "\\" + s, t)
    t = re.sub(r"\\mathrm\{d\}", "d", t)
    t = re.sub(r"\\(ldots|cdots|dots)(?![a-zA-Z])|\.\.\.", r"\\dots", t)
    t = re.sub(r"\\mid(?![a-zA-Z])|\\vert(?![a-zA-Z])", "|", t)
    t = re.sub(r"\\(text|mathrm|operatorname)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\\boxed\{(.*)\}$", r"\1", t)
    t = re.sub(r"\\operatorname\*?\{arg\s*(min|max)\}", r"\\arg\\\1", t)
    t = t.replace("\\Vert", "\\|").replace("\\lVert", "\\|").replace("\\rVert", "\\|")
    t = re.sub(r"\s+", "", t)
    for _ in range(3):
        t = re.sub(r"\{(\\[A-Za-z]+|[A-Za-z0-9])\}", r"\1", t)   # {a} → a, {\xi} → \xi
    t = re.sub(r"\\underset\{([^{}]*)\}\{\\arg\s*\\?(min|max)\}", r"\\arg\\\2_{\1}", t)
    # d x/dt scritto come \frac{dx}{dt} o \frac{d}{dt}x: stessa cosa
    t = re.sub(r"\\frac\{d([^{}]+)\}\{d([a-z])\}", r"\\frac{d}{d\2}\1", t)
    t = re.sub(r"\\frac\{d\}\{d([a-z])\}", r"\\fracd{d\1}", t)
    return t.rstrip(".,;")


def _has_ctrl(obj) -> bool:
    """Backspace/form feed in una stringa: un \\b o \\f di LaTeX (\\beta, \\frac) letto come escape JSON."""
    if isinstance(obj, str):
        return "\x08" in obj or "\x0c" in obj
    if isinstance(obj, dict):
        return any(_has_ctrl(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_ctrl(v) for v in obj)
    return False


def _json(text: str):
    """JSON dalla risposta del modello, che a volte lo fa precedere da una spiegazione con parentesi
    sue ("B has an unclosed `_{i`…"): si prova un eventuale blocco ```json, poi ogni '[' o '{' in
    ordine, finché uno si decodifica. Il LaTeX dentro le stringhe a volte arriva con i backslash non
    raddoppiati: "\\vec" rende il JSON non valido (si ripara raddoppiandoli), ma "\\frac" o "\\beta" sono
    escape JSON validi (\\f, \\b) e la formula diventerebbe "rac" in silenzio: anche quei caratteri di
    controllo fanno riparare. Se non si recupera niente: None (formule contese → incerte)."""
    dec = json.JSONDecoder()
    fence = re.search(r"```(?:json)?\s*([\[{].*?)```", text, re.S)
    sources = [fence.group(1)] if fence else []
    sources.append(text)
    for src in sources:
        for m in re.finditer(r"[\[{]", src):
            for cand in (src[m.start():], _fix_backslashes(src[m.start():])):
                try:
                    d, _ = dec.raw_decode(cand)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, (list, dict)) and d and not _has_ctrl(d):
                    return d
    return None


def _fix_backslashes(s: str) -> str:
    return re.sub(r'\\(?!["\\/]|u[0-9a-fA-F]{4})', r"\\\\", s)


def _cached_call(cache: Path, prompt: str, system: str, model: str, timeout: int) -> tuple[str, float]:
    """`claude -p` con Read; la risposta grezza resta accanto all'immagine (rifare la riconciliazione
    non costa nuove letture). Ritorna (testo, costo della chiamata; 0 se dalla cache)."""
    if cache.exists():
        return json.loads(cache.read_text())["result"], 0.0
    from src.notes_gen.notes_gen import run_claude_json
    d = run_claude_json(prompt, system, model, timeout, tools="Read")
    cache.write_text(json.dumps({"model": model, "result": d.get("result", "")}, ensure_ascii=False))
    return d.get("result", ""), float(d.get("total_cost_usd") or 0)


def _read(img: Path, n: int, model: str, timeout: int, tag: str) -> tuple[list[dict], float]:
    text, cost = _cached_call(img.with_name(f"{img.stem}.read_{tag}_{model}.json"),
                              READ_PROMPT.format(img=img.resolve(), n=n), READ_SYSTEM, model, timeout)
    data = _json(text) or {}
    items = data.get("items", []) if isinstance(data, dict) else []
    items = [i for i in items if isinstance(i, dict)]
    for i in items:
        try:
            i["panel"] = int(i.get("panel", 1))
        except (TypeError, ValueError):
            i["panel"] = 1
    return items, cost


def _align(fa: list[str], fb: list[str]) -> list[tuple[int | None, int | None]]:
    """Coppie (i in A, j in B): abbinamento globale per somiglianza (i due lettori non elencano le
    formule nello stesso ordine), ≥ ALIGN_MIN; None = vista da un solo lettore."""
    na, nb = [normalize(x) for x in fa], [normalize(x) for x in fb]
    cand = sorted(((1.0 if x == y else difflib.SequenceMatcher(None, x, y).ratio(), i, j)
                   for i, x in enumerate(na) for j, y in enumerate(nb)), reverse=True)
    pa, pb, pairs = set(), set(), []
    for r, i, j in cand:
        if r < ALIGN_MIN:
            break
        if i not in pa and j not in pb:
            pa.add(i)
            pb.add(j)
            pairs.append((i, j))
    pairs += [(i, None) for i in range(len(na)) if i not in pa]
    pairs += [(None, j) for j in range(len(nb)) if j not in pb]
    return sorted(pairs, key=lambda p: (p[0] if p[0] is not None else len(na) + p[1]))


def _disputes(ia: list[dict], ib: list[dict]) -> tuple[list, list, list]:
    """Formule dei due lettori (stesso pannello), concordi marcate verified 2/2, discordanze (i, j)."""
    fa = [x for x in ia if x.get("kind") == "formula" and x.get("latex")]
    fb = [x for x in ib if x.get("kind") == "formula" and x.get("latex")]
    out = []
    for i, j in _align([x["latex"] for x in fa], [x["latex"] for x in fb]):
        if i is not None and j is not None and normalize(fa[i]["latex"]) == normalize(fb[j]["latex"]):
            fa[i].update(status="verified", check="2/2")
        else:
            out.append((i, j))
    return fa, fb, out


def _settle(ia: list[dict], ib: list[dict], fa: list, fb: list, disputes: list, verdicts: list[dict]) -> list[dict]:
    """Applica i verdetti dell'arbitro e ritorna gli item riconciliati del pannello: le formule hanno
    "status" verified (2 letture uguali, o arbitro d'accordo con una) / uncertain / unreadable."""
    for (i, j), v in zip(disputes, verdicts):
        verdict = str(v.get("verdict", "")).lower()
        a = fa[i] if i is not None else None
        b = fb[j] if j is not None else None
        if verdict in ("a", "both") and a:
            a.update(status="verified", check="2/3", alt=b["latex"] if b else None)
        elif verdict == "b" and b:
            if a:
                a.update(latex=b["latex"], status="verified", check="2/3", alt=None)
            else:
                b.update(status="verified", check="2/3", _insert=True)
        elif verdict == "absent":
            if a:
                a.update(status="dropped")
        else:
            best = v.get("latex") or (a or b)["latex"]
            tgt = a or b
            tgt.update(latex=best, status="uncertain" if v.get("latex") else "unreadable",
                       check="0/3", alt=[x["latex"] for x in (a, b) if x and x["latex"] != best] or None)
            if not a:
                b["_insert"] = True
    # item finali nell'ordine della lettura A, con le formule viste solo da B inserite vicino
    out = [x for x in ia if x.get("status") != "dropped"]
    for x in fb:
        if x.pop("_insert", False):
            j = ib.index(x)
            prev = next((ib[k] for k in range(j - 1, -1, -1) if ib[k] in out), None)
            out.insert(out.index(prev) + 1 if prev is not None else 0, x)
    for x in out:
        if x.get("kind") == "formula" and "status" not in x:
            x["status"] = "uncertain"
        if x.get("kind") == "formula" and "[?]" in x.get("latex", "") and x["status"] == "verified":
            x.update(status="uncertain", check=x.get("check", "") + " [?]")   # un simbolo illeggibile
    # la stessa formula da più strade (una per lettore, giudicate separatamente): resta la migliore
    rank = {"verified": 0, "uncertain": 1, "unreadable": 2}
    best: dict[str, dict] = {}
    for x in out:
        if x.get("kind") == "formula":
            k = normalize(x["latex"])
            if k not in best or rank[x["status"]] < rank[best[k]["status"]]:
                best[k] = x
    return [x for x in out if x.get("kind") != "formula" or best.get(normalize(x["latex"])) is x]


def read_sheet(img: Path, n: int, models: tuple[str, str, str], timeout: int = 600,
               log=print) -> tuple[dict[int, list[dict]], float]:
    """Due letture indipendenti del foglio + un arbitro per tutte le discordanze del foglio.
    Ritorna {pannello: item riconciliati} e il costo."""
    ma, mb, mj = models
    with ThreadPoolExecutor(2) as ex:
        ra, rb = ex.submit(_read, img, n, ma, timeout, "a"), ex.submit(_read, img, n, mb, timeout, "b")
        (ia, ca), (ib, cb) = ra.result(), rb.result()
    cost = ca + cb
    per = {}
    for k in range(1, n + 1):
        pa = [x for x in ia if x["panel"] == k]
        pb = [x for x in ib if x["panel"] == k]
        per[k] = (pa, pb, *_disputes(pa, pb))
    flat = [(k, i, j) for k, (_, _, fa, fb, ds) in per.items() for i, j in ds]
    verdicts = {}
    if flat:
        txt = "\n".join(
            f"#{q} (panel {k})  A: {per[k][2][i]['latex'] if i is not None else '(not transcribed)'}\n"
            f"    B: {per[k][3][j]['latex'] if j is not None else '(not transcribed)'}"
            for q, (k, i, j) in enumerate(flat, 1))
        key = hashlib.sha1(txt.encode()).hexdigest()[:8]
        text, c = _cached_call(img.with_name(f"{img.stem}.judge_{mj}_{key}.json"),
                               JUDGE_PROMPT.format(img=img.resolve(), disputes=txt), JUDGE_SYSTEM, mj, timeout)
        cost += c
        verdicts = {v.get("k"): v for v in (_json(text) or []) if isinstance(v, dict)}
    out, q = {}, 0
    for k, (pa, pb, fa, fb, ds) in per.items():
        vs = [verdicts.get(q + r + 1, {}) for r in range(len(ds))]
        q += len(ds)
        out[k] = _settle(pa, pb, fa, fb, ds, vs)
    n_f = [x for items in out.values() for x in items if x.get("kind") == "formula"]
    log(f"board: {img.name} ({n} pannelli): {len(n_f)} formule — "
        f"{sum(x.get('check') == '2/2' for x in n_f)} concordi, "
        f"{sum(str(x.get('check', '')).startswith('2/3') and x['status'] == 'verified' for x in n_f)} decise "
        f"dall'arbitro, {sum(x['status'] != 'verified' for x in n_f)} incerte (${cost:.3f})")
    return out, cost


def read_shot(img: Path, models: tuple[str, str, str], timeout: int = 600, log=print) -> tuple[list[dict], float]:
    """Un'immagine sola (un pannello)."""
    out, cost = read_sheet(img, 1, models, timeout, log=log)
    return out[1], cost


SHEET_W, SHEET_ROWS, PANEL_H = 1900, 2, 640
MAX_BOARD_FIGS = 6
RENDER_S, RENDER_K = 300, 7                         # ritaglio: ultime 7 osservazioni libere negli ultimi 5 minuti


def pack_sheets(shots: list[Shot], out_dir: Path) -> list[tuple[Path, list[int]]]:
    """Impagina i ritagli in fogli numerati (righe larghe ≤ SHEET_W, ≤ SHEET_ROWS righe): una lettura
    per foglio invece che per ritaglio. Ritorna [(foglio, indici degli shot nell'ordine dei pannelli)]."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:
        font = ImageFont.load_default()
    ims = []
    for s in shots:
        im = Image.open(s.image).convert("RGB")
        sc = PANEL_H / im.height
        ims.append(im.resize((max(1, int(im.width * sc)), PANEL_H)) if abs(sc - 1) > 0.02 else im)
    sheets, rows, row, roww = [], [], [], 0
    for idx, im in enumerate(ims):
        w = min(im.width, SHEET_W)
        if row and roww + 12 + w > SHEET_W:
            rows.append(row)
            row, roww = [], 0
            if len(rows) == SHEET_ROWS:
                sheets.append(rows)
                rows = []
        row.append(idx)
        roww += (12 if roww else 0) + w
    if row:
        rows.append(row)
    if rows:
        sheets.append(rows)
    out = []
    for n, rows in enumerate(sheets, 1):
        width = max(sum(min(ims[i].width, SHEET_W) for i in r) + 12 * (len(r) - 1) for r in rows)
        sheet = Image.new("RGB", (width, len(rows) * PANEL_H + 12 * (len(rows) - 1)), "white")
        d = ImageDraw.Draw(sheet)
        order, y, k = [], 0, 0
        for r in rows:
            x = 0
            for i in r:
                k += 1
                im = ims[i].crop((0, 0, min(ims[i].width, SHEET_W), PANEL_H))
                sheet.paste(im, (x, y))
                d.rectangle((x, y, x + 58, y + 46), fill="white", outline="black", width=2)
                d.text((x + 10, y + 4), str(k), fill="black", font=font)
                order.append(i)
                x += im.width + 12
            y += PANEL_H + 12
        path = out_dir / f"sheet_{n:02d}.png"
        sheet.save(path)
        out.append((path, order))
    return out


# ── pipeline ─────────────────────────────────────────────────────────────────

@dataclass
class BoardReading:
    segments: list[tuple[float, float]]
    shots: list[Shot]

    def prompt_text(self) -> str:
        parts = []
        for k, s in enumerate(self.shots, 1):
            lines = [f"[board {k} · written by {s.mmss} · figure board_{k:02d}]"]
            for x in s.items:
                if x.get("kind") == "formula":
                    if x["status"] == "covered":
                        continue                     # pezzo di una formula verificata altrove
                    tag = {"verified": "VERIFIED", "uncertain": "UNCERTAIN", "unreadable": "UNREADABLE"}[x["status"]]
                    lines.append(f"  {tag}{' boxed' if x.get('boxed') else ''}: {x['latex']}")
                elif x.get("text"):
                    lines.append(f"  text: {x['text']}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def figures(self, latex_dir: Path, max_figs: int = MAX_BOARD_FIGS) -> list[dict]:
        """Foto della lavagna come figure: solo i ritagli con formule non verificate (il lettore le
        controlla sulla foto; quelle verificate non ne hanno bisogno), al massimo max_figs, prima
        quelli con più formule incerte. Con tutte (40 ritagli) gli appunti avevano 20+ figure."""
        import os
        bad = [(sum(x.get("status") in ("uncertain", "unreadable") for x in s.items), k, s)
               for k, s in enumerate(self.shots, 1)]
        bad = sorted((b for b in bad if b[0]), key=lambda b: -b[0])[:max_figs]
        return [{"timestamp": s.t1, "board": k, "uncertain": True, "caption": f"Blackboard, {s.mmss}",
                 "latex_path": os.path.relpath(s.image, latex_dir)}
                for _, k, s in sorted(bad, key=lambda b: b[1])]

    def verified(self) -> list[str]:
        return [x["latex"] for s in self.shots for x in s.items
                if x.get("kind") == "formula" and x.get("status") == "verified"]


def prepare_board(video: Path, out_dir: Path, log=print) -> tuple[list, list[Shot], list] | None:
    """Fase senza token: analisi del video, ritagli a piena risoluzione e fogli numerati. In cache in
    <out_dir>/analysis.json (rifarla costa ~2 min di CPU). Ritorna (tratti, shot, fogli) o None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "analysis.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("version") == VERSION:
                shots = [Shot(**s) for s in d["shots"]]
                if not shots:
                    return None                   # analizzato: nessuna lavagna
                if all(Path(s.image).exists() for s in shots):
                    sheets = [(Path(p), o) for p, o in d["sheets"]]
                    return [tuple(x) for x in d["segments"]], shots, sheets
        except Exception:
            pass
    segs, shots, box = analyse(video, log=log)
    if not shots:
        cache.write_text(json.dumps({"version": VERSION, "segments": [], "shots": [], "sheets": []}))
        return None
    for k, s in enumerate(shots, 1):
        s.image = str(render(video, s, box, out_dir / f"board_{k:02d}.png"))
    sheets = pack_sheets(shots, out_dir)
    log(f"board: {len(shots)} ritagli in {len(sheets)} fogli")
    cache.write_text(json.dumps({"version": VERSION, "segments": segs, "box": list(map(float, box)),
                                 "shots": [asdict(s) for s in shots],
                                 "sheets": [(str(p), o) for p, o in sheets]}, ensure_ascii=False, indent=1))
    return segs, shots, sheets


def resolve_fragments(shots: list[Shot]) -> int:
    """Formule non verificate che sono solo pezzi di formule verificate altrove nella lezione (un
    ritaglio che taglia la formula al bordo; lo stesso contenuto, intero, è verificato in un altro
    ritaglio), o frammenti senza contenuto (\\Phi), [?]i=1): status "covered", fuori da prompt e
    statistiche. Un pezzo è "dentro" se tutte le sue parti tra i [?] (≥ 3 caratteri normalizzati)
    stanno nella STESSA formula verificata. Ritorna quante formule ha coperto."""
    verified = [normalize(x["latex"]) for s in shots for x in s.items
                if x.get("kind") == "formula" and x.get("status") == "verified"]
    n = 0
    for s in shots:
        for x in s.items:
            if x.get("kind") != "formula" or x.get("status") not in ("uncertain", "unreadable"):
                continue
            parts = [q for q in normalize(x["latex"]).split("[?]") if len(q) >= 3]
            useful = sum(len(q) for q in parts)
            if useful < 6 or any(all(q in v for q in parts) for v in verified):
                x["status"] = "covered"
                n += 1
    return n


def read_board(video: Path, out_dir: Path, models=("sonnet", "opus", "opus"), timeout: int = 600,
               workers: int = 3, log=print) -> BoardReading | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "board.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("version") == VERSION and list(d.get("models", [])) == list(models):
                shots = [Shot(**s) for s in d["shots"]]
                if all(Path(s.image).exists() for s in shots):
                    resolve_fragments(shots)
                    log(f"board: dalla cache ({len(shots)} keyframe)")
                    return BoardReading([tuple(x) for x in d["segments"]], shots)
        except Exception:
            pass
    prep = prepare_board(video, out_dir, log=log)
    if not prep:
        return None
    segs, shots, sheets = prep

    def one(sheet: Path, order: list[int]):
        res, cost = read_sheet(sheet, len(order), models, timeout, log=log)
        for k, i in enumerate(order, 1):
            shots[i].items = res.get(k, [])
        shots[order[0]].cost += cost

    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(one, sh, order) for sh, order in sheets]:
            f.result()
    total = sum(s.cost for s in shots)
    cache.write_text(json.dumps({"version": VERSION, "models": list(models), "segments": segs,
                                 "shots": [asdict(s) for s in shots]}, ensure_ascii=False, indent=1))
    covered = resolve_fragments(shots)
    log(f"board: {len(shots)} keyframe letti (${total:.2f}); {covered} frammenti coperti da formule verificate")
    return BoardReading(segs, shots)


def reading_stats(board: BoardReading) -> dict:
    """Per il cancello prima degli appunti: ritagli senza nulla di letto, quota di formule non verificate
    (i frammenti coperti da una formula verificata non contano)."""
    f = [x for s in board.shots for x in s.items if x.get("kind") == "formula" and x.get("status") != "covered"]
    return {"shots": len(board.shots),
            "empty_shots": sum(not s.items for s in board.shots),
            "formulas": len(f),
            "verified": sum(x.get("status") == "verified" for x in f),
            "uncertain_frac": round(sum(x.get("status") != "verified" for x in f) / max(1, len(f)), 2),
            "cost": round(sum(s.cost for s in board.shots), 2)}


# ── controllo a valle ────────────────────────────────────────────────────────

def _math_snippets(tex: str) -> list[str]:
    envs = r"equation\*?|align\*?|gather\*?|multline\*?"
    out = re.findall(rf"\\begin\{{({envs})\}}(.*?)\\end\{{\1\}}", tex, re.S)
    snippets = [b for _, b in out]
    snippets += re.findall(r"\\\[(.*?)\\\]", tex, re.S)
    snippets += re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", tex, re.S)
    # righe di align/gather e \label/\tag via
    parts = []
    for s in snippets:
        s = re.sub(r"\\(label|tag)\{[^}]*\}|\\nonumber|\\notag", "", s)
        parts.append(s)
        parts += [p for p in re.split(r"\\\\", s) if p.strip()]
    return [normalize(p.replace("&", "")) for p in parts]


def _fragment(nf: str) -> bool:
    """Formula tagliata al bordo di un ritaglio ("\\dots)=f_h(\\mu)", "f_h(\\mu)-A"): parentesi
    sbilanciate o che comincia/finisce con un operatore. Non va cercata intatta negli appunti."""
    t = nf.replace("\\left", "").replace("\\right", "")
    if any(t.count(a) != t.count(b) for a, b in ("()", "[]", "{}")):
        return True
    return bool(re.match(r"^([=+\-*/,)\]]|\\dots)", t) or re.search(r"([=+\-*/,(\[]|\\dots)$", t))


def check_notes(tex: str, board: BoardReading) -> list[dict]:
    """Per ogni formula verificata della lavagna: la migliore corrispondenza nel .tex degli appunti.
    ratio 1.0 = copiata identica (a meno di scrittura LaTeX); < 0.85 = assente o cambiata."""
    snippets = _math_snippets(tex)
    report, seen = [], set()
    for f in board.verified():
        nf = normalize(f)
        if nf in seen or _fragment(nf):
            continue
        seen.add(nf)
        if len(re.findall(r"\\[A-Za-z]+|.", nf)) < 5:   # simboli isolati (\lambda_j, \vec a): non significativi
            continue
        best, where = 0.0, ""
        for s in snippets:
            if nf in s:
                best, where = 1.0, s
                break
            sm = difflib.SequenceMatcher(None, nf, s)
            # la formula può stare dentro un'espressione più lunga: confronta col blocco più simile
            m = sm.find_longest_match(0, len(nf), 0, len(s))
            lo = max(0, m.b - m.a)
            r = difflib.SequenceMatcher(None, nf, s[lo:lo + len(nf) + 4]).ratio()
            if r > best:
                best, where = r, s[lo:lo + len(nf) + 4]
        report.append({"board": f, "ratio": round(best, 3), "notes": where})
    return report
