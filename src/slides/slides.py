"""
slides.py — Trova ed estrae le slide di una lezione (es. dalla cartella di WeBeep Sync)
per usarle come supporto a trascrizione e appunti.

Layout atteso (WeBeep Sync): <slides_dir>/<NOME CORSO>/<sezione>/<modulo>/*.pdf
  - la cartella corso è scelta per somiglianza con il nome corso dell'archivio
  - il deck giusto è scelto con un punteggio: parole dell'argomento della lezione
    (nome file + prima pagina) + termini distintivi del deck presenti nella trascrizione
    (termini che compaiono in quel deck ma non negli altri del corso)

Estrazione (PyMuPDF): testo pagina per pagina, immagini raster sopra soglia
(scartando loghi/sfondi ripetuti), render intero delle pagine con grafica vettoriale.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path

STOP = set("""the a an and or of to in on for with by from at as is are was were be been this that these those it its
we you they he she our your their which what when where how why not no yes can could may might will would shall should
also than then there here into onto over under between about after before during while each per via more most very
some any all both such same other another one two three first second last new used using use use used based""".split())
STOP |= set("""il lo la i gli le un uno una di del della dei delle a al alla ai alle da dal dalla in nel nella nei nelle
con su sul sulla per tra fra e ed o che chi cui non si è sono come anche più molto ogni questo questa questi queste""".split())
# parole comuni (inglese) che non dicono nulla sull'argomento: escluse dal calcolo dei termini distintivi
STOP |= set("""about above according across actually after again against allow allows almost along already also
although always among another anything around ask asked available away back become been before begin behind being below
better between beyond both bring called came cannot case cases certain change changes chapter come comes coming common
completely consider consists could course day days define defined depends described design develop developed different
difficult done down during each early either else end enough even ever every everything example examples exam fact far
few final finally find first follow following form found four from further general generally get give given goes going
good got great group hand happen happens hard have having help here high however idea important inside instead just keep
kind know large last later least less let like likely little long look looking made main make makes making many matter
maybe mean means might minutes more moreover most mostly move much must name near need needed never next nothing now number
obtain obtained often once only open option order other others out over own part parts people perhaps place point points
possible present presented presentation probably problem problems project put question questions quite rather really
reason related require respect result results right said same say second see seem seen several shall short should show
shown side simple simply since slide slides small some something sometimes soon start started still student students
study such sure take taken tell than that their them then there therefore these thing things think third those though
three through thus time today together too took toward two type types under understand until upon used useful uses
using usually various very want way ways week well went were what whatever when where whether which while whole whose
why will within without work works would write year years yes yet you your""".split())

MIN_FIG_W, MIN_FIG_H = 180, 120        # px: sotto è un'icona/logo
REPEAT_PAGES = 3                       # immagine presente su ≥ N pagine = logo/sfondo
MAX_FIGS_PER_PAGE = 2
BACKGROUND_FRACTION = 0.85             # immagine che copre ≥ 85% della pagina = sfondo
SKIP_FIRST_PAGES = 1                   # slide titolo: niente figure
VECTOR_MIN_DRAWINGS = 25               # pagina con molti tracciati vettoriali → render intero
RENDER_DPI = 130


@dataclass
class SlidePage:
    page: int
    title: str
    text: str
    figures: list[str] = field(default_factory=list)     # path relativi a out_dir


@dataclass
class SlideDeck:
    pdf: str
    out_dir: str
    pages: list[SlidePage]
    captions: dict = field(default_factory=dict)   # nome figura → didascalia (triage); assente = non valutata
    dropped: list = field(default_factory=list)    # figure scartate dal triage

    # ── viste per i consumatori ────────────────────────────────────────────
    def _relevant_pages(self, transcript: str | None) -> set[int]:
        """Pagine con almeno un termine specifico in comune con la trascrizione (tutte se assente)."""
        if not transcript:
            return {p.page for p in self.pages}
        tr = _tokens(transcript, 5)
        return {p.page for p in self.pages if _tokens(p.title + " " + p.text, 5) & tr}

    def prompt_text(self, max_chars: int = 40_000, transcript: str | None = None) -> str:
        """Testo delle slide per il prompt. Con la trascrizione, solo le pagine pertinenti
        (un deck può coprire più lezioni: le altre non devono contaminare gli appunti)."""
        keep = self._relevant_pages(transcript)
        parts = []
        for p in self.pages:
            body = p.text.strip()
            if not body or p.page not in keep:
                continue
            parts.append(f"[slide {p.page}: {p.title}]\n{body}")
        out = "\n\n".join(parts)
        return out if len(out) <= max_chars else out[:max_chars] + "\n[... slides truncated ...]"

    def figure_list(self, transcript: str | None = None, max_figures: int | None = None) -> list[dict]:
        """
        Figure da offrire al modello. Con la trascrizione: punteggio di pertinenza
        (2 × termini della didascalia + termini della slide in comune con il parlato, ≥5 lettere),
        tenute solo quelle con punteggio ≥ 2, al massimo max_figures (le più pertinenti), in ordine di slide.
        """
        tr = _tokens(transcript, 5) if transcript else None
        # quanto il termine è discusso a lezione: occorrenze nella trascrizione (saturazione a 5)
        tf: Counter = Counter(t for t in _norm(transcript).split() if len(t) >= 5) if transcript else Counter()
        # specificità di un termine nel deck: 1/(numero di slide in cui compare) → "robot" vale poco, "jellyfish" molto
        df: Counter = Counter()
        for p in self.pages:
            df.update(_tokens(p.title + " " + p.text + " " + " ".join(self.captions.get(f, "") for f in p.figures), 5))
        cands = []
        for p in self.pages:
            ctx = _tokens(p.title + " " + p.text, 5)
            for f in p.figures:
                if f in self.dropped:
                    continue
                cap = self.captions.get(f, "")
                hint = cap or p.title
                score = 0.0
                if tr is not None:
                    cap_t = _tokens(cap, 5)
                    score = sum((2.0 if t in cap_t else 1.0) / df[t] * min(tf[t], 5) / 5 for t in (cap_t | ctx) & tr)
                    if score < 0.5:
                        continue
                cands.append((score, p.page, {"slide": p.page, "path": str(Path(self.out_dir) / f), "hint": hint}))
        if max_figures:
            cands = sorted(cands, key=lambda c: -c[0])[:max_figures]
        return [c[2] for c in sorted(cands, key=lambda c: c[1])]

    def key_terms(self, max_chars: int = 500) -> str:
        """Titoli + termini tecnici (maiuscole/multiword) per l'initial_prompt di Whisper."""
        terms: list[str] = []
        seen = set()
        for p in self.pages:
            for t in [p.title] + re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Za-z]+-[A-Za-z]+)\b", p.text):
                k = t.strip().lower()
                if 3 < len(k) < 60 and k not in seen and not k.split()[0] in STOP:
                    seen.add(k)
                    terms.append(t.strip())
        s = ", ".join(terms)
        return s[:max_chars]

    def save(self) -> Path:
        path = Path(self.out_dir) / "deck.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return path

    @staticmethod
    def load(out_dir: Path) -> "SlideDeck | None":
        path = Path(out_dir) / "deck.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return SlideDeck(pdf=d["pdf"], out_dir=d["out_dir"], pages=[SlidePage(**p) for p in d["pages"]],
                         captions=d.get("captions", {}), dropped=d.get("dropped", []))


# ── ricerca ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _tokens(s: str, min_len: int = 3) -> set[str]:
    return {t for t in _norm(s).split() if len(t) >= min_len and t not in STOP and not t.isdigit()}


def find_course_dir(slides_root: Path, course_name: str, min_ratio: float = 0.6) -> Path | None:
    if not slides_root.is_dir():
        return None
    target = _norm(course_name)
    best, best_r = None, 0.0
    for d in slides_root.iterdir():
        if not d.is_dir():
            continue
        r = SequenceMatcher(None, target, _norm(d.name)).ratio()
        if target in _norm(d.name) or _norm(d.name) in target:
            r = max(r, 0.95)
        if r > best_r:
            best, best_r = d, r
    return best if best_r >= min_ratio else None


DECK_EXT = (".pdf", ".pptx", ".ppt", ".odp")
CONVERT_DIR = Path("output/slides/_converted")


def list_decks(course_dir: Path) -> list[Path]:
    return sorted(p for p in course_dir.rglob("*") if p.suffix.lower() in DECK_EXT
                  and not p.name.startswith("~$") and p.stat().st_size > 10_000)


def as_pdf(deck: Path) -> Path | None:
    """PDF del deck: il file stesso se è un PDF, altrimenti conversione LibreOffice (cache per mtime+size)."""
    if deck.suffix.lower() == ".pdf":
        return deck
    import hashlib
    import shutil
    import subprocess
    st = deck.stat()
    key = hashlib.md5(f"{deck.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]
    out = CONVERT_DIR / f"{deck.stem}__{key}.pdf"
    if out.exists():
        return out
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    CONVERT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONVERT_DIR / f"_tmp_{key}"
    tmp.mkdir(exist_ok=True)
    try:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(deck)],
                       capture_output=True, timeout=300, env={**os.environ, "HOME": os.environ.get("HOME", "/tmp")})
        produced = next(tmp.glob("*.pdf"), None)
        if produced:
            produced.replace(out)
            return out
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _first_page_text(deck: Path) -> str:
    try:
        import pymupdf as fitz
        pdf = as_pdf(deck)
        if not pdf:
            return ""
        with fitz.open(pdf) as doc:
            return doc[0].get_text() if len(doc) else ""
    except Exception:
        return ""


def _deck_terms(deck: Path, max_pages: int = 80) -> Counter:
    try:
        import pymupdf as fitz
        fitz.TOOLS.mupdf_display_errors(False)
        pdf = as_pdf(deck)
        if not pdf:
            return Counter()
        c: Counter = Counter()
        with fitz.open(pdf) as doc:
            for page in list(doc)[:max_pages]:
                c.update(_tokens(page.get_text()))
        return c
    except Exception:
        return Counter()


def match_deck(decks: list[Path], topic: str, transcript: str | None = None,
               min_score: float = 0.15) -> tuple[Path, float, list[tuple[Path, float]]] | None:
    """
    Sceglie il deck della lezione. Punteggio in [0,1]:
      - 40%: overlap parole dell'argomento con nome file + prima pagina
      - 60%: frazione dei termini DISTINTIVI del deck (presenti in ≤ 1/3 dei deck del corso)
             che compaiono nella trascrizione (se disponibile)
    """
    if not decks:
        return None
    topic_tok = _tokens(topic)
    tr_tok = _tokens(transcript) if transcript else set()
    terms = {d: _deck_terms(d) for d in decks}
    df: Counter = Counter()
    for c in terms.values():
        df.update(set(c))
    n = len(decks)
    scored = []
    for d in decks:
        name_tok = _tokens(d.stem) | _tokens(_first_page_text(d))
        s_topic = len(topic_tok & name_tok) / max(len(topic_tok), 1) if topic_tok else 0.0
        distinctive = {t for t in terms[d] if df[t] <= max(1, n // 3) and len(t) >= 5}
        s_tr = (sum(1 for t in distinctive if t in tr_tok) / max(len(distinctive), 1)) if tr_tok else 0.0
        score = 0.4 * s_topic + 0.6 * s_tr if tr_tok else s_topic
        scored.append((d, round(score, 3)))
    scored.sort(key=lambda x: -x[1])
    best, sc = scored[0]
    if sc < min_score:
        return None
    return best, sc, scored


# ── estrazione ───────────────────────────────────────────────────────────────

def _page_title(page) -> str:
    """Titolo = righe scritte con il font più grande della pagina (in ordine di lettura)."""
    try:
        d = page.get_text("dict")
    except Exception:
        return ""
    spans = [(s["size"], s["text"].strip()) for b in d.get("blocks", []) if b.get("type") == 0
             for l in b.get("lines", []) for s in l.get("spans", []) if s["text"].strip()]
    if not spans:
        return ""
    big = max(sz for sz, _ in spans)
    title = " ".join(t for sz, t in spans if sz >= 0.9 * big)
    return re.sub(r"\s+", " ", title)[:90]


def extract_deck(deck_path: Path, out_dir: Path, max_pages: int = 150) -> SlideDeck:
    import pymupdf as fitz
    fitz.TOOLS.mupdf_display_errors(False)   # annotazioni video/Screen dei pptx → warning inutili

    pdf = as_pdf(deck_path)
    if not pdf:
        raise RuntimeError(f"impossibile convertire {deck_path.name} in PDF (LibreOffice mancante?)")
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    pages = list(doc)[:max_pages]

    # 1) pre-scan: quante pagine contiene ogni immagine (per scartare loghi/sfondi)
    occurrences: Counter = Counter()
    per_page_imgs: list[list[tuple[int, str]]] = []
    for page in pages:
        imgs = []
        seen_xref = set()
        # get_image_info: solo le immagini disegnate sulla pagina (get_images elenca le risorse,
        # che LibreOffice condivide tra tutte le pagine → falsi "loghi ripetuti")
        page_area = max(page.rect.width * page.rect.height, 1)
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref", 0)
            if not xref or xref in seen_xref:
                continue
            seen_xref.add(xref)
            bb = info.get("bbox")
            if bb:
                frac = max(0, (bb[2] - bb[0])) * max(0, (bb[3] - bb[1])) / page_area
                if frac >= BACKGROUND_FRACTION or frac < 0.02:
                    continue
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < MIN_FIG_W or pix.height < MIN_FIG_H:
                    continue
                h = hashlib.md5(pix.samples[:65536] + f"{pix.width}x{pix.height}".encode()).hexdigest()
            except Exception:
                continue
            imgs.append((xref, h))
            occurrences[h] += 1
        per_page_imgs.append(imgs)

    result: list[SlidePage] = []
    for idx, page in enumerate(pages):
        pno = idx + 1
        text = page.get_text().strip()
        title = _page_title(page) or f"slide {pno}"
        figs: list[str] = []

        seen_h = set()
        for xref, h in per_page_imgs[idx]:
            if idx < SKIP_FIRST_PAGES or occurrences[h] >= REPEAT_PAGES or h in seen_h or len(figs) >= MAX_FIGS_PER_PAGE:
                continue
            seen_h.add(h)
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:          # CMYK → RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                name = f"fig_p{pno:03d}_{len(figs) + 1}.png"
                pix.save(out_dir / name)
                figs.append(name)
            except Exception:
                continue

        # pagina con grafica vettoriale (schemi, plot) e nessuna raster utile → render intero
        if not figs:
            try:
                n_draw = sum(len(d.get("items", [])) for d in page.get_drawings())
            except Exception:
                n_draw = 0
            if n_draw >= VECTOR_MIN_DRAWINGS:
                name = f"slide_p{pno:03d}.png"
                page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False).save(out_dir / name)
                figs.append(name)

        result.append(SlidePage(page=pno, title=title, text=text, figures=figs))

    doc.close()
    deck = SlideDeck(pdf=str(deck_path), out_dir=str(out_dir), pages=result)
    deck.save()
    return deck


# ── entry point comodo ───────────────────────────────────────────────────────

def apply_triage(deck: SlideDeck, model: str = "haiku", log=print) -> SlideDeck:
    """Provino + Haiku: scarta figure decorative e assegna didascalie. Salva in deck.json."""
    from src.slides.triage import triage_figures
    figs = [Path(deck.out_dir) / f for p in deck.pages for f in p.figures]
    if not figs:
        return deck
    verdict = triage_figures(figs, Path(deck.out_dir), model=model, log=log)
    deck.dropped = [n for n, v in verdict.items() if v is None]
    deck.captions = {n: v for n, v in verdict.items() if v}
    deck.save()
    return deck


def locate_and_extract(slides_root: Path, course_name: str, topic: str, out_dir: Path,
                       transcript: str | None = None, log=print, triage: bool = True,
                       triage_model: str = "haiku") -> SlideDeck | None:
    """Cerca il deck della lezione e lo estrae in out_dir (riusa deck.json se già fatto)."""
    cached = SlideDeck.load(out_dir)
    if cached and Path(cached.pdf).exists():
        return cached
    course_dir = find_course_dir(slides_root, course_name)
    if not course_dir:
        log(f"slides: nessuna cartella per '{course_name}' in {slides_root}")
        return None
    decks = list_decks(course_dir)
    m = match_deck(decks, topic, transcript)
    if not m:
        log(f"slides: nessun deck convincente per '{topic}' tra {len(decks)} PDF in {course_dir.name}")
        return None
    best, score, ranking = m
    if len(ranking) > 1 and ranking[1][1] > 0 and score - ranking[1][1] < 0.05:
        log(f"slides: ambiguo ({best.name} {score} vs {ranking[1][0].name} {ranking[1][1]}) — salto")
        return None
    log(f"slides: {best.relative_to(course_dir)} (score {score})")
    deck = extract_deck(best, out_dir)
    n_fig = sum(len(p.figures) for p in deck.pages)
    log(f"slides: {len(deck.pages)} pagine, {n_fig} figure → {out_dir}")
    if triage and n_fig:
        deck = apply_triage(deck, model=triage_model, log=log)
    return deck
