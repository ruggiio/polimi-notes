"""
slides.py — Trova ed estrae le slide di una lezione (es. dalla cartella di WeBeep Sync)
per usarle come supporto a trascrizione e appunti.

Layout atteso (WeBeep Sync): <slides_dir>/<NOME CORSO>/<sezione>/<modulo>/*.pdf
  - la cartella corso è scelta per somiglianza con il nome corso dell'archivio
  - i deck della lezione (1..3: una lezione può finire un deck e iniziarne un altro) sono
    scelti per evidenza nella trascrizione (select_decks): termini che esistono solo in quel
    deck del corso e vengono pronunciati + pagine con più termini rari in comune col parlato.
    Il deck migliore entra sempre: la scelta non viene mai saltata per "ambiguità", perché il
    filtro a livello di pagina/figura (LectureSlides) tiene poi solo ciò che è stato discusso.
  - override manuale: chiave "decks": [nomi file] nel sidecar .json del video, oppure
    output/slides/<lezione>/selection.json

Estrazione (PyMuPDF): testo pagina per pagina, immagini raster sopra soglia
(scartando loghi/sfondi ripetuti), render intero delle pagine con grafica vettoriale.
Cache per deck in output/slides/_decks/<nome>__<hash>/ (testo, figure, triage): un deck usato
da più lezioni si estrae e si passa a Haiku una volta sola.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
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

    # ── viste per i consumatori (calcolate da LectureSlides: un deck = caso particolare) ──
    def prompt_text(self, max_chars: int = 40_000, transcript: str | None = None) -> str:
        return LectureSlides([self]).prompt_text(max_chars, transcript)

    def figure_list(self, transcript: str | None = None, max_figures: int | None = None) -> list[dict]:
        return LectureSlides([self]).figure_list(transcript, max_figures)

    def key_terms(self, max_chars: int = 500) -> str:
        return LectureSlides([self]).key_terms(max_chars)

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


@dataclass
class LectureSlides:
    """
    Le slide di una lezione: uno o più deck, in ordine di evidenza. Stesse viste di un
    SlideDeck ma calcolate sull'insieme: le pagine vengono etichettate col deck di origine
    quando i deck sono più di uno, e le figure competono in un unico pool.
    """
    decks: list[SlideDeck]
    selection: list[dict] = field(default_factory=list)   # diagnostica di select_decks / video
    fallback: str = ""                                     # come sono state scelte le figure se il filtro era vuoto
    timeline: object = None                                # video_match.Timeline: quali pagine e quando (se il video c'era)

    def _shown(self) -> dict[tuple[str, int], list[tuple[float, float]]]:
        """{(deck path, pagina): [(t0, t1), ...]} dalle slide riconosciute nel video."""
        out: dict[tuple[str, int], list[tuple[float, float]]] = {}
        if self.timeline:
            for sp in self.timeline.spans:
                out.setdefault((sp.deck, sp.page), []).append((sp.t0, sp.t1))
        return out

    @property
    def pages(self) -> list[SlidePage]:
        return [p for d in self.decks for p in d.pages]

    @staticmethod
    def label(d: SlideDeck) -> str:
        return Path(d.pdf).stem[:40]

    def _items(self) -> list[tuple[SlideDeck, SlidePage]]:
        return [(d, p) for d in self.decks for p in d.pages]

    def prompt_text(self, max_chars: int = 40_000, transcript: str | None = None) -> str:
        """Testo delle slide per il prompt. Con la trascrizione, solo le pagine pertinenti
        (≥ 30% della pagina più pertinente), le più pertinenti prima nel riempire il budget di
        caratteri, poi in ordine di deck e di slide (un deck copre più lezioni, una lezione può
        usarne più d'uno: il resto non deve contaminare gli appunti)."""
        items = [(d, p) for d, p in self._items() if p.text.strip()]
        shown = self._shown()
        if shown:
            # video: solo le pagine mostrate, in ordine di apparizione, con l'intervallo di tempo
            from src.slides.video_match import mmss
            timed = []
            for d, p in items:
                spans = shown.get((d.pdf, p.page))
                if spans:
                    timed.append((min(t0 for t0, _ in spans), d, p, spans))
            timed.sort(key=lambda x: x[0])
            parts, used = [], 0
            multi = len(self.decks) > 1
            for _, d, p, spans in timed:
                when = ", ".join(f"{mmss(a)}–{mmss(b)}" for a, b in spans[:3])
                where = f"{when} · {self.label(d)} · slide {p.page}" if multi else f"{when} · slide {p.page}"
                part = f"[{where}: {p.title}]\n{p.text.strip()}"
                if used + len(part) > max_chars:
                    break
                parts.append(part)
                used += len(part) + 2
            return "\n\n".join(parts)
        if transcript:
            rel = _page_relevance([_tokens(p.title + " " + p.text, 5) for _, p in items], transcript)
            top = max(rel, default=0.0)
            ranked = sorted(range(len(items)), key=lambda i: -rel[i])
            keep, used = set(), 0
            for i in ranked:
                if rel[i] < 0.3 * top or rel[i] <= 0:
                    break
                n = len(items[i][1].text) + 40
                if used + n > max_chars:
                    continue
                keep.add(i)
                used += n
            items = [it for i, it in enumerate(items) if i in keep]
        multi = len(self.decks) > 1
        parts = []
        for d, p in items:
            where = f"{self.label(d)} · slide {p.page}" if multi else f"slide {p.page}"
            parts.append(f"[{where}: {p.title}]\n{p.text.strip()}")
        out = "\n\n".join(parts)
        return out if len(out) <= max_chars else out[:max_chars] + "\n[... slides truncated ...]"

    def figure_list(self, transcript: str | None = None, max_figures: int | None = None,
                    min_score: float = 0.5) -> list[dict]:
        """
        Figure da offrire al modello, in ordine di deck e di slide. Con la trascrizione: punteggio
        di pertinenza (2 × termini della didascalia + termini della slide in comune con il parlato,
        ≥5 lettere, pesati per specificità nell'insieme dei deck e per frequenza nel parlato);
        tenute quelle con punteggio ≥ min_score, al massimo max_figures (le più pertinenti).
        Mai vuota se esiste almeno una figura: senza candidate sopra soglia si tengono le migliori
        con punteggio > 0, e in mancanza le prime del deck principale (self.fallback dice quale).
        """
        tr = _tokens(transcript, 5) if transcript else None
        tf: Counter = Counter(t for t in _norm(transcript).split() if len(t) >= 5) if transcript else Counter()
        df: Counter = Counter()          # specificità: 1/(numero di slide in cui compare il termine)
        for d, p in self._items():
            df.update(_tokens(p.title + " " + p.text + " " + " ".join(d.captions.get(f, "") for f in p.figures), 5))
        shown = self._shown()
        cands: list[tuple[float, int, int, dict]] = []
        for di, d in enumerate(self.decks):
            for p in d.pages:
                spans = shown.get((d.pdf, p.page)) if shown else None
                if shown and not spans:
                    continue                       # video: la slide non è mai stata mostrata
                ctx = _tokens(p.title + " " + p.text, 5)
                for f in p.figures:
                    if f in d.dropped:
                        continue
                    cap = d.captions.get(f, "")
                    score = 0.0
                    if tr is not None:
                        cap_t = _tokens(cap, 5)
                        score = sum((2.0 if t in cap_t else 1.0) / df[t] * min(tf[t], 5) / 5 for t in (cap_t | ctx) & tr)
                    item = {"slide": p.page, "deck": self.label(d), "path": str(Path(d.out_dir) / f),
                            "hint": cap or p.title}
                    if spans:
                        # quanto è rimasta sullo schermo (minuti) conta più delle parole in comune
                        secs = sum(b - a for a, b in spans)
                        score = min_score + secs / 60 + score
                        item["timestamp"] = max(spans, key=lambda x: x[1] - x[0])[0]
                        item["shown_s"] = secs
                    cands.append((score, di, p.page, item))
        self.fallback = ""
        if tr is not None or shown:
            keep = [c for c in cands if c[0] >= min_score]
            if not keep:
                keep = sorted((c for c in cands if c[0] > 0), key=lambda c: -c[0])[:min(3, max_figures or 3)]
                self.fallback = "punteggio > 0" if keep else ""
            if not keep and cands:
                keep = [c for c in cands if c[1] == 0][:min(3, max_figures or 3)]
                self.fallback = "prime del deck principale"
            cands = keep
        if max_figures:
            cands = sorted(cands, key=lambda c: -c[0])[:max_figures]
        if shown:      # in ordine di apparizione nel video
            return [c[3] for c in sorted(cands, key=lambda c: c[3].get("timestamp", 0))]
        return [c[3] for c in sorted(cands, key=lambda c: (c[1], c[2]))]

    def key_terms(self, max_chars: int = 500) -> str:
        """Titoli + termini tecnici (maiuscole/multiword) per l'initial_prompt di Whisper."""
        terms: list[str] = []
        seen = set()
        for _, p in self._items():
            for t in [p.title] + re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Za-z]+-[A-Za-z]+)\b", p.text):
                k = t.strip().lower()
                if 3 < len(k) < 60 and k not in seen and not k.split()[0] in STOP:
                    seen.add(k)
                    terms.append(t.strip())
        s = ", ".join(terms)
        return s[:max_chars]


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
NOTES_SUBDIR = "Appunti"       # appunti generati: stanno nella cartella del corso ma non sono un deck
CONVERT_DIR = Path("output/slides/_converted")


def _deck_key(p: Path) -> tuple[Path, str]:
    """(cartella logica, stem): `X.pptx` e la sua conversione `pdf/X.pdf` (tools/pptx2pdf.py) coincidono."""
    folder = p.parent.parent if p.parent.name == "pdf" and p.suffix.lower() == ".pdf" else p.parent
    return folder, p.stem.lower()


def list_decks(course_dir: Path, notes_subdir: str = NOTES_SUBDIR) -> list[Path]:
    found = sorted(p for p in course_dir.rglob("*") if p.suffix.lower() in DECK_EXT
                   and not p.name.startswith("~$") and p.stat().st_size > 10_000
                   # i nostri PDF di appunti sono sotto la cartella del corso: non sono slide
                   and not (notes_subdir and notes_subdir in p.relative_to(course_dir).parts))
    # sorgente + conversione insieme renderebbero il match "ambiguo" (stesso punteggio):
    # si tiene il PDF se aggiornato (niente riconversione), altrimenti il sorgente
    by_key: dict[tuple[Path, str], Path] = {}
    for d in found:
        k = _deck_key(d)
        other = by_key.get(k)
        if other is None:
            by_key[k] = d
        elif (d.suffix.lower() == ".pdf") != (other.suffix.lower() == ".pdf"):
            pdf, src = (d, other) if d.suffix.lower() == ".pdf" else (other, d)
            by_key[k] = pdf if pdf.stat().st_mtime >= src.stat().st_mtime else src
    return sorted(by_key.values())


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


DECKS_DIR = Path("output/slides/_decks")


def deck_cache_dir(deck: Path) -> Path:
    """Cartella cache del deck (testo, figure, triage), invalidata da path+mtime+size."""
    st = deck.stat()
    key = hashlib.md5(f"{deck.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", deck.stem).strip("_")[:60] or "deck"
    return DECKS_DIR / f"{safe}__{key}"


def deck_text(deck: Path, max_pages: int = 150) -> list[tuple[str, str]]:
    """[(titolo, testo)] pagina per pagina; cache in <deck_cache_dir>/text.json."""
    cache = deck_cache_dir(deck) / "text.json"
    if cache.exists():
        try:
            return [tuple(x) for x in json.loads(cache.read_text())["pages"]]
        except Exception:
            pass
    pages: list[tuple[str, str]] = []
    try:
        import pymupdf as fitz
        fitz.TOOLS.mupdf_display_errors(False)
        pdf = as_pdf(deck)
        if pdf:
            with fitz.open(pdf) as doc:
                for page in list(doc)[:max_pages]:
                    pages.append((_page_title(page), page.get_text()))
    except Exception:
        return pages
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"pdf": str(deck), "pages": pages}, ensure_ascii=False))
    return pages


def cache_source(cache_dir: Path) -> Path | None:
    """Deck sorgente di una cartella cache (da deck.json o text.json), None se illeggibile."""
    for name in ("deck.json", "text.json"):
        f = cache_dir / name
        if f.exists():
            try:
                return Path(json.loads(f.read_text())["pdf"])
            except Exception:
                return None
    return None


def _page_relevance(page_tokens: list[set[str]], transcript: str) -> list[float]:
    """
    Pertinenza di ogni pagina alla lezione: Σ sui termini (≥5 lettere) in comune col parlato di
    idf(termine) × quanto è detto (occorrenze nel parlato, saturazione a 3). idf calcolata
    sulle pagine passate (tutto il corso in select_decks): "camera" in un corso di visione
    vale poco, "eritrea" molto.
    """
    tr = _tokens(transcript, 5)
    tf: Counter = Counter(t for t in _norm(transcript).split() if len(t) >= 5)
    df: Counter = Counter()
    for pt in page_tokens:
        df.update(pt)
    n = max(len(page_tokens), 1)
    return [sum(math.log(n / df[t]) * min(tf[t], 3) / 3 for t in pt & tr) for pt in page_tokens]


def select_decks(decks: list[Path], topic: str, transcript: str | None = None,
                 max_decks: int = 3) -> list[tuple[Path, dict]]:
    """
    Deck usati nella lezione, in ordine di evidenza, [(deck, info)]. Vuoto solo senza alcun indizio.

    Con la trascrizione: pertinenza di ogni pagina del corso (_page_relevance), poi per deck
      mean   = pertinenza media delle pagine (premia il deck seguito slide per slide, non il
               deck lungo che condivide vocabolario sparso)
      strong = pagine con pertinenza ≥ 60% della media del deck migliore
    Il migliore per mean entra sempre (a parità: argomento nel nome file/prima pagina). Un altro
    deck entra se mean ≥ 35% del migliore, strong ≥ 3 e almeno il 15% delle sue pagine è
    ≥ 50% della media del migliore: una lezione che finisce un deck e ne inizia un altro, o
    una panoramica che mostra esempi da più deck; non il deck "cugino" citato di passaggio.
    Senza trascrizione (glossario per Whisper): sovrapposizione dell'argomento con nome file +
    prima pagina; a parità entrano tutti i pari.
    """
    if not decks:
        return []
    topic_tok = _tokens(topic)
    texts = {d: deck_text(d) for d in decks}
    page_tok = {d: [_tokens(t + " " + x, 5) for t, x in texts[d]] for d in decks}

    def topic_score(d: Path) -> float:
        first = texts[d][0][0] + " " + texts[d][0][1] if texts[d] else ""
        return len(topic_tok & (_tokens(d.stem) | _tokens(first))) / max(len(topic_tok), 1) if topic_tok else 0.0

    if not transcript:
        scored = sorted(((d, {"topic": round(topic_score(d), 2), "pages": len(texts[d])}) for d in decks),
                        key=lambda x: (-x[1]["topic"], str(x[0])))
        top = scored[0][1]["topic"]
        return [x for x in scored if x[1]["topic"] == top and top > 0][:max_decks]

    flat = [pt for d in decks for pt in page_tok[d]]
    rel = _page_relevance(flat, transcript)
    per_deck: dict[Path, list[float]] = {}
    i = 0
    for d in decks:
        per_deck[d] = rel[i:i + len(page_tok[d])]
        i += len(page_tok[d])
    mean = {d: (sum(r) / len(r) if r else 0.0) for d, r in per_deck.items()}
    best = max(decks, key=lambda d: (mean[d], topic_score(d)))
    mb = mean[best]
    if mb <= 0:
        return []

    def info(d: Path) -> dict:
        r = per_deck[d]
        return {"mean": round(mean[d] / mb, 2), "strong": sum(x >= 0.6 * mb for x in r),
                "frac": round(sum(x >= 0.5 * mb for x in r) / max(len(r), 1), 2),
                "topic": round(topic_score(d), 2), "pages": len(r)}

    chosen = [(best, info(best))]
    others = sorted((d for d in decks if d != best), key=lambda d: -mean[d])
    for d in others:
        if len(chosen) >= max_decks:
            break
        inf = info(d)
        if inf["mean"] >= 0.35 and inf["strong"] >= 3 and inf["frac"] >= 0.15:
            chosen.append((d, inf))
    return chosen


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
    shutil.rmtree(Path(deck.out_dir) / "_triage", ignore_errors=True)   # provini: servivano solo a Haiku
    deck.dropped = [n for n, v in verdict.items() if v is None]
    deck.captions = {n: v for n, v in verdict.items() if v}
    deck.save()
    return deck


def load_deck(deck_path: Path, triage: bool = True, triage_model: str = "haiku", log=print) -> SlideDeck:
    """SlideDeck estratto (e passato al triage) una volta sola, nella cache per deck."""
    out_dir = deck_cache_dir(deck_path)
    deck = SlideDeck.load(out_dir)
    if deck is None:
        deck = extract_deck(deck_path, out_dir)
        n_fig = sum(len(p.figures) for p in deck.pages)
        log(f"slides: {deck_path.name}: {len(deck.pages)} pagine, {n_fig} figure → {out_dir}")
        if triage and n_fig:
            deck = apply_triage(deck, model=triage_model, log=log)
    return deck


def _resolve_forced(decks: list[Path], forced: list[str]) -> list[Path]:
    """Nomi file (o loro sottostringhe) → deck del corso, nell'ordine dato."""
    out = []
    for name in forced:
        key = _norm(Path(name).stem)
        hit = next((d for d in decks if _norm(d.stem) == key), None) or \
              next((d for d in decks if key and key in _norm(d.stem)), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def video_timeline(video: Path | None, decks: list[Path], out_dir: Path, log=print):
    """Timeline delle slide nel video (cache out_dir/timeline.json), None se non c'è video."""
    from src.slides.video_match import Timeline, match_video
    cached = Timeline.load(out_dir / "timeline.json")
    have_video = bool(video) and Path(video).exists()
    if cached is not None and (not have_video or set(cached.decks or []) == {d.name for d in decks}):
        return cached          # (con deck nuovi in cartella e video ancora presente si rifà)
    if not have_video:
        return None
    tl = match_video(Path(video), decks, log=log)
    if tl is not None:
        tl.save(out_dir / "timeline.json")
    return tl


def locate_and_extract(slides_root: Path, course_name: str, topic: str, out_dir: Path,
                       transcript: str | None = None, log=print, triage: bool = True,
                       triage_model: str = "haiku", forced: list[str] | None = None,
                       max_decks: int = 3, video: Path | None = None,
                       notes_subdir: str = NOTES_SUBDIR) -> LectureSlides | None:
    """
    Slide della lezione (LectureSlides) oppure None se il corso non ha una cartella o nessun
    deck mostra indizi. Scelta dei deck, in ordine di affidabilità:
      1. "decks" forzati (sidecar del video) o selection.json già salvato;
      2. il video (video_match): deck e pagine effettivamente mostrati, con i tempi;
      3. la trascrizione (select_decks), se il video manca o non mostra slide riconoscibili.
    out_dir = output/slides/<lezione>: selection.json e timeline.json (riusati ai giri dopo).
    Senza trascrizione (stadio trascrizione) la scelta non viene salvata e i deck sono solo testo.
    """
    from src.slides.video_match import mmss
    course_dir = find_course_dir(slides_root, course_name)
    if not course_dir:
        log(f"slides: nessuna cartella per '{course_name}' in {slides_root}")
        return None
    decks = list_decks(course_dir, notes_subdir)
    if not decks:
        log(f"slides: nessun deck in {course_dir}")
        return None
    sel_path = out_dir / "selection.json"
    chosen: list[tuple[Path, dict]] = []
    source = ""
    if forced:
        chosen = [(d, {"forced": True}) for d in _resolve_forced(decks, forced)]
        source = "forzati"
        if not chosen:
            log(f"slides: deck forzati non trovati in {course_dir.name}: {forced}")
    elif transcript and sel_path.exists():
        try:
            saved = json.loads(sel_path.read_text())
            chosen = [(d, {"saved": True}) for d in _resolve_forced(decks, saved.get("decks", []))]
            source = saved.get("source", "salvati")
        except Exception:
            chosen = []

    timeline = video_timeline(video, decks, out_dir, log=log)
    if timeline is not None:
        matched = timeline.duration - timeline.unmatched
        if timeline.unmatched >= max(60.0, 0.2 * timeline.duration):
            log(f"video: {mmss(timeline.unmatched)} su {mmss(timeline.duration)} senza slide riconoscibile "
                f"(lavagna, o un deck non ancora in {course_dir.name})")
        if chosen and source not in ("video", "forzati") and matched >= 60:
            chosen = []                # una scelta salvata dalla sola trascrizione cede al video
        if not chosen and matched >= 60:
            by_deck = timeline.by_deck()
            chosen = [(Path(d), {"video": f"{i['seconds'] / 60:.1f} min", "pages": len(i["pages"])})
                      for d, i in by_deck.items() if i["seconds"] >= 60 or len(i["pages"]) >= 2][:max_decks]
            source = "video"
    if not chosen:
        chosen = select_decks(decks, topic, transcript, max_decks=max_decks)
        source = "trascrizione" if transcript else "argomento"
        timeline = None            # il video non ha riconosciuto questi deck: niente tempi
    if not chosen:
        log(f"slides: nessun deck con indizi per '{topic}' tra {len(decks)} in {course_dir.name}")
        return None
    log(f"slides ({source}): " + " + ".join(
        f"{d.name} ({', '.join(f'{k}={v}' for k, v in info.items() if k != 'pages')})" for d, info in chosen))
    if timeline is not None and not any(sp.deck == str(d) for d, _ in chosen for sp in timeline.spans):
        timeline = None

    if not transcript:      # solo testo: glossario per Whisper, niente estrazione/triage
        text_decks = [SlideDeck(pdf=str(d), out_dir=str(deck_cache_dir(d)),
                                pages=[SlidePage(page=i + 1, title=t, text=x) for i, (t, x) in enumerate(deck_text(d))])
                      for d, _ in chosen]
        return LectureSlides(text_decks, [dict(info, deck=d.name) for d, info in chosen], timeline=timeline)

    out_dir.mkdir(parents=True, exist_ok=True)
    sel_path.write_text(json.dumps({"course_dir": str(course_dir), "topic": topic, "source": source,
                                    "decks": [d.name for d, _ in chosen],
                                    "scores": [dict(info, deck=d.name) for d, info in chosen]},
                                   indent=2, ensure_ascii=False))
    loaded = [load_deck(d, triage=triage, triage_model=triage_model, log=log) for d, _ in chosen]
    return LectureSlides(loaded, [dict(info, deck=d.name) for d, info in chosen], timeline=timeline)
