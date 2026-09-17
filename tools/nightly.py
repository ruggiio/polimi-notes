#!/usr/bin/env python3
"""
nightly.py — Job notturno idempotente: scarica → trascrive → genera appunti.

Ogni stadio guarda i file su disco e fa solo ciò che manca:
  1. fetch      archivio PoliMi (corsi in config.auto.courses) → output/videos/<stem>.mp4 + .json
  2. transcribe ogni mp4 senza output/transcripts/<stem>.txt → faster-whisper (GPU)
  3. notes      ogni transcript senza PDF → notes_gen (backend da config) → output/notes/*.pdf
                con contesto RAG dalle lezioni precedenti del corso (rag.enabled); dopo il PDF
                il transcript viene indicizzato in output/rag (ChromaDB, embedding ONNX locali)
  4. cleanup    per le lezioni con PDF: via l'mp4 (se keep_videos false), file di lavoro LaTeX,
                provini del triage, cache di deck obsoleti

Stato/fallimenti in output/auto/state.json; log in output/auto/nightly.log.
Se la sessione PoliMi è scaduta lo stadio fetch viene saltato (con notifica) e gli
altri due proseguono. Uscita: 0 ok, 3 se serve il login manuale, 1 su errori.

  .venv/bin/python tools/nightly.py                 # tutto
  .venv/bin/python tools/nightly.py --no-fetch      # solo trascrizione + note
  .venv/bin/python tools/nightly.py --max-downloads 1 --max-notes 1 --dry-run
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                       # i moduli usano path relativi (config/courses, output/…)
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.downloader.archive import (LoginRequired, PortalUnavailable, Recording, download_media,  # noqa: E402
                                    goto_archive, open_context, open_recording, safe_name,
                                    search_archive, session_expiry)
from src.course_profiles import _slugify, extract_glossary, load_profile  # noqa: E402

EXIT_OK, EXIT_ERR, EXIT_LOGIN = 0, 1, 3


# ── infrastruttura ───────────────────────────────────────────────────────────

class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def notify(title: str, body: str, enabled: bool = True):
    if not enabled or not shutil.which("notify-send"):
        return
    try:
        subprocess.run(["notify-send", "-a", "polimi-notes", title, body], timeout=10)
    except Exception:
        pass


class State:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {}
        self.d.setdefault("downloaded", {})
        self.d.setdefault("failures", {})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def fail(self, key: str, err: str) -> int:
        e = self.d["failures"].setdefault(key, {"n": 0})
        e["n"] += 1
        e["last"] = f"{datetime.now():%Y-%m-%d %H:%M} {err[:300]}"
        self.save()
        return e["n"]

    def failures(self, key: str) -> int:
        return self.d["failures"].get(key, {}).get("n", 0)

    def clear_fail(self, key: str):
        if key in self.d["failures"]:
            del self.d["failures"][key]
            self.save()


def current_aa() -> int:
    t = date.today()
    return t.year if t.month >= 9 else t.year - 1


def course_config(auto: dict, course: str) -> dict:
    """Voce di auto.courses con quel nome (case-insensitive), altrimenti {}."""
    return next((c for c in auto.get("courses", []) if c.get("name", "").lower() == course.lower()), {})


def lecture_meta(video: Path) -> dict:
    """Corso/data/argomento dal sidecar .json, altrimenti dal nome file YYYY-MM-DD_CORSO_argomento."""
    side = video.with_suffix(".json")
    if side.exists():
        d = json.loads(side.read_text())
        course = re.sub(r"^\d+\s*-\s*", "", d.get("course", "")).split(" (")[0].strip() or "Unknown Course"
        return {"course": course, "date": d.get("date_iso") or str(date.today()), "topic": d.get("topic", ""),
                "decks": d.get("decks") or []}     # override manuale dei deck di slide
    m = re.match(r"(\d{4}-\d{2}-\d{2})_([^_]+)_?(.*)", video.stem)
    if m:
        return {"course": m.group(2), "date": m.group(1), "topic": m.group(3)}
    return {"course": "Unknown Course", "date": str(date.today()), "topic": ""}


# ── stadio 1: fetch ──────────────────────────────────────────────────────────

def stage_fetch(cfg: dict, state: State, log: Log, videos_dir: Path, tr_dir: Path,
                max_downloads: int | None, dry_run: bool) -> tuple[int, bool]:
    """Ritorna (n_scaricati, login_richiesto)."""
    from playwright.sync_api import sync_playwright

    auto = cfg["auto"]
    profile = Path(cfg["download"].get("profile_dir", "config/chrome_profile"))
    email = os.environ.get("POLIMI_EMAIL", "")
    if not email:
        log("fetch: POLIMI_EMAIL mancante in .env — salto")
        return 0, False

    exp = session_expiry(profile)
    if exp:
        left_h = (exp - datetime.now()).total_seconds() / 3600
        log(f"fetch: sessione PoliMi valida fino a {exp:%Y-%m-%d %H:%M} ({left_h:.0f} h)")
        if left_h < 48:
            notify("polimi-notes: sessione in scadenza",
                   f"Scade il {exp:%d/%m %H:%M}. Rinnova con tools/sso_login.py", auto.get("notify", True))

    n_done = 0
    with sync_playwright() as pw:
        ctx = open_context(pw, profile, headless=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for course in auto.get("courses", []):
                aa = course.get("aa") or current_aa()
                goto_archive(page)
                recs = search_archive(page, aa=aa, course=course["match"], kind=auto.get("kind"))
                new = [r for r in recs if r.transfer_id not in state.d["downloaded"]
                       and state.failures(f"fetch:{r.transfer_id}") < auto.get("max_failures", 3)]
                log(f"fetch: {course['name']} A.A. {aa}/{aa + 1 - 2000}: {len(recs)} registrazioni, {len(new)} nuove")
                for rec in new:
                    stem = f"{rec.date_iso}_{safe_name(rec.course_name)}_{safe_name(rec.topic, 40)}"
                    out = videos_dir / f"{stem}.mp4"
                    if out.exists() or (tr_dir / f"{stem}.txt").exists():
                        # scaricato fuori dal job (es. tools/fetch_lecture.py): registra e salta
                        state.d["downloaded"][rec.transfer_id] = {"stem": stem, "at": "pre-existing"}
                        state.save()
                        log(f"fetch: già presente {stem}")
                        continue
                    if max_downloads is not None and n_done >= max_downloads:
                        log("fetch: raggiunto --max-downloads")
                        return n_done, False
                    log(f"fetch: ↓ {rec.transfer_id} {rec.date} — {rec.topic[:60]}")
                    if dry_run:
                        continue
                    try:
                        info = open_recording(ctx, page, rec, email)
                        t0 = time.time()
                        download_media(info, out)
                        out.with_suffix(".json").write_text(json.dumps(
                            {**rec.to_dict(), "webex_title": info.title, "playback_url": info.playback_url},
                            indent=2, ensure_ascii=False))
                        state.d["downloaded"][rec.transfer_id] = {"stem": stem, "at": f"{datetime.now():%Y-%m-%d %H:%M}"}
                        state.clear_fail(f"fetch:{rec.transfer_id}")
                        state.save()
                        n_done += 1
                        log(f"fetch: ✓ {out.name} ({out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
                    except LoginRequired:
                        raise
                    except Exception as e:
                        n = state.fail(f"fetch:{rec.transfer_id}", f"{type(e).__name__}: {e}")
                        log(f"fetch: ✗ {rec.transfer_id}: {type(e).__name__}: {str(e)[:200]} (fallimento {n})")
                    # torna all'archivio per la prossima riga (la scheda Webex è stata chiusa)
                    goto_archive(page)
                    search_archive(page, aa=aa, course=course["match"], kind=auto.get("kind"))
        except LoginRequired as e:
            log(f"fetch: LOGIN RICHIESTO — {e}")
            notify("polimi-notes: login richiesto",
                   "La sessione PoliMi è scaduta: esegui tools/sso_login.py", auto.get("notify", True))
            return n_done, True
        except PortalUnavailable as e:
            log(f"fetch: portale PoliMi non disponibile, riprovo al prossimo giro — {e}")
            return n_done, False
        finally:
            ctx.close()
    return n_done, False


# ── stadio 1b: slide dai link WeBeep (cartelle OneDrive) ─────────────────────

def stage_slides_sync(cfg: dict, state: State, log: Log, dry_run: bool) -> bool:
    """Scarica i file nuovi/modificati dalle cartelle linkate. Ritorna True se serve login."""
    from playwright.sync_api import sync_playwright
    from src.slides.onedrive import sync_shared_folder

    auto = cfg["auto"]
    jobs = [(c["name"], link) for c in auto.get("courses", []) for link in (c.get("slides_links") or [])]
    if not jobs or not auto.get("slides", False):
        return False
    root = Path(os.path.expanduser(auto.get("slides_dir", "~/Scrivania/POLI")))
    profile = Path(cfg["download"].get("profile_dir", "config/chrome_profile"))
    with sync_playwright() as pw:
        ctx = open_context(pw, profile, headless=True)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for course, link in jobs:
                if dry_run:
                    log(f"slides-sync: (dry) {course} ← {link}")
                    continue
                key = f"slides-sync:{link}"
                if state.failures(key) >= auto.get("max_failures", 3):
                    log(f"slides-sync: salto {link} (troppi fallimenti)")
                    continue
                try:
                    got = sync_shared_folder(ctx, page, link, root / course / "_links", log=log,
                                             max_size_mb=auto.get("slides_max_mb"))
                    state.clear_fail(key)
                    log(f"slides-sync: {course}: {len(got)} file nuovi/aggiornati")
                except LoginRequired:
                    raise
                except Exception as e:
                    n = state.fail(key, f"{type(e).__name__}: {e}")
                    log(f"slides-sync: ✗ {course}: {type(e).__name__}: {str(e)[:200]} (fallimento {n})")
        except LoginRequired as e:
            log(f"slides-sync: LOGIN RICHIESTO — {e}")
            return True
        finally:
            ctx.close()
    return False


# ── slide della lezione ──────────────────────────────────────────────────────

def find_slides(cfg: dict, log: Log, stem: str, meta: dict, transcript: str | None,
                video: Path | None = None):
    """LectureSlides della lezione (1..slides_max_decks deck, cache in output/slides/_decks) oppure None.
    Con il video: deck e pagine riconosciuti nei frame (con i tempi). Override manuale:
    "decks": [nomi file] nel sidecar .json del video."""
    auto = cfg["auto"]
    if not auto.get("slides", False):
        return None
    root = Path(os.path.expanduser(auto.get("slides_dir", "~/Documenti/WeBeep Sync")))
    if not root.is_dir():
        return None
    from src.slides.slides import locate_and_extract
    try:
        return locate_and_extract(root, meta["course"], meta["topic"] or stem,
                                  Path("output/slides") / _slugify(stem), transcript, log=log,
                                  triage=auto.get("slides_triage", True),
                                  triage_model=auto.get("slides_triage_model", "haiku"),
                                  forced=meta.get("decks") or None,
                                  max_decks=auto.get("slides_max_decks", 3),
                                  video=video if auto.get("slides_video", True) else None)
    except Exception as e:
        log(f"slides: errore {type(e).__name__}: {str(e)[:150]}")
        return None


# ── stadio 2: transcribe ─────────────────────────────────────────────────────

def stage_transcribe(cfg: dict, state: State, log: Log, videos_dir: Path, tr_dir: Path,
                     dry_run: bool) -> int:
    tcfg = cfg["transcription"]
    auto = cfg["auto"]
    todo = sorted(v for v in videos_dir.glob("*.mp4") if not (tr_dir / f"{v.stem}.txt").exists())
    log(f"transcribe: {len(todo)} video senza trascrizione")
    if not todo or dry_run:
        for v in todo:
            log(f"transcribe: (dry) {v.name}")
        return 0
    from src.transcriber.transcriber import get_device, transcribe
    device = tcfg.get("device") or get_device()
    n = 0
    for v in todo:
        key = f"transcribe:{v.stem}"
        if state.failures(key) >= auto.get("max_failures", 3):
            log(f"transcribe: salto {v.name} (troppi fallimenti)")
            continue
        meta = lecture_meta(v)
        course_cfg = course_config(auto, meta["course"])
        # initial_prompt = glossario della scheda corso + titoli/termini delle slide (se trovate)
        glossary = extract_glossary(load_profile(meta["course"]))
        deck = find_slides(cfg, log, v.stem, meta, None, video=v)
        prompt = ", ".join(x for x in (glossary, deck.key_terms() if deck else "") if x) or None
        t0 = time.time()
        try:
            # lingua: per corso (auto.courses[].language) > globale > scelta per confidenza di decodifica
            r = transcribe(v, tr_dir, model_name=tcfg.get("model", "medium"),
                           language=course_cfg.get("language") or tcfg.get("language"),
                           language_candidates=tuple(tcfg.get("language_candidates", ["it", "en"])),
                           device=device, initial_prompt=prompt)
            n += 1
            state.clear_fail(key)
            log(f"transcribe: ✓ {v.stem} lang={r['language']} words={len(r['text'].split())} ({time.time() - t0:.0f}s)")
        except Exception as e:
            k = state.fail(key, f"{type(e).__name__}: {e}")
            log(f"transcribe: ✗ {v.stem}: {type(e).__name__}: {str(e)[:200]} (fallimento {k})")
    return n


# ── RAG di corso (contesto dalle lezioni precedenti) ─────────────────────────

def open_rag(cfg: dict, log: Log):
    """CourseRAG se rag.enabled e chromadb installato, altrimenti None (con log del perché)."""
    rcfg = cfg.get("rag", {})
    if not rcfg.get("enabled", False):
        return None
    try:
        from src.rag.rag import CourseRAG
        return CourseRAG(db_path=rcfg.get("db_path", "output/rag"),
                         chunk_size=rcfg.get("chunk_size", 500), chunk_overlap=rcfg.get("chunk_overlap", 50))
    except Exception as e:
        log(f"rag: non disponibile ({type(e).__name__}: {str(e)[:120]}) — appunti senza contesto di corso")
        return None


def rag_queries(text: str, topic: str, words: int = 150) -> list[str]:
    """Argomento + tre campioni (inizio, metà, fine) della lezione: l'inizio da solo è spesso burocrazia."""
    w = text.split()
    if len(w) <= 3 * words:
        return [f"{topic}. {text}"]
    mid = len(w) // 2
    return [f"{topic}. " + " ".join(w[:words]), " ".join(w[mid - words // 2: mid + words // 2]),
            " ".join(w[-words:])]


def rag_index(rag, log: Log, txt: Path, meta: dict) -> bool:
    """Indicizza il transcript se non lo è già (id = corso+data+chunk: upsert idempotente)."""
    if rag.is_indexed(meta["course"], meta["date"]):
        return False
    n = rag.add_lecture(txt.read_text(encoding="utf-8"), meta["course"], meta["date"])
    log(f"rag: indicizzata {txt.stem} ({n} chunk)")
    return True


def notes_language(ncfg: dict, course_cfg: dict, txt: Path) -> str | None:
    """Lingua degli appunti: auto.courses[].notes_language > notes.language > lingua della lezione
    (da Whisper, nei segmenti). 'lecture' = quella della lezione. None se ignota."""
    want = course_cfg.get("notes_language") or ncfg.get("language") or "lecture"
    if want != "lecture":
        return want
    seg = txt.with_name(txt.stem + "_segments.json")
    try:
        return json.loads(seg.read_text())["language"] if seg.exists() else None
    except Exception:
        return None


def timed_transcript(txt: Path, every: int = 60) -> str:
    """Trascrizione con un marcatore [mm:ss] all'inizio di ogni minuto (dai segmenti Whisper),
    così slide e figure con i tempi si allineano al parlato. '' se i segmenti mancano."""
    seg = txt.with_name(txt.stem + "_segments.json")
    if not seg.exists():
        return ""
    try:
        segments = json.loads(seg.read_text())["segments"]
    except Exception:
        return ""
    parts, next_mark = [], 0
    for sg in segments:
        if sg["start"] >= next_mark:
            m = int(sg["start"] // every) * every
            parts.append(f"[{m // 60:02d}:{m % 60:02d}]")
            next_mark = m + every
        parts.append(sg["text"].strip())
    return " ".join(parts)


# ── stadio 3: notes ──────────────────────────────────────────────────────────

def _pdf_for(meta: dict, pdf_dir: Path) -> Path:
    from src.notes_gen.notes_gen import _make_pdf_filename
    return pdf_dir / _make_pdf_filename(meta["course"], meta["date"], meta["topic"] or None)


def stage_notes(cfg: dict, state: State, log: Log, videos_dir: Path, tr_dir: Path,
                max_notes: int | None, dry_run: bool) -> int:
    from src.notes_gen.notes_gen import generate_notes
    ncfg = cfg["notes"]
    auto = cfg["auto"]
    backend = ncfg["backend"]
    bcfg = dict(ncfg.get(backend, {}))
    bcfg["auto_fix_latex"] = ncfg.get("auto_fix_latex", False)
    bcfg["use_tools"] = False
    latex_dir = Path(ncfg["latex"]["output_dir"])
    pdf_dir = Path(ncfg["latex"].get("pdf_output_dir", "output/notes"))

    todo = []
    done = []
    for txt in sorted(tr_dir.glob("*.txt")):
        meta = lecture_meta(videos_dir / f"{txt.stem}.mp4")
        (done if _pdf_for(meta, pdf_dir).exists() else todo).append((txt, meta))
    log(f"notes: {len(todo)} trascrizioni senza appunti (backend={backend})")
    rag = open_rag(cfg, log) if (todo or done) and not dry_run else None
    if rag:
        # recupero: lezioni con PDF ma non ancora nell'indice (es. transcript fatti prima del RAG)
        for txt, meta in done:
            try:
                rag_index(rag, log, txt, meta)
            except Exception as e:
                log(f"rag: ✗ {txt.stem}: {type(e).__name__}: {str(e)[:120]}")
    n = 0
    for txt, meta in todo:
        if max_notes is not None and n >= max_notes:
            log("notes: raggiunto --max-notes")
            break
        key = f"notes:{txt.stem}"
        if state.failures(key) >= auto.get("max_failures", 3):
            log(f"notes: salto {txt.stem} (troppi fallimenti)")
            continue
        log(f"notes: → {meta['course']} {meta['date']} — {meta['topic'][:50]}")
        if dry_run:
            continue
        # modello per corso (es. opus per i corsi più matematici), altrimenti quello del backend
        course_cfg = course_config(auto, meta["course"])
        run_cfg = dict(bcfg, **({"model": course_cfg["model"]} if course_cfg.get("model") else {}))
        transcript_text = txt.read_text(encoding="utf-8")
        deck = find_slides(cfg, log, txt.stem, meta, transcript_text, video=videos_dir / f"{txt.stem}.mp4")
        slides_text, figures, prompt_transcript = None, None, None
        if deck:
            slides_text = deck.prompt_text(transcript=transcript_text)
            figures = [{k: f[k] for k in ("slide", "deck", "timestamp") if k in f}
                       | {"caption": f["hint"], "latex_path": os.path.relpath(f["path"], latex_dir)}
                       for f in deck.figure_list(transcript_text, auto.get("slides_max_figures", 8))]
            log(f"slides: {len(figures)} figure candidate"
                f"{' (fallback: ' + deck.fallback + ')' if deck.fallback else ''}, "
                f"{len(slides_text)} chars di testo {'con tempi' if deck.timeline else 'pertinente'}")
            if deck.timeline:
                # slide e figure hanno i tempi: la trascrizione va nel prompt con un marcatore al minuto
                prompt_transcript = timed_transcript(txt) or None
        rag_context = None
        if rag and rag.course_exists(meta["course"]):
            try:
                rag_context = rag.query_context(rag_queries(transcript_text, meta["topic"]), meta["course"],
                                                n_results=cfg.get("rag", {}).get("n_results", 5),
                                                exclude_date=meta["date"]) or None
                if rag_context:
                    log(f"rag: {rag_context.count('[Lecture ')} passaggi dalle lezioni precedenti "
                        f"({len(rag_context)} chars)")
            except Exception as e:
                log(f"rag: ✗ query: {type(e).__name__}: {str(e)[:120]}")
        t0 = time.time()
        try:
            generate_notes(
                merged_data=[], output_dir=latex_dir, stem=txt.stem,
                course_name=meta["course"], lecture_date=meta["date"],
                backend=backend, backend_config=run_cfg,
                compile_pdf_flag=ncfg["latex"].get("compile_pdf", True),
                transcript_path=txt, pdf_output_dir=pdf_dir, suffix=meta["topic"] or None,
                figures=figures, slides_text=slides_text, rag_context=rag_context,
                transcript_text=prompt_transcript,
                language=notes_language(ncfg, course_cfg, txt),
            )
            pdf = _pdf_for(meta, pdf_dir)
            if pdf.exists():
                n += 1
                state.clear_fail(key)
                if rag:
                    try:
                        rag_index(rag, log, txt, meta)
                    except Exception as e:
                        log(f"rag: ✗ {txt.stem}: {type(e).__name__}: {str(e)[:120]}")
                from src.notes_gen.notes_gen import LAST_LAYOUT, LAST_USAGE as u
                usage = (f", {u.get('model')} {u.get('in')}→{u.get('out')} tok ${u.get('cost_usd')} eq"
                         if u.get("model") else "")
                log(f"notes: ✓ {pdf.name} ({time.time() - t0:.0f}s{usage})")
                if LAST_LAYOUT.get("overfull"):
                    log(f"notes: ⚠ impaginazione: {LAST_LAYOUT['overfull']} righe/tabelle oltre il margine "
                        f"(max {LAST_LAYOUT['worst_pt']:.0f}pt) in {pdf.name}")
            else:
                k = state.fail(key, "PDF non prodotto (errore LaTeX?)")
                log(f"notes: ✗ {txt.stem}: PDF non prodotto (fallimento {k})")
        except Exception as e:
            k = state.fail(key, f"{type(e).__name__}: {e}")
            log(f"notes: ✗ {txt.stem}: {type(e).__name__}: {str(e)[:200]} (fallimento {k})")
    return n


# ── stadio 4: cleanup ────────────────────────────────────────────────────────

def _rm(path: Path, dry_run: bool) -> int:
    """Rimuove file o cartella; ritorna i byte liberati."""
    if path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if not dry_run:
            shutil.rmtree(path, ignore_errors=True)
    else:
        size = path.stat().st_size
        if not dry_run:
            path.unlink(missing_ok=True)
    return size


def stage_cleanup(cfg: dict, log: Log, videos_dir: Path, tr_dir: Path, dry_run: bool) -> int:
    """
    Dopo gli appunti non serve più niente di pesante. Per ogni lezione che HA il PDF:
      - l'mp4 (135-260 MB) se auto.keep_videos è false — il sidecar .json resta, è lui che dà
        corso/data/argomento agli stadi successivi; fino al PDF il video resta, per poter
        ritrascrivere (es. lingua sbagliata)
    Sempre: file di lavoro di pdflatex (.aux/.log/.out), provini del triage, cache di deck
    spostati/aggiornati/cancellati in output/slides/_decks e conversioni pptx→pdf orfane.
    Ritorna i byte liberati.
    """
    auto = cfg["auto"]
    ncfg = cfg["notes"]
    latex_dir = Path(ncfg["latex"]["output_dir"])
    pdf_dir = Path(ncfg["latex"].get("pdf_output_dir", "output/notes"))
    freed = 0
    tag = "(dry) " if dry_run else ""

    if not auto.get("keep_videos", True):
        for v in sorted(videos_dir.glob("*.mp4")):
            if (tr_dir / f"{v.stem}.txt").exists() and _pdf_for(lecture_meta(v), pdf_dir).exists():
                mb = _rm(v, dry_run) / 1e6
                freed += mb * 1e6
                log(f"cleanup: {tag}rimosso {v.name} ({mb:.0f} MB, PDF presente)")

    for f in latex_dir.glob("*"):
        if f.suffix in (".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk") and f.is_file():
            freed += _rm(f, dry_run)

    if auto.get("slides", False):
        from src.slides.slides import CONVERT_DIR, DECKS_DIR, cache_source, deck_cache_dir, list_decks
        if DECKS_DIR.is_dir():
            for d in sorted(DECKS_DIR.iterdir()):
                if not d.is_dir():
                    continue
                src = cache_source(d)
                if src is None or not src.exists() or deck_cache_dir(src) != d:
                    freed += _rm(d, dry_run)
                    log(f"cleanup: {tag}cache slide obsoleta {d.name}")
                elif (d / "_triage").is_dir():
                    freed += _rm(d / "_triage", dry_run)
        if CONVERT_DIR.is_dir():
            root = Path(os.path.expanduser(auto.get("slides_dir", "~/Documenti/WeBeep Sync")))
            if root.is_dir():
                valid = set()
                for course_dir in root.iterdir():
                    if course_dir.is_dir():
                        for deck in list_decks(course_dir):
                            if deck.suffix.lower() != ".pdf":
                                st = deck.stat()
                                import hashlib
                                key = hashlib.md5(f"{deck.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]
                                valid.add(f"{deck.stem}__{key}.pdf")
                for f in CONVERT_DIR.glob("*.pdf"):
                    if f.name not in valid:
                        freed += _rm(f, dry_run)
                        log(f"cleanup: {tag}conversione orfana {f.name}")

    if freed:
        log(f"cleanup: {tag}liberati {freed / 1e6:.0f} MB")
    return freed


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-transcribe", action="store_true")
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--max-downloads", type=int, default=None)
    ap.add_argument("--max-notes", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    auto = cfg.setdefault("auto", {})
    log = Log(Path(auto.get("log_file", "output/auto/nightly.log")))
    state = State(Path(auto.get("state_file", "output/auto/state.json")))
    videos_dir = Path(cfg["download"]["output_dir"])
    tr_dir = Path(cfg["transcription"]["output_dir"])
    videos_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)

    lock_path = Path(auto.get("state_file", "output/auto/state.json")).with_name("nightly.lock")
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("un'altra istanza è in esecuzione — esco")
        return EXIT_OK

    log(f"══ nightly start (fetch={not a.no_fetch} transcribe={not a.no_transcribe} notes={not a.no_notes}"
        f"{' DRY-RUN' if a.dry_run else ''})")
    t0 = time.time()
    login_needed = False
    errors = 0
    try:
        if not a.no_fetch:
            _, login_needed = stage_fetch(cfg, state, log, videos_dir, tr_dir, a.max_downloads, a.dry_run)
            login_needed = stage_slides_sync(cfg, state, log, a.dry_run) or login_needed
        if not a.no_transcribe:
            stage_transcribe(cfg, state, log, videos_dir, tr_dir, a.dry_run)
        if not a.no_notes:
            stage_notes(cfg, state, log, videos_dir, tr_dir, a.max_notes, a.dry_run)
        if not a.no_cleanup and auto.get("cleanup", True):
            stage_cleanup(cfg, log, videos_dir, tr_dir, a.dry_run)
    except Exception as e:
        errors += 1
        log(f"ERRORE non gestito: {type(e).__name__}: {e}")
        notify("polimi-notes: errore", f"{type(e).__name__}: {str(e)[:120]}", auto.get("notify", True))
    state.save()
    log(f"══ nightly end in {time.time() - t0:.0f}s (login_needed={login_needed}, errors={errors})")
    if errors:
        return EXIT_ERR
    return EXIT_LOGIN if login_needed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
