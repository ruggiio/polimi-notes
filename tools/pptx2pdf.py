#!/usr/bin/env python3
"""
pptx2pdf.py — Converte presentazioni (.pptx/.ppt/.odp) in PDF con LibreOffice headless.

  .venv/bin/python tools/pptx2pdf.py ~/Scrivania/POLI -r        # tutti i corsi: <cartella>/pdf/<nome>.pdf
  .venv/bin/python tools/pptx2pdf.py lezione.pptx               # → pdf/lezione.pdf accanto al sorgente
  .venv/bin/python tools/pptx2pdf.py lezione.pptx --subdir .    # → lezione.pdf nella stessa cartella
  .venv/bin/python tools/pptx2pdf.py *.pptx --outdir out/       # destinazione unica per tutti
  .venv/bin/python tools/pptx2pdf.py deck.pptx --force          # riconverti anche se il PDF è aggiornato
  .venv/bin/python tools/pptx2pdf.py deck.pptx --compress       # immagini JPEG q80, max 150 dpi

Destinazione (in ordine di priorità): --outdir se dato, altrimenti <cartella del sorgente>/<--subdir>
(default "pdf").

Salta i PDF già presenti e più recenti del sorgente. Usa un profilo LibreOffice temporaneo,
così funziona anche con LibreOffice aperto in GUI (altrimenti la conversione fallisce in silenzio).

I video incorporati nel pptx finiscono nel PDF come allegati di annotazioni "Screen" (un deck da
1 GB resta da 1 GB): vengono rimossi dopo la conversione (serve pymupdf → usare .venv/bin/python;
--keep-video per tenerli). Il contenuto della slide non cambia.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

EXTS = (".pptx", ".ppt", ".odp", ".pps", ".ppsx")


def collect(paths: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            found += [f for f in it if f.is_file() and f.suffix.lower() in EXTS]
        elif p.is_file() and p.suffix.lower() in EXTS:
            found.append(p)
        else:
            print(f"  ! ignorato (non trovato o estensione non supportata): {raw}", file=sys.stderr)
    # "~$nome.pptx" sono lock file di PowerPoint, non presentazioni
    return sorted({f.resolve() for f in found if not f.name.startswith("~$")})


def needs_convert(src: Path, dst: Path, force: bool) -> bool:
    return force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime


# opzioni del filtro PDF di Impress (JSON, LibreOffice ≥ 7.4): le slide con video/screenshot
# escono altrimenti con immagini lossless a piena risoluzione (centinaia di MB)
COMPRESS = ('pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"false"},'
            '"Quality":{"type":"long","value":"80"},'
            '"ReduceImageResolution":{"type":"boolean","value":"true"},'
            '"MaxImageResolution":{"type":"long","value":"150"}}')


def convert_batch(soffice: str, files: list[Path], outdir: Path, profile: Path, timeout: int,
                  compress: bool = False) -> None:
    """Una sola invocazione di soffice per gruppo di file con la stessa destinazione (avvio lento)."""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [soffice, f"-env:UserInstallation=file://{profile}", "--headless",
           "--convert-to", COMPRESS if compress else "pdf", "--outdir", str(outdir), *map(str, files)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print(f"  ! soffice exit {r.returncode}: {r.stderr.strip()[:300]}", file=sys.stderr)


def strip_videos(pdf: Path) -> int | None:
    """Toglie le annotazioni Screen (video) e i loro allegati; ritorna i MB risparmiati, None se pymupdf manca."""
    try:
        import pymupdf as fitz
    except ImportError:
        return None
    fitz.TOOLS.mupdf_display_errors(False)   # "cannot create appearance stream for Screen annotations"
    before = pdf.stat().st_size
    tmp = pdf.with_suffix(".tmp.pdf")
    with fitz.open(pdf) as doc:
        n = 0
        for page in doc:
            for a in list(page.annots()):
                if a.type[1] == "Screen":
                    page.delete_annot(a)
                    n += 1
        if not n:
            return 0
        # il tag tree (PDF accessibile) punta ancora alle annotazioni tolte e terrebbe vivi i video:
        # per delle slide è superfluo, via anche quello. garbage=4 elimina poi gli stream orfani.
        cat = doc.pdf_catalog()
        doc.xref_set_key(cat, "StructTreeRoot", "null")
        doc.xref_set_key(cat, "MarkInfo", "null")
        doc.save(tmp, garbage=4, deflate=True)
    tmp.replace(pdf)
    return round((before - pdf.stat().st_size) / 1048576)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="file o cartelle")
    ap.add_argument("-o", "--outdir", help="cartella di destinazione unica (sovrascrive --subdir)")
    ap.add_argument("-s", "--subdir", default="pdf",
                    help='sottocartella accanto al sorgente (default "pdf"; "." = stessa cartella)')
    ap.add_argument("-r", "--recursive", action="store_true", help="scendi nelle sottocartelle")
    ap.add_argument("-f", "--force", action="store_true", help="riconverti anche se il PDF esiste ed è aggiornato")
    ap.add_argument("-c", "--compress", action="store_true",
                    help="immagini JPEG q80 e max 150 dpi invece di lossless")
    ap.add_argument("--keep-video", action="store_true", help="non rimuovere i video incorporati dal PDF")
    ap.add_argument("--timeout", type=int, default=600, help="secondi per batch (default 600)")
    a = ap.parse_args()

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("LibreOffice non trovato (soffice/libreoffice non in PATH)", file=sys.stderr)
        return 2

    files = collect(a.paths, a.recursive)
    if not files:
        print("nessuna presentazione da convertire")
        return 1

    outdir_fixed = Path(a.outdir).expanduser().resolve() if a.outdir else None
    jobs: dict[Path, list[Path]] = defaultdict(list)   # destinazione → sorgenti
    skipped = 0
    for src in files:
        dst_dir = outdir_fixed or (src.parent / a.subdir).resolve()
        if needs_convert(src, dst_dir / f"{src.stem}.pdf", a.force):
            jobs[dst_dir].append(src)
        else:
            skipped += 1

    todo = sum(len(v) for v in jobs.values())
    print(f"{len(files)} presentazioni: {todo} da convertire, {skipped} già aggiornate")

    ok, warned = 0, False
    with tempfile.TemporaryDirectory(prefix="lo-profile-") as profile:
        for dst_dir, srcs in jobs.items():
            for s in srcs:
                print(f"  → {s.name}")
            try:
                convert_batch(soffice, srcs, dst_dir, Path(profile), a.timeout, a.compress)
            except subprocess.TimeoutExpired:
                print(f"  ! timeout dopo {a.timeout}s su {dst_dir}", file=sys.stderr)
            for s in srcs:
                pdf = dst_dir / f"{s.stem}.pdf"
                if not (pdf.exists() and pdf.stat().st_mtime >= s.stat().st_mtime):
                    print(f"  ✗ non prodotto: {pdf}", file=sys.stderr)
                    continue
                ok += 1
                if a.keep_video:
                    continue
                saved = strip_videos(pdf)
                if saved is None and not warned:
                    print("  ! pymupdf assente: video non rimossi (usa .venv/bin/python)", file=sys.stderr)
                    warned = True
                elif saved:
                    print(f"    video rimossi da {pdf.name}: −{saved} MB → {pdf.stat().st_size / 1048576:.1f} MB")

    print(f"fatto: {ok}/{todo} convertiti")
    return 0 if ok == todo else 1


if __name__ == "__main__":
    sys.exit(main())
