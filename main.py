#!/usr/bin/env python3
"""
main.py — PoliMi Lecture Notes Pipeline CLI

Usage examples:
  python main.py run "https://..." --course "Analisi 2" --date "2024-03-15"
  python main.py run URL --no-ocr
  python main.py run URL --no-download --video my.mp4
  python main.py run URL --no-download --no-transcribe --video my.mp4
  python main.py transcribe-only my_lecture.mp4
  python main.py notes-only transcript.txt --ocr ocr.json
"""

import json
import os
import shutil
from pathlib import Path
from datetime import date

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

app = typer.Typer(
    name="polimi-notes",
    help="Download, transcribe, and convert PoliMi lectures to LaTeX notes.",
    rich_markup_mode="rich",
)
console = Console()

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        console.print(f"[red]Config not found:[/red] {path}")
        raise typer.Exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def get_credentials(cfg: dict) -> tuple[str, str]:
    username = os.environ.get("POLIMI_USER") or cfg["auth"].get("username", "")
    password = os.environ.get("POLIMI_PASS") or cfg["auth"].get("password", "")
    if not username or not password:
        username = username or typer.prompt("PoliMi username (Person Code)")
        password = password or typer.prompt("PoliMi password", hide_input=True)
    return username, password


def _cleanup_before_run(cfg: dict, stem: str):
    """
    Clean up files from the previous run before starting a new one.
    Keeps output/notes/ (final PDFs) and output/latex/figures/ untouched here
    (figures are cleaned by figure_extractor at the start of figure extraction).
    """
    console.print("[dim]Cleaning up previous run files...[/dim]")

    # Remove OCR frames directory
    frames_dir = Path(cfg["ocr"]["output_dir"]) / stem
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
        console.print(f"[dim]  Removed {frames_dir}[/dim]")

    # Remove OCR json
    ocr_json = Path(cfg["ocr"]["output_dir"]) / f"{stem}_ocr.json"
    if ocr_json.exists():
        ocr_json.unlink()
        console.print(f"[dim]  Removed {ocr_json}[/dim]")

    # Remove old transcript
    transcript_txt = Path(cfg["transcription"]["output_dir"]) / f"{stem}.txt"
    transcript_json = Path(cfg["transcription"]["output_dir"]) / f"{stem}_segments.json"
    if transcript_txt.exists():
        transcript_txt.unlink()
        console.print(f"[dim]  Removed {transcript_txt}[/dim]")
    if transcript_json.exists():
        transcript_json.unlink()
        console.print(f"[dim]  Removed {transcript_json}[/dim]")

    # Remove old .tex (not figures — those are cleaned by figure_extractor)
    tex_path = Path(cfg["notes"]["latex"]["output_dir"]) / "lecture_notes.tex"
    if tex_path.exists():
        tex_path.unlink()
        console.print(f"[dim]  Removed {tex_path}[/dim]")

    # Remove saved date (will be recreated on next download)
    date_file = Path(cfg["download"]["output_dir"]) / "lecture_date.txt"
    if date_file.exists():
        date_file.unlink()
        console.print(f"[dim]  Removed {date_file}[/dim]")


def _cleanup_video(video_path: Path):
    """Delete the video file after successful pipeline completion."""
    if video_path and video_path.exists():
        size_mb = video_path.stat().st_size / 1_000_000
        video_path.unlink()
        console.print(f"[dim]  Removed video ({size_mb:.0f} MB): {video_path}[/dim]")


@app.command()
def run(
    webex_url: str = typer.Argument(..., help="Webex recording playback URL"),
    course: str = typer.Option("Unknown Course", "--course", "-c"),
    lecture_date: str = typer.Option(str(date.today()), "--date", "-d"),
    config_path: Path = typer.Option(CONFIG_PATH, "--config"),
    no_download: bool = typer.Option(False, "--no-download"),
    video: Path = typer.Option(None, "--video"),
    no_transcribe: bool = typer.Option(False, "--no-transcribe"),
    no_ocr: bool = typer.Option(False, "--no-ocr"),
    no_notes: bool = typer.Option(False, "--no-notes"),
    headless: bool = typer.Option(True, "--headless/--headed"),
    backend: str = typer.Option(None, "--backend"),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Skip automatic cleanup of previous run files"),
    suffix: str = typer.Option(None, "--suffix", "-s", help="Optional suffix for the PDF filename (e.g. 'Part 1', 'Stability Analysis')"),
):
    """
    [bold green]Full pipeline:[/bold green] Download → Transcribe → OCR → LaTeX notes → PDF.
    """
    cfg = load_config(config_path)

    console.print(Panel.fit(
        f"[bold]PoliMi Lecture Notes Pipeline[/bold]\n"
        f"Course: [cyan]{course}[/cyan]  |  Date: [cyan]{lecture_date}[/cyan]"
        + (f"  |  Suffix: [cyan]{suffix}[/cyan]" if suffix else ""),
        border_style="blue"
    ))

    # ── Step 0: Cleanup previous run ─────────────────────────────────────────
    if not no_cleanup and not no_download:
        # Only clean up if we're doing a fresh full run
        _cleanup_before_run(cfg, "lecture")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if no_download:
        if not video or not video.exists():
            console.print("[red]--no-download requires --video <path>[/red]")
            raise typer.Exit(1)
        video_path = video
        console.print(f"[dim]Using existing video: {video_path}[/dim]")

        # Try to restore the date saved during the original download
        date_file = Path(cfg["download"]["output_dir"]) / "lecture_date.txt"
        if date_file.exists() and lecture_date == str(date.today()):
            lecture_date = date_file.read_text().strip()
            console.print(f"[green]✓ Using saved date:[/green] {lecture_date}")
    else:
        from src.downloader.downloader import download_lecture
        username, password = get_credentials(cfg)
        console.print(Rule("[bold]Step 1/4: Download[/bold]"))
        video_path, extracted_date = download_lecture(
            webex_url=webex_url,
            username=username,
            password=password,
            output_dir=Path(cfg["download"]["output_dir"]),
            cookies_file=Path(cfg["download"]["cookies_file"]),
            headless=headless,
        )
        if extracted_date and lecture_date == str(date.today()):
            lecture_date = extracted_date
            console.print(f"[green]✓ Using extracted date:[/green] {lecture_date}")

    stem = video_path.stem

    # ── Step 2: Transcribe ────────────────────────────────────────────────────
    if no_transcribe:
        transcript_txt = Path(cfg["transcription"]["output_dir"]) / f"{stem}.txt"
        transcript_json = Path(cfg["transcription"]["output_dir"]) / f"{stem}_segments.json"

        if not transcript_txt.exists():
            console.print(f"[red]Transcript not found:[/red] {transcript_txt}")
            console.print("[yellow]Run without --no-transcribe to generate it first.[/yellow]")
            raise typer.Exit(1)

        console.print(Rule("[bold]Step 2/4: Transcription[/bold]"))
        console.print(f"[dim]Using existing transcript: {transcript_txt}[/dim]")

        text = transcript_txt.read_text(encoding="utf-8")
        segments = []
        if transcript_json.exists():
            with open(transcript_json) as f:
                data = json.load(f)
                segments = data.get("segments", [])
                lang = data.get("language", "unknown")
            console.print(f"[dim]Loaded {len(segments)} segments, language: {lang}[/dim]")
        else:
            segments = [{"id": 0, "start": 0, "end": 9999, "text": text}]

        transcript_result = {
            "text": text,
            "segments": segments,
            "language": "unknown",
            "txt_path": transcript_txt,
            "json_path": transcript_json,
        }
    else:
        from src.transcriber.transcriber import transcribe, get_device
        console.print(Rule("[bold]Step 2/4: Transcription[/bold]"))
        tcfg = cfg["transcription"]
        device = get_device() if tcfg["device"] == "cuda" else "cpu"
        transcript_result = transcribe(
            video_path=video_path,
            output_dir=Path(tcfg["output_dir"]),
            model_name=tcfg["model"],
            language=tcfg.get("language"),
            device=device,
        )

    # ── Step 3: OCR ───────────────────────────────────────────────────────────
    merged_data = []
    frames = []

    if not no_ocr:
        from src.ocr.ocr import extract_frames, run_ocr, align_ocr_with_transcript, save_ocr_results
        console.print(Rule("[bold]Step 3/4: OCR Analysis[/bold]"))

        ocrcfg = cfg["ocr"]
        frames = extract_frames(
            video_path=video_path,
            output_dir=Path(ocrcfg["output_dir"]) / stem,
            interval_sec=ocrcfg["frame_interval_sec"],
            scene_threshold=ocrcfg["scene_change_threshold"],
        )
        ocr_results = run_ocr(
            frames=frames,
            languages=ocrcfg["languages"],
            gpu=ocrcfg["gpu"],
        )
        save_ocr_results(ocr_results, Path(ocrcfg["output_dir"]), stem)
        merged_data = align_ocr_with_transcript(ocr_results, transcript_result["segments"])

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                console.print("[dim]GPU memory freed after OCR[/dim]")
        except Exception:
            pass
    else:
        console.print("[dim]OCR skipped (--no-ocr)[/dim]")
        merged_data = [
            {"timestamp_sec": s["start"], "speech": s["text"], "ocr_text": "", "is_slide": False}
            for s in transcript_result["segments"]
        ]
        ocr_json_path = Path(cfg["ocr"]["output_dir"]) / f"{stem}_ocr.json"
        if ocr_json_path.exists():
            try:
                with open(ocr_json_path) as f:
                    ocr_raw = json.load(f)
                frames = [
                    (r["timestamp_sec"], Path(r["image_path"]))
                    for r in ocr_raw
                    if Path(r["image_path"]).exists()
                ]
                if frames:
                    console.print(f"[dim]Loaded {len(frames)} frames from saved OCR json[/dim]")
            except Exception:
                pass

    # ── Step 3.5: Figure Extraction (optional) ──────────────────────────────
    figures = []
    figures_enabled = cfg.get("figures", {}).get("enabled", False)
    if figures_enabled and not no_notes and frames:
        from src.ocr.figure_extractor import extract_figures
        console.print(Rule("[bold]Step 3.5/4: Figure Extraction[/bold]"))
        figures_dir = Path(cfg["notes"]["latex"]["output_dir"]) / "figures"
        figures = extract_figures(
            frames=frames,
            merged_data=merged_data,
            figures_output_dir=figures_dir,
            api_key=cfg.get("figures", {}).get("api_key", ""),
            model=cfg.get("figures", {}).get("model", "claude-sonnet-4-20250514"),
            max_candidates=cfg.get("figures", {}).get("max_candidates", 30),
        )
        console.print(f"[dim]Figures selected: {len(figures)}[/dim]")
    elif figures_enabled and not frames:
        console.print("[yellow]⚠ Figure extraction enabled but no frames available[/yellow]")

    # ── Step 4: Notes Generation ──────────────────────────────────────────────
    if not no_notes:
        from src.notes_gen.notes_gen import generate_notes
        console.print(Rule("[bold]Step 4/4: LaTeX Notes Generation[/bold]"))

        ncfg = cfg["notes"]
        active_backend = backend or ncfg["backend"]
        backend_config = ncfg.get(active_backend, {})
        backend_config["math_only_ocr"] = cfg.get("ocr", {}).get("math_only_filter", True)
        pdf_output_dir = Path(ncfg["latex"].get("pdf_output_dir", "output/notes"))

        tex_path = generate_notes(
            merged_data=merged_data,
            output_dir=Path(ncfg["latex"]["output_dir"]),
            stem=stem,
            course_name=course,
            lecture_date=lecture_date,
            backend=active_backend,
            backend_config=backend_config,
            compile_pdf_flag=ncfg["latex"]["compile_pdf"],
            transcript_path=transcript_result["txt_path"],
            pdf_output_dir=pdf_output_dir,
            figures=figures if figures else None,
            suffix=suffix,
        )

        # ── Step 5: Cleanup video (heavy file, no longer needed) ──────────────
        if not no_download and not no_cleanup:
            console.print(Rule("[bold]Step 5/4: Cleanup[/bold]"))
            _cleanup_video(video_path)
            console.print("[dim]OCR frames kept for potential rerun[/dim]")

        console.print(Panel.fit(
            f"[bold green]✓ Pipeline complete![/bold green]\n\n"
            f"  Transcript: {transcript_result['txt_path']}\n"
            f"  Notes .tex: {tex_path}\n"
            f"  Notes .pdf: {pdf_output_dir}/",
            border_style="green"
        ))
    else:
        console.print("[dim]Notes generation skipped (--no-notes)[/dim]")


@app.command()
def transcribe_only(
    video: Path = typer.Argument(..., help="Path to video file"),
    model: str = typer.Option("medium", "--model", "-m"),
    language: str = typer.Option(None, "--language", "-l"),
    config_path: Path = typer.Option(CONFIG_PATH, "--config"),
):
    """Transcribe a single video file (no download or notes)."""
    cfg = load_config(config_path)
    from src.transcriber.transcriber import transcribe, get_device
    device = get_device()
    transcribe(
        video_path=video,
        output_dir=Path(cfg["transcription"]["output_dir"]),
        model_name=model,
        language=language,
        device=device,
    )


@app.command()
def notes_only(
    transcript: Path = typer.Argument(..., help="Path to .txt transcript file"),
    ocr_json: Path = typer.Option(None, "--ocr", help="Path to OCR JSON file"),
    course: str = typer.Option("Unknown Course", "--course", "-c"),
    lecture_date: str = typer.Option(str(date.today()), "--date", "-d"),
    backend: str = typer.Option("claude", "--backend", "-b"),
    config_path: Path = typer.Option(CONFIG_PATH, "--config"),
):
    """Generate LaTeX notes from an existing transcript (and optional OCR)."""
    from src.notes_gen.notes_gen import generate_notes
    from src.ocr.ocr import FrameOCRResult, align_ocr_with_transcript

    cfg = load_config(config_path)
    text = transcript.read_text(encoding="utf-8")
    segments = [{"id": 0, "start": 0, "end": 9999, "text": text}]

    if ocr_json and ocr_json.exists():
        with open(ocr_json) as f:
            ocr_raw = json.load(f)
        ocr_results = [FrameOCRResult(**r) for r in ocr_raw]
        merged_data = align_ocr_with_transcript(ocr_results, segments)
    else:
        merged_data = [{"timestamp_sec": 0, "speech": text, "ocr_text": "", "is_slide": False}]

    ncfg = cfg["notes"]
    active_backend = backend or ncfg["backend"]
    pdf_output_dir = Path(ncfg["latex"].get("pdf_output_dir", "output/notes"))

    generate_notes(
        merged_data=merged_data,
        output_dir=Path(ncfg["latex"]["output_dir"]),
        stem=transcript.stem,
        course_name=course,
        lecture_date=lecture_date,
        backend=active_backend,
        backend_config=ncfg.get(active_backend, {}),
        compile_pdf_flag=ncfg["latex"]["compile_pdf"],
        transcript_path=transcript,
        pdf_output_dir=pdf_output_dir,
    )


if __name__ == "__main__":
    app()