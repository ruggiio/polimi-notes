"""
transcriber.py — GPU-accelerated Whisper transcription

Uses OpenAI Whisper (local) to transcribe lecture audio.
Outputs both a plain .txt and a timestamped .json for alignment with OCR frames.
"""

import json
from pathlib import Path

import whisper
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]✓ GPU detected:[/green] {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    console.print("[yellow]⚠ No CUDA GPU found — falling back to CPU (slow)[/yellow]")
    return "cpu"


def recommend_model(device: str) -> str:
    """Suggest the best Whisper model for the available hardware."""
    if device == "cpu":
        return "base"  # Fast enough on CPU
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram >= 10:
        return "large-v2"
    elif vram >= 5:
        return "medium"
    elif vram >= 3:
        return "small"
    return "base"


def transcribe(
    video_path: Path,
    output_dir: Path,
    model_name: str = "large-v2",
    language: str | None = None,  # None = auto-detect
    device: str = "cuda",
) -> dict:
    """
    Transcribe the audio track of a video file using Whisper.

    Args:
        video_path:  Path to the .mp4 (or any audio/video) file
        output_dir:  Where to write transcript files
        model_name:  Whisper model size
        language:    Force language (e.g. "it", "en") or None for auto
        device:      "cuda" or "cpu"

    Returns:
        dict with keys: 'text' (full string), 'segments' (timestamped list),
                        'language' (detected), 'txt_path', 'json_path'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    console.print(f"\n[bold cyan]── Transcription ──────────────────────────────[/bold cyan]")
    console.print(f"Model:    [yellow]{model_name}[/yellow]")
    console.print(f"Device:   [yellow]{device}[/yellow]")
    console.print(f"File:     {video_path.name}")

    # Load model (downloads on first use, ~3 GB for large-v2)
    console.print(f"[dim]Loading Whisper model '{model_name}'...[/dim]")
    model = whisper.load_model(model_name, device=device)

    # Transcribe
    console.print("[cyan]Transcribing... (this takes a few minutes for a 1h lecture)[/cyan]")
    result = model.transcribe(
        str(video_path),
        language=language,
        verbose=False,          # Don't spam stdout
        task="transcribe",      # "transcribe" keeps original language; "translate" → English
        fp16=(device == "cuda"),  # Use fp16 on GPU for speed
        word_timestamps=True,   # Finer granularity for alignment
    )

    # Free GPU memory immediately after transcription
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        console.print("[dim]GPU memory freed after transcription[/dim]")

    detected_lang = result.get("language", "unknown")
    full_text = result["text"].strip()
    segments = result["segments"]

    console.print(f"[green]✓ Transcription complete[/green]  "
                  f"Language: {detected_lang}  |  "
                  f"Segments: {len(segments)}  |  "
                  f"Words: {len(full_text.split())}")

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Plain text
    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text(full_text, encoding="utf-8")

    # Timestamped JSON (useful for syncing with OCR frames)
    json_path = output_dir / f"{stem}_segments.json"
    serialisable_segments = [
        {
            "id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
        }
        for s in segments
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": detected_lang,
                "source": str(video_path),
                "segments": serialisable_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    console.print(f"  Transcript → [blue]{txt_path}[/blue]")
    console.print(f"  Segments   → [blue]{json_path}[/blue]")

    return {
        "text": full_text,
        "segments": serialisable_segments,
        "language": detected_lang,
        "txt_path": txt_path,
        "json_path": json_path,
    }
