"""
transcriber.py — GPU-accelerated Whisper transcription

Uses faster-whisper (CTranslate2 backend) to transcribe lecture audio.
Outputs both a plain .txt and a timestamped .json for alignment with OCR frames.
"""

import json
from pathlib import Path

import torch
from rich.console import Console

console = Console()


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]\u2713 GPU detected:[/green] {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return "cuda"
    console.print("[yellow]\u26a0 No CUDA GPU found \u2014 falling back to CPU (slow)[/yellow]")
    return "cpu"


def recommend_model(device: str) -> str:
    """Suggest the best Whisper model for the available hardware."""
    if device == "cpu":
        return "base"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram >= 10:
        return "large-v3"
    elif vram >= 5:
        return "medium"
    elif vram >= 3:
        return "small"
    return "base"


def transcribe(
    video_path: Path,
    output_dir: Path,
    model_name: str = "large-v3",
    language: str | None = None,
    device: str = "cuda",
    initial_prompt: str | None = None,
) -> dict:
    """
    Transcribe the audio track of a video file using faster-whisper.

    Args:
        video_path:     Path to the .mp4 (or any audio/video) file
        output_dir:     Where to write transcript files
        model_name:     Whisper model size
        language:       Force language (e.g. "it", "en") or None for auto
        device:         "cuda" or "cpu"
        initial_prompt: Domain terms to anchor recognition. When None, the
                        "## Glossario" of the matching course profile in
                        config/courses/ is used (matched on the filename).

    Returns:
        dict with keys: 'text', 'segments', 'language', 'txt_path', 'json_path'
    """
    from faster_whisper import WhisperModel

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    if initial_prompt is None:
        try:
            from src.course_profiles import extract_glossary, find_profile_for_file
            initial_prompt = extract_glossary(find_profile_for_file(stem)) or None
        except Exception:
            initial_prompt = None

    console.print(f"\n[bold cyan]\u2500\u2500 Transcription \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/bold cyan]")
    console.print(f"Model:    [yellow]{model_name}[/yellow]")
    console.print(f"Device:   [yellow]{device}[/yellow]")
    console.print(f"File:     {video_path.name}")
    if initial_prompt:
        console.print(f"Glossary: [dim]{initial_prompt[:80]}{'...' if len(initial_prompt) > 80 else ''}[/dim]")

    # Load model — float16 is fastest on modern GPUs; fall back to int8_float16
    console.print(f"[dim]Loading faster-whisper model '{model_name}'...[/dim]")
    if device == "cuda":
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            actual_compute = "float16"
        except Exception:
            model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
            actual_compute = "int8_float16"
    else:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        actual_compute = "int8"

    console.print(f"[green]\u2713 Model loaded[/green]  device={device}  compute_type={actual_compute}")

    # Transcribe
    console.print("[cyan]Transcribing...[/cyan]")
    segments_gen, info = model.transcribe(
        str(video_path),
        language=language,
        initial_prompt=initial_prompt,
        beam_size=2,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = info.language
    segments_list = list(segments_gen)
    full_text = " ".join(s.text.strip() for s in segments_list)

    # Free GPU memory immediately
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        console.print("[dim]GPU memory freed after transcription[/dim]")

    console.print(f"[green]\u2713 Transcription complete[/green]  "
                  f"Language: {detected_lang}  |  "
                  f"Segments: {len(segments_list)}  |  "
                  f"Words: {len(full_text.split())}")

    # ── Save outputs ───────────────────────────────────────────────────────────────────────────

    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text(full_text, encoding="utf-8")

    json_path = output_dir / f"{stem}_segments.json"
    serialisable_segments = [
        {
            "id": i,
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
        }
        for i, s in enumerate(segments_list)
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

    console.print(f"  Transcript \u2192 [blue]{txt_path}[/blue]")
    console.print(f"  Segments   \u2192 [blue]{json_path}[/blue]")

    return {
        "text": full_text,
        "segments": serialisable_segments,
        "language": detected_lang,
        "txt_path": txt_path,
        "json_path": json_path,
    }
