"""
transcriber.py — GPU-accelerated Whisper transcription

Uses faster-whisper (CTranslate2 backend) to transcribe lecture audio.
Outputs both a plain .txt and a timestamped .json for alignment with OCR frames.
"""

import ctypes
import glob
import json
import os
import site
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

# faster-whisper runs on CTranslate2, which dlopen()s cuBLAS/cuDNN at runtime.
# The pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 ship them, but not on
# the loader path: preload them so no LD_LIBRARY_PATH fiddling is needed.
_CUDA_LIB_GLOBS = (
    "nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*",
    "nvidia/cudnn/lib/libcudnn_graph.so.*", "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.*",
    "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.*", "nvidia/cudnn/lib/libcudnn_heuristic.so.*",
    "nvidia/cudnn/lib/libcudnn_ops.so.*", "nvidia/cudnn/lib/libcudnn_cnn.so.*",
    "nvidia/cudnn/lib/libcudnn_adv.so.*", "nvidia/cudnn/lib/libcudnn.so.*",
)
_preloaded = False


def _preload_cuda_libs() -> None:
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for pat in _CUDA_LIB_GLOBS:
            for lib in sorted(glob.glob(os.path.join(sp, pat))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _gpu_info() -> tuple[str, float] | None:
    """(name, VRAM GB) of GPU 0 via nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        name, mib = out.rsplit(",", 1)
        return name.strip(), float(mib) / 1024
    except Exception:
        return None


def get_device() -> str:
    """Return 'cuda' if CTranslate2 sees a CUDA GPU, else 'cpu'."""
    _preload_cuda_libs()
    import ctranslate2
    if ctranslate2.get_cuda_device_count() > 0:
        info = _gpu_info()
        if info:
            console.print(f"[green]\u2713 GPU detected:[/green] {info[0]} ({info[1]:.1f} GB VRAM)")
        else:
            console.print("[green]\u2713 CUDA GPU detected[/green]")
        return "cuda"
    console.print("[yellow]\u26a0 No CUDA GPU found \u2014 falling back to CPU (slow)[/yellow]")
    return "cpu"


def recommend_model(device: str) -> str:
    """Suggest the best Whisper model for the available hardware."""
    if device == "cpu":
        return "base"
    info = _gpu_info()
    vram = info[1] if info else 4.0
    if vram >= 10:
        return "large-v3"
    elif vram >= 5:
        return "medium"
    elif vram >= 3:
        return "small"
    return "base"


def _audio_duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip() or 0)
    except Exception:
        return 0.0


def choose_language(model, audio_path: Path, candidates=("it", "en"), n_windows: int = 3,
                    window_s: int = 30) -> tuple[str, dict[str, float]]:
    """
    Sceglie la lingua tra i candidati decodificando qualche finestra con ciascuna.

    Il rilevatore di Whisper si fa ingannare dall'accento: un docente italiano che fa
    lezione in inglese viene rilevato "it" (p≈0.8) su ogni finestra, e la decodifica in
    italiano produce una pseudo-traduzione ("un approccio di l'ansano" = hands-on).
    La lingua giusta si riconosce invece dalla confidenza della decodifica: media
    dell'avg_logprob dei segmenti, pesata sulla durata, su n finestre sparse nella lezione.
    Ritorna (lingua, {lingua: punteggio}).
    """
    import tempfile
    candidates = [c for c in dict.fromkeys(candidates) if c]
    if len(candidates) < 2:
        return (candidates[0] if candidates else None), {}
    dur = _audio_duration(audio_path)
    if dur <= 0:
        return None, {}
    starts = [dur * (i + 1) / (n_windows + 1) for i in range(n_windows)]
    scores: dict[str, list[tuple[float, float]]] = {c: [] for c in candidates}
    for t0 in starts:
        wav = Path(tempfile.mktemp(suffix=".wav"))
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(t0 - window_s / 2, 0):.1f}",
                            "-t", str(window_s), "-i", str(audio_path), "-ac", "1", "-ar", "16000",
                            str(wav)], check=True, capture_output=True, timeout=120)
            for lang in candidates:
                segs, _ = model.transcribe(str(wav), language=lang, beam_size=1, vad_filter=True,
                                           condition_on_previous_text=False)
                for s in segs:
                    scores[lang].append((s.avg_logprob, max(s.end - s.start, 0.1)))
        except Exception:
            continue
        finally:
            wav.unlink(missing_ok=True)
    mean = {c: (sum(lp * w for lp, w in v) / sum(w for _, w in v)) if v else float("-inf")
            for c, v in scores.items()}
    best = max(mean, key=mean.get)
    return (best if mean[best] > float("-inf") else None), {c: round(x, 3) for c, x in mean.items()}


def transcribe(
    video_path: Path,
    output_dir: Path,
    model_name: str = "large-v3",
    language: str | None = None,
    device: str = "cuda",
    initial_prompt: str | None = None,
    language_candidates=("it", "en"),
) -> dict:
    """
    Transcribe the audio track of a video file using faster-whisper.

    Args:
        video_path:     Path to the .mp4 (or any audio/video) file
        output_dir:     Where to write transcript files
        model_name:     Whisper model size
        language:       Force language (e.g. "it", "en") or None for auto
        language_candidates: with language=None, the languages compared by
                        decoding confidence (see choose_language); () = trust
                        Whisper's own detector
        device:         "cuda" or "cpu"
        initial_prompt: Domain terms to anchor recognition. When None, the
                        "## Glossario" of the matching course profile in
                        config/courses/ is used (matched on the filename).

    Returns:
        dict with keys: 'text', 'segments', 'language', 'txt_path', 'json_path'
    """
    _preload_cuda_libs()
    from faster_whisper import WhisperModel

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    if initial_prompt is None:
        try:
            from src.course_profiles import (
                extract_glossary, find_profile_for_file, load_profile,
            )
            # POLIMI_COURSE (set e.g. by the Colab notebook) beats the
            # filename heuristic, since video names rarely contain the
            # full course name.
            course = os.environ.get("POLIMI_COURSE", "")
            profile = load_profile(course) if course else find_profile_for_file(stem)
            initial_prompt = extract_glossary(profile) or None
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

    lang_scores: dict[str, float] = {}
    if not language and language_candidates:
        language, lang_scores = choose_language(model, video_path, language_candidates)
        if language:
            console.print(f"Language: [yellow]{language}[/yellow] (decoding confidence: "
                          + ", ".join(f"{c} {v}" for c, v in lang_scores.items()) + ")")

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

    # Free GPU memory immediately (CTranslate2 releases it when the model is dropped)
    del model

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
                "language_scores": lang_scores,
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
