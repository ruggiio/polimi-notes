"""
ocr.py — Frame extraction + OCR for lecture slides and blackboard content
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import easyocr
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass
class FrameOCRResult:
    frame_index: int
    timestamp_sec: float
    image_path: str
    raw_text: str
    blocks: list
    is_slide: bool


# ── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = 10.0,
    scene_threshold: float = 0.4,
) -> list[tuple[float, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    console.print(f"\n[bold cyan]── Frame Extraction ────────────────────────────[/bold cyan]")
    console.print(f"Duration: {duration_sec/60:.1f} min  |  FPS: {fps:.1f}  |  Interval: {interval_sec}s")

    frame_step = max(1, int(fps * interval_sec))
    frames_out: list[tuple[float, Path]] = []
    prev_hist = None

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        is_new_scene = True
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            is_new_scene = diff > scene_threshold

        if is_new_scene:
            fname = output_dir / f"frame_{frame_idx:07d}_{timestamp:.1f}s.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frames_out.append((timestamp, fname))
            prev_hist = hist

        frame_idx += frame_step

    cap.release()
    console.print(f"[green]✓ Extracted {len(frames_out)} unique frames[/green]")
    return frames_out


# ── Slide vs. Blackboard Heuristic ───────────────────────────────────────────

def _is_slide(frame_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    return bool(mean_brightness > 128)


# ── OCR ───────────────────────────────────────────────────────────────────────

def run_ocr(
    frames: list[tuple[float, Path]],
    languages: list[str] = ("it", "en"),
    gpu: bool = True,
) -> list[FrameOCRResult]:
    console.print(f"\n[bold cyan]── OCR Analysis ────────────────────────────────[/bold cyan]")
    console.print(f"Languages: {languages}  |  GPU: {gpu}  |  Frames: {len(frames)}")
    console.print("[dim]Loading EasyOCR model...[/dim]")

    reader = easyocr.Reader(list(languages), gpu=gpu)
    results: list[FrameOCRResult] = []

    for i, (timestamp, frame_path) in enumerate(track(frames, description="Running OCR...")):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        is_slide_frame = _is_slide(frame_bgr)
        if not is_slide_frame:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_bgr = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)

        ocr_output = reader.readtext(frame_bgr)

        blocks = [
            {"text": str(text), "confidence": float(conf)}
            for (_, text, conf) in ocr_output
            if float(conf) > 0.3 and len(str(text).strip()) > 1
        ]
        raw_text = " ".join(b["text"] for b in blocks)

        results.append(FrameOCRResult(
            frame_index=i,
            timestamp_sec=float(timestamp),
            image_path=str(frame_path),
            raw_text=raw_text,
            blocks=blocks,
            is_slide=bool(is_slide_frame),
        ))

    text_frames = [r for r in results if r.raw_text.strip()]
    console.print(f"[green]✓ OCR complete:[/green] {len(text_frames)}/{len(frames)} frames contain text")

    # Free GPU memory after OCR
    del reader
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            console.print("[dim]GPU memory freed after OCR[/dim]")
    except ImportError:
        pass

    return text_frames


# ── Alignment: OCR ↔ Transcript ───────────────────────────────────────────────

def align_ocr_with_transcript(
    ocr_results: list[FrameOCRResult],
    transcript_segments: list[dict],
) -> list[dict]:
    merged = []
    for ocr in ocr_results:
        t = ocr.timestamp_sec
        nearby_speech = [
            seg["text"]
            for seg in transcript_segments
            if abs(seg["start"] - t) < 30 or (seg["start"] <= t <= seg["end"])
        ]
        merged.append({
            "timestamp_sec": t,
            "speech": " ".join(nearby_speech),
            "ocr_text": ocr.raw_text,
            "is_slide": ocr.is_slide,
            "image_path": ocr.image_path,
        })

    merged.sort(key=lambda x: x["timestamp_sec"])
    return merged


# ── Save ──────────────────────────────────────────────────────────────────────

def _json_convert(obj):
    import numpy as np
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def save_ocr_results(
    results: list[FrameOCRResult],
    output_dir: Path,
    stem: str,
) -> Path:
    out_path = output_dir / f"{stem}_ocr.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            ensure_ascii=False,
            indent=2,
            default=_json_convert,
        )
    console.print(f"  OCR data → [blue]{out_path}[/blue]")
    return out_path