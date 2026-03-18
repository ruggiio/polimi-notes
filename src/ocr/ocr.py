"""
ocr.py — Frame extraction + OCR optimized for handwritten tablet notes

Key differences from slides version:
  - Higher contrast preprocessing for handwritten content
  - Lower scene threshold (handwriting changes more gradually)
  - Adaptive thresholding for better ink detection
  - Larger frame interval (handwriting stays on screen longer)
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
    interval_sec: float = 15.0,           # step size for scene-change sampling
    scene_threshold: float = 0.25,        # histogram distance threshold for scene change
    max_force_interval_sec: float = 45.0, # always capture at least every N seconds
) -> list[tuple[float, Path]]:
    """
    Extract frames by scene change detection with a guaranteed periodic fallback.

    Two complementary mechanisms:
    - Scene change: saves a frame whenever the histogram difference from the last
      saved frame exceeds `scene_threshold`. Catches content transitions.
    - Forced capture: saves a frame unconditionally every `max_force_interval_sec`
      seconds. Ensures no lecture segment longer than N seconds is skipped, which
      is critical for slowly-evolving handwritten content that never triggers a
      scene change threshold.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    console.print(f"\n[bold cyan]── Frame Extraction ────────────────────────────[/bold cyan]")
    console.print(
        f"Duration: {duration_sec/60:.1f} min  |  FPS: {fps:.1f}  |  "
        f"Interval: {interval_sec}s  |  Max gap: {max_force_interval_sec}s"
    )
    console.print(f"[dim]Mode: handwriting-optimized (scene change + forced periodic)[/dim]")

    frame_step = int(fps * interval_sec)
    frames_out: list[tuple[float, Path]] = []
    prev_hist = None
    last_saved_timestamp: float = -max_force_interval_sec  # trigger save on first frame

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        scene_changed = True
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            scene_changed = diff > scene_threshold

        force_save = (timestamp - last_saved_timestamp) >= max_force_interval_sec

        if scene_changed or force_save:
            fname = output_dir / f"frame_{frame_idx:07d}_{timestamp:.1f}s.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames_out.append((timestamp, fname))
            prev_hist = hist
            last_saved_timestamp = timestamp

        frame_idx += frame_step

    cap.release()
    min_expected = int(duration_sec / max_force_interval_sec)
    console.print(
        f"[green]✓ Extracted {len(frames_out)} unique frames[/green] "
        f"[dim](≥{min_expected} from forced periodic capture)[/dim]"
    )
    return frames_out


# ── Handwriting Preprocessing ─────────────────────────────────────────────────

def _preprocess_for_handwriting(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline optimized for handwritten tablet notes:
    1. Convert to grayscale
    2. Adaptive thresholding to handle uneven lighting/pen pressure
    3. Morphological operations to connect broken strokes
    4. Return as BGR for EasyOCR
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Check if this is a dark background (digital tablet often uses dark mode)
    mean_brightness = gray.mean()
    if mean_brightness < 100:
        # Dark background (e.g. dark mode tablet) — invert
        gray = cv2.bitwise_not(gray)

    # Adaptive thresholding — handles varying ink density and lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )

    # Slight dilation to connect broken pen strokes
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Convert back to BGR for EasyOCR
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _is_handwritten(frame_bgr: np.ndarray) -> bool:
    """
    Heuristic: handwritten tablet notes tend to have:
    - Medium brightness (not pure white like slides, not black)
    - High edge density (lots of ink strokes)
    Returns True if this looks like handwritten content.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    edges = cv2.Canny(gray, 30, 100)
    edge_density = edges.mean()
    # Handwriting: medium brightness OR dark background with high edge density
    return edge_density > 5 or mean_brightness < 50


# ── OCR ───────────────────────────────────────────────────────────────────────

def run_ocr(
    frames: list[tuple[float, Path]],
    languages: list[str] = ("it", "en"),
    gpu: bool = True,
) -> list[FrameOCRResult]:
    console.print(f"\n[bold cyan]── OCR Analysis ────────────────────────────────[/bold cyan]")
    console.print(f"Languages: {languages}  |  GPU: {gpu}  |  Frames: {len(frames)}")
    console.print(f"[dim]Mode: handwriting-optimized preprocessing[/dim]")
    console.print("[dim]Loading EasyOCR model...[/dim]")

    reader = easyocr.Reader(list(languages), gpu=gpu)
    results: list[FrameOCRResult] = []

    for i, (timestamp, frame_path) in enumerate(track(frames, description="Running OCR...")):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        is_handwritten_frame = _is_handwritten(frame_bgr)

        if is_handwritten_frame:
            # Apply handwriting-specific preprocessing
            processed = _preprocess_for_handwriting(frame_bgr)
        else:
            # Slide-like content — use standard preprocessing
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean()
            if mean_brightness < 128:
                frame_bgr = cv2.cvtColor(
                    cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR
                )
            processed = frame_bgr

        ocr_output = reader.readtext(processed)

        blocks = [
            {"text": text, "confidence": round(conf, 3)}
            for (_, text, conf) in ocr_output
            if conf > 0.25 and len(text.strip()) > 1  # lower confidence threshold for handwriting
        ]
        raw_text = " ".join(b["text"] for b in blocks)

        results.append(FrameOCRResult(
            frame_index=i,
            timestamp_sec=timestamp,
            image_path=str(frame_path),
            raw_text=raw_text,
            blocks=blocks,
            is_slide=not is_handwritten_frame,
        ))

    text_frames = [r for r in results if r.raw_text.strip()]
    console.print(f"[green]✓ OCR complete:[/green] {len(text_frames)}/{len(frames)} frames contain text")

    # Free GPU memory
    del reader
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            console.print("[dim]GPU memory freed after OCR[/dim]")
    except Exception:
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

def save_ocr_results(
    results: list[FrameOCRResult],
    output_dir: Path,
    stem: str,
) -> Path:
    out_path = output_dir / f"{stem}_ocr.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    console.print(f"  OCR data → [blue]{out_path}[/blue]")
    return out_path