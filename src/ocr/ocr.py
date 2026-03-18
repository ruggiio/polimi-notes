"""
ocr.py — Frame extraction + OCR for lecture slides and blackboard content
"""

import base64
import json
import os
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
    duration_sec = total_frames / fps

    console.print(f"\n[bold cyan]── Frame Extraction ────────────────────────────[/bold cyan]")
    console.print(f"Duration: {duration_sec/60:.1f} min  |  FPS: {fps:.1f}  |  Interval: {interval_sec}s")

    frame_step = int(fps * interval_sec)
    frames_out: list[tuple[float, Path]] = []
    prev_hist = None

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


# ── Frame Type Classification ────────────────────────────────────────────────

def _detect_frame_type(img: np.ndarray) -> str:
    """
    Classify a frame as 'slide', 'blackboard', or 'tablet'.

    - slide:      bright background (mean > 180) and low variance (std < 60)
    - blackboard: dark background (mean < 80)
    - tablet:     everything else (handwritten tablet notes)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    std_brightness = float(gray.std())

    if mean_brightness > 180 and std_brightness < 60:
        return "slide"
    elif mean_brightness < 80:
        return "blackboard"
    else:
        return "tablet"


# ── Slide vs. Blackboard Heuristic (legacy, kept for compatibility) ──────────

def _is_slide(frame_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    return bool(mean_brightness > 128)


# ── Claude Vision OCR ────────────────────────────────────────────────────────

def _ocr_with_vision(img: np.ndarray, client, model: str) -> list[dict]:
    """
    Use Claude Vision API to OCR handwritten or blackboard content.
    Returns list of {"text": str, "confidence": 1.0} blocks.
    """
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Transcribe ALL text visible in this image exactly as written, "
                        "preserving spatial layout. Include mathematical formulas, labels "
                        "on diagrams, and any handwritten annotations. Return only the "
                        "transcribed text, nothing else."
                    ),
                },
            ],
        }],
    )

    text = response.content[0].text.strip()
    if not text:
        return []
    return [{"text": text, "confidence": 1.0}]


# ── OCR ───────────────────────────────────────────────────────────────────────

def run_ocr(
    frames: list[tuple[float, Path]],
    languages: list[str] = ("it", "en"),
    gpu: bool = True,
    vision_for_handwriting: bool = False,
) -> list[FrameOCRResult]:
    console.print(f"\n[bold cyan]── OCR Analysis ────────────────────────────────[/bold cyan]")
    console.print(f"Languages: {languages}  |  GPU: {gpu}  |  Frames: {len(frames)}")
    console.print("[dim]Loading EasyOCR model...[/dim]")

    reader = easyocr.Reader(list(languages), gpu=gpu)
    results: list[FrameOCRResult] = []

    # Set up Claude Vision client if vision_for_handwriting is enabled
    vision_client = None
    vision_model = None
    if vision_for_handwriting:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                vision_client = anthropic.Anthropic(api_key=api_key)
                vision_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
                console.print("[cyan]Vision OCR enabled for handwriting/blackboard frames[/cyan]")
            except ImportError:
                console.print("[yellow]⚠ anthropic package not installed — falling back to EasyOCR for all frames[/yellow]")
        else:
            console.print("[dim]Vision OCR: no ANTHROPIC_API_KEY set — using EasyOCR for all frames[/dim]")

    for i, (timestamp, frame_path) in enumerate(track(frames, description="Running OCR...")):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        frame_type = _detect_frame_type(frame_bgr)
        is_slide_frame = frame_type == "slide"

        # Route based on frame type
        if frame_type == "slide":
            # Existing EasyOCR pipeline for slides
            ocr_output = reader.readtext(frame_bgr)
            blocks = [
                {"text": str(text), "confidence": float(conf)}
                for (_, text, conf) in ocr_output
                if float(conf) > 0.3 and len(str(text).strip()) > 1
            ]
        elif frame_type in ("tablet", "blackboard") and vision_client is not None:
            # Claude Vision for handwriting/blackboard
            ocr_frame = frame_bgr
            if frame_type == "blackboard":
                # Apply CLAHE contrast enhancement for blackboard frames
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                ocr_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            try:
                blocks = _ocr_with_vision(ocr_frame, vision_client, vision_model)
            except Exception as e:
                console.print(f"[yellow]⚠ Vision OCR failed for frame {i}, falling back to EasyOCR: {e}[/yellow]")
                # Fall back to EasyOCR
                if frame_type == "blackboard":
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    frame_bgr = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
                ocr_output = reader.readtext(frame_bgr)
                blocks = [
                    {"text": str(text), "confidence": float(conf)}
                    for (_, text, conf) in ocr_output
                    if float(conf) > 0.3 and len(str(text).strip()) > 1
                ]
        else:
            # EasyOCR fallback for tablet/blackboard when vision is not available
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
