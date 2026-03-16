"""
figure_extractor.py — Claude Vision-based figure selection and captioning

Pipeline:
  1. Take candidate frames from OCR extraction
  2. Pre-process frames: crop black borders, remove webcam overlay
  3. Filter for visually complex frames (diagrams, graphs, microstructures)
  4. Send to Claude Vision API to decide if important + generate caption
  5. Copy selected frames to output/latex/figures/
  6. Return list of (timestamp, filename, caption) for use in notes_gen
"""

import base64
import json
import os
import shutil
from pathlib import Path

import anthropic
import cv2
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


# ── Frame Pre-processing ──────────────────────────────────────────────────────

def _crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Automatically crop black borders around the slide content.
    Works for any slide layout — finds the largest non-black rectangle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find bounding box of non-black content
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img  # No content found, return original

    x, y, w, h = cv2.boundingRect(coords)

    # Add small padding (2px) to avoid cutting content edges
    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    cropped = img[y:y+h, x:x+w]
    return cropped


def _remove_webcam_overlay(img: np.ndarray, webcam_fraction: float = 0.22) -> np.ndarray:
    """
    Remove the professor's webcam overlay, which in Webex recordings
    is always positioned in the top-right corner.

    Fills the webcam area with the average background color of the slide
    to avoid leaving a black patch that looks odd.

    Args:
        img:              Input frame (BGR)
        webcam_fraction:  Fraction of width/height to mask (default 22%)
    """
    h, w = img.shape[:2]
    result = img.copy()

    # Webcam is in top-right corner
    webcam_w = int(w * webcam_fraction)
    webcam_h = int(h * webcam_fraction)

    # Sample background color from the slide (use top-left area as reference)
    sample_region = img[0:50, 0:50]
    bg_color = np.median(sample_region.reshape(-1, 3), axis=0).astype(np.uint8)

    # Fill webcam area with background color
    result[0:webcam_h, w - webcam_w:w] = bg_color

    return result


def _preprocess_frame(frame_path: Path) -> np.ndarray | None:
    """
    Full pre-processing pipeline for a single frame:
    1. Load image
    2. Remove webcam overlay (top-right)
    3. Crop black borders
    Returns processed numpy array or None on error.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    # Step 1: Remove webcam overlay
    img = _remove_webcam_overlay(img)

    # Step 2: Crop black borders
    img = _crop_black_borders(img)

    return img


def _save_processed_frame(img: np.ndarray, dest_path: Path) -> bool:
    """Save processed frame as high-quality JPEG."""
    try:
        cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception as e:
        console.print(f"[yellow]⚠ Could not save {dest_path}: {e}[/yellow]")
        return False


# ── Visual Complexity Filter ──────────────────────────────────────────────────

def _visual_complexity(img: np.ndarray) -> float:
    """
    Estimate visual complexity using edge density.
    High edge density = diagrams, graphs, microstructures.
    Returns 0.0 (simple) to 1.0 (complex).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean()) / 255.0


def _prefilter_frames(
    frames: list[tuple[float, Path]],
    complexity_threshold: float = 0.05,
    max_candidates: int = 30,
) -> list[tuple[float, Path, np.ndarray]]:
    """
    Pre-process and pre-filter frames by visual complexity.
    Returns list of (timestamp, original_path, processed_image).
    """
    scored = []
    for timestamp, frame_path in frames:
        processed = _preprocess_frame(frame_path)
        if processed is None:
            continue
        score = _visual_complexity(processed)
        scored.append((score, timestamp, frame_path, processed))

    # Sort by complexity descending, take top N
    scored.sort(reverse=True)
    candidates = [(t, p, img) for (_, t, p, img) in scored[:max_candidates]]

    console.print(
        f"[dim]Visual pre-filter: {len(frames)} frames → "
        f"{len(candidates)} candidates[/dim]"
    )
    return candidates


# ── Claude Vision Analysis ────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """You are an expert at analyzing lecture slide images and deciding which ones 
contain visual content that is important for student lecture notes.

You will be shown a frame from a lecture recording (already cropped to show only the slide content).
Decide if it contains a figure, diagram, graph, chart, microstructure image, table, or any other 
visual content that would be valuable in lecture notes and CANNOT be adequately described in text alone.

Respond with EXACTLY this JSON format (no other text):
{
  "include": true or false,
  "caption": "A concise, informative caption for the figure (max 15 words). Empty string if include=false.",
  "reason": "Brief reason for your decision (1 sentence)"
}

Include if: diagram, graph, chart, microstructure image, schematic, table with data, photo of real application
Exclude if: pure text slide, blank/transition frame, text already captured by transcript"""


def _encode_array_to_base64(img: np.ndarray) -> str:
    """Encode numpy image array to base64 JPEG string."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buffer).decode("utf-8")


def _analyze_frame_with_vision(
    img: np.ndarray,
    timestamp: float,
    nearby_transcript: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
) -> dict | None:
    """Send a processed frame to Claude Vision API for analysis."""
    try:
        img_b64 = _encode_array_to_base64(img)
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)

        message = client.messages.create(
            model=model,
            max_tokens=200,
            system=VISION_SYSTEM_PROMPT,
            messages=[
                {
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
                                f"Frame at {mins:02d}:{secs:02d} in the lecture.\n"
                                f"Context (what was being said): "
                                f"{nearby_transcript[:200]}"
                            ),
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        return json.loads(response_text)

    except Exception as e:
        console.print(f"[yellow]⚠ Vision API error at {timestamp:.0f}s: {e}[/yellow]")
        return None


# ── Main Figure Extraction ────────────────────────────────────────────────────

def extract_figures(
    frames: list[tuple[float, Path]],
    merged_data: list[dict],
    figures_output_dir: Path,
    api_key: str = "",
    model: str = "claude-sonnet-4-20250514",
    complexity_threshold: float = 0.05,
    max_candidates: int = 30,
) -> list[dict]:
    """
    Select important figures from lecture frames using Claude Vision.
    Frames are pre-processed (webcam removed, black borders cropped) before
    analysis and before saving.

    Returns list of dicts: {timestamp, filename, caption, latex_path}
    """
    figures_output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up figures from previous run
    old_figures = (
        list(figures_output_dir.glob("*.jpg")) +
        list(figures_output_dir.glob("*.png"))
    )
    if old_figures:
        for f in old_figures:
            f.unlink()
        console.print(f"[dim]Cleaned {len(old_figures)} figures from previous run[/dim]")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        console.print("[yellow]⚠ No API key for Vision — skipping figure extraction[/yellow]")
        return []

    console.print(f"\n[bold cyan]── Figure Extraction ───────────────────────────[/bold cyan]")
    console.print(f"Input frames: {len(frames)}  |  Max candidates: {max_candidates}")

    # Step 1: Pre-process + pre-filter by visual complexity
    candidates = _prefilter_frames(frames, complexity_threshold, max_candidates)

    # Build transcript lookup for context
    def get_nearby_transcript(timestamp: float) -> str:
        for entry in merged_data:
            if abs(entry.get("timestamp_sec", 0) - timestamp) < 30:
                return entry.get("speech", "")
        return ""

    # Step 2: Analyze with Claude Vision
    client = anthropic.Anthropic(api_key=key)
    selected_figures = []
    fig_count = 0

    console.print(f"[cyan]Analyzing {len(candidates)} candidates with Claude Vision...[/cyan]")

    for timestamp, frame_path, processed_img in track(candidates, description="Analyzing..."):
        nearby_text = get_nearby_transcript(timestamp)
        result = _analyze_frame_with_vision(
            processed_img, timestamp, nearby_text, client, model
        )

        if result and result.get("include"):
            fig_count += 1
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            fig_filename = f"fig_{fig_count:02d}_{mins:02d}m{secs:02d}s.jpg"
            dest_path = figures_output_dir / fig_filename

            # Save the PROCESSED image (cropped, webcam removed)
            if _save_processed_frame(processed_img, dest_path):
                selected_figures.append({
                    "timestamp": timestamp,
                    "filename": fig_filename,
                    "caption": result.get("caption", ""),
                    "reason": result.get("reason", ""),
                    "latex_path": f"figures/{fig_filename}",
                })
                console.print(
                    f"  [green]✓[/green] [{mins:02d}:{secs:02d}] "
                    f"{result.get('caption', '')[:60]}"
                )
        else:
            reason = result.get("reason", "excluded") if result else "API error"
            console.print(
                f"  [dim]✗ [{int(timestamp//60):02d}:{int(timestamp%60):02d}] "
                f"{reason}[/dim]"
            )

    console.print(
        f"[green]✓ Figure extraction complete:[/green] "
        f"{len(selected_figures)}/{len(candidates)} figures selected"
    )

    # Save manifest
    manifest_path = figures_output_dir / "figures_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(selected_figures, f, indent=2)

    return selected_figures