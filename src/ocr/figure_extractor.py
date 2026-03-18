"""
figure_extractor.py — Claude Vision figure selection for handwritten lecture notes

Adapted for tablet handwriting:
- Looks for handwritten diagrams, sketches, graphs, tables
- Ignores pure text pages (OCR handles those)
- Handles dark mode tablets
"""

import base64
import json
import os
import shutil
from pathlib import Path

import re as _re

import anthropic
import cv2
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


# ── Transcript Visual-Keyword Scoring ────────────────────────────────────────

# Italian keywords/phrases that indicate the professor is ACTIVELY SHOWING or
# DRAWING visual content — NOT general topic nouns (which fire during the entire
# lecture for engineering courses).
_VISUAL_KEYWORDS_IT = _re.compile(
    r"|".join([
        # Direct visual-object references (high confidence)
        r"\bnello schema\b", r"\bnella figura\b", r"\bnel grafico\b",
        r"\bnel diagramma\b", r"\bnella tabella\b",
        r"\blo schema\b", r"\bla figura\b", r"\bil grafico\b",
        r"\bil diagramma\b", r"\bla tabella\b",
        # Drawing / showing ACTIONS (high confidence — professor is drawing NOW)
        r"\bdisegn\w+", r"\btracci\w+",
        # Explicit pointing at visual content
        r"\bcome (?:potete )?ved\w+", r"\bcome si vede\b",
        r"\bvedete qui\b", r"\bguardate\b",
        r"\ba sinistra\b", r"\ba destra\b", r"\bin alto\b", r"\bin basso\b",
        # Multi-word phrases that strongly indicate a drawing is being discussed
        r"\bschema a blocchi\b", r"\bdiagramma di flusso\b",
        r"\bschema impiantistic\w+",
    ]),
    _re.IGNORECASE,
)


def _compute_transcript_visual_scores(merged_data: list[dict]) -> dict[float, float]:
    """
    Score each transcript entry for visual-context keywords.

    Returns ``{timestamp_sec: score}`` where *score* ∈ [0.0, 1.0].
    A high score means the professor is likely discussing visual content.
    """
    scores: dict[float, float] = {}
    for entry in merged_data:
        text = entry.get("speech", "")
        ts = entry.get("timestamp_sec", 0.0)
        if not text:
            scores[ts] = 0.0
            continue
        matches = _VISUAL_KEYWORDS_IT.findall(text)
        # Normalize: ≥3 keywords in one segment → max score
        scores[ts] = min(len(matches) / 3.0, 1.0)
    return scores


def _get_visual_score_at(
    timestamp: float,
    visual_scores: dict[float, float],
    window_sec: float = 90.0,
) -> float:
    """Return the **max** visual score within ±*window_sec* of *timestamp*."""
    best = 0.0
    for ts, score in visual_scores.items():
        if abs(ts - timestamp) <= window_sec:
            if score > best:
                best = score
    return best


# ── Frame Pre-processing ──────────────────────────────────────────────────────

def _crop_black_borders(img: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Crop dark borders around the tablet content area."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    return img[y:y+h, x:x+w]


def _remove_webcam_overlay(img: np.ndarray, webcam_fraction: float = 0.22) -> np.ndarray:
    """Remove webcam overlay from top-right corner."""
    h, w = img.shape[:2]
    result = img.copy()
    webcam_w = int(w * webcam_fraction)
    webcam_h = int(h * webcam_fraction)
    sample_region = img[0:50, 0:200]
    bg_color = np.median(sample_region.reshape(-1, 3), axis=0).astype(np.uint8)
    result[0:webcam_h, w - webcam_w:w] = bg_color
    return result


def _preprocess_frame(frame_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    img = _remove_webcam_overlay(img)
    img = _crop_black_borders(img)
    return img


def _save_processed_frame(img: np.ndarray, dest_path: Path) -> bool:
    try:
        cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    except Exception as e:
        console.print(f"[yellow]⚠ Could not save {dest_path}: {e}[/yellow]")
        return False


# ── Visual Complexity Filter ──────────────────────────────────────────────────

def _visual_complexity(img: np.ndarray) -> float:
    """
    Complexity score combining std deviation and edge density.
    - std_dev captures overall content richness
    - edge_density specifically detects diagrams, drawings, and structured content
      (diagrams produce long structured edges; uniform text areas score lower)
    Edge density is weighted higher so simple block diagrams on white backgrounds
    are not filtered out in favour of dense-text frames.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = float(gray.std())
    edges = cv2.Canny(gray, 30, 100)
    edge_density = float(edges.mean()) / 255.0  # Normalize to [0, 1]
    # Edges weighted 60% — better at detecting diagrams over pure text
    score = (std_dev / 80.0) * 0.4 + edge_density * 0.6
    return min(score, 1.0)


def _prefilter_frames(
    frames: list[tuple[float, Path]],
    complexity_threshold: float = 0.05,
    max_candidates: int = 150,
) -> list[tuple[float, Path, np.ndarray]]:
    """
    Select candidate frames ensuring both quality and temporal coverage.

    1. Score all frames by visual complexity (std_dev + edge density).
    2. Filter out frames below `complexity_threshold`.
    3. If more than `max_candidates` remain, use temporal bucketing to pick
       the most complex frame from each equal-sized time window.

    Note: near-duplicate collapsing is handled AFTER Vision selection by
    `_deduplicate_figures`, not here — applying a gap filter here would chain
    all frames into one cluster since they are typically only 15-45s apart.
    """
    scored = []
    for timestamp, frame_path in frames:
        processed = _preprocess_frame(frame_path)
        if processed is None:
            continue
        score = _visual_complexity(processed)
        if score >= complexity_threshold:
            scored.append((score, timestamp, frame_path, processed))

    if not scored:
        console.print(
            f"[dim]Visual pre-filter: {len(frames)} frames → 0 candidates "
            f"(all below threshold {complexity_threshold:.2f})[/dim]"
        )
        return []

    # If few enough, skip bucketing and use all
    if len(scored) <= max_candidates:
        candidates = [(t, p, img) for (_, t, p, img) in sorted(scored, key=lambda x: x[1])]
    else:
        # Temporal bucketing: divide lecture into max_candidates equal time windows
        # and pick the most complex frame from each non-empty window.
        max_time = max(t for _, t, _, _ in scored)
        bucket_size = (max_time + 1.0) / max_candidates
        buckets: dict[int, list] = {}
        for score, timestamp, frame_path, processed in scored:
            idx = min(int(timestamp / bucket_size), max_candidates - 1)
            buckets.setdefault(idx, []).append((score, timestamp, frame_path, processed))

        candidates = []
        for bucket in buckets.values():
            best = max(bucket, key=lambda x: x[0])
            candidates.append((best[1], best[2], best[3]))
        candidates.sort(key=lambda x: x[0])

    console.print(
        f"[dim]Visual pre-filter: {len(frames)} frames → "
        f"{len(candidates)} candidates[/dim]"
    )
    return candidates


def _deduplicate_figures(figures: list[dict], min_gap_sec: float = 90.0) -> list[dict]:
    """
    Collapse near-duplicate consecutive figures after Vision selection.

    Groups selected figures that are within `min_gap_sec` of each other into
    clusters (same drawing session) and keeps only the last entry (most complete
    state of the drawing). Files for removed duplicates are deleted from disk.
    """
    if len(figures) <= 1:
        return figures

    clusters: list[list] = [[figures[0]]]
    for fig in figures[1:]:
        if fig["timestamp"] - clusters[-1][-1]["timestamp"] < min_gap_sec:
            clusters[-1].append(fig)
        else:
            clusters.append([fig])

    result = [cluster[-1] for cluster in clusters]
    removed = len(figures) - len(result)
    if removed:
        console.print(
            f"[dim]Post-selection dedup: {len(figures)} → {len(result)} figures "
            f"({removed} near-duplicates removed)[/dim]"
        )
    return result


# ── Claude Vision Analysis ────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """Sei un esperto nell'analizzare frame video di lezioni universitarie con appunti scritti a mano su tablet.

Ti verrà mostrato un frame di una registrazione di lezione. Il professore scrive in italiano su un tablet.

OBIETTIVO: selezionare SOLO i frame che contengono DISEGNI, DIAGRAMMI, SCHEMI o GRAFICI disegnati a mano — cioè contenuto con struttura spaziale che non può essere rappresentato come puro testo.

PRIMA DI TUTTO, ESCLUDI IMMEDIATAMENTE se il frame mostra una di queste cose:
- Interfaccia di Microsoft Teams, Zoom, o altra piattaforma di videoconferenza (barre laterali, lista canali, chat, partecipanti)
- Pagine web, browser, siti di e-learning, elenchi di partecipanti o studenti
- Solo webcam del professore o avatar/iniziali su sfondo scuro (es. cerchio con lettere "DL", "MR" ecc.)
- Sfondo nero/vuoto o schermata di caricamento
- Solo testo stampato o digitale (non scritto a mano)

Se il frame ha superato il filtro sopra, allora valuta il CONTENUTO SCRITTO A MANO:

INCLUDI il frame SE il contenuto scritto a mano contiene ALMENO UNO di questi elementi visivi:
- Schizzi o disegni a mano con forme geometriche (rettangoli, cerchi, linee che rappresentano componenti)
- Schemi a blocchi collegati da frecce che mostrano flussi o relazioni
- Grafici con assi (anche abbozzati) — curve, andamenti, diagrammi
- Schemi impiantistici, elettrici, idraulici, meccanici, termici, P&ID
- Tabelle strutturate con righe E colonne
- Layout spaziali che mostrano disposizione fisica di componenti
- Qualsiasi combinazione di forme geometriche + frecce + etichette

ESCLUDI se il contenuto scritto a mano è SOLO:
- Equazioni/formule matematiche (senza diagrammi o schemi accanto)
- Testo scritto, elenchi puntati, titoli, definizioni
- Note senza elementi grafici/spaziali

NOTA: molti frame hanno testo + disegno insieme sulla stessa pagina. Se c'è anche un piccolo schizzo o schema nel frame, INCLUDI. In caso di dubbio tra testo+disegno, INCLUDI.

Rispondi con ESATTAMENTE questo formato JSON (nessun altro testo):
{
  "include": true o false,
  "caption": "Una didascalia concisa e informativa in italiano (max 12 parole). Stringa vuota se include=false.",
  "reason": "Breve motivazione della tua decisione (1 frase in italiano)"
}
"""

def _encode_array_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buffer).decode("utf-8")


def _analyze_frame_with_vision(
    img: np.ndarray,
    timestamp: float,
    nearby_transcript: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-20250514",
    visual_score: float = 0.0,
) -> dict | None:
    try:
        img_b64 = _encode_array_to_base64(img)
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)

        # Build the user message with optional transcript-boost hint
        context_parts = [
            f"Frame at {mins:02d}:{secs:02d} in the lecture.",
            f"Context (what was being said): {nearby_transcript[:800]}",
        ]
        if visual_score >= 0.5:
            context_parts.append(
                "\n⚠️ CONTESTO IMPORTANTE: l'analisi della trascrizione indica "
                "che il professore sta descrivendo contenuto visivo/spaziale "
                "(diagrammi, schemi, grafici) in questo momento. Valuta con "
                "particolare attenzione se il frame contiene gli elementi visivi "
                "di cui sta parlando — in caso di dubbio, INCLUDI."
            )
        elif visual_score >= 0.2:
            context_parts.append(
                "\n📝 Nota: la trascrizione suggerisce possibile discussione "
                "di contenuto visivo in questo periodo della lezione."
            )

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
                            "text": "\n".join(context_parts),
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
    complexity_threshold: float = 0.1,
    max_candidates: int = 30,
) -> list[dict]:
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
    console.print(f"[dim]Mode: handwritten diagram detection (two-signal: Vision + transcript)[/dim]")

    candidates = _prefilter_frames(frames, complexity_threshold, max_candidates)

    # ── Transcript visual-keyword scoring ─────────────────────────────────
    visual_scores = _compute_transcript_visual_scores(merged_data)
    n_high = sum(1 for s in visual_scores.values() if s >= 0.5)
    console.print(
        f"[dim]Transcript visual scores: {n_high} high-confidence windows "
        f"(≥0.5) out of {len(visual_scores)} segments[/dim]"
    )

    def get_nearby_transcript(timestamp: float) -> str:
        """Aggregate all speech within ±45s of *timestamp*."""
        parts = []
        for entry in merged_data:
            if abs(entry.get("timestamp_sec", 0) - timestamp) < 45:
                speech = entry.get("speech", "").strip()
                if speech:
                    parts.append(speech)
        return " ".join(parts)

    client = anthropic.Anthropic(api_key=key)
    selected_figures = []
    fig_count = 0

    console.print(f"[cyan]Analyzing {len(candidates)} candidates with Claude Vision...[/cyan]")

    for timestamp, frame_path, processed_img in track(candidates, description="Analyzing..."):
        nearby_text = get_nearby_transcript(timestamp)
        vscore = _get_visual_score_at(timestamp, visual_scores)

        result = _analyze_frame_with_vision(
            processed_img, timestamp, nearby_text, client, model,
            visual_score=vscore,
        )

        # ── Decision: Vision only (transcript is a soft hint inside the prompt) ──
        # The transcript score influences Claude Vision's judgement via the prompt
        # hint, but does NOT override Vision's decision.  This avoids force-including
        # Teams UI / empty frames that happen while the professor talks about
        # engineering topics in general.
        include = bool(result and result.get("include"))

        mins = int(timestamp // 60)
        secs = int(timestamp % 60)

        if include:
            fig_count += 1
            fig_filename = f"fig_{fig_count:02d}_{mins:02d}m{secs:02d}s.jpg"
            dest_path = figures_output_dir / fig_filename

            if _save_processed_frame(processed_img, dest_path):
                caption = result.get("caption", "") if result else ""
                reason = result.get("reason", "") if result else ""
                selected_figures.append({
                    "timestamp": timestamp,
                    "filename": fig_filename,
                    "caption": caption,
                    "reason": reason,
                    "latex_path": f"figures/{fig_filename}",
                })
                score_tag = f" [dim](vs={vscore:.1f})[/dim]" if vscore > 0 else ""
                console.print(
                    f"  [green]✓[/green] [{mins:02d}:{secs:02d}]{score_tag} "
                    f"{caption[:60]}"
                )
        else:
            reason = result.get("reason", "excluded") if result else "API error"
            score_tag = f" [dim](vs={vscore:.1f})[/dim]" if vscore > 0 else ""
            console.print(
                f"  [dim]✗ [{mins:02d}:{secs:02d}]{score_tag} "
                f"{reason}[/dim]"
            )

    console.print(
        f"[green]✓ Figure extraction complete:[/green] "
        f"{len(selected_figures)}/{len(candidates)} figures selected"
    )

    # Deduplicate: collapse clusters of near-identical consecutive figures,
    # keeping only the last (most complete drawing state) from each cluster.
    selected_figures = _deduplicate_figures(selected_figures, min_gap_sec=90.0)

    # Renumber filenames after deduplication so they stay sequential
    renamed = []
    for new_idx, fig in enumerate(selected_figures, start=1):
        old_path = figures_output_dir / fig["filename"]
        mins = int(fig["timestamp"] // 60)
        secs = int(fig["timestamp"] % 60)
        new_filename = f"fig_{new_idx:02d}_{mins:02d}m{secs:02d}s.jpg"
        new_path = figures_output_dir / new_filename
        if old_path.exists() and old_path != new_path:
            old_path.rename(new_path)
        renamed.append({**fig, "filename": new_filename, "latex_path": f"figures/{new_filename}"})
    selected_figures = renamed

    manifest_path = figures_output_dir / "figures_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(selected_figures, f, indent=2)

    return selected_figures