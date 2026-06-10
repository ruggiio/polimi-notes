"""
course_profiles.py — Per-course "scheda corso" support

A course profile is a free-form markdown file in config/courses/ that captures
a course's terminology, notation conventions and preferred LaTeX style.
It is used in two places:

  1. Notes generation: the whole profile is injected into the prompt as a
     style guide, so notation stays consistent across lectures.
  2. Transcription: the "## Glossario" section (if present) becomes the
     faster-whisper initial_prompt, anchoring recognition of domain terms.

File naming: config/courses/<slug>.md, where <slug> is the course name
lowercased with every non-alphanumeric run replaced by "_"
(e.g. "Model Order Reduction" -> model_order_reduction.md).
Files starting with "_" (like _template.md) are ignored.
"""

import re
from pathlib import Path

PROFILES_DIR = Path("config/courses")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_profile(course_name: str) -> str:
    """Return the profile markdown for a course, or "" if none exists."""
    path = PROFILES_DIR / f"{_slugify(course_name)}.md"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def find_profile_for_file(stem: str) -> str:
    """
    Best-effort profile lookup when only a filename is available
    (e.g. during transcription, before the course name is known).
    Picks the profile whose slug tokens best overlap the filename;
    returns "" unless at least half of the tokens match.
    """
    if not PROFILES_DIR.is_dir():
        return ""
    stem_norm = _slugify(stem)
    best_path, best_score = None, 0.0
    for path in sorted(PROFILES_DIR.glob("*.md")):
        if path.name.startswith("_"):
            continue
        tokens = [t for t in path.stem.split("_") if len(t) >= 4]
        if not tokens:
            continue
        score = sum(1 for t in tokens if t in stem_norm) / len(tokens)
        if score > best_score:
            best_path, best_score = path, score
    if best_path is not None and best_score >= 0.5:
        return best_path.read_text(encoding="utf-8").strip()
    return ""


def extract_glossary(profile: str, max_chars: int = 800) -> str:
    """
    Return the "## Glossario" section of a profile as one comma-separated
    line, suitable for faster-whisper's initial_prompt (~224-token budget).
    """
    m = re.search(
        r"^##\s*Glossario\s*$(.*?)(?=^##\s|\Z)",
        profile,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    lines = [ln.strip().lstrip("-*").strip() for ln in m.group(1).splitlines()]
    terms = ", ".join(ln for ln in lines if ln)
    return terms[:max_chars]
