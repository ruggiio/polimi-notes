"""
eg2.py — EmbeddingGemma 2 (google/embeddinggemma-2): testo e immagini nello stesso spazio a 768 d.

Usato per: riconoscere le slide nei frame del video (video_match), stimare quali pagine sono
state discusse dalla trascrizione (LectureSlides), embedding del RAG (src/rag).
Banco del 07/10/2026 (VB3DM + MOR): trascrizione→pagina 45% pagina esatta contro 31% del
lessicale; frame→pagina 100% d'accordo col vecchio matcher a pixel sui frame che riconosceva,
più slide titolo, build Beamer, PDF viewer a tutto schermo; lavagna mai presa per slide.

Un solo modello per processo (encoder audio mai caricato: 440M con la visione, 270M senza),
bf16 su GPU (fp16 dà NaN senza errore), float32 su CPU. release() libera la VRAM: va chiamato
prima di Whisper, che sulla A2000 da 4 GB non ci sta insieme.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-2"
DIM = 768
_model = None
_vision = False


def available() -> bool:
    try:
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def model(vision: bool = True):
    """Il SentenceTransformer condiviso; con vision=False basta il solo testo (ma se il modello
    con la visione è già caricato si riusa quello)."""
    global _model, _vision
    if _model is not None and (_vision or not vision):
        return _model
    release()
    import torch
    from sentence_transformers import SentenceTransformer
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    cfg = {"audio_config": None} if vision else {"audio_config": None, "vision_config": None}
    _model = SentenceTransformer(MODEL_ID, device="cuda" if cuda else "cpu",
                                 model_kwargs={"torch_dtype": dtype}, config_kwargs=cfg)
    _vision = vision
    return _model


def release() -> None:
    global _model, _vision
    if _model is None:
        return
    _model, _vision = None, False
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def encode_texts(texts: list[str], prompt_name: str | None = None, batch_size: int = 16) -> np.ndarray:
    """(n, 768) normalizzati. prompt_name: "SearchQuery", "Document", "SentenceSimilarity"…"""
    if not texts:
        return np.empty((0, DIM), np.float32)
    m = model(vision=False)
    return m.encode(texts, prompt_name=prompt_name, batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def encode_images(images: list, batch_size: int = 8) -> np.ndarray:
    """(n, 768) normalizzati da immagini PIL RGB."""
    if not images:
        return np.empty((0, DIM), np.float32)
    m = model(vision=True)
    return m.encode([{"image": im} for im in images], batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def page_embeddings(deck: Path) -> np.ndarray:
    """(pagine, 768) delle pagine del deck renderizzate; cache in <deck_cache_dir>/eg2_pages.npy.
    Vuoto per i notebook e per i deck che non si convertono in PDF."""
    from PIL import Image
    from src.slides.slides import as_pdf, deck_cache_dir, is_notebook
    if is_notebook(deck):
        return np.empty((0, DIM), np.float32)
    cache = deck_cache_dir(deck) / "eg2_pages.npy"
    if cache.exists():
        try:
            return np.load(cache)
        except Exception:
            pass
    import pymupdf as fitz
    fitz.TOOLS.mupdf_display_errors(False)
    pdf = as_pdf(deck)
    if not pdf:
        return np.empty((0, DIM), np.float32)
    out = []
    with fitz.open(pdf) as doc:
        batch = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            batch.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            if len(batch) == 16:
                out.append(encode_images(batch)); batch = []
        if batch:
            out.append(encode_images(batch))
    arr = np.concatenate(out) if out else np.empty((0, DIM), np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def transcript_windows(transcript: str, words: int = 150) -> list[str]:
    """Finestre consecutive di ~words parole (~1 minuto di parlato)."""
    w = transcript.split()
    return [" ".join(w[i:i + words]) for i in range(0, len(w), words) if len(w[i:i + words]) >= 30]
