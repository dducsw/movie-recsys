"""
recsys_core.embeddings
----------------------
Pinned semantic encoder for Qdrant dense vectors.

Primary: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, cosine).
Fallback: deterministic TF-IDF hash — keeps tests/offline working
without downloading weights.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_encoder = None


def build_movie_texts(movies_df: pd.DataFrame) -> list[str]:
    """Build semantic text per movie: title + genres + overview."""
    texts: list[str] = []
    for _, row in movies_df.iterrows():
        title = str(row.get("title", "") or "")
        genres = str(row.get("genres", "") or "").replace("|", " ")
        overview = str(row.get("overview", "") or "")
        texts.append(f"{title}. Genres: {genres}. {overview}".strip())
    return texts


def _fallback_hash_encode(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic hashing fallback (L2-normalized)."""
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vectors[i, h % dim] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm
    return vectors


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts to L2-normalized (N, 384) float32 matrix."""
    global _encoder
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    try:
        if _encoder is None:
            from sentence_transformers import SentenceTransformer

            _encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Loaded semantic encoder %s.", EMBEDDING_MODEL_NAME)
        vectors = np.asarray(
            _encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        return vectors
    except ImportError:
        logger.warning("sentence-transformers missing, using hash fallback.")
    except Exception as exc:  # noqa: BLE001 - encoder must never crash serving
        logger.warning("Semantic encoder failed (%s), using hash fallback.", exc)
    return _fallback_hash_encode(texts)
