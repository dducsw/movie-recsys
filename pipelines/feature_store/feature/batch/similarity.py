"""pipeline/feature/batch/similarity.py
-------------------------------------
Item-item similarity for Redis precompute (TF-IDF cosine + iALS dot, fused).

Semantic (plot) embeddings live in Qdrant (384-dim, see backfill_qdrant.py)
and are intentionally NOT mixed here — different spaces must not be averaged.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("BatchSimilarity")


def _top_k_from_scores(
    scores: np.ndarray, m_ids: list[int], self_idx: int, k: int = 20,
    min_score: float = 0.1,
) -> list[dict[str, Any]]:
    order = np.argsort(scores)[::-1]
    out: list[dict[str, Any]] = []
    for idx in order:
        if idx == self_idx or float(scores[idx]) <= min_score:
            continue
        out.append({"movieId": m_ids[idx], "score": round(float(scores[idx]), 4)})
        if len(out) >= k:
            break
    return out


def compute_item_similarity_matrix(
    movies_df: pd.DataFrame,
    genre_vectors: np.ndarray,
    als_factors_map: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    """Compute Redis-ready similar lists + lightweight movie meta."""
    logger.info("Computing item similarity (TF-IDF cosine + iALS dot)...")
    als_factors_map = als_factors_map or {}

    m_ids = [int(r["movieId"]) for _, r in movies_df.iterrows()]
    movie_meta = {
        int(row["movieId"]): {
            "title": str(row["title"]) if pd.notna(row["title"]) else "",
            "genres": str(row["genres"]) if pd.notna(row["genres"]) else "",
        }
        for _, row in movies_df.iterrows()
    }

    tfidf_norm = genre_vectors.astype(np.float32)
    norms = np.linalg.norm(tfidf_norm, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf_norm = tfidf_norm / norms
    tfidf_sim = cosine_similarity(tfidf_norm)

    als_matrix = None
    if als_factors_map:
        als_matrix = np.stack(
            [als_factors_map.get(mid, np.zeros(32, dtype=np.float32)) for mid in m_ids]
        ).astype(np.float32)
        als_sim = als_matrix @ als_matrix.T
    else:
        als_sim = np.zeros_like(tfidf_sim)

    similarities: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, mid in enumerate(m_ids):
        fused = 0.6 * tfidf_sim[i] + 0.4 * als_sim[i]
        similarities[mid] = _top_k_from_scores(fused, m_ids, i, k=20, min_score=0.1)
    return dict(similarities), movie_meta
