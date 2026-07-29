"""
pipeline/feature/batch/fallback.py
-----------------------------------
Fallback NumPy/Pandas Batch Pipeline for environments without PySpark runtime.
"""

import os
import math
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from pipeline.feature.config import MOVIES_CSV

logger = logging.getLogger("BatchFallback")


def run_pandas_fallback_batch_pipeline() -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    """Fallback NumPy/Pandas Batch Pipeline if PySpark runtime is unavailable."""
    logger.info("--- Running Fallback Pandas/NumPy Batch Pipeline ---")

    if not os.path.exists(MOVIES_CSV):
        logger.error(f"Movies dataset not found at {MOVIES_CSV}")
        return {}, {}

    movies_df = pd.read_csv(MOVIES_CSV)
    if "genres" not in movies_df.columns and len(movies_df.columns) >= 3:
        movies_df.columns = ["movieId", "title", "genres"] + list(movies_df.columns[3:])
    elif "movieid" in movies_df.columns:
        movies_df.rename(columns={"movieid": "movieId"}, inplace=True)

    genre_set = set()
    for raw_genres in movies_df["genres"].dropna():
        for g in str(raw_genres).split("|"):
            if g and g != "(no genres listed)":
                genre_set.add(g)
    genre_list = sorted(list(genre_set))[:20]

    m_ids = []
    vectors = []
    movie_meta = {}

    for _, row in movies_df.iterrows():
        m_id = int(row["movieId"])
        genres_str = str(row["genres"]) if pd.notna(row["genres"]) else ""
        title = str(row["title"]) if pd.notna(row["title"]) else ""

        vec = [0.0] * 20
        if genres_str and genres_str != "(no genres listed)":
            m_genres = set(genres_str.split("|"))
            for idx, g in enumerate(genre_list):
                if g in m_genres:
                    vec[idx] = 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]

        m_ids.append(m_id)
        vectors.append(vec)
        movie_meta[m_id] = {
            "title": title,
            "genres": genres_str,
            "vector": vec
        }

    matrix = np.array(vectors, dtype=np.float32)
    N = len(m_ids)
    similarities = defaultdict(list)

    chunk_size = 500
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = matrix[start_idx:end_idx]
        sim_chunk = np.dot(chunk, matrix.T)

        for i_local in range(end_idx - start_idx):
            i_global = start_idx + i_local
            id_a = m_ids[i_global]
            scores = sim_chunk[i_local]
            top_indices = np.argpartition(scores, -min(25, N))[-min(25, N):]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for idx in top_indices:
                if idx == i_global:
                    continue
                score = float(scores[idx])
                if score > 0.1:
                    similarities[id_a].append({"movieId": m_ids[idx], "score": round(score, 4)})
                if len(similarities[id_a]) >= 20:
                    break

    return similarities, movie_meta
