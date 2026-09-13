"""
pipeline/feature/batch/similarity.py
-------------------------------------
Vector consolidation and item-item similarity matrix calculation.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("BatchSimilarity")


def compute_item_similarity_matrix(
    movies_df,
    genre_vectors: np.ndarray,
    als_factors_map: Dict[int, np.ndarray] = None
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    """
    Consolidate TF-IDF genre vectors & ALS factors, and compute item similarity matrix via chunked dot-products.
    """
    logger.info("Extracting movie vectors & calculating similarity matrix...")
    if als_factors_map is None:
        als_factors_map = {}

    m_ids = []
    vectors = []
    movie_meta = {}

    for idx, row in movies_df.iterrows():
        m_id = int(row["movieId"])
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        genres = str(row["genres"]) if pd.notna(row["genres"]) else ""

        genre_vec = genre_vectors[idx]

        # Combine with ALS factor vector if available
        if m_id in als_factors_map:
            combined = 0.5 * genre_vec + 0.5 * als_factors_map[m_id]
            norm = np.linalg.norm(combined)
            final_vec = (combined / norm).tolist() if norm > 0 else genre_vec.tolist()
        else:
            final_vec = genre_vec.tolist()

        # Ensure exact dimension size 20
        if len(final_vec) < 20:
            final_vec.extend([0.0] * (20 - len(final_vec)))
        else:
            final_vec = final_vec[:20]

        m_ids.append(m_id)
        vectors.append(final_vec)
        movie_meta[m_id] = {
            "title": title,
            "genres": genres,
            "vector": final_vec
        }

    matrix = np.array(vectors, dtype=np.float32)  # Shape (N, 20)
    N = len(m_ids)
    similarities = defaultdict(list)

    # Chunked dot product to prevent NxN memory allocation OOM on large datasets
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
