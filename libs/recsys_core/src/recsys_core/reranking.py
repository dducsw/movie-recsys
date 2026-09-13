"""
recsys_core.reranking
---------------------
Diversity and reranking algorithms including Maximal Marginal Relevance (MMR) and dynamic lambda.
"""

from typing import List, Dict, Any, Optional, Set
import math
from collections import Counter
import numpy as np


def calculate_user_lambda(
    user_liked_genres: List[str],
    all_genres: List[str],
    base_min: float = 0.4,
    base_max: float = 0.9
) -> float:
    """
    Compute dynamic MMR exploration parameter (lambda) based on Shannon entropy of liked genres.
    Higher entropy (broad taste) -> lower lambda (more diversification).
    Lower entropy (focused taste) -> higher lambda (more exploitation).
    """
    if not user_liked_genres:
        return 0.7

    counts = Counter(user_liked_genres)
    total = len(user_liked_genres)

    entropy = 0.0
    for g, cnt in counts.items():
        p = cnt / total
        entropy -= p * math.log2(p)

    max_entropy = math.log2(len(all_genres)) if len(all_genres) > 1 else 1.0
    norm_entropy = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0
    dynamic_lambda = base_max - (base_max - base_min) * norm_entropy
    return float(dynamic_lambda)


def maximal_marginal_relevance(
    candidates: List[Dict[str, Any]],
    limit: int = 20,
    lmbda: float = 0.7,
    relevance_key: str = "rank_score"
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) for Stage 3 recommendation reranking.
    Balances ranking relevance against pairwise genre similarity in the selected set.
    """
    if not candidates:
        return []

    # Extract genre sets
    movie_genres = []
    for c in candidates:
        g_str = c.get("genres") or ""
        g_set = set(g_str.split("|")) if g_str else set()
        movie_genres.append(g_set)

    # Normalize relevance scores to [0, 1]
    scores = np.array([float(c.get(relevance_key, 0.0)) for c in candidates])
    min_s, max_s = scores.min(), scores.max()
    norm_scores = (scores - min_s) / (max_s - min_s + 1e-6) if max_s > min_s else np.ones_like(scores)

    def jaccard_sim(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidates)))

    # Seed with top ranked candidate
    first_idx = int(np.argmax(norm_scores))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    while len(selected_indices) < limit and remaining_indices:
        best_mmr = -float("inf")
        best_idx = -1

        for idx in remaining_indices:
            rel = norm_scores[idx]
            max_sim = max([jaccard_sim(movie_genres[idx], movie_genres[s_idx]) for s_idx in selected_indices])
            mmr_score = lmbda * rel - (1.0 - lmbda) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx != -1:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break

    return [candidates[i] for i in selected_indices]
