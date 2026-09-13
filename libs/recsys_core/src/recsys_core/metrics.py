"""
recsys_core.metrics
-------------------
Mathematical definitions of offline recommendation metrics (accuracy & beyond-accuracy).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def evaluate_ranking_metrics(
    eval_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0
) -> Dict[str, float]:
    """
    Calculate Hit Ratio@k, NDCG@k, and Mean Reciprocal Rank (MRR).

    Args:
        eval_df: DataFrame containing userId, movieId, rating, and pred columns.
        k: ranking cutoff depth.
        relevance_threshold: minimum true rating considered relevant.
    """
    hit_ratios, ndcgs, rrs = [], [], []

    for uid, g in eval_df.groupby("userId"):
        if len(g) < 1:
            continue
        g = g.sort_values("pred", ascending=False)
        rel = (g["rating"] >= relevance_threshold).astype(int).tolist()

        # Hit Ratio @ K
        hit = 1.0 if any(rel[:k]) else 0.0
        hit_ratios.append(hit)

        # NDCG @ K
        dcg = sum([r / np.log2(idx + 2) for idx, r in enumerate(rel[:k])])
        ideal_rel = sorted(rel, reverse=True)
        idcg = sum([r / np.log2(idx + 2) for idx, r in enumerate(ideal_rel[:k])])
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg)

        # MRR (Reciprocal Rank)
        rr = 0.0
        for idx, r in enumerate(rel):
            if r == 1:
                rr = 1.0 / (idx + 1)
                break
        rrs.append(rr)

    return {
        f"Hit Ratio@{k}": float(np.mean(hit_ratios)) if hit_ratios else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "MRR": float(np.mean(rrs)) if rrs else 0.0,
    }


def calculate_beyond_accuracy_metrics(
    recommendations: Dict[int, List[int]],
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    movie_features: Optional[pd.DataFrame] = None,
    k: int = 10,
    item_col: str = "movieId"
) -> Tuple[float, float, float]:
    """
    Calculate Intra-List Diversity, Novelty, and Catalog Coverage.

    Returns:
        mean_diversity: Average pairwise genre dissimilarity (1 - cosine_similarity) in Top-K
        mean_novelty: Average novelty (self-information: -log2(popularity))
        coverage: Proportion of catalog recommended across all evaluated users
    """
    item_counts = train_df[item_col].value_counts().to_dict()
    total_ratings = len(train_df)
    item_popularity = {iid: (count / total_ratings) for iid, count in item_counts.items()}

    if movie_features is None:
        movies_copy = movies_df.copy()
        movies_copy["genres_clean"] = movies_copy["genres"].str.replace("|", " ", regex=False)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = tfidf.fit_transform(movies_copy["genres_clean"])
        movie_features = pd.DataFrame(tfidf_matrix.toarray(), index=movies_copy[item_col])

    novelty_scores = []
    diversity_scores = []
    all_recommended_items = set()

    for uid, top_k in recommendations.items():
        top_k = list(top_k)[:k]
        if not top_k:
            continue

        all_recommended_items.update(top_k)

        # Novelty: -log2(popularity)
        user_novelty = []
        for iid in top_k:
            pop = item_popularity.get(iid, 1 / total_ratings)
            user_novelty.append(-np.log2(pop))
        novelty_scores.append(np.mean(user_novelty))

        # Intra-list Diversity
        valid_items = [iid for iid in top_k if iid in movie_features.index]
        if len(valid_items) > 1:
            feats = movie_features.loc[valid_items].values
            sim_matrix = cosine_similarity(feats)
            n_items = len(valid_items)
            diffs = []
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    diffs.append(1.0 - sim_matrix[i, j])
            diversity_scores.append(np.mean(diffs) if diffs else 0.0)
        else:
            diversity_scores.append(0.0)

    total_items = movies_df[item_col].nunique()
    coverage = len(all_recommended_items) / total_items if total_items > 0 else 0.0
    mean_novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0
    mean_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

    return mean_diversity, mean_novelty, float(coverage)
