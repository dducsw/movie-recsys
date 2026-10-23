"""
recsys_core.metrics
-------------------
Mathematical definitions of offline recommendation metrics (accuracy & beyond-accuracy).

Canonical metric keys
---------------------
``HR@{k}``, ``NDCG@{k}``, ``MRR@{k}``.

The legacy alias ``Hit Ratio@{k}`` is still emitted so downstream notebooks keep
working, but ``HR@{k}`` is the single source of truth and is what
:mod:`recsys_core.quality_gate` reads.  Never introduce a third spelling.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# Canonical per-user metric column names produced by :func:`per_user_ranking_scores`.
USER_SCORE_COLUMNS = ("hr", "ndcg", "rr")

# Mapping from canonical metric key -> per-user column.
_METRIC_TO_COLUMN = {"hr": "hr", "ndcg": "ndcg", "rr": "rr"}


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def per_user_ranking_scores(
    eval_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> pd.DataFrame:
    """Compute per-user Hit Ratio@k / NDCG@k / RR.

    Users whose holdout contains **no relevant item** are excluded: ranking
    metrics are undefined for them (NDCG has a zero ideal denominator and a
    Hit Ratio of 0 would only penalise the model for an impossible query).
    The number of excluded users is reported by
    :func:`evaluate_ranking_metrics` as ``n_users_skipped``.

    Args:
        eval_df: DataFrame containing userId, movieId, rating, and pred columns.
        k: ranking cutoff depth.
        relevance_threshold: minimum true rating considered relevant.

    Returns:
        DataFrame with columns ``userId``, ``hr``, ``ndcg``, ``rr`` — one row
        per *evaluable* user.
    """
    required = {"userId", "rating", "pred"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df missing columns: {sorted(missing)}.")

    rows: List[Dict[str, Any]] = []
    for uid, g in eval_df.groupby("userId"):
        g = g.sort_values("pred", ascending=False)
        rel = (g["rating"] >= relevance_threshold).astype(int).tolist()

        # No relevant item in this user's holdout -> metrics are undefined.
        if sum(rel) == 0:
            continue

        # Hit Ratio @ K
        hit = 1.0 if any(rel[:k]) else 0.0

        # NDCG @ K  (binary gain, IDCG from the same candidate set)
        dcg = sum([r / np.log2(idx + 2) for idx, r in enumerate(rel[:k])])
        ideal_rel = sorted(rel, reverse=True)
        idcg = sum([r / np.log2(idx + 2) for idx, r in enumerate(ideal_rel[:k])])
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        # Reciprocal rank of the first relevant item (over the full ranking).
        rr = 0.0
        for idx, r in enumerate(rel):
            if r == 1:
                rr = 1.0 / (idx + 1)
                break

        rows.append({"userId": uid, "hr": hit, "ndcg": ndcg, "rr": rr})

    return pd.DataFrame(rows, columns=["userId", "hr", "ndcg", "rr"])


def evaluate_ranking_metrics(
    eval_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> Dict[str, float]:
    """
    Calculate Hit Ratio@k, NDCG@k, and Mean Reciprocal Rank (MRR).

    Keys returned (canonical names):
        ``HR@{k}``, ``NDCG@{k}``, ``MRR@{k}``
        plus the legacy alias ``Hit Ratio@{k}`` (same value as ``HR@{k}``).
        ``n_users_evaluated``   - number of users included in the averages.
        ``n_users_skipped``     - users with no relevant holdout item (undefined).
    """
    scores = per_user_ranking_scores(eval_df, k=k, relevance_threshold=relevance_threshold)

    n_evaluated = len(scores)
    n_total = int(eval_df["userId"].nunique()) if "userId" in eval_df.columns else 0

    if n_evaluated == 0:
        hr = ndcg = rr = 0.0
    else:
        hr = float(scores["hr"].mean())
        ndcg = float(scores["ndcg"].mean())
        rr = float(scores["rr"].mean())

    return {
        f"HR@{k}": hr,
        f"NDCG@{k}": ndcg,
        f"MRR@{k}": rr,
        # Legacy alias — kept so existing notebooks/report code keeps working.
        f"Hit Ratio@{k}": hr,
        "n_users_evaluated": float(n_evaluated),
        "n_users_skipped": float(n_total - n_evaluated),
    }


def bootstrap_ranking_ci(
    per_user_scores: pd.DataFrame,
    metric: str = "ndcg",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a per-user ranking metric.

    Users are resampled with replacement (the sampling unit is the user, since
    a user's holdout rows are not independent).  Use this before declaring that
    a metric delta is real rather than sampling noise.

    Args:
        per_user_scores: output of :func:`per_user_ranking_scores`.
        metric: one of ``"hr"``, ``"ndcg"``, ``"rr"``.
        n_boot: number of bootstrap resamples (>= 1000 recommended).
        seed: RNG seed for reproducibility.
        alpha: two-sided significance level.

    Returns:
        Dict with ``mean``, ``ci_low``, ``ci_high``, ``n``, ``n_boot``.
    """
    if metric not in _METRIC_TO_COLUMN:
        raise ValueError(f"metric must be one of {sorted(_METRIC_TO_COLUMN)}.")
    values = per_user_scores[_METRIC_TO_COLUMN[metric]].to_numpy(dtype=float)
    if len(values) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_boot": n_boot}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(len(values)),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_delta(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    metric: str = "ndcg",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap of the per-user difference ``metric(A) - metric(B)``.

    Pairing on ``userId`` removes between-user variance, which makes the
    interval far tighter than comparing two independent means.

    Returns:
        Dict with ``delta``, ``ci_low``, ``ci_high``, ``n_users`` and
        ``significant`` (True when the CI excludes 0).
    """
    if metric not in _METRIC_TO_COLUMN:
        raise ValueError(f"metric must be one of {sorted(_METRIC_TO_COLUMN)}.")
    col = _METRIC_TO_COLUMN[metric]

    merged = scores_a[["userId", col]].merge(
        scores_b[["userId", col]], on="userId", suffixes=("_a", "_b")
    )
    if merged.empty:
        return {
            "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "n_users": 0, "significant": False,
        }

    diff = (merged[f"{col}_a"] - merged[f"{col}_b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_users": int(len(diff)),
        "significant": bool(lo > 0.0 or hi < 0.0),
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

    # Coverage denominator must only count items that actually exist in the
    # catalog, otherwise coverage can exceed 1.0.
    catalog_ids = set(movies_df[item_col].dropna().unique())
    recommended_in_catalog = all_recommended_items & catalog_ids
    total_items = len(catalog_ids)
    coverage = len(recommended_in_catalog) / total_items if total_items > 0 else 0.0
    mean_novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0
    mean_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

    return mean_diversity, mean_novelty, float(coverage)
