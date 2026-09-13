"""
recsys_core
===========
Shared core library for MovieNex Recommendation System.
Provides unified feature definitions, ranking metrics, fusion algorithms, and MMR reranking.
"""

from .features import (
    RANKING_FEATURES,
    extract_movie_release_year,
    build_movie_meta,
    build_user_profiles,
)
from .metrics import (
    rmse_mae,
    evaluate_ranking_metrics,
    calculate_beyond_accuracy_metrics,
)
from .reranking import (
    maximal_marginal_relevance,
    calculate_user_lambda,
)
from .fusion import (
    reciprocal_rank_fusion,
    BM25,
)

__all__ = [
    "RANKING_FEATURES",
    "extract_movie_release_year",
    "build_movie_meta",
    "build_user_profiles",
    "rmse_mae",
    "evaluate_ranking_metrics",
    "calculate_beyond_accuracy_metrics",
    "maximal_marginal_relevance",
    "calculate_user_lambda",
    "reciprocal_rank_fusion",
    "BM25",
]
