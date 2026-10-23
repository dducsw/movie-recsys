"""
recsys_core
===========
Shared core library for MovieNex Recommendation System.
Provides unified feature definitions, ranking metrics, fusion algorithms, and MMR reranking.
"""

from .embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    build_movie_texts,
    encode_texts,
)
from .features import (
    RANKING_FEATURES,
    TFIDF_MAX_FEATURES,
    build_movie_meta,
    build_ranking_row,
    build_user_profiles,
    extract_movie_release_year,
)
from .fusion import (
    BM25,
    reciprocal_rank_fusion,
)
from .metrics import (
    calculate_beyond_accuracy_metrics,
    evaluate_ranking_metrics,
    rmse_mae,
)
from .quality_gate import QualityGateConfig, QualityGateError, check_quality_gate
from .reranking import (
    calculate_user_lambda,
    maximal_marginal_relevance,
)
from .retrieval import IALSConfig, build_confidence_matrix, train_ials_factors
from .sampling import sample_negatives
from .two_tower import TwoTowerConfig, TwoTowerRetrieval

__all__ = [
    "RANKING_FEATURES",
    "TFIDF_MAX_FEATURES",
    "extract_movie_release_year",
    "build_movie_meta",
    "build_ranking_row",
    "build_user_profiles",
    "rmse_mae",
    "evaluate_ranking_metrics",
    "calculate_beyond_accuracy_metrics",
    "maximal_marginal_relevance",
    "calculate_user_lambda",
    "reciprocal_rank_fusion",
    "BM25",
    "IALSConfig",
    "build_confidence_matrix",
    "train_ials_factors",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_DIM",
    "build_movie_texts",
    "encode_texts",
    "sample_negatives",
    "QualityGateConfig",
    "QualityGateError",
    "check_quality_gate",
    "TwoTowerConfig",
    "TwoTowerRetrieval",
]
