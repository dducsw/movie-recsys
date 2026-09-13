"""
libs/recsys_core/tests/test_core.py
----------------------------------
Unit tests for recsys_core features, metrics, reranking, and fusion algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from recsys_core import (
    RANKING_FEATURES,
    extract_movie_release_year,
    build_movie_meta,
    build_user_profiles,
    rmse_mae,
    evaluate_ranking_metrics,
    maximal_marginal_relevance,
    calculate_user_lambda,
    reciprocal_rank_fusion,
)


def test_schema_and_features():
    assert len(RANKING_FEATURES) == 8
    row = {"title": "The Matrix (1999)"}
    assert extract_movie_release_year(row) == 1999


def test_rmse_and_metrics():
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([3.0, 5.0])
    rmse, mae = rmse_mae(y_true, y_pred)
    assert pytest.approx(rmse, abs=1e-4) == np.sqrt(0.5)
    assert pytest.approx(mae, abs=1e-4) == 0.5


def test_mmr_diversity():
    candidates = [
        {"movieId": 1, "title": "M1", "rank_score": 0.9, "genres": "Action"},
        {"movieId": 2, "title": "M2", "rank_score": 0.88, "genres": "Action"},
        {"movieId": 3, "title": "M3", "rank_score": 0.70, "genres": "Comedy"},
    ]
    # With lmbda=0.2 (high diversity weight), movie 3 (Comedy) should be promoted over movie 2 (Action duplicate)
    reranked = maximal_marginal_relevance(candidates, limit=2, lmbda=0.2)
    assert len(reranked) == 2
    assert reranked[0]["movieId"] == 1
    assert reranked[1]["movieId"] == 3


def test_rrf():
    fused = reciprocal_rank_fusion([1, 2], [2, 1], k=60)
    assert len(fused) == 2
