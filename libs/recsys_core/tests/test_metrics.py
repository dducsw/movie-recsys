"""
Unit tests for recsys_core.metrics
Testing mathematical correctness, edge cases (empty data, no relevant items, perfect rankings),
and beyond-accuracy metrics (ILD, novelty, coverage).
"""

import pytest
import numpy as np
import pandas as pd
from recsys_core.metrics import (
    rmse_mae,
    evaluate_ranking_metrics,
    calculate_beyond_accuracy_metrics,
)


def test_rmse_mae_exact_match():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rmse, mae = rmse_mae(y_true, y_pred)
    assert rmse == 0.0
    assert mae == 0.0


def test_rmse_mae_known_delta():
    y_true = np.array([2.0, 4.0])
    y_pred = np.array([1.0, 5.0])
    rmse, mae = rmse_mae(y_true, y_pred)
    # diffs: [-1.0, 1.0] -> squared: [1.0, 1.0] -> mean: 1.0 -> sqrt: 1.0
    assert pytest.approx(rmse, abs=1e-5) == 1.0
    assert pytest.approx(mae, abs=1e-5) == 1.0


def test_evaluate_ranking_metrics_empty():
    empty_df = pd.DataFrame(columns=["userId", "movieId", "rating", "pred"])
    res = evaluate_ranking_metrics(empty_df, k=10)
    assert res["Hit Ratio@10"] == 0.0
    assert res["NDCG@10"] == 0.0
    assert res["MRR"] == 0.0


def test_evaluate_ranking_metrics_no_relevant_items():
    # True ratings are all below threshold (4.0)
    df = pd.DataFrame({
        "userId": [1, 1, 1],
        "movieId": [10, 20, 30],
        "rating": [1.0, 2.0, 3.0],
        "pred": [4.5, 4.0, 3.5]
    })
    res = evaluate_ranking_metrics(df, k=10, relevance_threshold=4.0)
    assert res["Hit Ratio@10"] == 0.0
    assert res["NDCG@10"] == 0.0
    assert res["MRR"] == 0.0


def test_evaluate_ranking_metrics_perfect_ranking():
    # User has 2 relevant items (5.0, 4.0) and 1 irrelevant (2.0)
    # Model places relevant items at rank 1 and rank 2
    df = pd.DataFrame({
        "userId": [1, 1, 1],
        "movieId": [10, 20, 30],
        "rating": [5.0, 4.0, 2.0],
        "pred": [0.95, 0.85, 0.10]
    })
    res = evaluate_ranking_metrics(df, k=2, relevance_threshold=4.0)
    assert res["Hit Ratio@2"] == 1.0
    assert pytest.approx(res["NDCG@2"], abs=1e-4) == 1.0
    assert res["MRR"] == 1.0


def test_evaluate_ranking_metrics_imperfect_ranking():
    # Relevant item at rank 2:
    # rank 1: irrelevant (rating 1.0, pred 0.9)
    # rank 2: relevant (rating 5.0, pred 0.8)
    df = pd.DataFrame({
        "userId": [1, 1],
        "movieId": [10, 20],
        "rating": [1.0, 5.0],
        "pred": [0.9, 0.8]
    })
    res = evaluate_ranking_metrics(df, k=2, relevance_threshold=4.0)
    assert res["Hit Ratio@2"] == 1.0
    assert res["MRR"] == 0.5  # 1 / rank 2 = 0.5
    # dcg = 1 / log2(2 + 1) = 1 / log2(3) = ~0.6309
    # idcg = 1 / log2(0 + 2) = 1 / 1 = 1.0
    # ndcg = dcg / idcg = 1 / log2(3)
    assert pytest.approx(res["NDCG@2"], abs=1e-4) == 1.0 / np.log2(3)


def test_calculate_beyond_accuracy_metrics():
    # Catalog of 4 movies
    movies_df = pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "genres": ["Action", "Action", "Comedy", "Drama"]
    })
    train_df = pd.DataFrame({
        "userId": [100, 100, 101, 102],
        "movieId": [1, 2, 1, 3],
        "rating": [4.0, 5.0, 3.0, 4.0]
    })
    recommendations = {
        100: [1, 3],  # Action & Comedy
        101: [2, 4],  # Action & Drama
    }
    mean_diversity, mean_novelty, coverage = calculate_beyond_accuracy_metrics(
        recommendations=recommendations,
        train_df=train_df,
        movies_df=movies_df,
        k=2
    )

    # All 4 items recommended across users 100 & 101 -> coverage = 4 / 4 = 1.0
    assert pytest.approx(coverage, abs=1e-4) == 1.0
    assert mean_novelty > 0.0
    assert mean_diversity > 0.0
