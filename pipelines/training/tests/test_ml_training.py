"""
evaluation/tests/test_ml_training.py
------------------------------------
Unit tests for ML ranking metrics, RRF fusion, and dynamic MMR parameters.
"""

import pytest
import numpy as np
import pandas as pd

try:
    from pipelines.training.ml_training.src.train import (
        rmse_mae,
        evaluate_ranking_metrics,
    )
    from pipelines.training.recsys_utils import (
        reciprocal_rank_fusion,
        calculate_user_lambda,
    )
except ImportError:
    from train import (
        rmse_mae,
        evaluate_ranking_metrics,
    )
    from recsys_utils import (
        reciprocal_rank_fusion,
        calculate_user_lambda,
    )


def test_rmse_mae():
    y_true = np.array([3.0, 4.0, 5.0])
    y_pred = np.array([3.0, 3.0, 5.0])  # errors: 0, -1, 0

    rmse, mae = rmse_mae(y_true, y_pred)
    assert pytest.approx(rmse, abs=1e-4) == np.sqrt(1.0 / 3.0)
    assert pytest.approx(mae, abs=1e-4) == 1.0 / 3.0


def test_evaluate_ranking_metrics_perfect_ranking():
    # User 1 has 3 items, item 101 is highly relevant (5.0) and ranked 1st by model (score 0.95)
    eval_df = pd.DataFrame([
        {"userId": 1, "movieId": 101, "rating": 5.0, "pred": 0.95},
        {"userId": 1, "movieId": 102, "rating": 2.0, "pred": 0.50},
        {"userId": 1, "movieId": 103, "rating": 1.0, "pred": 0.10},
    ])

    metrics = evaluate_ranking_metrics(eval_df, k=10, relevance_threshold=4.0)

    assert metrics["Hit Ratio@10"] == 1.0
    assert metrics["NDCG@10"] == 1.0
    assert metrics["MRR"] == 1.0


def test_evaluate_ranking_metrics_no_hits():
    # User 1 has no items meeting relevance threshold >= 4.0
    eval_df = pd.DataFrame([
        {"userId": 1, "movieId": 101, "rating": 3.0, "pred": 0.95},
        {"userId": 1, "movieId": 102, "rating": 2.0, "pred": 0.50},
    ])

    metrics = evaluate_ranking_metrics(eval_df, k=10, relevance_threshold=4.0)

    assert metrics["Hit Ratio@10"] == 0.0
    assert metrics["NDCG@10"] == 0.0
    assert metrics["MRR"] == 0.0


def test_reciprocal_rank_fusion():
    als = [1, 2, 3]
    bm25 = [2, 1, 4]
    k = 60

    fused = reciprocal_rank_fusion(als, bm25, k=k)
    fused_dict = dict(fused)

    # Item 1: rank 1 in als (1/61), rank 2 in bm25 (1/62)
    expected_score_1 = (1.0 / 61.0) + (1.0 / 62.0)
    assert pytest.approx(fused_dict[1], abs=1e-6) == expected_score_1

    # Item 2: rank 2 in als (1/62), rank 1 in bm25 (1/61)
    expected_score_2 = (1.0 / 62.0) + (1.0 / 61.0)
    assert pytest.approx(fused_dict[2], abs=1e-6) == expected_score_2

    # Top items must be 1 and 2
    top_ids = [x[0] for x in fused[:2]]
    assert set(top_ids) == {1, 2}


def test_calculate_user_lambda():
    all_genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]

    # 1. Empty history fallback
    assert calculate_user_lambda([], all_genres) == 0.7

    # 2. Focused taste (single genre repeated) -> low entropy -> higher lambda (near base_max=0.9)
    focused_lambda = calculate_user_lambda(["Action", "Action", "Action", "Action"], all_genres)
    assert focused_lambda == 0.9

    # 3. Diverse taste (all genres equally represented) -> high entropy -> lower lambda (near base_min=0.4)
    diverse_lambda = calculate_user_lambda(["Action", "Comedy", "Drama", "Horror", "Sci-Fi"], all_genres)
    assert pytest.approx(diverse_lambda, abs=1e-4) == 0.4
