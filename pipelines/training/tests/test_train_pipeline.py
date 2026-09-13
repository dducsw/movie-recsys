"""
pipelines/training/tests/test_train_pipeline.py
-----------------------------------------------
Integration tests for continuous training pipeline: retrieval, LightGBM ranker,
and metric quality gate evaluation.
"""

import pytest
import numpy as np
import pandas as pd

from recsys_core.features import RANKING_FEATURES
import pipelines.training.train_pipeline as tp


def test_ranking_features_schema():
    """Verify 8 standard ranking features schema matching serving."""
    expected_cols = [
        "popularity",
        "vote_average",
        "genre_overlap",
        "release_year",
        "user_activity",
        "user_bias",
        "als_score",
        "cb_score",
    ]
    assert RANKING_FEATURES == expected_cols
    assert len(RANKING_FEATURES) == 8


@pytest.fixture
def sample_data():
    """Small sample data fixture for continuous training integration tests."""
    df_movies = pd.DataFrame([
        {"movieId": 1, "title": "Toy Story (1995)", "genres": "Animation|Children|Comedy", "popularity": 25.0, "vote_average": 8.1, "release_year": 1995},
        {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy", "popularity": 18.0, "vote_average": 7.2, "release_year": 1995},
        {"movieId": 3, "title": "Heat (1995)", "genres": "Action|Crime|Thriller", "popularity": 30.0, "vote_average": 8.3, "release_year": 1995},
        {"movieId": 4, "title": "Sabrina (1995)", "genres": "Comedy|Romance", "popularity": 12.0, "vote_average": 6.8, "release_year": 1995},
    ])
    train_ratings = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1000},
        {"userId": 1, "movieId": 2, "rating": 3.0, "timestamp": 1001},
        {"userId": 1, "movieId": 3, "rating": 5.0, "timestamp": 1002},
        {"userId": 2, "movieId": 2, "rating": 4.0, "timestamp": 1003},
        {"userId": 2, "movieId": 3, "rating": 2.0, "timestamp": 1004},
        {"userId": 2, "movieId": 4, "rating": 4.5, "timestamp": 1005},
        {"userId": 3, "movieId": 1, "rating": 5.0, "timestamp": 1006},
        {"userId": 3, "movieId": 4, "rating": 3.5, "timestamp": 1007},
    ])
    test_ratings = pd.DataFrame([
        {"userId": 1, "movieId": 4, "rating": 4.0, "timestamp": 1008},
        {"userId": 2, "movieId": 1, "rating": 3.0, "timestamp": 1009},
    ])
    return df_movies, train_ratings, test_ratings


def test_train_retrieval_models_smoke(sample_data, tmp_path, monkeypatch):
    """Verify retrieval model training creates TF-IDF matrix and collaborative factors."""
    monkeypatch.setattr(tp, "MODELS_DIR", str(tmp_path))
    df_movies, train_ratings, _ = sample_data

    vectorizer, tfidf_matrix, als_map = tp.train_retrieval_models(df_movies, train_ratings)

    assert tfidf_matrix.shape[0] == len(df_movies)
    assert isinstance(als_map, dict)
    assert (tmp_path / "tfidf_vectorizer.pkl").exists()
    assert (tmp_path / "tfidf_matrix.pkl").exists()
    assert (tmp_path / "als_model.pkl").exists()


def test_train_ranking_model_schema(sample_data, tmp_path, monkeypatch):
    """Verify LightGBM LambdaRanker fits 8 features and saves artifact."""
    monkeypatch.setattr(tp, "MODELS_DIR", str(tmp_path))
    df_movies, train_ratings, _ = sample_data

    als_map = {
        1: np.array([0.1, 0.2]),
        2: np.array([0.3, 0.4]),
        3: np.array([0.5, 0.1]),
        4: np.array([0.2, 0.3]),
    }
    ranker = tp.train_ranking_model(
        df_movies,
        train_ratings,
        als_map=als_map,
        params={"n_estimators": 5, "min_child_samples": 1, "num_leaves": 4}
    )

    assert ranker.n_features_ == 8
    assert (tmp_path / "lgb_ranker.pkl").exists()


def test_evaluate_pipeline_returns_valid_metrics(sample_data, tmp_path, monkeypatch):
    """Verify end-to-end evaluation computes valid ranking metrics and latency."""
    monkeypatch.setattr(tp, "MODELS_DIR", str(tmp_path))
    df_movies, train_ratings, test_ratings = sample_data

    als_map = {
        1: np.array([0.1, 0.2]),
        2: np.array([0.3, 0.4]),
        3: np.array([0.5, 0.1]),
        4: np.array([0.2, 0.3]),
    }
    ranker = tp.train_ranking_model(
        df_movies,
        train_ratings,
        als_map=als_map,
        params={"n_estimators": 5, "min_child_samples": 1, "num_leaves": 4}
    )

    metrics = tp.evaluate_pipeline(ranker, df_movies, test_ratings, train_ratings, als_map=als_map, k=2)

    assert "HR@2" in metrics
    assert "NDCG@2" in metrics
    assert 0.0 <= metrics["HR@2"] <= 1.0
    assert 0.0 <= metrics["NDCG@2"] <= 1.0
    assert metrics["evaluated_users"] == 2
    assert metrics["avg_latency_ms"] < 30.0
