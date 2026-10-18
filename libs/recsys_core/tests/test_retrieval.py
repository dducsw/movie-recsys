"""Tests for recsys_core.retrieval (true iALS) — clean senior style."""

import numpy as np
import pandas as pd

from recsys_core.retrieval import (
    IALSConfig,
    build_confidence_matrix,
    train_ials_factors,
)


def _sample_ratings() -> pd.DataFrame:
    return pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 4.0},
        {"userId": 1, "movieId": 2, "rating": 3.0},
        {"userId": 2, "movieId": 2, "rating": 5.0},
        {"userId": 2, "movieId": 3, "rating": 2.0},
    ])


def test_confidence_matrix_shape_and_values():
    ratings = _sample_ratings()
    matrix, users, items = build_confidence_matrix(ratings, alpha=40.0)
    assert matrix.shape == (len(users), len(items))  # [users x items] for implicit
    # c = 1 + 40 * r
    assert matrix.max() == 1.0 + 40.0 * 5.0


def test_train_ials_returns_normalized_factors():
    ratings = _sample_ratings()
    factors = train_ials_factors(
        ratings, config=IALSConfig(factors=8, iterations=5))
    assert set(factors.keys()) == {1, 2, 3}
    for vec in factors.values():
        assert vec.shape == (8,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


def test_train_ials_empty():
    empty = pd.DataFrame(columns=["userId", "movieId", "rating"])
    assert train_ials_factors(empty) == {}
