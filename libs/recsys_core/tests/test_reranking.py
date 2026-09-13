"""
Unit tests for recsys_core.reranking
Testing dynamic MMR entropy lambda and Maximal Marginal Relevance reranking.
"""

import pytest
from recsys_core.reranking import (
    calculate_user_lambda,
    maximal_marginal_relevance,
)


def test_calculate_user_lambda_empty_genres():
    # Fallback to default lambda 0.7
    lmbda = calculate_user_lambda([], ["Action", "Comedy", "Drama"])
    assert lmbda == 0.7


def test_calculate_user_lambda_focused_taste():
    # Only likes Action -> entropy = 0 -> maximum exploitation (base_max = 0.9)
    all_genres = ["Action", "Comedy", "Drama", "Sci-Fi"]
    lmbda = calculate_user_lambda(["Action", "Action", "Action"], all_genres, base_min=0.4, base_max=0.9)
    assert pytest.approx(lmbda, abs=1e-3) == 0.9


def test_calculate_user_lambda_broad_taste():
    # Broad taste across all genres -> high entropy -> minimum exploitation / high diversity (close to base_min)
    all_genres = ["Action", "Comedy", "Drama", "Sci-Fi"]
    liked = ["Action", "Comedy", "Drama", "Sci-Fi"]
    lmbda = calculate_user_lambda(liked, all_genres, base_min=0.4, base_max=0.9)
    # Uniform distribution over 4 items gives maximum entropy
    assert pytest.approx(lmbda, abs=1e-3) == 0.4


def test_maximal_marginal_relevance_empty_candidates():
    reranked = maximal_marginal_relevance([])
    assert reranked == []


def test_maximal_marginal_relevance_limit_greater_than_candidates():
    candidates = [
        {"movieId": 1, "rank_score": 0.9, "genres": "Action"},
        {"movieId": 2, "rank_score": 0.8, "genres": "Drama"},
    ]
    reranked = maximal_marginal_relevance(candidates, limit=5)
    assert len(reranked) == 2
    assert reranked[0]["movieId"] == 1
    assert reranked[1]["movieId"] == 2


def test_maximal_marginal_relevance_penalizes_redundancy():
    candidates = [
        {"movieId": 1, "rank_score": 1.0, "genres": "Action"},
        {"movieId": 2, "rank_score": 0.98, "genres": "Action"},
        {"movieId": 3, "rank_score": 0.75, "genres": "Comedy"},
    ]
    # Under low lambda (e.g. 0.2), diversity penalty dominates over the slight relevance advantage of movie 2
    reranked = maximal_marginal_relevance(candidates, limit=2, lmbda=0.2)
    assert len(reranked) == 2
    assert reranked[0]["movieId"] == 1
    assert reranked[1]["movieId"] == 3
