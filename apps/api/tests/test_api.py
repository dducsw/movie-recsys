import pytest
import sys
import os
import time

from fastapi.testclient import TestClient
from main import app
from app.services.auth_service import hash_password, verify_password, create_access_token, decode_access_token
from app.services.ml_training_service import MLTrainingModelService, get_model_bundle
from app.chatbot.agents.detect_intent import heuristic_detect_intent

client = TestClient(app)


# ── 1. Basic Health & Root ─────────────────────────────────────────────────────

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200


# ── 2. Authentication & Security Unit Tests ────────────────────────────────────

def test_password_hashing_and_verification():
    raw_pass = "SeniorRecSys2026!#"
    hashed = hash_password(raw_pass)
    assert hashed != raw_pass
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_jwt_token_lifecycle():
    user_id = 999
    username = "senior_engineer"
    token = create_access_token(user_id=user_id, username=username)
    assert token is not None
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == str(user_id)
    assert payload["username"] == username


def test_auth_registration_validation():
    # Password too short (< 6 chars)
    res = client.post("/api/auth/register", json={
        "email": "valid@movienex.ai",
        "password": "123"
    })
    assert res.status_code == 400
    assert "Mật khẩu" in res.json()["detail"] or "Password" in res.json()["detail"]

    # Invalid email format
    res_bad_email = client.post("/api/auth/register", json={
        "email": "not-an-email",
        "password": "valid_password_123"
    })
    assert res_bad_email.status_code == 422


def test_protected_routes_without_auth():
    # Commenting requires auth
    res_comment = client.post("/api/movies/1/comments", json={"content": "Great movie!"})
    assert res_comment.status_code == 401

    # Liking a comment requires auth
    res_like = client.post("/api/comments/1/like")
    assert res_like.status_code == 401

    # Deleting a comment requires auth
    res_delete = client.delete("/api/comments/1")
    assert res_delete.status_code == 401


# ── 3. ML Training & Ranking Integration Tests ─────────────────────────────────

@pytest.mark.skipif(get_model_bundle() is None, reason="models.joblib not present in evaluation/ml_training")
def test_ml_training_bundle_loaded():
    """Verify that models.joblib in evaluation/ml_training is loaded properly."""
    bundle = get_model_bundle()
    assert bundle is not None, "Failed to load models.joblib from evaluation/ml_training"
    assert "catboost" in bundle or "lgbm" in bundle
    assert "movie_feats" in bundle
    assert "user_feats" in bundle
    assert len(bundle["movie_feats"]) > 1000


@pytest.mark.skipif(get_model_bundle() is None, reason="models.joblib not present in evaluation/ml_training")
def test_ml_training_candidate_scoring():
    """Verify that MLTrainingModelService scores candidates within SLA (< 50ms)."""
    candidate_ids = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
    t0 = time.perf_counter()
    scores = MLTrainingModelService.score_candidates(
        candidate_ids=candidate_ids,
        user_id=1,
        target_genres={"Action", "Sci-Fi"}
    )
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(scores, dict)
    assert len(scores) > 0
    assert duration_ms < 60.0, f"Inference took {duration_ms:.2f}ms, exceeding 60ms SLA"
    for mid, score in scores.items():
        assert isinstance(score, float)


@pytest.mark.skipif(get_model_bundle() is None, reason="models.joblib not present in evaluation/ml_training")
def test_ml_training_recommend_for_user():
    """Verify full-catalog recommendation for user."""
    top_recs = MLTrainingModelService.recommend_for_user(user_id=1, top_k=5)
    assert isinstance(top_recs, list)
    assert len(top_recs) == 5
    for item in top_recs:
        assert "movieId" in item
        assert "rank_score" in item


def test_recsys_stage_2_ranking_and_serving(monkeypatch):
    """Verify that RecsysService Stage 2 LightGBM ranker scores 8 features properly."""
    from app.services.recsys import RecsysService, get_ranker_model
    from app.models.movie import MovieModel

    # Verify ranker has 8 standardized features
    ranker = get_ranker_model()
    if ranker is not None and hasattr(ranker, "feature_name_"):
        assert len(ranker.feature_name_) == 8
        assert "als_score" in ranker.feature_name_
        assert "cb_score" in ranker.feature_name_

    # Mock get_by_ids to test Stage 2 ranking logic without live DB container
    dummy_movies = [
        {"movieId": 1, "title": "Toy Story", "popularity": 30.0, "vote_average": 8.0, "genres": "Animation|Children|Comedy", "release_date": "1995-10-30"},
        {"movieId": 2, "title": "Jumanji", "popularity": 20.0, "vote_average": 7.0, "genres": "Adventure|Children|Fantasy", "release_date": "1995-12-15"},
        {"movieId": 3, "title": "Grumpier Old Men", "popularity": 10.0, "vote_average": 6.5, "genres": "Comedy|Romance", "release_date": "1995-12-22"}
    ]
    monkeypatch.setattr(MovieModel, "get_by_ids", lambda ids: dummy_movies)

    from app.services.feature_store import FeatureStoreService
    monkeypatch.setattr(FeatureStoreService, "get_online_movie_features", lambda ids: {})

    candidates = [
        {"movieId": 1, "retrieval_score": 0.9},
        {"movieId": 2, "retrieval_score": 0.8},
        {"movieId": 3, "retrieval_score": 0.7}
    ]
    # Warm up OpenMP / LightGBM C++ runtime
    RecsysService._stage_2_ranking(candidates, target_genres={"Animation", "Children"}, liked_count=2)

    t0 = time.perf_counter()
    ranked = RecsysService._stage_2_ranking(candidates, target_genres={"Animation", "Children"}, liked_count=2)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(ranked, list)
    assert len(ranked) == 3
    assert duration_ms < 50.0, f"Stage 2 ranking took {duration_ms:.2f}ms, exceeding 50ms SLA"
    for m in ranked:
        assert "rank_score" in m
        assert isinstance(m["rank_score"], float)


# ── 4. Chatbot NLP & Intent Detection Tests ───────────────────────────────────

def test_chatbot_heuristic_intent_detection():
    # Similar intent
    res_similar = heuristic_detect_intent("Can you recommend movies similar to Inception?")
    assert res_similar.intent == "similar"
    assert "inception" in res_similar.target_title.lower()

    # Genre intent
    res_genre = heuristic_detect_intent("Suggest some exciting action and sci-fi movies")
    assert res_genre.intent == "genre"
    assert "Action" in res_genre.genres or "Sci-Fi" in res_genre.genres

    # Vietnamese similar intent
    res_vn_similar = heuristic_detect_intent("Có phim nào tương tự Interstellar không?")
    assert res_vn_similar.intent == "similar"
    assert "interstellar" in res_vn_similar.target_title.lower()

    # Personalized intent
    res_personal = heuristic_detect_intent("Gợi ý cho tôi vài bộ phim hay nên xem tối nay")
    assert res_personal.intent == "personalized"

    # Vietnamese comedy intent: MUST NOT match Horror
    res_comedy = heuristic_detect_intent("tôi muốn tìm phim hài mà vui vẻ")
    assert res_comedy.intent == "genre"
    assert "Comedy" in res_comedy.genres
    assert "Horror" not in res_comedy.genres
    assert res_comedy.language == "vi"

    # English comedy intent
    res_en_comedy = heuristic_detect_intent("recommend me some comedy movies")
    assert res_en_comedy.intent == "genre"
    assert "Comedy" in res_en_comedy.genres
    assert "Horror" not in res_en_comedy.genres
    assert res_en_comedy.language == "en"


def test_cold_start_recommendations(monkeypatch):
    """Verify cold start recommendation strategy with genre matching and trending fallback."""
    from app.services.recsys import RecsysService
    from app.models.movie import MovieModel

    dummy_trending = [
        {"movieId": 1, "title": "Toy Story", "genres": "Animation|Children", "popularity": 50.0, "vote_average": 8.0},
        {"movieId": 2, "title": "Die Hard", "genres": "Action|Thriller", "popularity": 45.0, "vote_average": 8.2},
        {"movieId": 3, "title": "The Lion King", "genres": "Animation|Drama", "popularity": 40.0, "vote_average": 8.5},
    ]
    monkeypatch.setattr(MovieModel, "get_trending", lambda page, limit: dummy_trending)

    # 1. Matching preferred genre "Action" -> Die Hard should appear first
    recs_action = RecsysService.get_cold_start_recommendations(preferred_genres=["Action"], limit=2)
    assert len(recs_action) == 2
    assert recs_action[0]["movieId"] == 2

    # 2. No preferred genres -> global trending order
    recs_global = RecsysService.get_cold_start_recommendations(preferred_genres=None, limit=2)
    assert len(recs_global) == 2
    assert recs_global[0]["movieId"] == 1

