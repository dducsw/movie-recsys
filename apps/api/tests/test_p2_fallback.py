"""P2 tests: cursor pagination + infra fallback. Simple, no DB needed."""

from unittest import mock

from app.deps import decode_cursor, encode_cursor
from app.services import recsys as recsys_mod


def test_cursor_roundtrip():
    cur = encode_cursor(12.5, 42)
    assert decode_cursor(cur) == (12.5, 42)
    assert decode_cursor(None) is None


def test_cursor_invalid_400():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        decode_cursor("not-a-cursor!!!")
    assert exc.value.status_code == 400


def test_recsys_falls_back_when_redis_qdrant_down():
    """Stage 1 must degrade to DB trending, never 500, when infra is down."""
    fake_trending = [
        {"movieId": 1, "title": "A", "genres": "Action",
         "popularity": 10.0, "vote_average": 8.0, "poster_url": "",
         "release_date": "2020"},
        {"movieId": 2, "title": "B", "genres": "Drama",
         "popularity": 9.0, "vote_average": 7.0, "poster_url": "",
         "release_date": "2021"},
    ]
    with mock.patch.object(recsys_mod, "get_redis_client", return_value=None), \
         mock.patch.object(recsys_mod, "get_qdrant_client", return_value=None), \
         mock.patch("app.models.movie.MovieModel.get_by_id",
                    return_value=fake_trending[0]), \
         mock.patch("app.models.movie.MovieModel.get_trending",
                    return_value=fake_trending), \
         mock.patch("app.models.movie.MovieModel.get_by_ids",
                    side_effect=lambda ids: [m for m in fake_trending
                                             if m["movieId"] in ids]):
        out = recsys_mod.RecsysService.get_similar_movies(1, limit=2)
        assert isinstance(out, list)


def test_ranker_missing_uses_heuristic():
    """Stage 2 must use heuristic score when ranker model is missing."""
    cands = [{"movieId": 1, "retrieval_score": 0.9},
             {"movieId": 2, "retrieval_score": 0.1}]
    movies = [
        {"movieId": 1, "title": "A", "genres": "Action",
         "popularity": 10.0, "vote_average": 8.0, "release_date": "2020"},
        {"movieId": 2, "title": "B", "genres": "Drama",
         "popularity": 5.0, "vote_average": 9.0, "release_date": "2021"},
    ]
    with mock.patch.object(recsys_mod, "get_ranker_model", return_value=None), \
         mock.patch("app.models.movie.MovieModel.get_by_ids", return_value=movies), \
         mock.patch("app.services.recsys.MLTrainingModelService.score_candidates",
                    return_value={}), \
         mock.patch("app.services.recsys.FeatureStoreService.get_online_movie_features",
                    return_value={}):
        ranked = recsys_mod.RecsysService._stage_2_ranking(cands, {"Action"})
        assert ranked[0]["movieId"] == 1  # genre_overlap wins via heuristic
        assert "rank_score" in ranked[0]


def test_emit_never_blocks_when_queue_full():
    """Event queue must drop, not raise, under pressure."""
    from app.services import event_producer as ep

    for i in range(1200):
        ep.emit(ep.EventType.CLICK, user_id="u", movie_id=i)
