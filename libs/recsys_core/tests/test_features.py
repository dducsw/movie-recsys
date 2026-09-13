"""
Unit tests for recsys_core.features
Testing feature extraction, movie metadata precomputation, and user profile generation.
"""

import pytest
import numpy as np
import pandas as pd
from recsys_core.features import (
    RANKING_FEATURES,
    extract_movie_release_year,
    build_movie_meta,
    build_user_profiles,
)


def test_ranking_features_schema():
    assert len(RANKING_FEATURES) == 8
    expected = [
        "popularity", "vote_average", "genre_overlap", "release_year",
        "user_activity", "user_bias", "als_score", "cb_score"
    ]
    assert RANKING_FEATURES == expected


def test_extract_movie_release_year_from_release_date():
    assert extract_movie_release_year({"release_date": "1994-09-23"}) == 1994
    assert extract_movie_release_year({"release_date": "2020"}) == 2020


def test_extract_movie_release_year_from_title():
    assert extract_movie_release_year({"title": "Inception (2010)"}) == 2010
    assert extract_movie_release_year({"title": "Avatar"}) == 2010  # fallback


def test_build_movie_meta_and_user_profiles():
    df_movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Toy Story (1995)", "Jumanji (1995)"],
        "genres": ["Animation|Children|Comedy", "Adventure|Children"],
        "popularity": [25.0, 15.0],
        "vote_average": [8.0, 7.0]
    })
    als_map = {
        1: np.array([0.1, 0.2]),
        2: np.array([0.3, 0.4])
    }
    meta = build_movie_meta(df_movies, als_map=als_map)
    assert len(meta) == 2
    assert meta[1]["release_year"] == 1995
    assert "Animation" in meta[1]["genres"]
    assert np.array_equal(meta[1]["als_vec"], np.array([0.1, 0.2]))

    train_ratings = pd.DataFrame({
        "userId": [10, 10],
        "movieId": [1, 2],
        "rating": [4.0, 5.0]
    })
    profiles = build_user_profiles(train_ratings, meta, global_avg_rating=3.5)
    assert 10 in profiles
    assert profiles[10]["bias"] == pytest.approx(4.5 - 3.5)
    assert profiles[10]["activity"] == pytest.approx(np.log1p(2))
    assert "Comedy" in profiles[10]["genres"]
    assert profiles[10]["als_vec"] is not None
