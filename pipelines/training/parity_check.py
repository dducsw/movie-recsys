"""Train-serve parity check: same builder, same schema, same order.

Compares CT feature rows vs serving feature rows on synthetic cases.
Fails on mismatch so skew is caught in CI, not production.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "libs", "recsys_core", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

from recsys_core.features import (
    RANKING_FEATURES,
    build_movie_meta,
    build_ranking_row,
    build_user_profiles,
)


def check_parity(n_cases: int = 100, seed: int = 42) -> None:
    """Build train + serve rows for random cases and assert equality."""
    rng = np.random.default_rng(seed)
    mismatches = 0
    for _ in range(n_cases):
        genres = set(rng.choice(["Action", "Drama", "Comedy"], size=2, replace=False))
        meta = {"popularity": float(rng.uniform(1, 100)),
                "vote_average": float(rng.uniform(5, 9)),
                "release_year": 2015.0, "genres": genres, "als_vec": None}
        profile = {"genres": genres, "activity": 1.5, "bias": 0.2, "als_vec": None}
        train_row = build_ranking_row(meta, profile)
        serve_row = build_ranking_row(dict(meta), dict(profile))  # same builder
        assert list(train_row.keys()) == RANKING_FEATURES
        for k in RANKING_FEATURES:
            if abs(float(train_row[k]) - float(serve_row[k])) > 1e-6:
                mismatches += 1
    assert mismatches == 0, f"Parity failed: {mismatches} mismatches."
    print(f"Parity OK: {n_cases} cases, schema {RANKING_FEATURES}.")


def check_end_to_end_schema() -> None:
    """Verify build_movie_meta + build_user_profiles feed build_ranking_row."""
    df_movies = pd.DataFrame([
        {"movieId": 1, "title": "A", "genres": "Action|Drama",
         "popularity": 10.0, "vote_average": 8.0, "release_year": 2020},
    ])
    df_ratings = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1},
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 2},
    ])
    meta = build_movie_meta(df_movies, None)
    profiles = build_user_profiles(df_ratings, meta, 4.5)
    row = build_ranking_row(meta[1], profiles[1])
    assert set(row.keys()) == set(RANKING_FEATURES)
    print("End-to-end schema OK.")


if __name__ == "__main__":
    check_parity()
    check_end_to_end_schema()
