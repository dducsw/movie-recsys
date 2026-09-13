"""
recsys_core.features
--------------------
Standardized ranking feature schema and feature precomputation helpers.
"""

import re
from typing import Dict, Any, Optional, Set
import numpy as np
import pandas as pd

# Standard 8-feature schema matching Stage 2 serving and continuous training
RANKING_FEATURES = [
    "popularity",
    "vote_average",
    "genre_overlap",
    "release_year",
    "user_activity",
    "user_bias",
    "als_score",
    "cb_score",
]


def extract_movie_release_year(row: Dict[str, Any] | pd.Series) -> int:
    """Extract 4-digit release year from release_date or parenthesized year in title."""
    if "release_date" in row and pd.notna(row["release_date"]):
        s = str(row["release_date"]).strip()
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
    title = str(row.get("title", ""))
    m = re.search(r'\((\d{4})\)', title)
    if m:
        return int(m.group(1))
    return 2010


def build_movie_meta(df_movies: pd.DataFrame, als_map: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, Dict[str, Any]]:
    """Precompute movie metadata dictionary containing release year, genres, and ALS vector."""
    records = df_movies.to_dict("records")
    has_year = "release_year" in df_movies.columns

    movie_meta = {}
    for r in records:
        mid = int(r["movieId"])
        g_str = str(r.get("genres", "") or "")
        g_set = set(g_str.split("|")) if g_str else set()
        rel_year = float(r["release_year"]) if has_year else float(extract_movie_release_year(r))
        movie_meta[mid] = {
            "popularity": float(r.get("popularity", 1.0)),
            "vote_average": float(r.get("vote_average", 5.0)),
            "release_year": rel_year,
            "genres": g_set,
            "als_vec": als_map.get(mid) if als_map else None,
        }
    return movie_meta


def build_user_profiles(
    train_ratings: pd.DataFrame,
    movie_meta: Dict[int, Dict[str, Any]],
    global_avg_rating: Optional[float] = None
) -> Dict[int, Dict[str, Any]]:
    """Precompute user profile statistics and genre/ALS affinities from historical ratings."""
    if global_avg_rating is None:
        global_avg_rating = float(train_ratings["rating"].mean())

    user_profiles = {}
    for uid, g in train_ratings.groupby("userId"):
        # Log-transformed activity accounts for diminishing returns and tames long-tail skew
        u_activity = float(np.log1p(len(g)))
        u_bias = float(g["rating"].mean() - global_avg_rating)
        liked_mids = g[g["rating"] >= 3.0]["movieId"].values
        u_genres = set()
        u_als_vecs = []
        for mid in liked_mids:
            if mid in movie_meta:
                u_genres.update(movie_meta[mid]["genres"])
                if movie_meta[mid]["als_vec"] is not None:
                    u_als_vecs.append(movie_meta[mid]["als_vec"])
        avg_als_vec = np.mean(u_als_vecs, axis=0) if u_als_vecs else None
        user_profiles[uid] = {
            "activity": u_activity,
            "bias": u_bias,
            "genres": u_genres,
            "als_vec": avg_als_vec,
        }
    return user_profiles
