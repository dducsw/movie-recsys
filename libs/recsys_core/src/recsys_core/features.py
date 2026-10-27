"""
recsys_core.features
--------------------
Standardized ranking feature schema and feature precomputation helpers.
"""

import re
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Standard 11-feature schema matching Stage 2 serving and continuous training.
#
# NOTE on `retrieval_score`: it is the *collaborative/ANN retrieval affinity*
# score.  At training time it is the iALS user-item dot product; at serving
# time it is the ANN similarity of the candidate (Qdrant cosine / Redis
# precomputed similarity).  Both are "how strongly this candidate matches the
# user's collaborative signal" and the model is trained on the same column
# name, so train/serve stay schema-compatible.  The previous name `als_score`
# was misleading because serving never had iALS factors in memory.
RANKING_FEATURES = [
    "popularity",
    "vote_average",
    "bayesian_rating",
    "genre_overlap",
    "genre_affinity_pct",
    "release_year",
    "recency_decay_score",
    "user_activity",
    "user_bias",
    "retrieval_score",
    "cb_score",
]

# Single TF-IDF vocab size shared by CT and batch (no skew).
TFIDF_MAX_FEATURES = 50

# Canonical genre vocabulary (MovieLens 19 genres).  Used by the MMR/λ entropy
# calculation, the cold-start ranking features and the ML bundle cold-start
# priors.  Always import it from here — never re-declare a genre list.
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX",
]
NUM_GENRES = len(ALL_GENRES)

# Reference year for the recency-decay feature.  Overridable so that
# back-tests stay reproducible across calendar years; defaults to the current
# year (never hard-code a literal year).
DEFAULT_REFERENCE_YEAR: Optional[int] = None

# Half-life of the recency decay in years (exp(-ln2/15 * years_ago)).
RECENCY_DECAY_HALFLIFE_YEARS = 15.0


def reference_year() -> int:
    """Reference year used by the recency-decay feature (current year by default)."""
    return DEFAULT_REFERENCE_YEAR if DEFAULT_REFERENCE_YEAR is not None else date.today().year


def bayesian_rating(vote_count: float, vote_average: float,
                    min_votes: float = 25.0, prior_mean: float = 5.0) -> float:
    """IMDb-style weighted rating: WR = (v/(v+m))*R + (m/(v+m))*C.

    Single source of truth — used by the CT feature builder and the serving
    fallback scoring so the two never drift.
    """
    v = float(vote_count)
    if v <= 0.0:
        return float(vote_average)
    m, c = float(min_votes), float(prior_mean)
    return float((v / (v + m)) * vote_average + (m / (v + m)) * c)


def build_ranking_row(meta: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, float]:
    """Unified 11-feature row. Shared by training and serving (100% parity)."""
    prof_genres = profile.get("genres", set()) or set()
    movie_genres = meta.get("genres", set()) or set()
    overlap = float(len(prof_genres.intersection(movie_genres)))
    genre_affinity_pct = float(overlap / max(len(prof_genres), 1))
    cb_score = genre_affinity_pct

    # Recency decay feature: exponential decay with a ~15 year half-life.
    rel_year = float(meta.get("release_year", 2010.0))
    years_ago = max(0.0, float(reference_year()) - rel_year)
    recency_decay = float(np.exp(-np.log(2.0) / RECENCY_DECAY_HALFLIFE_YEARS * years_ago))

    # Collaborative / ANN retrieval affinity.  Training passes iALS factors,
    # serving passes the ANN similarity directly via meta["retrieval_score"].
    retrieval_score = meta.get("retrieval_score")
    if retrieval_score is None:
        retrieval_score = 0.0
        if profile.get("als_vec") is not None and meta.get("als_vec") is not None:
            try:
                retrieval_score = float(np.dot(profile["als_vec"], meta["als_vec"]))
            except Exception:
                retrieval_score = 0.0

    return {
        "popularity": float(meta.get("popularity", 1.0)),
        "vote_average": float(meta.get("vote_average", 5.0)),
        "bayesian_rating": float(meta.get("bayesian_rating", meta.get("vote_average", 5.0))),
        "genre_overlap": overlap,
        "genre_affinity_pct": genre_affinity_pct,
        "release_year": rel_year,
        "recency_decay_score": recency_decay,
        "user_activity": float(profile.get("activity", 1.0)),
        "user_bias": float(profile.get("bias", 0.0)),
        "retrieval_score": float(retrieval_score),
        "cb_score": cb_score,
    }


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
    """Precompute movie metadata dictionary containing release year, genres, Bayesian rating, and ALS vector."""
    records = df_movies.to_dict("records")
    has_year = "release_year" in df_movies.columns

    # Bayesian weighted rating parameters (IMDb formula): WR = (v / (v + m)) * R + (m / (v + m)) * C
    # m = minimum votes threshold (25), C = global prior rating (~5.0)
    m_prior = 25.0
    c_prior = 5.0

    movie_meta = {}
    for r in records:
        mid = int(r["movieId"])
        g_str = str(r.get("genres", "") or "")
        g_set = set(g_str.split("|")) if g_str else set()
        rel_year = float(r["release_year"]) if has_year else float(extract_movie_release_year(r))
        vote_avg = float(r.get("vote_average", 5.0))
        vote_cnt = float(r.get("vote_count", r.get("popularity", 10.0)))

        movie_meta[mid] = {
            "popularity": float(r.get("popularity", 1.0)),
            "vote_average": vote_avg,
            "bayesian_rating": bayesian_rating(vote_cnt, vote_avg, m_prior, c_prior),
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
