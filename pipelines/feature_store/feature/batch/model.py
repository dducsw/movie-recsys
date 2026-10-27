"""pipeline/feature/batch/model.py
--------------------------------
Pandas & scikit-learn data loading, genre TF-IDF extraction,
and true implicit-ALS item factors (Hu-Koren-Volinsky confidence).
Zero Spark/Kafka dependency: batch runs on Pandas, streaming on Redis Streams.
"""

import os
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pipelines.feature_store.feature.config import (
    MOVIES_CSV, RATINGS_CSV,
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)

logger = logging.getLogger("BatchModel")


def load_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load movies and ratings DataFrames into Pandas (Postgres SQL with CSV fallback)."""
    movies_df = None
    ratings_df = None

    # 1. Try Postgres connection
    try:
        import psycopg2
        logger.info(f"Connecting to PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=3
        )
        movies_df = pd.read_sql("SELECT movieid AS \"movieId\", title, overview FROM movies", conn)
        # Fetch genres mapping if available
        try:
            genres_df = pd.read_sql("""
                SELECT mg.movie_id AS "movieId", STRING_AGG(g.name, '|') AS genres
                FROM movie_genres mg
                JOIN genres g ON mg.genre_id = g.id
                GROUP BY mg.movie_id
            """, conn)
            movies_df = movies_df.merge(genres_df, on="movieId", how="left")
        except Exception:
            pass
        conn.close()
        logger.info(f"Loaded {len(movies_df)} movies from PostgreSQL.")
    except Exception as e:
        logger.warning(f"PostgreSQL direct load skipped/failed ({e}). Falling back to CSV...")

    # 2. Fallback to CSV
    if movies_df is None or movies_df.empty:
        # Check potential paths
        candidate_paths = [
            MOVIES_CSV,
            os.path.join(os.path.dirname(MOVIES_CSV), "..", "crawler", "movies_crawled.csv"),
            os.path.join(os.path.dirname(MOVIES_CSV), "movies.csv"),
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                logger.info(f"Reading movies from CSV: {path}")
                movies_df = pd.read_csv(path)
                break

    if movies_df is not None:
        if "movieid" in movies_df.columns:
            movies_df.rename(columns={"movieid": "movieId"}, inplace=True)
        if "genres" not in movies_df.columns:
            # Only the legacy 3-column CSV (movieId, title, genres) may be
            # positional-renamed. Anything else (e.g. a Postgres load whose
            # genre join failed, carrying `overview` in 3rd place) must fail
            # loudly — silently turning overview text into "genres" corrupts
            # TF-IDF, similarity and Qdrant vectors for the whole catalog.
            legacy_csv = {"movieId", "title"}.issubset(movies_df.columns) \
                and len(movies_df.columns) == 3
            if not legacy_csv:
                raise ValueError(
                    "Movies source has no 'genres' column — refusing to guess one; "
                    f"got columns {list(movies_df.columns)}.")
            movies_df.columns = ["movieId", "title", "genres"] + list(movies_df.columns[3:])
        movies_df["genres"] = movies_df["genres"].fillna("(no genres listed)")

    # Load ratings CSV
    if os.path.exists(RATINGS_CSV):
        try:
            ratings_df = pd.read_csv(RATINGS_CSV)
            if "movieid" in ratings_df.columns:
                ratings_df.rename(columns={"movieid": "movieId"}, inplace=True)
            if "userid" in ratings_df.columns:
                ratings_df.rename(columns={"userid": "userId"}, inplace=True)
            logger.info(f"Loaded {len(ratings_df)} ratings from {RATINGS_CSV}.")
        except Exception as e:
            logger.warning(f"Failed to load ratings CSV: {e}")

    return movies_df, ratings_df


def extract_genre_tfidf_features(movies_df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """Compute TF-IDF genre vectors. Vocab size shared with CT (no skew)."""
    logger.info("Computing Genre TF-IDF vectors using Scikit-learn...")
    from recsys_core.features import TFIDF_MAX_FEATURES

    genres_series = movies_df["genres"].fillna("(no genres listed)").astype(str)
    # Replace pipe delimiter with spaces so TfidfVectorizer tokenizes them as separate words
    genres_text = genres_series.str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, token_pattern=r"(?u)\b[\w-]+\b")
    tfidf_matrix = tfidf.fit_transform(genres_text).toarray().astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_matrix = tfidf_matrix / norms

    # Ensure stable width matching shared constant
    if norm_matrix.shape[1] < TFIDF_MAX_FEATURES:
        pad_width = TFIDF_MAX_FEATURES - norm_matrix.shape[1]
        norm_matrix = np.pad(norm_matrix, ((0, 0), (0, pad_width)), mode='constant')

    return norm_matrix, tfidf.get_feature_names_out().tolist()


def train_als_model(ratings_df: Optional[pd.DataFrame]) -> Dict[int, np.ndarray]:
    """Train true iALS item factors (confidence c = 1 + alpha * r).

    Delegates entirely to ``recsys_core.retrieval.train_ials_factors`` — the
    core module owns both the implicit-ALS primary path and the
    TruncatedSVD fallback, so there is exactly one implementation offline
    and online (the previous local SVD copy drifted from the core one).
    """
    if ratings_df is None or ratings_df.empty:
        return {}
    from recsys_core.retrieval import IALSConfig, train_ials_factors

    factors = train_ials_factors(ratings_df, config=IALSConfig())
    logger.info("Collaborative factors computed for %d movies.", len(factors))
    return factors
