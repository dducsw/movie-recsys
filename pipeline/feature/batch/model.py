"""
pipeline/feature/batch/model.py
--------------------------------
Pandas & Scikit-learn data loading, genre TF-IDF feature extraction,
and collaborative item factor modeling (Zero PySpark dependency).
"""

import os
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.feature.config import (
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
        if "genres" not in movies_df.columns and len(movies_df.columns) >= 3:
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
    """Compute 20-dim TF-IDF feature vectors on movie genres using Scikit-Learn."""
    logger.info("Computing Genre TF-IDF vectors using Scikit-learn...")
    genres_series = movies_df["genres"].fillna("(no genres listed)").astype(str)
    # Replace pipe delimiter with spaces so TfidfVectorizer tokenizes them as separate words
    genres_text = genres_series.str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(max_features=20, token_pattern=r"(?u)\b[\w-]+\b")
    tfidf_matrix = tfidf.fit_transform(genres_text).toarray().astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_matrix = tfidf_matrix / norms

    # Ensure shape is (N, 20)
    if norm_matrix.shape[1] < 20:
        pad_width = 20 - norm_matrix.shape[1]
        norm_matrix = np.pad(norm_matrix, ((0, 0), (0, pad_width)), mode='constant')

    return norm_matrix, tfidf.get_feature_names_out().tolist()


def train_als_model(ratings_df: Optional[pd.DataFrame]) -> Dict[int, np.ndarray]:
    """Train Collaborative Filtering / item factor vectors if ratings available."""
    als_factors_map = {}
    if ratings_df is None or ratings_df.empty:
        return als_factors_map

    try:
        logger.info(f"Extracting item collaborative factors from {len(ratings_df)} ratings...")
        # Simple SVD / factor representation using scipy/numpy
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD

        user_ids = ratings_df["userId"].astype("category").cat.codes
        movie_ids = ratings_df["movieId"].astype("category").cat.codes
        unique_movies = ratings_df["movieId"].astype("category").cat.categories

        rating_vals = ratings_df["rating"].values.astype(np.float32)
        n_users = user_ids.max() + 1
        n_movies = movie_ids.max() + 1

        interaction_matrix = csr_matrix((rating_vals, (user_ids, movie_ids)), shape=(n_users, n_movies))

        n_components = min(20, min(n_users, n_movies) - 1)
        if n_components >= 2:
            svd = TruncatedSVD(n_components=20 if n_components >= 20 else n_components, random_state=42)
            item_factors = svd.fit_transform(interaction_matrix.T)

            # Pad to 20 if needed
            if item_factors.shape[1] < 20:
                item_factors = np.pad(item_factors, ((0, 0), (0, 20 - item_factors.shape[1])), mode='constant')

            for idx, m_id in enumerate(unique_movies):
                vec = item_factors[idx].astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                als_factors_map[int(m_id)] = vec

            logger.info(f"Collaborative SVD factors computed for {len(als_factors_map)} movies.")
    except Exception as e:
        logger.warning(f"Collaborative factor extraction skipped: {e}")

    return als_factors_map
