"""
pipeline/feature/batch/model.py
--------------------------------
PySpark data loading, ALS Collaborative Filtering model training,
and TF-IDF genre feature vector extraction.
"""

import os
import logging
from typing import Dict, Tuple, Any
import numpy as np

from pipeline.feature.config import (
    MOVIES_CSV, RATINGS_CSV,
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)

logger = logging.getLogger("BatchModel")


def load_spark_datasets(spark) -> Tuple[Any, Any]:
    """Load movies and ratings DataFrames into PySpark (Postgres JDBC with CSV fallback)."""
    from pyspark.sql.functions import col

    movies_sp_df = None
    ratings_sp_df = None

    jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    try:
        logger.info(f"Attempting PySpark JDBC load from Postgres: {jdbc_url}")
        movies_sp_df = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("dbtable", "movies")
            .option("user", POSTGRES_USER)
            .option("password", POSTGRES_PASSWORD)
            .option("driver", "org.postgresql.Driver")
            .load()
        )
        if "movieid" in movies_sp_df.columns:
            movies_sp_df = movies_sp_df.withColumnRenamed("movieid", "movieId")
        logger.info(f"Loaded movies from Postgres via PySpark. Count: {movies_sp_df.count()}")
    except Exception as e:
        logger.warning(f"PySpark JDBC load failed ({e}). Loading from CSV fallback...")

    if movies_sp_df is None:
        if not os.path.exists(MOVIES_CSV):
            logger.error(f"Movies CSV dataset not found at {MOVIES_CSV}")
            return None, None
        movies_sp_df = spark.read.csv(MOVIES_CSV, header=True, inferSchema=True)

    sp_cols = movies_sp_df.columns
    if "genres" not in sp_cols and len(sp_cols) >= 3:
        movies_sp_df = (
            movies_sp_df
            .withColumnRenamed(sp_cols[0], "movieId")
            .withColumnRenamed(sp_cols[1], "title")
            .withColumnRenamed(sp_cols[2], "genres")
        )
    elif "movieid" in sp_cols:
        movies_sp_df = movies_sp_df.withColumnRenamed("movieid", "movieId")

    if os.path.exists(RATINGS_CSV):
        ratings_sp_df = spark.read.csv(RATINGS_CSV, header=True, inferSchema=True)
        if "movieid" in ratings_sp_df.columns:
            ratings_sp_df = ratings_sp_df.withColumnRenamed("movieid", "movieId")
        if "userid" in ratings_sp_df.columns:
            ratings_sp_df = ratings_sp_df.withColumnRenamed("userid", "userId")

    return movies_sp_df, ratings_sp_df


def extract_genre_tfidf_features(movies_sp_df):
    """Compute PySpark ML Genre TF-IDF vectors."""
    from pyspark.sql.functions import col, when
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Normalizer

    logger.info("Computing Genre TF-IDF vectors using PySpark ML Pipeline...")
    movies_clean = movies_sp_df.withColumn(
        "genres_clean",
        when(col("genres").isNotNull(), col("genres")).otherwise("(no genres listed)")
    )

    tokenizer = Tokenizer(inputCol="genres_clean", outputCol="words")
    words_df = tokenizer.transform(movies_clean)

    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=20)
    tf_df = hashing_tf.transform(words_df)

    idf = IDF(inputCol="raw_features", outputCol="idf_features")
    idf_model = idf.fit(tf_df)
    tfidf_df = idf_model.transform(tf_df)

    normalizer = Normalizer(inputCol="idf_features", outputCol="norm_features", p=2.0)
    norm_df = normalizer.transform(tfidf_df)
    return norm_df


def train_als_model(ratings_sp_df) -> Dict[int, np.ndarray]:
    """Train PySpark ALS Collaborative Filtering model and return item factor embeddings."""
    from pyspark.ml.recommendation import ALS

    als_factors_map = {}
    if ratings_sp_df is None:
        return als_factors_map

    try:
        logger.info("Training PySpark ALS Collaborative Filtering model...")
        als = ALS(
            rank=20,
            maxIter=10,
            regParam=0.1,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        als_model = als.fit(ratings_sp_df)
        item_factors = als_model.itemFactors.collect()
        for row in item_factors:
            m_id = int(row["id"])
            vec = np.array(row["features"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            als_factors_map[m_id] = vec
        logger.info(f"PySpark ALS trained successfully. Extracted factor vectors for {len(als_factors_map)} movies.")
    except Exception as e:
        logger.warning(f"PySpark ALS training failed or skipped: {e}")

    return als_factors_map
