"""
spark_batch.py
--------------
Batch pipeline for Movie Recommender System.
Calculates item latent vectors (PySpark ALS Collaborative Filtering)
and genre feature vectors (PySpark ML TF-IDF), matrix item-item similarity.
Exports precomputed results to Redis (Online Feature Cache) and Qdrant (Vector DB).
"""

import json
import logging
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkBatchPipeline")

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ml-latest-small")
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "movies"

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5435))
POSTGRES_DB = os.getenv("POSTGRES_DB", "movie_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")


def connect_redis():
    """Connect to Redis client."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return r
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        return None


def connect_qdrant():
    """Connect to Qdrant client."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=20, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")
        return None


def create_spark_session():
    """Initialize PySpark Session for Batch processing."""
    try:
        # Windows compatibility patch for PySpark py4j socketserver
        import socketserver
        if not hasattr(socketserver, "UnixStreamServer"):
            socketserver.UnixStreamServer = object

        from pyspark.sql import SparkSession
        logger.info(f"Initializing PySpark Session (Master: {SPARK_MASTER})...")
        builder = (
            SparkSession.builder
            .appName("RecSys-SparkBatchPipeline")
            .master(SPARK_MASTER)
            .config("spark.driver.memory", "2g")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        )
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info("PySpark Session created successfully.")
        return spark
    except Exception as e:
        logger.warning(f"PySpark initialization failed: {e}")
        return None


def run_pyspark_batch_pipeline(spark) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    """
    Core PySpark Batch Pipeline:
    1. Load movies & ratings datasets via PySpark.
    2. Fit PySpark ALS Collaborative Filtering model for latent factor embeddings.
    3. Generate PySpark ML Genre TF-IDF vectors.
    4. Compute matrix item-item cosine similarity.
    """
    import numpy as np
    from pyspark.sql.functions import col, split, coalesce, lit, concat_ws, when
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Normalizer

    logger.info("--- Running PySpark Batch Pipeline ---")

    # 1. Load Data with PySpark
    movies_sp_df = None
    ratings_sp_df = None

    # Try loading from Postgres JDBC first
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
            return {}, {}
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

    # 2. PySpark ML: Genre TF-IDF Feature Extraction
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

    # 3. PySpark ALS Collaborative Filtering Model
    als_factors_map = {}
    if ratings_sp_df is not None:
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

    # 4. Consolidate Vectors & Calculate Cosine Similarities
    logger.info("Extracting movie vectors & calculating similarity matrix...")
    movie_rows = norm_df.select("movieId", "title", "genres", "norm_features").collect()

    m_ids = []
    vectors = []
    movie_meta = {}

    for row in movie_rows:
        m_id = int(row["movieId"])
        title = str(row["title"]) if row["title"] else ""
        genres = str(row["genres"]) if row["genres"] else ""
        
        # Dense genre vector from PySpark DenseVector / SparseVector
        spark_vec = row["norm_features"].toArray() if hasattr(row["norm_features"], "toArray") else np.zeros(20)
        
        # Combine with ALS vector if present
        if m_id in als_factors_map:
            combined = 0.5 * spark_vec + 0.5 * als_factors_map[m_id]
            norm = np.linalg.norm(combined)
            final_vec = (combined / norm).tolist() if norm > 0 else spark_vec.tolist()
        else:
            final_vec = spark_vec.tolist()

        # Ensure exact size 20
        if len(final_vec) < 20:
            final_vec.extend([0.0] * (20 - len(final_vec)))
        else:
            final_vec = final_vec[:20]

        m_ids.append(m_id)
        vectors.append(final_vec)
        movie_meta[m_id] = {
            "title": title,
            "genres": genres,
            "vector": final_vec
        }

    matrix = np.array(vectors, dtype=np.float32)  # Shape (N, 20)
    N = len(m_ids)
    similarities = defaultdict(list)

    # Chunked dot product to prevent NxN memory allocation OOM on large datasets
    chunk_size = 500
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = matrix[start_idx:end_idx]
        sim_chunk = np.dot(chunk, matrix.T)

        for i_local in range(end_idx - start_idx):
            i_global = start_idx + i_local
            id_a = m_ids[i_global]
            scores = sim_chunk[i_local]
            top_indices = np.argpartition(scores, -min(25, N))[-min(25, N):]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for idx in top_indices:
                if idx == i_global:
                    continue
                score = float(scores[idx])
                if score > 0.1:
                    similarities[id_a].append({"movieId": m_ids[idx], "score": round(score, 4)})
                if len(similarities[id_a]) >= 20:
                    break

    return similarities, movie_meta



def run_pandas_fallback_batch_pipeline() -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    """Fallback NumPy/Pandas Batch Pipeline if PySpark runtime is unavailable."""
    import numpy as np
    logger.info("--- Running Fallback Pandas/NumPy Batch Pipeline ---")

    if not os.path.exists(MOVIES_CSV):
        logger.error(f"Movies dataset not found at {MOVIES_CSV}")
        return {}, {}

    movies_df = pd.read_csv(MOVIES_CSV)
    if "genres" not in movies_df.columns and len(movies_df.columns) >= 3:
        movies_df.columns = ["movieId", "title", "genres"] + list(movies_df.columns[3:])
    elif "movieid" in movies_df.columns:
        movies_df.rename(columns={"movieid": "movieId"}, inplace=True)

    genre_set = set()
    for raw_genres in movies_df["genres"].dropna():
        for g in str(raw_genres).split("|"):
            if g and g != "(no genres listed)":
                genre_set.add(g)
    genre_list = sorted(list(genre_set))[:20]

    m_ids = []
    vectors = []
    movie_meta = {}

    for _, row in movies_df.iterrows():
        m_id = int(row["movieId"])
        genres_str = str(row["genres"]) if pd.notna(row["genres"]) else ""
        title = str(row["title"]) if pd.notna(row["title"]) else ""

        vec = [0.0] * 20
        if genres_str and genres_str != "(no genres listed)":
            m_genres = set(genres_str.split("|"))
            for idx, g in enumerate(genre_list):
                if g in m_genres:
                    vec[idx] = 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]

        m_ids.append(m_id)
        vectors.append(vec)
        movie_meta[m_id] = {
            "title": title,
            "genres": genres_str,
            "vector": vec
        }

    matrix = np.array(vectors, dtype=np.float32)
    N = len(m_ids)
    similarities = defaultdict(list)

    chunk_size = 500
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk = matrix[start_idx:end_idx]
        sim_chunk = np.dot(chunk, matrix.T)

        for i_local in range(end_idx - start_idx):
            i_global = start_idx + i_local
            id_a = m_ids[i_global]
            scores = sim_chunk[i_local]
            top_indices = np.argpartition(scores, -min(25, N))[-min(25, N):]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            for idx in top_indices:
                if idx == i_global:
                    continue
                score = float(scores[idx])
                if score > 0.1:
                    similarities[id_a].append({"movieId": m_ids[idx], "score": round(score, 4)})
                if len(similarities[id_a]) >= 20:
                    break

    return similarities, movie_meta


def compute_offline_metrics(similarities: Dict[int, List[Dict[str, Any]]], ratings_csv: str) -> Dict[str, float]:
    """Compute Hit Ratio @ 10 and NDCG @ 10 on user interaction history using Leave-One-Out split."""
    if not os.path.exists(ratings_csv):
        return {"HR@10": 0.0, "NDCG@10": 0.0}
    try:
        df = pd.read_csv(ratings_csv)
        if "movieid" in df.columns:
            df.rename(columns={"movieid": "movieId", "userid": "userId"}, inplace=True)

        if "timestamp" in df.columns:
            df = df.sort_values(["userId", "timestamp"])

        user_groups = df.groupby("userId")
        hits = 0
        ndcgs = 0.0
        total_eval_users = 0

        for user_id, group in user_groups:
            if len(group) < 2:
                continue

            target_movie = int(group.iloc[-1]["movieId"])
            train_movies = set(group.iloc[:-1]["movieId"].astype(int))

            candidate_scores = defaultdict(float)
            for prev_m in train_movies:
                for sim in similarities.get(prev_m, []):
                    c_id = sim["movieId"]
                    if c_id not in train_movies:
                        candidate_scores[c_id] += sim["score"]

            if not candidate_scores:
                continue

            top_recs = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)[:10]

            total_eval_users += 1
            if target_movie in top_recs:
                hits += 1
                rank = top_recs.index(target_movie) + 1
                ndcgs += 1.0 / math.log2(rank + 1)

        hr_10 = round(hits / total_eval_users, 4) if total_eval_users > 0 else 0.0
        ndcg_10 = round(ndcgs / total_eval_users, 4) if total_eval_users > 0 else 0.0
        logger.info(f"--- Offline Evaluation Metrics ({total_eval_users} users evaluated) --- -> HR@10: {hr_10}, NDCG@10: {ndcg_10}")
        return {"HR@10": hr_10, "NDCG@10": ndcg_10}
    except Exception as e:
        logger.warning(f"Could not compute offline evaluation metrics: {e}")
        return {"HR@10": 0.0, "NDCG@10": 0.0}


def run_batch_pipeline():
    """Main batch pipeline execution with PySpark & fallback."""
    logger.info("Starting Spark Batch Pipeline for RecSys...")
    spark = create_spark_session()

    if spark:
        try:
            similarities, movie_meta = run_pyspark_batch_pipeline(spark)
            spark.stop()
        except Exception as e:
            logger.error(f"PySpark batch processing failed ({e}). Falling back to NumPy/Pandas...")
            similarities, movie_meta = run_pandas_fallback_batch_pipeline()
    else:
        similarities, movie_meta = run_pandas_fallback_batch_pipeline()

    if not similarities or not movie_meta:
        logger.error("Batch pipeline aborted due to empty result set.")
        return

    # Compute & Log Offline Metrics
    compute_offline_metrics(similarities, RATINGS_CSV)

    # 1. Export precomputed recommendations & metadata to Redis
    r = connect_redis()
    if r:
        logger.info("Writing precomputed similarities & metadata to Redis...")
        pipe = r.pipeline()
        for m_id, sim_list in similarities.items():
            pipe.set(f"movie:{m_id}:similar", json.dumps(sim_list))

        for m_id, meta in movie_meta.items():
            pipe.set(f"movie:{m_id}:meta", json.dumps({
                "title": meta["title"],
                "genres": meta["genres"]
            }))
        pipe.execute()
        logger.info(f"Successfully cached precomputed similarity for {len(similarities)} movies in Redis.")

    # 2. Export vector embeddings & payload to Qdrant
    qdrant = connect_qdrant()
    if qdrant:
        try:
            from qdrant_client.models import PointStruct
            logger.info("Upserting movie vector embeddings into Qdrant...")
            points = []
            for m_id, meta in movie_meta.items():
                points.append(PointStruct(
                    id=m_id,
                    vector=meta["vector"],
                    payload={"title": meta["title"], "genres": meta["genres"]}
                ))
            batch_size = 500
            for i in range(0, len(points), batch_size):
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i + batch_size])
            logger.info(f"Successfully upserted {len(points)} vectors into Qdrant collection '{COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")

    logger.info("Spark Batch Pipeline completed successfully!")


if __name__ == "__main__":
    run_batch_pipeline()
