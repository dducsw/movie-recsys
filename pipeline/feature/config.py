"""
pipeline/feature/config.py
--------------------------
Central configuration settings and environment variables for batch and streaming pipelines.
"""

import os

# Data Paths
PIPELINE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data", "ml-latest-small"))

MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

# Redis Online Feature Store
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Qdrant Vector Database
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "movies")

# Postgres Database
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5435))
POSTGRES_DB = os.getenv("POSTGRES_DB", "movie_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")

# PySpark Master
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Kafka Streaming
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "user-events")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "recsys-streaming-group")

# Event Score Weights for Real-time Trending
SCORE_WEIGHTS = {
    "like": 3.0,
    "click": 1.5,
    "detail_view": 1.0,
    "similar_movie_request": 0.8,
    "recommendation_request": 0.5,
}
