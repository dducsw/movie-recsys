"""
pipeline/feature/db.py
----------------------
Database connections and client initializers (Redis, Qdrant, PySpark with Delta Lake extension).
"""

import logging
from typing import Optional, Any
from pipeline.feature.config import (
    REDIS_HOST, REDIS_PORT,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    SPARK_MASTER
)

logger = logging.getLogger("PipelineDB")


def connect_redis() -> Optional[Any]:
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


def connect_qdrant() -> Optional[Any]:
    """Connect to Qdrant client and ensure collection exists."""
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


def create_spark_session(app_name: str = "RecSys-SparkPipeline") -> Optional[Any]:
    """Initialize PySpark Session with Delta Lake extension."""
    try:
        # Windows compatibility patch for PySpark py4j socketserver
        import socketserver
        if not hasattr(socketserver, "UnixStreamServer"):
            socketserver.UnixStreamServer = object

        from pyspark.sql import SparkSession
        logger.info(f"Initializing PySpark Session '{app_name}' with Delta Lake (Master: {SPARK_MASTER})...")
        builder = (
            SparkSession.builder
            .appName(app_name)
            .master(SPARK_MASTER)
            .config("spark.driver.memory", "2g")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        logger.info("PySpark Session created successfully with Delta Lake support.")
        return spark
    except Exception as e:
        logger.warning(f"PySpark initialization failed: {e}")
        return None
