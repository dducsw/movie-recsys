"""
pipeline/feature/db.py
----------------------
Database connections and client initializers (Redis, Qdrant, PySpark with Delta Lake extension).
"""

import logging
from typing import Optional, Any
from pipeline.feature.config import (
    REDIS_HOST, REDIS_PORT,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
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
