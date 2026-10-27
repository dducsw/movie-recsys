"""
pipeline/feature/db.py
----------------------
Database connections and client initializers (Redis, Qdrant).
"""

import logging
import os
from typing import Any

from pipelines.feature_store.feature.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_HOST,
    QDRANT_PORT,
    REDIS_HOST,
    REDIS_PORT,
)

logger = logging.getLogger("PipelineDB")


def connect_redis() -> Any | None:
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


def connect_qdrant() -> Any | None:
    """Connect to Qdrant and ensure collection matches pinned embedding dim.

    A dimension mismatch is a hard error by default — silently wiping the
    collection would destroy the serving index. Set QDRANT_RECREATE=1 to
    explicitly allow destructive recreation (e.g. after re-embedding).
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME in collections:
            dim = client.get_collection(COLLECTION_NAME).config.params.vectors.size
            if dim != EMBEDDING_DIM:
                if os.getenv("QDRANT_RECREATE", "").lower() in ("1", "true"):
                    logger.warning("Qdrant dim mismatch (%s != %s) — recreating "
                                   "collection (QDRANT_RECREATE).", dim, EMBEDDING_DIM)
                    client.delete_collection(COLLECTION_NAME)
                    collections.remove(COLLECTION_NAME)
                else:
                    raise RuntimeError(
                        f"Qdrant collection '{COLLECTION_NAME}' has dim {dim} "
                        f"but the pinned embedding dim is {EMBEDDING_DIM}. "
                        "Fix the collection or set QDRANT_RECREATE=1.")
        if COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})")
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")
        return None
