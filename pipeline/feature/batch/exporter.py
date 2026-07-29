"""
pipeline/feature/batch/exporter.py
-----------------------------------
Exporters for writing precomputed item similarities and vector embeddings to Redis and Qdrant.
"""

import json
import logging
from typing import Dict, List, Any

from pipeline.feature.config import COLLECTION_NAME
from pipeline.feature.db import connect_redis, connect_qdrant

logger = logging.getLogger("BatchExporter")


def export_to_redis(similarities: Dict[int, List[Dict[str, Any]]], movie_meta: Dict[int, Dict[str, Any]]):
    """Write precomputed similarities & movie metadata to Redis."""
    r = connect_redis()
    if not r:
        logger.warning("Skipping Redis export: client unavailable.")
        return

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


def export_to_qdrant(movie_meta: Dict[int, Dict[str, Any]]):
    """Upsert movie vector embeddings & payload into Qdrant vector DB."""
    qdrant = connect_qdrant()
    if not qdrant:
        logger.warning("Skipping Qdrant export: client unavailable.")
        return

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
