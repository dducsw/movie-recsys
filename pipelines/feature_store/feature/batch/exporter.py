"""pipeline/feature/batch/exporter.py
-----------------------------------
Exporters: Redis precomputed similarities + Qdrant semantic embeddings.

Batch jobs must fail loudly: a half-written offline store that silently
"succeeds" leaves serving indices stale and unexaminable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from pipelines.feature_store.feature.config import COLLECTION_NAME
from pipelines.feature_store.feature.db import connect_qdrant, connect_redis

logger = logging.getLogger("BatchExporter")

# Similarity caches expire so movies that leave the catalog do not linger
# forever (stale keys were invisible before — nothing ever overwrote them).
_SIM_TTL_SECONDS = 7 * 24 * 3600


def export_to_redis(
    similarities: dict[int, list[dict[str, Any]]],
    movie_meta: dict[int, dict[str, Any]],
) -> None:
    """Write precomputed similarities & minimal metadata to Redis (atomic)."""
    r = connect_redis()
    if not r:
        raise RuntimeError("Redis unavailable — batch export aborted.")
    logger.info("Writing similarities & metadata to Redis...")
    pipe = r.pipeline()
    for m_id, sim_list in similarities.items():
        pipe.setex(f"movie:{m_id}:similar", _SIM_TTL_SECONDS, json.dumps(sim_list))
    for m_id, meta in movie_meta.items():
        pipe.setex(f"movie:{m_id}:meta", _SIM_TTL_SECONDS,
                   json.dumps({"title": meta["title"], "genres": meta["genres"]}))
    pipe.execute()
    logger.info("Cached similarity for %d movies in Redis.", len(similarities))


def export_to_qdrant(
    movies_df: pd.DataFrame | dict[int, dict[str, Any]] | None = None,
    movie_meta: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Upsert pinned 384-dim semantic embeddings into Qdrant.

    Accepts either a movies DataFrame (preferred) or a legacy meta dict.
    Raises on failure: a silent empty export leaves the old index serving
    stale data with no alert.
    """
    qdrant = connect_qdrant()
    if not qdrant:
        raise RuntimeError("Qdrant unavailable — batch export aborted.")
    from qdrant_client.models import PointStruct
    from recsys_core.embeddings import build_movie_texts, encode_texts

    if isinstance(movies_df, pd.DataFrame):
        ids = [int(m) for m in movies_df["movieId"].tolist()]
        texts = build_movie_texts(movies_df)
        payloads = [
            {"title": str(r.get("title", "")),
             "genres": str(r.get("genres", ""))}
            for _, r in movies_df.iterrows()
        ]
    elif isinstance(movie_meta, dict) and movie_meta:
        ids = list(movie_meta.keys())
        texts = [f"{v.get('title', '')}. Genres: {v.get('genres', '')}"
                 for v in movie_meta.values()]
        payloads = [{"title": v.get("title", ""), "genres": v.get("genres", "")}
                    for v in movie_meta.values()]
    elif isinstance(movies_df, dict) and movies_df:
        # Back-compat: export_to_qdrant(movie_meta) positional call
        ids = list(movies_df.keys())
        texts = [f"{v.get('title', '')}. Genres: {v.get('genres', '')}"
                 for v in movies_df.values()]
        payloads = [{"title": v.get("title", ""), "genres": v.get("genres", "")}
                    for v in movies_df.values()]
    else:
        logger.warning("Skipping Qdrant export: no movies provided.")
        return

    vectors = encode_texts(texts)
    points = [PointStruct(id=mid, vector=vec.tolist(), payload=pay)
              for mid, vec, pay in zip(ids, vectors, payloads)]
    for i in range(0, len(points), 500):
        qdrant.upsert(collection_name=COLLECTION_NAME,
                      points=points[i:i + 500])
    logger.info("Upserted %d vectors into Qdrant '%s'.",
                len(points), COLLECTION_NAME)
