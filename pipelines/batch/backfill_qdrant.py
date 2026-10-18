"""
pipelines/batch/backfill_qdrant.py
---------------------------------
One-click backfill of pinned 384-dim semantic embeddings into Qdrant.

Usage:
    PYTHONPATH=. python pipelines/batch/backfill_qdrant.py [--limit N]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QdrantBackfill")


def main(limit: int = 0) -> None:
    """Load catalog and upsert semantic vectors."""
    from pipelines.feature_store.feature.batch.exporter import export_to_qdrant
    from pipelines.feature_store.feature.batch.model import load_datasets

    movies_df, _ = load_datasets()
    if movies_df is None or movies_df.empty:
        logger.error("Backfill aborted: no movies.")
        return
    if limit > 0:
        movies_df = movies_df.head(limit)
    logger.info("Backfilling %d movies to Qdrant (all-MiniLM-L6-v2, 384-dim)...",
                len(movies_df))
    export_to_qdrant(movies_df)
    logger.info("Backfill done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Qdrant semantic vectors.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only backfill first N movies (0 = all).")
    args = parser.parse_args()
    main(limit=args.limit)
