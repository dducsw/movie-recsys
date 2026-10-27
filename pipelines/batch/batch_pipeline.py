"""
batch_pipeline (lean batch)
---------------------------
Pandas + scikit-learn offline pipeline (no Spark):
TF-IDF genres + true iALS -> Redis similarities + Qdrant semantic vectors.
"""

from __future__ import annotations

import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipelines.feature_store.feature.batch.exporter import (
    export_to_qdrant,
    export_to_redis,
)
from pipelines.feature_store.feature.batch.metrics import compute_offline_metrics
from pipelines.feature_store.feature.batch.model import (
    extract_genre_tfidf_features,
    load_datasets,
    train_als_model,
)
from pipelines.feature_store.feature.batch.similarity import (
    compute_item_similarity_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchPipeline")


def run_batch_pipeline() -> None:
    """Run lean batch feature pipeline. Raises on any critical failure."""
    logger.info("Starting lean batch feature pipeline...")
    movies_df, ratings_df = load_datasets()
    if movies_df is None or movies_df.empty:
        raise RuntimeError("Batch aborted: no movies data.")

    logger.info("Loaded %d movies.", len(movies_df))
    genre_vectors, _ = extract_genre_tfidf_features(movies_df)
    als_map = train_als_model(ratings_df)
    similarities, movie_meta = compute_item_similarity_matrix(
        movies_df, genre_vectors, als_map)
    if not similarities or not movie_meta:
        raise RuntimeError("Batch aborted: empty result set.")

    compute_offline_metrics(similarities)
    # Exports are critical: a failure propagates so the scheduler sees exit 1
    # instead of a half-written offline store silently "succeeding".
    export_to_redis(similarities, movie_meta)
    export_to_qdrant(movies_df)  # 384-dim semantic vectors

    from pipelines.feature_store.feast_repo.materialize import run_materialization

    run_materialization()
    logger.info("Batch pipeline completed.")


def main() -> None:
    try:
        run_batch_pipeline()
    except Exception:
        logger.exception("Batch pipeline FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
