"""
spark_batch.py (batch_pipeline)
--------------------------------
Lean Batch pipeline entry point for Movie Recommender System.
Runs Scikit-Learn TF-IDF genre feature extraction & Collaborative Factor extraction,
computes item-item similarity matrix and offline evaluation metrics (HR@10, NDCG@10),
and exports precomputed features directly to Redis and Qdrant (Zero Spark overhead).
"""

import os
import sys
import logging

# Ensure project root is in sys.path when executed directly as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipelines.feature_store.feature.batch.model import load_datasets, extract_genre_tfidf_features, train_als_model
from pipelines.feature_store.feature.batch.similarity import compute_item_similarity_matrix
from pipelines.feature_store.feature.batch.metrics import compute_offline_metrics
from pipelines.feature_store.feature.batch.exporter import export_to_redis, export_to_qdrant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchPipeline")


def run_batch_pipeline():
    """Main batch pipeline execution with Pandas & Scikit-learn."""
    logger.info("Starting Lean Batch Feature Pipeline for RecSys...")

    movies_df, ratings_df = load_datasets()
    if movies_df is None or movies_df.empty:
        logger.error("Batch pipeline aborted: No movies data found.")
        return

    logger.info(f"Loaded {len(movies_df)} movies for feature processing.")
    genre_vectors, feature_names = extract_genre_tfidf_features(movies_df)
    als_factors_map = train_als_model(ratings_df)

    similarities, movie_meta = compute_item_similarity_matrix(movies_df, genre_vectors, als_factors_map)

    if not similarities or not movie_meta:
        logger.error("Batch pipeline aborted due to empty result set.")
        return

    # Compute & Log Offline Evaluation Metrics
    compute_offline_metrics(similarities)

    # Export precomputed recommendations & vector embeddings
    export_to_redis(similarities, movie_meta)
    export_to_qdrant(movie_meta)

    # Synchronize Features with Feast Feature Store
    try:
        from pipelines.feature_store.feast_repo.materialize import run_materialization
        run_materialization()
    except Exception as e:
        logger.warning(f"Skipping Feast materialization: {e}")

    logger.info("Batch Feature Pipeline completed successfully!")


if __name__ == "__main__":
    run_batch_pipeline()
