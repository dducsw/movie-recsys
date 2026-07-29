"""
spark_batch.py
--------------
Batch pipeline entry point for Movie Recommender System.
Runs PySpark ALS & TF-IDF feature extraction, matrix item-item similarity,
computes offline evaluation metrics (HR@10, NDCG@10), and exports to Redis and Qdrant.
"""

import os
import sys
import logging

# Ensure project root is in sys.path when executed directly as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature.db import create_spark_session
from pipeline.feature.batch.model import load_spark_datasets, extract_genre_tfidf_features, train_als_model
from pipeline.feature.batch.similarity import compute_item_similarity_matrix
from pipeline.feature.batch.fallback import run_pandas_fallback_batch_pipeline
from pipeline.feature.batch.metrics import compute_offline_metrics
from pipeline.feature.batch.exporter import export_to_redis, export_to_qdrant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkBatchPipeline")


def run_batch_pipeline():
    """Main batch pipeline execution with PySpark & fallback."""
    logger.info("Starting Spark Batch Pipeline for RecSys...")
    spark = create_spark_session("RecSys-SparkBatchPipeline")

    similarities, movie_meta = {}, {}

    if spark:
        try:
            movies_sp_df, ratings_sp_df = load_spark_datasets(spark)
            if movies_sp_df is not None:
                norm_df = extract_genre_tfidf_features(movies_sp_df)
                als_factors_map = train_als_model(ratings_sp_df)
                similarities, movie_meta = compute_item_similarity_matrix(norm_df, als_factors_map)
            spark.stop()
        except Exception as e:
            logger.error(f"PySpark batch processing failed ({e}). Falling back to NumPy/Pandas...")
            similarities, movie_meta = run_pandas_fallback_batch_pipeline()
    else:
        similarities, movie_meta = run_pandas_fallback_batch_pipeline()

    if not similarities or not movie_meta:
        logger.error("Batch pipeline aborted due to empty result set.")
        return

    # Compute & Log Offline Evaluation Metrics
    compute_offline_metrics(similarities)

    # Export precomputed recommendations & vector embeddings
    export_to_redis(similarities, movie_meta)
    export_to_qdrant(movie_meta)

    logger.info("Spark Batch Pipeline completed successfully!")


if __name__ == "__main__":
    run_batch_pipeline()
