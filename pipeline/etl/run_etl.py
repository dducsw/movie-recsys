"""
pipeline/etl/run_etl.py
-----------------------
Main Lakehouse ETL Pipeline orchestrator (Bronze -> Silver -> Gold).
Runs entity extraction from PostgreSQL, applies Silver layer conformed cleaning,
aggregates Gold ML feature matrices, and syncs artifacts to SeaweedFS S3-compatible storage.
"""

import os
import sys
import time
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.etl.bronze.movies import extract_raw_movies
from pipeline.etl.bronze.genres import extract_raw_genres
from pipeline.etl.bronze.directors import extract_raw_directors
from pipeline.etl.bronze.actors import extract_raw_actors
from pipeline.etl.bronze.tags import extract_raw_tags
from pipeline.etl.bronze.events import extract_raw_events

from pipeline.etl.silver.movies import transform_silver_movies
from pipeline.etl.silver.events import transform_silver_events

from pipeline.etl.gold.movies import build_gold_movies_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LakehouseETL")


def run_lakehouse_pipeline():
    """Run full Medallion Lakehouse ETL Pipeline."""
    start_time = time.time()
    logger.info("==================================================")
    logger.info("STARTING MEDALLION LAKEHOUSE ETL PIPELINE")
    logger.info("==================================================")

    # 1. BRONZE LAYER (Raw Extraction)
    logger.info("\n--- [STEP 1/3] BRONZE LAYER: Extracting Raw Entities from PostgreSQL ---")
    extract_raw_movies()
    extract_raw_genres()
    extract_raw_directors()
    extract_raw_actors()
    extract_raw_tags()
    extract_raw_events()

    # 2. SILVER LAYER (Conformed & Cleaned Transformation)
    logger.info("\n--- [STEP 2/3] SILVER LAYER: Cleaning & Denormalizing Conformed Datasets ---")
    transform_silver_movies()
    transform_silver_events()

    # 3. GOLD LAYER (ML Feature Tables & Downstream Sync)
    logger.info("\n--- [STEP 3/3] GOLD LAYER: Building Feature Matrices & Syncing Storage ---")
    build_gold_movies_features()

    elapsed = round(time.time() - start_time, 2)
    logger.info("==================================================")
    logger.info(f"MEDALLION LAKEHOUSE ETL COMPLETED IN {elapsed}s")
    logger.info("==================================================")


if __name__ == "__main__":
    run_lakehouse_pipeline()
