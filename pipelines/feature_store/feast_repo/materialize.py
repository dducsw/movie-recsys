"""
pipeline/feature_store/materialize.py
--------------------------------------
Synchronizes offline features from PostgreSQL into the Redis online feature store via Feast.
Can be executed standalone or called from the batch feature pipeline.
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

# Ensure project root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeastMaterialize")


def run_materialization():
    """Apply definitions and materialize features to Redis."""
    try:
        from feast import FeatureStore
        from pipelines.feature_store.feast_repo.features import (
            movie_entity, user_entity, movie_stats_view, user_stats_view
        )
        repo_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Initializing Feast FeatureStore from repo: {repo_path}")
        store = FeatureStore(repo_path=repo_path)

        logger.info("Applying Feast entity and feature view definitions to registry...")
        store.apply(objects=[movie_entity, user_entity, movie_stats_view, user_stats_view])

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=730) # past 2 years

        logger.info(f"Materializing features to Redis online store from {start_date} to {end_date}...")
        store.materialize(start_date=start_date, end_date=end_date)
        logger.info("Feast feature materialization to Redis completed successfully!")
        return True
    except Exception as e:
        logger.warning(f"Feast materialization encounter: {e}. (Will continue with pipeline)")
        return False


if __name__ == "__main__":
    run_materialization()
