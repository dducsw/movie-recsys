"""
web_app/backend/app/services/feature_store.py
---------------------------------------------
Online Feature Store Service leveraging Feast with graceful fallback.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_feast_store = None
_feast_initialized = False


def get_feast_store():
    """Lazy initialize Feast FeatureStore singleton."""
    global _feast_store, _feast_initialized
    if _feast_initialized:
        return _feast_store

    _feast_initialized = True
    try:
        from feast import FeatureStore
        curr = os.path.dirname(os.path.abspath(__file__))
        repo_root = curr
        while repo_root and not os.path.exists(os.path.join(repo_root, "pipeline", "feature_store")):
            parent = os.path.dirname(repo_root)
            if parent == repo_root:
                break
            repo_root = parent

        feast_repo_path = os.path.join(repo_root, "pipeline", "feature_store")
        if os.path.exists(feast_repo_path):
            _feast_store = FeatureStore(repo_path=feast_repo_path)
            logger.info(f"Connected to Feast Feature Store at {feast_repo_path}")
        else:
            logger.warning(f"Feast repo not found at {feast_repo_path}. Using fallback.")
            _feast_store = None
    except Exception as e:
        logger.warning(f"Failed to initialize Feast Feature Store: {e}. Using fallback.")
        _feast_store = None

    return _feast_store


class FeatureStoreService:
    """Provides low-latency online feature retrieval for recommendation scoring."""

    @staticmethod
    def get_online_movie_features(movie_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch online features for candidate movie IDs from Feast (Redis online store).
        Returns mapping: {movie_id: {popularity, vote_average, release_year, vote_count}}
        """
        if not movie_ids:
            return {}

        store = get_feast_store()
        if not store:
            return {}

        # Quick circuit-breaker: verify Redis is reachable before invoking Feast to avoid long socket timeouts
        try:
            from app.services.recsys import get_redis_client
            if not get_redis_client():
                return {}
        except Exception:
            return {}

        try:
            entity_rows = [{"movieId": mid} for mid in movie_ids]
            features_to_fetch = [
                "movie_stats:popularity",
                "movie_stats:vote_average",
                "movie_stats:release_year",
                "movie_stats:vote_count",
            ]
            response = store.get_online_features(
                features=features_to_fetch,
                entity_rows=entity_rows
            ).to_dict()

            results = {}
            m_ids = response.get("movieId", [])
            pops = response.get("popularity", [])
            votes = response.get("vote_average", [])
            years = response.get("release_year", [])
            counts = response.get("vote_count", [])

            for idx, mid in enumerate(m_ids):
                results[mid] = {
                    "popularity": pops[idx] if idx < len(pops) and pops[idx] is not None else 0.0,
                    "vote_average": votes[idx] if idx < len(votes) and votes[idx] is not None else 0.0,
                    "release_year": years[idx] if idx < len(years) and years[idx] is not None else 2010,
                    "vote_count": counts[idx] if idx < len(counts) and counts[idx] is not None else 0,
                }
            return results
        except Exception as e:
            logger.warning(f"Feast online feature fetch error: {e}")
            return {}
