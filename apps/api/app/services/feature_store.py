"""
web_app/backend/app/services/feature_store.py
---------------------------------------------
Online Feature Store Service leveraging Feast with graceful fallback.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_feast_store: Optional[Any] = None
_feast_lock = threading.Lock()

# A transient failure (e.g. Redis/Redis-backed repo briefly unavailable) must
# degrade Feast for a bounded window, not forever — so unlike the old
# "set the initialized flag first" pattern, failures are retried after a TTL.
_FEAST_RETRY_SECONDS = 60.0
_FEAST_LAST_FAILURE = 0.0


def get_feast_store():
    """Lazily initialize the Feast FeatureStore singleton (retry-safe, locked)."""
    global _feast_store, _FEAST_LAST_FAILURE

    if _feast_store is not None:
        return _feast_store

    with _feast_lock:
        if _feast_store is not None:
            return _feast_store
        if _FEAST_LAST_FAILURE and (time.time() - _FEAST_LAST_FAILURE) < _FEAST_RETRY_SECONDS:
            return None

        try:
            from feast import FeatureStore

            curr = os.path.dirname(os.path.abspath(__file__))
            repo_root = curr
            while repo_root and not (
                os.path.exists(os.path.join(repo_root, "pipelines", "feature_store", "feast_repo"))
                or os.path.exists(os.path.join(repo_root, ".git"))
            ):
                parent = os.path.dirname(repo_root)
                if parent == repo_root:
                    break
                repo_root = parent

            feast_repo_path = os.path.join(repo_root, "pipelines", "feature_store", "feast_repo")
            _feast_store = FeatureStore(repo_path=feast_repo_path)
            logger.info("Connected to Feast Feature Store at %s", feast_repo_path)
        except Exception as e:
            logger.warning("Failed to initialize Feast Feature Store: %s. Retrying in %.0fs.",
                           e, _FEAST_RETRY_SECONDS)
            _feast_store = None
            _FEAST_LAST_FAILURE = time.time()
            return None

        return _feast_store


def reset_feast_state() -> None:
    """Test hook: drop the cached Feast store so state is not shared across tests."""
    global _feast_store, _FEAST_LAST_FAILURE
    with _feast_lock:
        _feast_store = None
        _FEAST_LAST_FAILURE = 0.0


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
            from app.services.redis_factory import get_redis

            if not get_redis(db=0, timeout=0.5):
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
