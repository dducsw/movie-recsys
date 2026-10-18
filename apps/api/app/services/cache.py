import json
import logging
from typing import Any, Optional

from app.services.redis_factory import get_redis

logger = logging.getLogger(__name__)

CACHE_DB = 1


def get_cache_client():
    """App object cache (Redis db=1). Fail-open."""
    return get_redis(db=CACHE_DB, timeout=0.5)


def cache_get(key: str) -> Optional[Any]:
    client = get_cache_client()
    if not client:
        return None
    try:
        val = client.get(key)
        if val:
            return json.loads(val)
    except Exception as e:
        logger.debug(f"Cache get error for {key}: {e}")
    return None


def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> bool:
    client = get_cache_client()
    if not client:
        return False
    try:
        client.setex(key, ttl_seconds, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.debug(f"Cache set error for {key}: {e}")
    return False


def cache_delete(key: str) -> bool:
    client = get_cache_client()
    if not client:
        return False
    try:
        client.delete(key)
        return True
    except Exception:
        return False


class RecsysCacheWrapper:
    """Wrapper providing .get() and .set() interface matching recsys service usage."""
    def get(self, key: str) -> Optional[Any]:
        return cache_get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        return cache_set(key, value, ttl_seconds=ttl_seconds)

    def delete(self, key: str) -> bool:
        return cache_delete(key)


recsys_cache = RecsysCacheWrapper()
