import os
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client = None


def get_cache_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client if _redis_client is not False else None

    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        client = redis.Redis(
            host=host,
            port=port,
            db=1,  # use db 1 for app object caching
            decode_responses=True,
            socket_timeout=0.5,
            socket_connect_timeout=0.5
        )
        client.ping()
        _redis_client = client
        logger.info(f"Connected to Redis Cache at {host}:{port}")
    except Exception as e:
        logger.warning(f"Redis Cache unavailable ({e}), caching disabled.")
        _redis_client = False

    return _redis_client if _redis_client is not False else None


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
