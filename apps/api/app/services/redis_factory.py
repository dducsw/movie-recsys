"""Single Redis factory. One place for host/port/timeout/db."""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_clients: dict[int, object] = {}


def _settings() -> tuple[str, int]:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", 6379))
    return host, port


def get_redis(db: int = 0, timeout: float = 0.5):
    """Return shared Redis client for db, or None if unavailable."""
    with _lock:
        if db in _clients:
            client = _clients[db]
            return None if client is False else client
        try:
            import redis

            host, port = _settings()
            client = redis.Redis(
                host=host, port=port, db=db, decode_responses=True,
                socket_timeout=timeout, socket_connect_timeout=timeout,
            )
            client.ping()
            _clients[db] = client
            logger.info("Redis connected db=%s at %s:%s", db, host, port)
            return client
        except Exception as exc:
            logger.warning("Redis unavailable db=%s (%s)", db, exc)
            _clients[db] = False
            return None


def reset_redis_for_tests() -> None:
    """Clear cached clients (tests only)."""
    with _lock:
        _clients.clear()
