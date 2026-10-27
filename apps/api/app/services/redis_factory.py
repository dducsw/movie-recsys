"""Single Redis factory. One place for host/port/timeout/db."""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# A transient outage must degrade Redis for a bounded window, not forever —
# failures are retried after this TTL (same contract as Feast / the bundle).
_RETRY_SECONDS = 60.0

_lock = threading.Lock()
_clients: dict[int, object] = {}
_last_failure: dict[int, float] = {}


def _settings() -> tuple[str, int]:
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", 6379))
    return host, port


def get_redis(db: int = 0, timeout: float = 0.5):
    """Return shared Redis client for ``db``, or None if unavailable."""
    with _lock:
        client = _clients.get(db)
        if client is not None:
            return client
        if time.time() - _last_failure.get(db, 0.0) < _RETRY_SECONDS:
            return None
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
            logger.warning("Redis unavailable db=%s (%s); retrying in %.0fs",
                           db, exc, _RETRY_SECONDS)
            _last_failure[db] = time.time()
            return None


def reset_redis_for_tests() -> None:
    """Clear cached clients (tests only)."""
    with _lock:
        _clients.clear()
        _last_failure.clear()
