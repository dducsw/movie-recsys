import time
import threading
from typing import Any, Optional

class SimpleTTLCache:
    """Thread-safe in-memory TTL cache fallback for RecSys results."""
    def __init__(self, default_ttl_seconds: int = 300):
        self._cache = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            val, expire_time = item
            if time.time() > expire_time:
                del self._cache[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expire_time = time.time() + ttl
        with self._lock:
            self._cache[key] = (value, expire_time)

    def clear(self):
        with self._lock:
            self._cache.clear()

recsys_cache = SimpleTTLCache(default_ttl_seconds=300)
