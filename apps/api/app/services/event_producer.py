"""
event_producer.py
-----------------
Event producer singleton. Emit user-interaction events lên Redis stream "user-events-stream"
hoặc fallback logger. Non-blocking qua background thread.
"""

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Redis Event Stream Helper ──────────────────────────────────────────────────
_redis_client = None

def _get_redis():
    """Khởi tạo kết nối Redis cho event streaming."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client if _redis_client is not False else None

    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        client = redis.Redis(host=host, port=port, db=0, decode_responses=True, socket_timeout=1.0)
        client.ping()
        _redis_client = client
        logger.info("EventProducer connected to Redis stream at %s:%d", host, port)
    except Exception as exc:
        logger.warning("EventProducer Redis init failed (events fallback to logger): %s", exc)
        _redis_client = False

    return _redis_client if _redis_client is not False else None


# ── Public API ─────────────────────────────────────────────────────────────────
STREAM_KEY = "user-events-stream"


def _send_in_background(record: dict[str, Any]) -> None:
    try:
        r = _get_redis()
        if r:
            # Flatten extra for Redis stream entry
            payload = {
                "userId": str(record.get("userId") or ""),
                "movieId": str(record.get("movieId") or ""),
                "timestamp": str(record.get("timestamp") or ""),
                "event_type": str(record.get("event_type") or ""),
                "session_id": str(record.get("session_id") or ""),
                "source": str(record.get("source") or "webapp"),
                "extra": json.dumps(record.get("extra") or {}, default=str),
            }
            r.xadd(STREAM_KEY, payload, maxlen=100000, approximate=True)
        logger.debug("Emitted event '%s' for user '%s'", record.get("event_type"), record.get("userId"))
    except Exception as exc:
        logger.warning("Failed to emit event '%s': %s", record.get("event_type"), exc)


def emit(
    event_type: str,
    *,
    user_id: str = "anonymous",
    movie_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Gửi một event lên Kafka. Non-blocking (chạy ngầm), không làm chậm HTTP response.
    """
    record = {
        "userId":     user_id,
        "movieId":    movie_id,
        "timestamp":  time.time(),          # epoch float — align với simulator
        "event_type": event_type,
        "session_id": user_id,              # alias, giữ compat với simulator field
        "source":     "webapp",             # phân biệt real vs simulated data
        "extra":      extra or {},
    }
    threading.Thread(target=_send_in_background, args=(record,), daemon=True).start()


# ── Event type constants ───────────────────────────────────────────────────────
class EventType:
    # Align với simulator — dùng chung khi Spark join batch + stream
    CLICK                   = "click"               # ≈ simulator: click (từ listing)
    DETAIL_VIEW             = "detail_view"         # ≈ simulator: detail_view

    # Webapp-only events (không có trong simulator)
    MOVIE_SEARCH            = "movie_search"
    TRENDING_BROWSE         = "trending_browse"
    LATEST_BROWSE           = "latest_browse"
    RECOMMENDATION_REQUEST  = "recommendation_request"
    SIMILAR_MOVIE_REQUEST   = "similar_movie_request"
    CHATBOT_MESSAGE         = "chatbot_message"
    IMPRESSION              = "impression"
    RATING                  = "rating"
    WATCHLIST               = "watchlist"
