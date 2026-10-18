"""
event_producer.py
-----------------
Non-blocking event producer. Uses a bounded queue + single worker
(no thread-per-request). Fail-open when Redis is down.
"""

import json
import logging
import queue
import threading
import time
from typing import Any

from app.services.redis_factory import get_redis

logger = logging.getLogger(__name__)

STREAM_KEY = "user-events-stream"
STREAM_DB = 0
_QUEUE: queue.Queue = queue.Queue(maxsize=1000)
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _send_batch(records: list[dict[str, Any]]) -> None:
    r = get_redis(db=STREAM_DB, timeout=1.0)
    if not r:
        return
    try:
        pipe = r.pipeline()
        for record in records:
            payload = {
                "userId": str(record.get("userId") or ""),
                "movieId": str(record.get("movieId") or ""),
                "timestamp": str(record.get("timestamp") or ""),
                "event_type": str(record.get("event_type") or ""),
                "session_id": str(record.get("session_id") or ""),
                "source": str(record.get("source") or "webapp"),
                "extra": json.dumps(record.get("extra") or {}, default=str),
            }
            pipe.xadd(STREAM_KEY, payload, maxlen=100000, approximate=True)
        pipe.execute()
    except Exception as exc:
        logger.warning("Failed to emit %d events: %s", len(records), exc)


def _worker() -> None:
    while True:
        batch = [_QUEUE.get()]
        try:
            while len(batch) < 50:
                batch.append(_QUEUE.get_nowait())
        except queue.Empty:
            pass
        try:
            _send_batch(batch)
        finally:
            for _ in batch:
                _QUEUE.task_done()
        if not batch:
            time.sleep(0.2)


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        t = threading.Thread(target=_worker, daemon=True, name="event-producer")
        t.start()
        _WORKER_STARTED = True


def emit(
    event_type: str,
    *,
    user_id: str = "anonymous",
    movie_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Enqueue event. Never blocks request, drops when full."""
    _ensure_worker()
    record = {
        "userId": user_id,
        "movieId": movie_id,
        "timestamp": time.time(),
        "event_type": event_type,
        "session_id": user_id,
        "source": "webapp",
        "extra": extra or {},
    }
    try:
        _QUEUE.put_nowait(record)
    except queue.Full:
        logger.warning("Event queue full, dropping '%s'", event_type)


# ── Event type constants ───────────────────────────────────────────────────────
class EventType:
    CLICK = "click"
    DWELL = "dwell"
    DETAIL_VIEW = "detail_view"
    MOVIE_SEARCH = "movie_search"
    TRENDING_BROWSE = "trending_browse"
    LATEST_BROWSE = "latest_browse"
    RECOMMENDATION_REQUEST = "recommendation_request"
    SIMILAR_MOVIE_REQUEST = "similar_movie_request"
    CHATBOT_MESSAGE = "chatbot_message"
    CHAT_PREFERENCE = "chat_preference"
    IMPRESSION = "impression"
    RATING = "rating"
    WATCHLIST = "watchlist"


def record_nearline_interaction(session_id: str, movie_id: int) -> None:
    """Record nearline interaction in Redis for instant session feed adaptation (TTL 10m)."""
    if not session_id or not movie_id:
        return
    r = get_redis(db=STREAM_DB, timeout=0.5)
    if r:
        try:
            key = f"session:{session_id}:recent"
            pipe = r.pipeline()
            pipe.lrem(key, 0, str(movie_id))
            pipe.lpush(key, str(movie_id))
            pipe.ltrim(key, 0, 19)
            pipe.expire(key, 600)  # 10 minutes session budget
            pipe.execute()
        except Exception as exc:
            logger.debug("Failed to record nearline interaction: %s", exc)


def get_nearline_interactions(session_id: str) -> list[int]:
    """Retrieve recent movie IDs interacted within current session."""
    if not session_id:
        return []
    r = get_redis(db=STREAM_DB, timeout=0.5)
    if r:
        try:
            key = f"session:{session_id}:recent"
            raw = r.lrange(key, 0, 19)
            return [int(mid) for mid in raw if str(mid).isdigit()]
        except Exception:
            pass
    return []
