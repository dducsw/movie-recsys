"""
pipeline/feature/streaming/processor.py
----------------------------------------
Core user interaction event processor and Redis pipeline updates for streaming pipeline.
"""

import json
import time
import logging
from typing import Dict, Any

from pipeline.feature.config import SCORE_WEIGHTS, REDIS_HOST, REDIS_PORT

logger = logging.getLogger("StreamingProcessor")


def apply_event_to_redis_pipeline(pipe, session_id: str, movie_id: Any, event_type: str, timestamp: float):
    """
    Append Redis pipeline commands for:
    1. Updating user's recent interactions list: user:{session_id}:recent
    2. Incrementing real-time trending score in sorted set: realtime:trending
    """
    if not session_id or not movie_id:
        return

    # 1. Recent interaction history list (keep last 50, TTL 7 days)
    user_key = f"user:{session_id}:recent"
    pipe.lpush(user_key, json.dumps({
        "movie_id": movie_id,
        "event": event_type,
        "ts": timestamp
    }))
    pipe.ltrim(user_key, 0, 49)
    pipe.expire(user_key, 86400 * 7)

    # 2. Real-time trending counter in Redis Sorted Set (TTL 24 hours)
    weight = SCORE_WEIGHTS.get(event_type, 1.0)
    pipe.zincrby("realtime:trending", weight, str(movie_id))
    pipe.expire("realtime:trending", 86400)


def process_event(event: Dict[str, Any], redis_client):
    """Process a single user interaction event (used by fallback consumer)."""
    if not redis_client:
        return

    session_id = event.get("session_id") or event.get("userId")
    movie_id = event.get("movieId")
    event_type = event.get("event_type")
    ts = event.get("timestamp") or time.time()

    if not session_id or not movie_id:
        return

    pipe = redis_client.pipeline()
    apply_event_to_redis_pipeline(pipe, session_id, movie_id, event_type, ts)
    pipe.execute()
