"""
pipeline/feature/streaming/redis_stream_job.py
-----------------------------------------------
Lightweight Real-time Streaming Consumer using Redis Streams.
Consumes user interaction events from Redis stream ('user-events-stream'),
calculates real-time trending scores, and updates online user profiles in Redis.
"""

import json
import time
import logging
from typing import Optional

from pipelines.feature_store.feature.config import (
    REDIS_HOST, REDIS_PORT, REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP
)
from pipelines.feature_store.feature.db import connect_redis
from pipelines.feature_store.feature.streaming.processor import process_event

logger = logging.getLogger("RedisStreamingJob")


def run_redis_streaming_consumer():
    """Consume events from Redis Stream and update online user/trending features."""
    logger.info(f"Starting Redis Stream Consumer on {REDIS_HOST}:{REDIS_PORT} stream '{REDIS_STREAM_KEY}'...")
    r = connect_redis()
    if not r:
        logger.error("Cannot start streaming consumer: Redis connection failed.")
        return

    # Create consumer group if not exists
    try:
        r.xgroup_create(REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP, id="0", mkstream=True)
        logger.info(f"Created Redis consumer group '{REDIS_CONSUMER_GROUP}' on stream '{REDIS_STREAM_KEY}'.")
    except Exception as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group '{REDIS_CONSUMER_GROUP}' already exists.")
        else:
            logger.warning(f"Error checking consumer group: {e}")

    consumer_name = "worker-1"
    logger.info(f"Listening for real-time events as consumer '{consumer_name}'... (Press Ctrl+C to stop)")

    last_id = ">"
    while True:
        try:
            # Read new messages from group
            entries = r.xreadgroup(
                REDIS_CONSUMER_GROUP,
                consumer_name,
                {REDIS_STREAM_KEY: last_id},
                count=50,
                block=2000
            )

            if not entries:
                continue

            for stream_name, messages in entries:
                for msg_id, payload in messages:
                    try:
                        # Extract extra payload if serialized
                        extra_raw = payload.get("extra")
                        extra = {}
                        if extra_raw:
                            try:
                                extra = json.loads(extra_raw)
                            except Exception:
                                pass

                        event = {
                            "userId": payload.get("userId"),
                            "movieId": int(payload.get("movieId")) if payload.get("movieId") and str(payload.get("movieId")).isdigit() else None,
                            "timestamp": float(payload.get("timestamp")) if payload.get("timestamp") else time.time(),
                            "event_type": payload.get("event_type"),
                            "session_id": payload.get("session_id") or payload.get("userId"),
                            "source": payload.get("source", "webapp"),
                            "extra": extra
                        }

                        logger.info(f"Processing event: type={event.get('event_type')}, user={event.get('userId')}, movie={event.get('movieId')}")
                        process_event(event, r)

                        # Acknowledge message
                        r.xack(REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP, msg_id)
                    except Exception as err:
                        logger.error(f"Error processing message {msg_id}: {err}")

        except KeyboardInterrupt:
            logger.info("Stopping Redis Stream consumer...")
            break
        except Exception as e:
            logger.error(f"Redis Stream reading error: {e}. Retrying in 2 seconds...")
            time.sleep(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_redis_streaming_consumer()
