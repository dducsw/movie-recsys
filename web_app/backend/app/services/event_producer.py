"""
event_producer.py
-----------------
Kafka producer singleton. Emit user-interaction events lên topic "user-events".

Schema align với data/simulator/sim_click_events.csv để Spark job
xử lý realtime và batch data chung một schema:

{
    "userId":     str,    # session_id từ chatbot hoặc "anonymous"
    "movieId":    int | null,  # ID phim liên quan (null nếu không liên quan)
    "timestamp":  float,  # Unix epoch (giây, float) — giống simulator
    "event_type": str,    # xem EventType bên dưới
    "session_id": str,    # duplicate userId cho compat với simulator
    "source":     "webapp",  # phân biệt với simulated data
    "extra":      dict    # payload bổ sung tuỳ event_type
}

Mapping event_type với simulator (sim_click_events.csv):
  Simulator           │ Webapp
  ────────────────────┼────────────────────────────────────────
  click               │ — (không có listing page)
  detail_view         │ detail_view  (GET /api/movies/{id})
  watch_start         │ — (không có player)
  watch_complete      │ — (không có player)
  ────────────────────┼────────────────────────────────────────
  (không có)          │ movie_search          (search query)
  (không có)          │ trending_browse       (duyệt trending)
  (không có)          │ latest_browse         (duyệt latest)
  (không có)          │ recommendation_request (liked list → rec)
  (không có)          │ similar_movie_request  (xem similar)
  (không có)          │ chatbot_message        (chat với bot)

extra payload per event_type:
  detail_view            → { title, genres }
  movie_search           → { query, result_count }
  trending_browse        → { page, limit }
  latest_browse          → { page, limit }
  recommendation_request → { liked_movie_ids: [int], result_count }
  similar_movie_request  → { limit }
  chatbot_message        → { message_len, result_movie_count, message_count }
"""

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_producer = None


def _get_producer():
    """Khởi tạo KafkaProducer lần đầu, tái dùng sau đó."""
    global _producer
    if _producer is not None:
        return _producer

    try:
        from kafka import KafkaProducer  # kafka-python
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        _producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            acks=0,               # fire-and-forget, không block HTTP response
            retries=3,
            request_timeout_ms=5_000,
        )
        logger.info("KafkaProducer connected to %s", bootstrap)
    except Exception as exc:
        # Kafka chưa sẵn sàng → log, trả None; app vẫn hoạt động bình thường
        logger.warning("KafkaProducer init failed (events will be dropped): %s", exc)
        _producer = None

    return _producer


# ── Public API ─────────────────────────────────────────────────────────────────
TOPIC = "user-events"


def emit(
    event_type: str,
    *,
    user_id: str = "anonymous",
    movie_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Gửi một event lên Kafka. Non-blocking, lỗi chỉ được log.

    Args:
        event_type: Tên event (dùng hằng số từ EventType).
        user_id:    Session ID hoặc "anonymous".
        movie_id:   ID phim liên quan; None nếu event không gắn phim cụ thể.
        extra:      Payload bổ sung tuỳ event_type.
    """
    producer = _get_producer()
    if producer is None:
        return

    record = {
        "userId":     user_id,
        "movieId":    movie_id,
        "timestamp":  time.time(),          # epoch float — align với simulator
        "event_type": event_type,
        "session_id": user_id,              # alias, giữ compat với simulator field
        "source":     "webapp",             # phân biệt real vs simulated data
        "extra":      extra or {},
    }

    try:
        producer.send(TOPIC, value=record)
    except Exception as exc:
        logger.warning("Failed to emit event '%s': %s", event_type, exc)


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
