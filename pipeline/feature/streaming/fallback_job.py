"""
pipeline/feature/streaming/fallback_job.py
-------------------------------------------
Fallback Kafka consumer loop using kafka-python if PySpark JVM / Kafka connector is unavailable.
"""

import json
import time
import logging

from pipeline.feature.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, KAFKA_GROUP_ID
from pipeline.feature.db import connect_redis
from pipeline.feature.streaming.processor import process_event

logger = logging.getLogger("FallbackStreamingJob")


def run_fallback_kafka_streaming():
    """Fallback Kafka consumer loop using kafka-python."""
    logger.info(f"Starting Fallback Kafka Consumer listening to {KAFKA_BOOTSTRAP} topic '{KAFKA_TOPIC}'...")
    r = connect_redis()

    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id=KAFKA_GROUP_ID,
        )

        logger.info(f"Successfully connected to Kafka topic '{KAFKA_TOPIC}'. Waiting for events...")
        for message in consumer:
            event = message.value
            logger.info(f"Received event: type={event.get('event_type')}, user={event.get('userId')}, movie={event.get('movieId')}")
            process_event(event, r)

    except Exception as e:
        logger.error(f"Fallback Kafka consumer error: {e}")
        logger.info("Retrying connection in 5 seconds...")
        time.sleep(5)
