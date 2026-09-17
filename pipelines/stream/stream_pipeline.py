"""
spark_streaming.py (stream_pipeline)
-------------------------------------
Real-time Streaming feature pipeline entry point.
Consumes user interaction events from Redis stream ('user-events-stream'),
calculates real-time trending scores, and updates online user profiles in Redis (Zero Kafka/Spark overhead).
"""

import os
import sys
import logging

# Ensure project root is in sys.path when executed directly as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipelines.feature_store.feature.streaming.redis_stream_job import run_redis_streaming_consumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StreamingPipeline")


def run_streaming_pipeline():
    """Main Streaming Pipeline entry point."""
    run_redis_streaming_consumer()


if __name__ == "__main__":
    run_streaming_pipeline()
