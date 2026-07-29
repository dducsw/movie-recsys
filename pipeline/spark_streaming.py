"""
spark_streaming.py
------------------
Real-time PySpark Structured Streaming pipeline entry point.
Consumes user interaction events from Kafka, processes micro-batch windows,
calculates real-time trending scores, and updates online user profiles in Redis.
"""

import os
import sys
import logging

# Ensure project root is in sys.path when executed directly as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature.streaming.spark_job import run_pyspark_structured_streaming
from pipeline.feature.streaming.fallback_job import run_fallback_kafka_streaming

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkStreamingPipeline")


def run_streaming_pipeline():
    """Main Streaming Pipeline entry point."""
    try:
        run_pyspark_structured_streaming()
    except Exception as e:
        logger.warning(f"PySpark Structured Streaming initialization failed: {e}")
        logger.info("Switching to standard Kafka Consumer loop fallback...")
        run_fallback_kafka_streaming()


if __name__ == "__main__":
    run_streaming_pipeline()
