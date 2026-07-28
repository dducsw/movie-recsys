"""
spark_streaming.py
------------------
Real-time PySpark Structured Streaming pipeline with Kafka consumer.
Consumes user interaction events from Kafka topic 'user-events',
processes micro-batch windows, calculates real-time trending scores,
and updates online user profiles in Redis online feature store.
"""

import json
import logging
import os
import time
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkStreamingPipeline")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = "user-events"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")


def connect_redis():
    """Connect to Redis client."""
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return r
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        return None


def process_event(event: Dict[str, Any], redis_client):
    """
    Process single user interaction event:
    1. Update user's recent interactions in Redis: user:{session_id}:recent
    2. Increment real-time trending count in Redis sorted set: realtime:trending
    """
    if not redis_client:
        return

    session_id = event.get("session_id") or event.get("userId")
    movie_id = event.get("movieId")
    event_type = event.get("event_type")

    if not session_id:
        return

    pipe = redis_client.pipeline()

    # 1. Update user recent interaction list
    if movie_id:
        user_key = f"user:{session_id}:recent"
        pipe.lpush(user_key, json.dumps({
            "movie_id": movie_id,
            "event": event_type,
            "ts": event.get("timestamp") or time.time()
        }))
        pipe.ltrim(user_key, 0, 49)  # Keep last 50 interactions
        pipe.expire(user_key, 86400 * 7)  # 7 days TTL

    # 2. Increment trending counter in Redis Sorted Set
    if movie_id:
        score_weights = {
            "like": 3.0,
            "click": 1.5,
            "detail_view": 1.0,
            "similar_movie_request": 0.8,
            "recommendation_request": 0.5,
        }
        score_weight = score_weights.get(event_type, 1.0)
        pipe.zincrby("realtime:trending", score_weight, str(movie_id))
        pipe.expire("realtime:trending", 86400)  # Reset/expire after 24 hours

    pipe.execute()


def process_partition(partition):
    """Worker-level execution for processing a partition of events to Redis in parallel."""
    import redis
    import json
    import time

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True, socket_timeout=2.0)
        r.ping()
    except Exception as e:
        logger.error(f"Worker partition failed to connect to Redis ({e})")
        return

    pipe = r.pipeline()
    count = 0

    score_weights = {
        "like": 3.0,
        "click": 1.5,
        "detail_view": 1.0,
        "similar_movie_request": 0.8,
        "recommendation_request": 0.5,
    }

    for row in partition:
        try:
            row_dict = row.asDict() if hasattr(row, "asDict") else row
            session_id = row_dict.get("session_id") or row_dict.get("userId")
            movie_id = row_dict.get("movieId")
            event_type = row_dict.get("event_type")
            ts = row_dict.get("timestamp") or time.time()

            if not session_id or not movie_id:
                continue

            # 1. Update user recent interaction list
            user_key = f"user:{session_id}:recent"
            pipe.lpush(user_key, json.dumps({
                "movie_id": movie_id,
                "event": event_type,
                "ts": ts
            }))
            pipe.ltrim(user_key, 0, 49)
            pipe.expire(user_key, 86400 * 7)

            # 2. Increment trending counter in Redis Sorted Set
            weight = score_weights.get(event_type, 1.0)
            pipe.zincrby("realtime:trending", weight, str(movie_id))
            pipe.expire("realtime:trending", 86400)

            count += 1
            if count % 100 == 0:
                pipe.execute()
                pipe = r.pipeline()
        except Exception as e:
            logger.error(f"Error processing streaming event row in partition: {e}")

    if count % 100 != 0:
        try:
            pipe.execute()
        except Exception as e:
            logger.error(f"Error executing final pipeline batch in partition: {e}")


def process_spark_micro_batch(batch_df, batch_id):
    """Distributed micro-batch processing using PySpark foreachPartition."""
    try:
        num_records = batch_df.count()
        if num_records > 0:
            logger.info(f"Processing PySpark Streaming micro-batch {batch_id} with {num_records} events via foreachPartition.")
            batch_df.rdd.foreachPartition(process_partition)
    except Exception as e:
        logger.error(f"Error executing micro-batch {batch_id}: {e}")



def run_pyspark_structured_streaming():
    """Run PySpark Structured Streaming pipeline consuming from Kafka."""
    logger.info("Initializing PySpark Structured Streaming with Kafka connector...")
    try:
        import socketserver
        if not hasattr(socketserver, "UnixStreamServer"):
            socketserver.UnixStreamServer = object

        from pyspark.sql import SparkSession
        from pyspark.sql.functions import from_json, col
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

        # Submit session with Kafka package dependency
        spark = (
            SparkSession.builder
            .appName("RecSys-SparkStructuredStreaming")
            .master(SPARK_MASTER)
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        # Schema matching webapp event_producer and simulator
        event_schema = StructType([
            StructField("userId", StringType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("event_type", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("source", StringType(), True),
        ])

        # Read stream from Kafka
        kafka_stream = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
            .option("subscribe", TOPIC)
            .option("startingOffsets", "latest")
            .load()
        )

        parsed_stream = (
            kafka_stream
            .selectExpr("CAST(value AS STRING) as json_str")
            .select(from_json(col("json_str"), event_schema).alias("data"))
            .select("data.*")
        )

        logger.info(f"Starting PySpark Structured Streaming query listening to Kafka '{TOPIC}'...")

        query = (
            parsed_stream.writeStream
            .foreachBatch(process_spark_micro_batch)
            .outputMode("update")
            .start()
        )

        query.awaitTermination()

    except Exception as e:
        logger.warning(f"PySpark Structured Streaming initialization failed: {e}")
        logger.info("Switching to standard Kafka Consumer loop fallback...")
        run_fallback_kafka_streaming()


def run_fallback_kafka_streaming():
    """Fallback Kafka consumer loop using kafka-python if PySpark JVM/Kafka connector is unavailable."""
    logger.info(f"Starting Fallback Kafka Consumer listening to {KAFKA_BOOTSTRAP} topic '{TOPIC}'...")
    r = connect_redis()

    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="recsys-streaming-group",
        )

        logger.info(f"Successfully connected to Kafka topic '{TOPIC}'. Waiting for events...")
        for message in consumer:
            event = message.value
            logger.info(f"Received event: type={event.get('event_type')}, user={event.get('userId')}, movie={event.get('movieId')}")
            process_event(event, r)

    except Exception as e:
        logger.error(f"Fallback Kafka consumer error: {e}")
        logger.info("Retrying connection in 5 seconds...")
        time.sleep(5)


def run_streaming_pipeline():
    """Main Streaming Pipeline execution router."""
    run_pyspark_structured_streaming()


if __name__ == "__main__":
    run_streaming_pipeline()
