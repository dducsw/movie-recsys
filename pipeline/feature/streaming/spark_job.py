"""
pipeline/feature/streaming/spark_job.py
----------------------------------------
PySpark Structured Streaming pipeline runner with Kafka connector.
"""

import logging
from pipeline.feature.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, SPARK_MASTER
from pipeline.feature.streaming.processor import process_spark_micro_batch

logger = logging.getLogger("SparkStreamingJob")


def run_pyspark_structured_streaming():
    """Run PySpark Structured Streaming pipeline consuming from Kafka."""
    logger.info("Initializing PySpark Structured Streaming with Kafka connector...")

    # Windows compatibility patch for PySpark py4j socketserver
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
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_stream = (
        kafka_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), event_schema).alias("data"))
        .select("data.*")
    )

    logger.info(f"Starting PySpark Structured Streaming query listening to Kafka '{KAFKA_TOPIC}'...")

    query = (
        parsed_stream.writeStream
        .foreachBatch(process_spark_micro_batch)
        .outputMode("update")
        .start()
    )

    query.awaitTermination()
