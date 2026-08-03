"""
pipeline/etl/silver/events.py
------------------------------
Silver transformation layer for interaction events (PySpark Medallion Architecture).
Transforms raw event data from Bronze to Silver layer with Incremental Loading, Audit Logging & Watermarking.
Uses Delta Lake format partitioned by event_type and saveAsTable.
"""

import os
import sys
from datetime import datetime

# Windows compatibility patch for PySpark py4j socketserver
import socketserver
if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = object

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline.utils.audit import init_audit_table, log_audit
from pipeline.utils.watermark import get_watermark, update_watermark
from pipeline.etl.config import BRONZE_DIR, SILVER_DIR
from pipeline.etl.schemas import EVENTS_SILVER_DDL, safe_read_pyspark_parquet, execute_spark_sql_ddl
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.catalog import register_table


def transform_events_table(spark: SparkSession, source_table: str, target_table: str) -> None:
    """Transforms raw event data from Bronze to Silver layer with Incremental Loading."""
    start_ts = datetime.now()
    pipeline_name = "Silver-Events"

    print(f"Transforming {source_table} to {target_table}...")

    try:
        init_audit_table(spark)

        # Execute Spark SQL DDL with Partitioning
        execute_spark_sql_ddl(spark, EVENTS_SILVER_DDL)

        # Get last watermark
        last_watermark = get_watermark(spark, pipeline_name)
        print(f"Processing data created after: {last_watermark}")

        bronze_events_dir = os.path.join(BRONZE_DIR, "events")
        silver_events_dir = os.path.join(SILVER_DIR, "events")

        # 1. Read from Bronze
        df_bronze = safe_read_pyspark_parquet(spark, bronze_events_dir)
        if df_bronze is None:
            print("No Bronze events data to process.")
            log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "SUCCESS", start_ts)
            return

        source_cnt = df_bronze.count()
        if source_cnt == 0:
            print("No new data to process.")
            log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "SUCCESS", start_ts)
            return

        # 2. Transform: Normalize fields, cast data types, deduplicate
        df_silver = (
            df_bronze
            .withColumn("userId", col("userId").cast("string"))
            .withColumn("movieId", col("movieId").cast("long"))
            .withColumn("event_type", col("event_type").cast("string"))
            .withColumn("timestamp", col("timestamp").cast("double"))
            .withColumn("updated_at", current_timestamp())
            .filter(col("movieId") > 0)
            .dropDuplicates(["userId", "movieId", "event_type"])
        )
        target_cnt = df_silver.count()

        # 3. Write to Silver Delta Table partitioned by event_type
        try:
            (
                df_silver.write
                .format("delta")
                .option("path", silver_events_dir)
                .partitionBy("event_type")
                .mode("overwrite")
                .saveAsTable(target_table)
            )
        except Exception:
            (
                df_silver.write
                .format("parquet")
                .option("path", silver_events_dir)
                .partitionBy("event_type")
                .mode("overwrite")
                .saveAsTable(target_table)
            )

        # 4. Sync & Register Catalog Metadata
        out_file = os.path.join(silver_events_dir, "part-0.parquet")
        if os.path.exists(out_file):
            upload_to_seaweedfs(out_file, "silver/events")
        register_table("events", "silver", silver_events_dir, target_cnt, str(df_silver.schema))

        # Update watermark table & audit
        update_watermark(spark, pipeline_name, datetime.now())
        log_audit(spark, pipeline_name, source_table, target_table, source_cnt, target_cnt, "SUCCESS", start_ts)
        print(f"Transformation of events completed.")

    except Exception as e:
        log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "FAILED", start_ts, str(e))
        raise e


def transform_silver_events():
    """Execution wrapper for PySpark Events transformation."""
    spark = (
        SparkSession.builder
        .appName("Silver-Transform-Events")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        transform_events_table(spark, "bronze.events", "silver_events")
    finally:
        spark.stop()


if __name__ == "__main__":
    transform_silver_events()
