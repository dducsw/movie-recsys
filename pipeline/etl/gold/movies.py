"""
pipeline/etl/gold/movies.py
----------------------------
Gold feature aggregation layer for movie datasets (PySpark Medallion Architecture).
Reads Silver conformed table directory, constructs production ML feature table,
saves to Catalog Delta Table partitioned by primary_genre using PySpark saveAsTable,
logs audit metrics, and syncs downstream CSV targets.
"""

import os
import sys
from datetime import datetime

# Windows compatibility patch for PySpark py4j socketserver
import socketserver
if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = object

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline.utils.audit import init_audit_table, log_audit
from pipeline.utils.watermark import get_watermark, update_watermark
from pipeline.etl.config import SILVER_DIR, GOLD_DIR, TARGET_MOVIES_CSV, TARGET_PROCESSED_CSV
from pipeline.etl.schemas import MOVIES_GOLD_DDL, safe_read_pyspark_parquet, execute_spark_sql_ddl
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.catalog import register_table


def transform_gold_movies_table(spark: SparkSession, source_table: str, target_table: str) -> None:
    """Aggregates Silver conformed movies into Gold production ML feature matrix."""
    start_ts = datetime.now()
    pipeline_name = "Gold-Movies"

    print(f"Transforming {source_table} to {target_table}...")

    try:
        init_audit_table(spark)

        # Execute Spark SQL DDL with Partitioning
        execute_spark_sql_ddl(spark, MOVIES_GOLD_DDL)

        # Get last watermark
        last_watermark = get_watermark(spark, pipeline_name)
        print(f"Processing data created after: {last_watermark}")

        silver_movies_dir = os.path.join(SILVER_DIR, "movies")
        gold_features_dir = os.path.join(GOLD_DIR, "movies_features")

        # 1. Read Silver Movies
        df_silver = safe_read_pyspark_parquet(spark, silver_movies_dir)
        if df_silver is None:
            print("No Silver movies data to aggregate.")
            log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "SUCCESS", start_ts)
            return

        source_cnt = df_silver.count()
        if source_cnt == 0:
            print("No Silver movie records to aggregate.")
            log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "SUCCESS", start_ts)
            return

        # 2. Select & Build Production Gold Feature Schema
        required_cols = ["movieId", "title", "genres", "primary_genre", "overview", "popularity", "vote_average", "vote_count", "director", "cast", "keywords"]
        existing_cols = [c for c in required_cols if c in df_silver.columns]

        df_gold = df_silver.select(*existing_cols)
        target_cnt = df_gold.count()

        # 3. Write to Gold Catalog Delta Table partitioned by primary_genre
        try:
            (
                df_gold.write
                .format("delta")
                .option("path", gold_features_dir)
                .partitionBy("primary_genre")
                .mode("overwrite")
                .saveAsTable(target_table)
            )
        except Exception:
            (
                df_gold.write
                .format("parquet")
                .option("path", gold_features_dir)
                .partitionBy("primary_genre")
                .mode("overwrite")
                .saveAsTable(target_table)
            )

        # Sync to SeaweedFS & Catalog
        out_file = os.path.join(gold_features_dir, "part-0.parquet")
        if os.path.exists(out_file):
            upload_to_seaweedfs(out_file, "gold/movies_features")
        register_table("movies_features", "gold", gold_features_dir, target_cnt, str(df_gold.schema))

        # 4. Sync downstream target CSV files for PySpark batch training
        try:
            pandas_df = df_gold.toPandas()
            os.makedirs(os.path.dirname(TARGET_MOVIES_CSV), exist_ok=True)
            pandas_df[["movieId", "title", "genres"]].to_csv(TARGET_MOVIES_CSV, index=False)
            os.makedirs(os.path.dirname(TARGET_PROCESSED_CSV), exist_ok=True)
            pandas_df.to_csv(TARGET_PROCESSED_CSV, index=False)
            print(f"Synced Gold feature CSVs to {TARGET_MOVIES_CSV} & {TARGET_PROCESSED_CSV}")
        except Exception as e:
            print(f"Warning syncing Gold pandas CSVs: {e}")

        # Update watermark & audit log
        update_watermark(spark, pipeline_name, datetime.now())
        log_audit(spark, pipeline_name, source_table, target_table, source_cnt, target_cnt, "SUCCESS", start_ts)
        print(f"Gold feature aggregation completed.")

    except Exception as e:
        log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "FAILED", start_ts, str(e))
        raise e


def build_gold_movies_features():
    """Execution wrapper for PySpark Gold Feature Aggregation."""
    spark = (
        SparkSession.builder
        .appName("Gold-Transform-Movies")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        transform_gold_movies_table(spark, "silver_movies", "gold_movies_features")
    finally:
        spark.stop()


if __name__ == "__main__":
    build_gold_movies_features()
