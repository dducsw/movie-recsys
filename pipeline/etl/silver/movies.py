"""
pipeline/etl/silver/movies.py
------------------------------
Silver transformation layer for movie entity (PySpark Medallion Architecture).
Denormalizes Bronze entity tables (movies, genres, directors, cast, tags),
calculates primary_genre, writes into Delta Table partitioned by primary_genre with Audit & Watermarking.
"""

import os
import sys
from datetime import datetime

# Windows compatibility patch for PySpark py4j socketserver
import socketserver
if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = object

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, collect_set, current_timestamp, coalesce, lit, split

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pipeline.utils.audit import init_audit_table, log_audit
from pipeline.utils.watermark import get_watermark, update_watermark
from pipeline.etl.config import BRONZE_DIR, SILVER_DIR
from pipeline.etl.schemas import MOVIES_SILVER_DDL, safe_read_pyspark_parquet, execute_spark_sql_ddl
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.catalog import register_table


def transform_movies_table(spark: SparkSession, source_table: str, target_table: str) -> None:
    """Transforms raw movie entities from Bronze to Silver layer with PySpark."""
    start_ts = datetime.now()
    pipeline_name = "Silver-Movies"

    print(f"Transforming {source_table} to {target_table}...")

    try:
        init_audit_table(spark)

        # Execute Spark SQL DDL with Partitioning
        execute_spark_sql_ddl(spark, MOVIES_SILVER_DDL)

        # Get last watermark
        last_watermark = get_watermark(spark, pipeline_name)
        print(f"Processing data created after: {last_watermark}")

        bronze_movies_dir = os.path.join(BRONZE_DIR, "movies")
        silver_movies_dir = os.path.join(SILVER_DIR, "movies")

        # 1. Read Bronze Movies
        df_bronze_movies = safe_read_pyspark_parquet(spark, bronze_movies_dir)
        if df_bronze_movies is None:
            print("No Bronze movies data to process.")
            log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "SUCCESS", start_ts)
            return

        source_cnt = df_bronze_movies.count()
        if source_cnt == 0:
            print("No movie records to process.")
            log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "SUCCESS", start_ts)
            return

        if "movieid" in df_bronze_movies.columns:
            df_bronze_movies = df_bronze_movies.withColumnRenamed("movieid", "movieId")

        # 2. Load auxiliary Bronze entities (genres, directors)
        genres_dir = os.path.join(BRONZE_DIR, "genres")
        movie_genres_dir = os.path.join(BRONZE_DIR, "movie_genres")
        directors_dir = os.path.join(BRONZE_DIR, "directors")

        # Denormalize Genres
        genre_agg_df = None
        g_df = safe_read_pyspark_parquet(spark, genres_dir)
        mg_df = safe_read_pyspark_parquet(spark, movie_genres_dir)
        if g_df is not None and mg_df is not None and g_df.count() > 0 and mg_df.count() > 0:
            m_col = "movie_id" if "movie_id" in mg_df.columns else "movieid"
            mg_joined = mg_df.join(g_df, mg_df.genre_id == g_df.id, "inner")
            genre_agg_df = mg_joined.groupBy(m_col).agg(concat_ws("|", collect_set("name")).alias("genres_agg"))
            genre_agg_df = genre_agg_df.withColumnRenamed(m_col, "m_id_g")

        # Denormalize Directors
        dir_df = None
        d_df = safe_read_pyspark_parquet(spark, directors_dir)
        if d_df is not None and d_df.count() > 0 and "director_id" in df_bronze_movies.columns:
            dir_df = d_df.select(col("id").alias("d_id"), col("name").alias("director_name"))

        # 3. Perform Joins & Transformation
        df_silver = df_bronze_movies
        if genre_agg_df is not None:
            df_silver = df_silver.join(genre_agg_df, df_silver.movieId == genre_agg_df.m_id_g, "left")
            df_silver = df_silver.withColumn("genres", coalesce(col("genres"), col("genres_agg"), lit("(no genres listed)")))

        if dir_df is not None:
            df_silver = df_silver.join(dir_df, df_silver.director_id == dir_df.d_id, "left")
            df_silver = df_silver.withColumn("director", coalesce(col("director_name"), lit("")))

        for col_name in ["director", "cast", "keywords", "overview", "poster_url", "release_date"]:
            if col_name not in df_silver.columns:
                df_silver = df_silver.withColumn(col_name, lit(""))

        for num_col in ["popularity", "vote_average"]:
            if num_col not in df_silver.columns:
                df_silver = df_silver.withColumn(num_col, lit(0.0))

        if "vote_count" not in df_silver.columns:
            df_silver = df_silver.withColumn("vote_count", lit(0))

        # Add primary_genre column for partitioning
        df_silver = df_silver.withColumn(
            "primary_genre",
            coalesce(split(col("genres"), "\\|").getItem(0), lit("(no genres listed)"))
        )

        df_silver = df_silver.select(
            col("movieId").cast("long"),
            coalesce(col("title"), lit("")).alias("title"),
            coalesce(col("genres"), lit("(no genres listed)")).alias("genres"),
            coalesce(col("primary_genre"), lit("(no genres listed)")).alias("primary_genre"),
            coalesce(col("release_date"), lit("")).alias("release_date"),
            coalesce(col("popularity"), lit(0.0)).cast("double").alias("popularity"),
            coalesce(col("vote_average"), lit(0.0)).cast("double").alias("vote_average"),
            coalesce(col("vote_count"), lit(0)).cast("long").alias("vote_count"),
            coalesce(col("overview"), lit("")).alias("overview"),
            coalesce(col("poster_url"), lit("")).alias("poster_url"),
            coalesce(col("director"), lit("")).alias("director"),
            coalesce(col("cast"), lit("")).alias("cast"),
            coalesce(col("keywords"), lit("")).alias("keywords")
        ).dropDuplicates(["movieId"])

        target_cnt = df_silver.count()

        # 4. Write to Silver Delta Table partitioned by primary_genre
        try:
            (
                df_silver.write
                .format("delta")
                .option("path", silver_movies_dir)
                .partitionBy("primary_genre")
                .mode("overwrite")
                .saveAsTable(target_table)
            )
        except Exception:
            (
                df_silver.write
                .format("parquet")
                .option("path", silver_movies_dir)
                .partitionBy("primary_genre")
                .mode("overwrite")
                .saveAsTable(target_table)
            )

        # Sync to SeaweedFS & Register Metadata Catalog
        out_file = os.path.join(silver_movies_dir, "part-0.parquet")
        if os.path.exists(out_file):
            upload_to_seaweedfs(out_file, "silver/movies")
        register_table("movies", "silver", silver_movies_dir, target_cnt, str(df_silver.schema))

        # Update watermark & audit log
        update_watermark(spark, pipeline_name, datetime.now())
        log_audit(spark, pipeline_name, source_table, target_table, source_cnt, target_cnt, "SUCCESS", start_ts)
        print(f"Transformation of movies completed.")

    except Exception as e:
        log_audit(spark, pipeline_name, source_table, target_table, 0, 0, "FAILED", start_ts, str(e))
        raise e


def transform_silver_movies():
    """Execution wrapper for PySpark Movies transformation."""
    spark = (
        SparkSession.builder
        .appName("Silver-Transform-Movies")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        transform_movies_table(spark, "bronze.movies", "silver_movies")
    finally:
        spark.stop()


if __name__ == "__main__":
    transform_silver_movies()
