"""
pipeline/etl/schemas.py
-----------------------
Spark SQL DDL Schema Definitions & Delta Lake Writers with Partitioning.
Provides native Spark SQL CREATE TABLE statements USING delta with PARTITIONED BY columns.
"""

import os
import logging

# Windows compatibility patch for PySpark py4j socketserver
import socketserver
if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = object

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, LongType, IntegerType, DoubleType, StringType, BooleanType, TimestampType
)

logger = logging.getLogger("ETL-Schemas")

# ==========================================
# SPARK SQL DDL STATEMENTS (Delta Lake Format with Partitioning)
# ==========================================

MOVIES_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_movies (
    movieId BIGINT,
    title STRING,
    release_date STRING,
    popularity DOUBLE,
    adult BOOLEAN,
    overview STRING,
    vote_average DOUBLE,
    vote_count BIGINT,
    poster_url STRING,
    director_id BIGINT,
    trailer_url STRING
) USING delta
"""

GENRES_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_genres (
    id BIGINT,
    name STRING
) USING delta
"""

MOVIE_GENRES_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_movie_genres (
    movie_id BIGINT,
    genre_id BIGINT
) USING delta
"""

DIRECTORS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_directors (
    id BIGINT,
    name STRING
) USING delta
"""

ACTORS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_actors (
    id BIGINT,
    name STRING
) USING delta
"""

MOVIE_ACTORS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_movie_actors (
    movie_id BIGINT,
    actor_id BIGINT,
    cast_order INT
) USING delta
"""

TAGS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_tags (
    id BIGINT,
    name STRING
) USING delta
"""

MOVIE_TAGS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_movie_tags (
    movie_id BIGINT,
    tag_id BIGINT
) USING delta
"""

EVENTS_BRONZE_DDL = """
CREATE TABLE IF NOT EXISTS bronze_events (
    userId STRING,
    movieId BIGINT,
    event_type STRING,
    timestamp DOUBLE
) USING delta
PARTITIONED BY (event_type)
"""

# ==========================================
# SPARK SQL DDL STATEMENTS (Silver Delta Layer with Partitioning)
# ==========================================

MOVIES_SILVER_DDL = """
CREATE TABLE IF NOT EXISTS silver_movies (
    movieId BIGINT,
    title STRING,
    genres STRING,
    primary_genre STRING,
    release_date STRING,
    popularity DOUBLE,
    vote_average DOUBLE,
    vote_count BIGINT,
    overview STRING,
    poster_url STRING,
    director STRING,
    cast STRING,
    keywords STRING
) USING delta
PARTITIONED BY (primary_genre)
"""

EVENTS_SILVER_DDL = """
CREATE TABLE IF NOT EXISTS silver_events (
    userId STRING,
    movieId BIGINT,
    event_type STRING,
    timestamp DOUBLE
) USING delta
PARTITIONED BY (event_type)
"""

# ==========================================
# SPARK SQL DDL STATEMENTS (Gold Delta Layer with Partitioning)
# ==========================================

MOVIES_GOLD_DDL = """
CREATE TABLE IF NOT EXISTS gold_movies_features (
    movieId BIGINT,
    title STRING,
    genres STRING,
    primary_genre STRING,
    overview STRING,
    popularity DOUBLE,
    vote_average DOUBLE,
    vote_count BIGINT,
    director STRING,
    cast STRING,
    keywords STRING
) USING delta
PARTITIONED BY (primary_genre)
"""

# ==========================================
# STRUCTTYPE DEFINITIONS
# ==========================================

MOVIES_BRONZE_SCHEMA = StructType([
    StructField("movieId", LongType(), False),
    StructField("title", StringType(), True),
    StructField("release_date", StringType(), True),
    StructField("popularity", DoubleType(), True),
    StructField("adult", BooleanType(), True),
    StructField("overview", StringType(), True),
    StructField("vote_average", DoubleType(), True),
    StructField("vote_count", LongType(), True),
    StructField("poster_url", StringType(), True),
    StructField("director_id", LongType(), True),
    StructField("trailer_url", StringType(), True),
])

GENRES_BRONZE_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("name", StringType(), True),
])

MOVIE_GENRES_BRONZE_SCHEMA = StructType([
    StructField("movie_id", LongType(), False),
    StructField("genre_id", LongType(), False),
])

DIRECTORS_BRONZE_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("name", StringType(), True),
])

ACTORS_BRONZE_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("name", StringType(), True),
])

MOVIE_ACTORS_BRONZE_SCHEMA = StructType([
    StructField("movie_id", LongType(), False),
    StructField("actor_id", LongType(), False),
    StructField("cast_order", IntegerType(), True),
])

TAGS_BRONZE_SCHEMA = StructType([
    StructField("id", LongType(), False),
    StructField("name", StringType(), True),
])

MOVIE_TAGS_BRONZE_SCHEMA = StructType([
    StructField("movie_id", LongType(), False),
    StructField("tag_id", LongType(), False),
])

EVENTS_BRONZE_SCHEMA = StructType([
    StructField("userId", StringType(), True),
    StructField("movieId", LongType(), True),
    StructField("event_type", StringType(), True),
    StructField("timestamp", DoubleType(), True),
])

MOVIES_SILVER_SCHEMA = StructType([
    StructField("movieId", LongType(), False),
    StructField("title", StringType(), False),
    StructField("genres", StringType(), False),
    StructField("primary_genre", StringType(), False),
    StructField("release_date", StringType(), True),
    StructField("popularity", DoubleType(), False),
    StructField("vote_average", DoubleType(), False),
    StructField("vote_count", LongType(), False),
    StructField("overview", StringType(), True),
    StructField("poster_url", StringType(), True),
    StructField("director", StringType(), True),
    StructField("cast", StringType(), True),
    StructField("keywords", StringType(), True),
])

EVENTS_SILVER_SCHEMA = StructType([
    StructField("userId", StringType(), False),
    StructField("movieId", LongType(), False),
    StructField("event_type", StringType(), False),
    StructField("timestamp", DoubleType(), False),
])

MOVIES_GOLD_SCHEMA = StructType([
    StructField("movieId", LongType(), False),
    StructField("title", StringType(), False),
    StructField("genres", StringType(), False),
    StructField("primary_genre", StringType(), False),
    StructField("overview", StringType(), True),
    StructField("popularity", DoubleType(), False),
    StructField("vote_average", DoubleType(), False),
    StructField("vote_count", LongType(), False),
    StructField("director", StringType(), True),
    StructField("cast", StringType(), True),
    StructField("keywords", StringType(), True),
])


def execute_spark_sql_ddl(spark, ddl: str):
    """Execute native Spark SQL DDL statement."""
    try:
        spark.sql(ddl)
        logger.info(f"Executed Spark SQL DDL: {ddl.strip().splitlines()[0]}")
    except Exception as e:
        logger.error(f"Error executing Spark SQL DDL: {e}")


def write_lakehouse_table(df: pd.DataFrame, schema: StructType, table_dir: str) -> str:
    """Align DataFrame to PySpark StructType schema and write into Lakehouse Table directory."""
    try:
        os.makedirs(table_dir, exist_ok=True)
        out_file = os.path.join(table_dir, "part-0.parquet")

        field_names = [f.name for f in schema.fields]

        if df is None or df.empty:
            empty_data = {name: pd.Series(dtype="object") for name in field_names}
            df = pd.DataFrame(empty_data)

        for field in schema.fields:
            col_name = field.name
            if col_name not in df.columns:
                if isinstance(field.dataType, (LongType, IntegerType)):
                    df[col_name] = 0
                elif isinstance(field.dataType, DoubleType):
                    df[col_name] = 0.0
                elif isinstance(field.dataType, BooleanType):
                    df[col_name] = False
                else:
                    df[col_name] = ""

        df = df[field_names].copy()

        pa_fields = []
        for f in schema.fields:
            if isinstance(f.dataType, LongType):
                pa_type = pa.int64()
            elif isinstance(f.dataType, IntegerType):
                pa_type = pa.int32()
            elif isinstance(f.dataType, DoubleType):
                pa_type = pa.float64()
            elif isinstance(f.dataType, BooleanType):
                pa_type = pa.bool_()
            else:
                pa_type = pa.string()
            pa_fields.append(pa.field(f.name, pa_type, nullable=f.nullable))

        pa_schema = pa.schema(pa_fields)
        table = pa.Table.from_pandas(df, schema=pa_schema, preserve_index=False)
        pq.write_table(table, out_file)
        logger.info(f"Wrote StructType Lakehouse Table ({len(df)} rows) -> {table_dir}")
        return out_file
    except Exception as e:
        logger.error(f"Failed to write Lakehouse table to {table_dir}: {e}")
        return ""


def safe_read_pyspark_parquet(spark, table_dir: str):
    """Safely read PySpark Parquet/Delta directory without throwing empty leaf file exception."""
    if not os.path.exists(table_dir):
        return None

    try:
        parquet_files = [f for f in os.listdir(table_dir) if f.endswith(".parquet") or f == "_delta_log"]
        if not parquet_files:
            return None

        df = spark.read.parquet(table_dir)
        return df
    except Exception as e:
        logger.warning(f"PySpark read failed for {table_dir}: {e}")
        return None
