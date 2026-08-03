"""
pipeline/etl/bronze/movies.py
------------------------------
Bronze raw extractor for 'movies' table.
Writes schema-enforced Lakehouse table directory data/lakehouse/bronze/movies/.
"""

import os
import logging
from pipeline.etl.config import BRONZE_DIR
from pipeline.etl.db import extract_table
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.schemas import MOVIES_BRONZE_SCHEMA, write_lakehouse_table
from pipeline.etl.catalog import register_table

logger = logging.getLogger("Bronze-Movies")


def extract_raw_movies():
    """Extract raw 'movies' table from Postgres to Bronze layer table directory."""
    table_dir = os.path.join(BRONZE_DIR, "movies")
    df = extract_table("movies")
    row_count = len(df) if not df.empty else 0

    out_file = write_lakehouse_table(df, MOVIES_BRONZE_SCHEMA, table_dir)
    if out_file:
        upload_to_seaweedfs(out_file, "bronze/movies")
        register_table("movies", "bronze", table_dir, row_count, str(MOVIES_BRONZE_SCHEMA))
    return table_dir
