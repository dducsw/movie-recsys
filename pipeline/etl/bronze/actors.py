"""
pipeline/etl/bronze/actors.py
------------------------------
Bronze raw extractor for 'actors' and 'movie_actors' tables.
Writes schema-enforced Lakehouse table directories data/lakehouse/bronze/actors/ and movie_actors/.
"""

import os
import logging
from pipeline.etl.config import BRONZE_DIR
from pipeline.etl.db import extract_table
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.schemas import ACTORS_BRONZE_SCHEMA, MOVIE_ACTORS_BRONZE_SCHEMA, write_lakehouse_table
from pipeline.etl.catalog import register_table

logger = logging.getLogger("Bronze-Actors")


def extract_raw_actors():
    """Extract raw 'actors' and 'movie_actors' tables to Bronze layer table directories."""
    actors_dir = os.path.join(BRONZE_DIR, "actors")
    ma_dir = os.path.join(BRONZE_DIR, "movie_actors")

    actors_df = extract_table("actors")
    a_file = write_lakehouse_table(actors_df, ACTORS_BRONZE_SCHEMA, actors_dir)
    if a_file:
        upload_to_seaweedfs(a_file, "bronze/actors")
        register_table("actors", "bronze", actors_dir, len(actors_df), str(ACTORS_BRONZE_SCHEMA))

    ma_df = extract_table("movie_actors")
    ma_file = write_lakehouse_table(ma_df, MOVIE_ACTORS_BRONZE_SCHEMA, ma_dir)
    if ma_file:
        upload_to_seaweedfs(ma_file, "bronze/movie_actors")
        register_table("movie_actors", "bronze", ma_dir, len(ma_df), str(MOVIE_ACTORS_BRONZE_SCHEMA))

    return actors_dir, ma_dir
