"""
pipeline/etl/bronze/genres.py
------------------------------
Bronze raw extractor for 'genres' and 'movie_genres' tables.
Writes schema-enforced Lakehouse table directories data/lakehouse/bronze/genres/ and movie_genres/.
"""

import os
import logging
from pipeline.etl.config import BRONZE_DIR
from pipeline.etl.db import extract_table
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.schemas import GENRES_BRONZE_SCHEMA, MOVIE_GENRES_BRONZE_SCHEMA, write_lakehouse_table
from pipeline.etl.catalog import register_table

logger = logging.getLogger("Bronze-Genres")


def extract_raw_genres():
    """Extract raw 'genres' and 'movie_genres' tables to Bronze layer table directories."""
    genres_dir = os.path.join(BRONZE_DIR, "genres")
    mg_dir = os.path.join(BRONZE_DIR, "movie_genres")

    genres_df = extract_table("genres")
    g_file = write_lakehouse_table(genres_df, GENRES_BRONZE_SCHEMA, genres_dir)
    if g_file:
        upload_to_seaweedfs(g_file, "bronze/genres")
        register_table("genres", "bronze", genres_dir, len(genres_df), str(GENRES_BRONZE_SCHEMA))

    mg_df = extract_table("movie_genres")
    mg_file = write_lakehouse_table(mg_df, MOVIE_GENRES_BRONZE_SCHEMA, mg_dir)
    if mg_file:
        upload_to_seaweedfs(mg_file, "bronze/movie_genres")
        register_table("movie_genres", "bronze", mg_dir, len(mg_df), str(MOVIE_GENRES_BRONZE_SCHEMA))

    return genres_dir, mg_dir
