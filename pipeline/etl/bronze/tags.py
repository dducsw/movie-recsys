"""
pipeline/etl/bronze/tags.py
----------------------------
Bronze raw extractor for 'tags' and 'movie_tags' tables.
Writes schema-enforced Lakehouse table directories data/lakehouse/bronze/tags/ and movie_tags/.
"""

import os
import logging
from pipeline.etl.config import BRONZE_DIR
from pipeline.etl.db import extract_table
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.schemas import TAGS_BRONZE_SCHEMA, MOVIE_TAGS_BRONZE_SCHEMA, write_lakehouse_table
from pipeline.etl.catalog import register_table

logger = logging.getLogger("Bronze-Tags")


def extract_raw_tags():
    """Extract raw 'tags' and 'movie_tags' tables to Bronze layer table directories."""
    tags_dir = os.path.join(BRONZE_DIR, "tags")
    mt_dir = os.path.join(BRONZE_DIR, "movie_tags")

    tags_df = extract_table("tags")
    t_file = write_lakehouse_table(tags_df, TAGS_BRONZE_SCHEMA, tags_dir)
    if t_file:
        upload_to_seaweedfs(t_file, "bronze/tags")
        register_table("tags", "bronze", tags_dir, len(tags_df), str(TAGS_BRONZE_SCHEMA))

    mt_df = extract_table("movie_tags")
    mt_file = write_lakehouse_table(mt_df, MOVIE_TAGS_BRONZE_SCHEMA, mt_dir)
    if mt_file:
        upload_to_seaweedfs(mt_file, "bronze/movie_tags")
        register_table("movie_tags", "bronze", mt_dir, len(mt_df), str(MOVIE_TAGS_BRONZE_SCHEMA))

    return tags_dir, mt_dir
