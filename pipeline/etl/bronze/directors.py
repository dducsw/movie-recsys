"""
pipeline/etl/bronze/directors.py
---------------------------------
Bronze raw extractor for 'directors' table.
Writes schema-enforced Lakehouse table directory data/lakehouse/bronze/directors/.
"""

import os
import logging
from pipeline.etl.config import BRONZE_DIR
from pipeline.etl.db import extract_table
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.schemas import DIRECTORS_BRONZE_SCHEMA, write_lakehouse_table
from pipeline.etl.catalog import register_table

logger = logging.getLogger("Bronze-Directors")


def extract_raw_directors():
    """Extract raw 'directors' table to Bronze layer table directory."""
    table_dir = os.path.join(BRONZE_DIR, "directors")
    df = extract_table("directors")

    out_file = write_lakehouse_table(df, DIRECTORS_BRONZE_SCHEMA, table_dir)
    if out_file:
        upload_to_seaweedfs(out_file, "bronze/directors")
        register_table("directors", "bronze", table_dir, len(df), str(DIRECTORS_BRONZE_SCHEMA))
    return table_dir
