"""
pipeline/etl/bronze/events.py
------------------------------
Bronze raw extractor for user interaction events.
Writes schema-enforced Lakehouse table directory data/lakehouse/bronze/events/.
"""

import os
import logging
import pandas as pd
from pipeline.etl.config import BRONZE_DIR
from pipeline.etl.db import extract_table
from pipeline.etl.storage import upload_to_seaweedfs
from pipeline.etl.schemas import EVENTS_BRONZE_SCHEMA, write_lakehouse_table
from pipeline.etl.catalog import register_table

logger = logging.getLogger("Bronze-Events")


def extract_raw_events():
    """Extract raw user interaction events table to Bronze layer table directory."""
    table_dir = os.path.join(BRONZE_DIR, "events")

    df = extract_table("user_events")
    if df.empty:
        df = extract_table("ratings")

    out_file = write_lakehouse_table(df, EVENTS_BRONZE_SCHEMA, table_dir)
    if out_file:
        upload_to_seaweedfs(out_file, "bronze/events")
        register_table("events", "bronze", table_dir, len(df), str(EVENTS_BRONZE_SCHEMA))
    return table_dir
