"""
pipeline/etl/db.py
------------------
PostgreSQL connection and data extraction helper.
"""

import logging
import pandas as pd
import psycopg2
from pipeline.etl.config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)

logger = logging.getLogger("ETL-DB")


def get_postgres_connection():
    """Establish connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL ({POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}): {e}")
        return None


def extract_table(table_name: str, query: str = None) -> pd.DataFrame:
    """Extract table from Postgres as a Pandas DataFrame."""
    conn = get_postgres_connection()
    if not conn:
        logger.warning(f"Returning empty DataFrame for '{table_name}' because Postgres is unreachable.")
        return pd.DataFrame()

    try:
        sql = query or f"SELECT * FROM {table_name}"
        logger.info(f"Extracting table '{table_name}' from Postgres...")
        df = pd.read_sql(sql, conn)
        conn.close()
        logger.info(f"Successfully extracted {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        logger.error(f"Error extracting table '{table_name}': {e}")
        if conn:
            conn.close()
        return pd.DataFrame()
