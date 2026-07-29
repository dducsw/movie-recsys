"""
pipeline/etl/config.py
----------------------
Configurations and directory paths for Medallion Lakehouse ETL Pipeline (Bronze, Silver, Gold).
Supports local Lakehouse storage and SeaweedFS (S3-compatible) object storage.
"""

import os

ETL_DIR = os.path.dirname(__file__)
PIPELINE_DIR = os.path.dirname(ETL_DIR)
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)

DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
LAKEHOUSE_DIR = os.path.join(DATA_DIR, "lakehouse")

BRONZE_DIR = os.path.join(LAKEHOUSE_DIR, "bronze")
SILVER_DIR = os.path.join(LAKEHOUSE_DIR, "silver")
GOLD_DIR = os.path.join(LAKEHOUSE_DIR, "gold")

TARGET_MOVIES_CSV = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
TARGET_PROCESSED_CSV = os.path.join(DATA_DIR, "processed", "movies_processed.csv")

# PostgreSQL Connection Parameters
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5435))
POSTGRES_DB = os.getenv("POSTGRES_DB", "movie_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")

# SeaweedFS S3 & Filer Object Storage Configuration
SEAWEEDFS_FILER_URL = os.getenv("SEAWEEDFS_FILER_URL", "http://localhost:8888")
SEAWEEDFS_S3_ENDPOINT = os.getenv("SEAWEEDFS_S3_ENDPOINT", "http://localhost:8333")
SEAWEEDFS_BUCKET = os.getenv("SEAWEEDFS_BUCKET", "recsys-data")
USE_SEAWEEDFS_SYNC = os.getenv("USE_SEAWEEDFS_SYNC", "true").lower() == "true"
