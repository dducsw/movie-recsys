"""
pipeline/etl/storage.py
-----------------------
SeaweedFS Object Storage sync helper for Medallion Lakehouse artifacts.
Uploads Parquet/Delta Lake files to SeaweedFS S3-compatible Filer endpoint.
"""

import os
import logging
import requests
from pipeline.etl.config import SEAWEEDFS_FILER_URL, SEAWEEDFS_BUCKET, USE_SEAWEEDFS_SYNC

logger = logging.getLogger("ETL-Storage")


def upload_to_seaweedfs(local_path: str, lakehouse_layer: str) -> bool:
    """Upload local Parquet/Delta file to SeaweedFS object storage."""
    if not USE_SEAWEEDFS_SYNC:
        return False

    if not os.path.exists(local_path):
        logger.warning(f"File not found for SeaweedFS upload: {local_path}")
        return False

    filename = os.path.basename(local_path)
    target_url = f"{SEAWEEDFS_FILER_URL}/{SEAWEEDFS_BUCKET}/lakehouse/{lakehouse_layer}/{filename}"
    file_size_kb = os.path.getsize(local_path) / 1024.0

    try:
        logger.info(f"Syncing '{filename}' ({file_size_kb:.1f} KB) to SeaweedFS Lakehouse ({lakehouse_layer})...")
        with open(local_path, "rb") as f:
            res = requests.put(target_url, data=f, timeout=30.0)

        if res.status_code in (200, 201):
            logger.info(f"Successfully uploaded '{filename}' to SeaweedFS object storage at {target_url}.")
            return True
        else:
            logger.warning(f"SeaweedFS upload response status {res.status_code}: {res.text}")
            return False
    except Exception as e:
        logger.warning(f"Could not connect to SeaweedFS Filer at {SEAWEEDFS_FILER_URL}: {e}")
        return False
