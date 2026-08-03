"""
pipeline/etl/catalog.py
-----------------------
Lakehouse Metadata Catalog Manager (Supports Apache Polaris REST Catalog & Local Manifest Registry).
Tracks table registration, layer lineage, PyArrow schemas, row counts, and checksums.
"""

import os
import json
import time
import logging
import requests
from typing import Dict, Any, List

from pipeline.etl.config import LAKEHOUSE_DIR

logger = logging.getLogger("ETL-Catalog")

POLARIS_URI = os.getenv("POLARIS_CATALOG_URI", "http://localhost:8181/api/catalog")
POLARIS_REALM = os.getenv("POLARIS_REALM", "recsys")
CATALOG_DIR = os.path.join(LAKEHOUSE_DIR, ".catalog")
MANIFEST_PATH = os.path.join(CATALOG_DIR, "manifest.json")


def _load_manifest() -> Dict[str, Any]:
    os.makedirs(CATALOG_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load local catalog manifest: {e}")
    return {"catalog_name": "recsys_lakehouse_catalog", "tables": {}, "last_updated": time.time()}


def _save_manifest(manifest: Dict[str, Any]):
    os.makedirs(CATALOG_DIR, exist_ok=True)
    manifest["last_updated"] = time.time()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def register_table_to_polaris(table_name: str, layer: str, file_path: str, row_count: int, fields: List[str]) -> bool:
    """Attempt table registration to Apache Polaris REST Catalog endpoint."""
    try:
        payload = {
            "table_name": table_name,
            "layer": layer,
            "location": file_path,
            "row_count": row_count,
            "schema_fields": fields,
            "catalog": "polaris",
            "timestamp": time.time()
        }
        res = requests.post(f"{POLARIS_URI}/v1/{POLARIS_REALM}/namespaces/{layer}/tables", json=payload, timeout=2.0)
        if res.status_code in (200, 201):
            logger.info(f"Successfully registered table '{layer}.{table_name}' to Apache Polaris Catalog!")
            return True
        else:
            logger.debug(f"Polaris REST Catalog endpoint returned status {res.status_code} (Polaris service offline).")
            return False
    except Exception:
        # Polaris REST endpoint is optional; fail gracefully to local manifest registry
        return False


def register_table(table_name: str, layer: str, file_path: str, row_count: int, schema_str: str = "") -> Dict[str, Any]:
    """Register Lakehouse table in Metadata Catalog (Polaris REST + Local Manifest)."""
    manifest = _load_manifest()

    table_key = f"{layer}.{table_name}"
    entry = {
        "table_name": table_name,
        "layer": layer,
        "location": file_path,
        "row_count": row_count,
        "schema": schema_str,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "timestamp": time.time()
    }

    manifest["tables"][table_key] = entry
    _save_manifest(manifest)
    logger.info(f"Registered table '{table_key}' ({row_count} rows) in Lakehouse Metadata Catalog -> {MANIFEST_PATH}")

    # Sync with Apache Polaris REST Catalog if available
    fields = [s.strip() for s in schema_str.split("\n") if s.strip()]
    register_table_to_polaris(table_name, layer, file_path, row_count, fields)

    return entry


def get_catalog_summary() -> Dict[str, Any]:
    """Retrieve catalog manifest summary."""
    return _load_manifest()
