"""
pipeline/utils/watermark.py
----------------------------
Watermark tracking utility for incremental loading in PySpark Medallion Lakehouse pipelines.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("PipelineWatermark")
WATERMARK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "lakehouse", ".catalog", "watermarks.json")
DEFAULT_MIN_WATERMARK = "1970-01-01 00:00:00"


def _load_watermarks() -> dict:
    os.makedirs(os.path.dirname(WATERMARK_FILE), exist_ok=True)
    if os.path.exists(WATERMARK_FILE):
        try:
            with open(WATERMARK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_watermark(spark=None, pipeline_name: str = "") -> str:
    """Retrieve last watermark timestamp string for given pipeline."""
    w_map = _load_watermarks()
    wm = w_map.get(pipeline_name, DEFAULT_MIN_WATERMARK)
    logger.info(f"Watermark for '{pipeline_name}': {wm}")
    return wm


def update_watermark(spark=None, pipeline_name: str = "", new_watermark: Any = None) -> None:
    """Update watermark timestamp for given pipeline."""
    if new_watermark is None:
        new_watermark = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(new_watermark, datetime):
        new_watermark = new_watermark.strftime("%Y-%m-%d %H:%M:%S")

    w_map = _load_watermarks()
    w_map[pipeline_name] = str(new_watermark)

    os.makedirs(os.path.dirname(WATERMARK_FILE), exist_ok=True)
    with open(WATERMARK_FILE, "w") as f:
        json.dump(w_map, f, indent=2)

    logger.info(f"Updated Watermark for '{pipeline_name}' -> {new_watermark}")
