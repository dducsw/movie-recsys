"""
pipeline/utils/audit.py
------------------------
Audit logging utility for tracking PySpark pipeline executions, row counts, and statuses.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("PipelineAudit")
AUDIT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "lakehouse", ".catalog", "audit_log.json")


def init_audit_table(spark=None) -> None:
    """Initialize audit log storage table/file."""
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    if not os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "w") as f:
            json.dump([], f)


def log_audit(
    spark,
    pipeline_name: str,
    source_table: str,
    target_table: str,
    source_count: int,
    target_count: int,
    status: str,
    start_ts: datetime,
    error_msg: Optional[str] = None
) -> None:
    """Record execution audit metrics into Lakehouse audit table/log."""
    end_ts = datetime.now()
    duration_sec = round((end_ts - start_ts).total_seconds(), 2)

    record = {
        "pipeline_name": pipeline_name,
        "source_table": source_table,
        "target_table": target_table,
        "source_count": source_count,
        "target_count": target_count,
        "status": status,
        "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": duration_sec,
        "error_message": error_msg or ""
    }

    logger.info(f"AUDIT [{status}] {pipeline_name}: {source_count} -> {target_count} records in {duration_sec}s")

    try:
        init_audit_table()
        records = []
        if os.path.exists(AUDIT_LOG_FILE):
            with open(AUDIT_LOG_FILE, "r") as f:
                records = json.load(f)
        records.append(record)
        with open(AUDIT_LOG_FILE, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist audit log to disk: {e}")
