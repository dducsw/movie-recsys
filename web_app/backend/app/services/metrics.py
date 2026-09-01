"""
web_app/backend/app/services/metrics.py
---------------------------------------
Prometheus metrics registry and telemetry collectors for MovieNex RecSys.
"""

import time
from contextlib import contextmanager
import logging
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Request Counter
RECSYS_REQUESTS_TOTAL = Counter(
    "recsys_requests_total",
    "Total recommendation requests processed",
    ["endpoint", "status"]
)

# Latency Histogram per Stage (Retrieval, Ranking, Reranking)
RECSYS_STAGE_LATENCY_SECONDS = Histogram(
    "recsys_stage_latency_seconds",
    "Execution latency in seconds for recommendation pipeline stages",
    ["stage"],
    buckets=(0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.250, 0.500, 1.0)
)

# Cache Hit & Miss Counters
RECSYS_CACHE_HITS = Counter(
    "recsys_cache_hits_total",
    "Total cache hits in recommendation service",
    ["cache_type"]
)

RECSYS_CACHE_MISSES = Counter(
    "recsys_cache_misses_total",
    "Total cache misses in recommendation service",
    ["cache_type"]
)

# Candidates Count Histogram
RECSYS_CANDIDATES_COUNT = Histogram(
    "recsys_candidates_count",
    "Number of candidates processed at each stage",
    ["stage"],
    buckets=(1, 5, 10, 20, 50, 80, 100, 150, 200, 300)
)


@contextmanager
def track_stage_latency(stage_name: str):
    """Context manager to measure and log execution duration of a pipeline stage."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        try:
            RECSYS_STAGE_LATENCY_SECONDS.labels(stage=stage_name).observe(elapsed)
        except Exception:
            pass
