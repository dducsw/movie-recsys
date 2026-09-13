"""
web_app/backend/app/services/metrics.py
---------------------------------------
Prometheus metrics registry and telemetry collectors for MovieNex RecSys.
Includes robust no-op mock fallback if prometheus_client is not installed.
"""

import time
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed. Using lightweight telemetry fallbacks.")

    class _MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass

    Counter = _MockMetric
    Histogram = _MockMetric

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
    ["stage"]
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
    ["stage"]
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
