"""
apps/api/app/services/metrics.py
--------------------------------
Prometheus metrics registry and telemetry collectors for the RecSys API.
Includes robust no-op mock fallback if prometheus_client is not installed.

Instrumentation contract
------------------------
* ``status`` on ``recsys_requests_total`` is one of ``success`` | ``fallback``
  | ``error``.  A request answered by a heuristic or trending fallback is
  **not** a success — reporting it as one makes the error rate permanently
  read 0 in Grafana.
* Every graceful-degradation path increments
  ``recsys_fallbacks_total{source, endpoint}``.
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed. Using lightweight telemetry fallbacks.")

    class _MockMetric:
        def __init__(self, *args, **kwargs):
            self._kwargs = kwargs

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    Counter = _MockMetric
    Histogram = _MockMetric


# Latency buckets tuned for a stage-scale of a few milliseconds; the
# prometheus defaults bottom out at 5 ms which makes p50/p95 unusable here.
STAGE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

# Request Counter — status is success | fallback | error
RECSYS_REQUESTS_TOTAL = Counter(
    "recsys_requests_total",
    "Total recommendation requests processed",
    ["endpoint", "status"],
)

# Latency Histogram per Stage (Retrieval, Ranking, Reranking), split by endpoint so
# the similar-movies and personalized pipelines are not aggregated together.
RECSYS_STAGE_LATENCY_SECONDS = Histogram(
    "recsys_stage_latency_seconds",
    "Execution latency in seconds for recommendation pipeline stages",
    ["stage", "endpoint"],
    buckets=STAGE_BUCKETS,
)

# Cache Hit & Miss Counters
RECSYS_CACHE_HITS = Counter(
    "recsys_cache_hits_total",
    "Total cache hits in recommendation service",
    ["cache_type"],
)

RECSYS_CACHE_MISSES = Counter(
    "recsys_cache_misses_total",
    "Total cache misses in recommendation service",
    ["cache_type"],
)

# Candidates Count Histogram
RECSYS_CANDIDATES_COUNT = Histogram(
    "recsys_candidates_count",
    "Number of candidates processed at each stage",
    ["stage", "endpoint"],
    buckets=(0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2500, 5000),
)

# Fallback Counters — one for every graceful-degradation path, tagged by the
# source that degraded and by the endpoint it affected.
RECSYS_FALLBACKS_TOTAL = Counter(
    "recsys_fallbacks_total",
    "Number of times a recommendation stage fell back to a degraded source",
    ["source", "endpoint"],
)


@contextmanager
def track_stage_latency(stage_name: str, endpoint: str = "unknown"):
    """Context manager to measure and log execution duration of a pipeline stage."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        try:
            RECSYS_STAGE_LATENCY_SECONDS.labels(
                stage=stage_name, endpoint=endpoint,
            ).observe(elapsed)
        except Exception:
            pass
