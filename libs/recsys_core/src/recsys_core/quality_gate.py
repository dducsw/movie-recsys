"""
recsys_core.quality_gate
------------------------
Declarative CT quality gates: accuracy + beyond-accuracy + latency.
"""

from __future__ import annotations

from dataclasses import dataclass


class QualityGateError(AssertionError):
    """Raised when a CT run fails a promotion gate."""


@dataclass(frozen=True)
class QualityGateConfig:
    """Promotion thresholds for the CT pipeline."""

    min_hr: float = 0.15
    min_ndcg: float = 0.12
    min_ild: float = 0.30
    max_latency_ms: float = 30.0


def check_quality_gate(
    metrics: dict[str, float],
    config: QualityGateConfig | None = None,
    k: int = 10,
) -> None:
    """Validate ``metrics`` against ``config``; raise ``QualityGateError`` on breach."""
    cfg = config or QualityGateConfig()
    hr = float(metrics.get(f"HR@{k}", 0.0))
    ndcg = float(metrics.get(f"NDCG@{k}", 0.0))
    ild = float(metrics.get(f"ILD@{k}", metrics.get("ILD", 0.0)))
    latency = float(metrics.get("avg_latency_ms", float("inf")))

    failures = []
    if hr < cfg.min_hr:
        failures.append(f"HR@{k}={hr:.4f} < min {cfg.min_hr}")
    if ndcg < cfg.min_ndcg:
        failures.append(f"NDCG@{k}={ndcg:.4f} < min {cfg.min_ndcg}")
    if ild < cfg.min_ild:
        failures.append(f"ILD@{k}={ild:.4f} < min {cfg.min_ild}")
    if latency > cfg.max_latency_ms:
        failures.append(f"latency={latency:.2f}ms > max {cfg.max_latency_ms}ms")
    if failures:
        raise QualityGateError("Quality gate FAILED: " + "; ".join(failures))
