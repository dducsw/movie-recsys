"""
recsys_core.quality_gate
------------------------
Declarative CT quality gates: absolute thresholds + incumbent comparison.

The gate is the *last* thing that runs before an artifact may be promoted to
production.  Two classes of check are supported:

* **Absolute thresholds** — the new model must be good enough on its own
  (``min_hr``, ``min_ndcg``, ``min_ild``, ``max_latency_ms``).
* **Incumbent comparison** — the new model must be *better than what is
  currently serving* (``min_rmse_improvement_pct``) and must not regress the
  ranking guardrail (``max_ndcg_regression``).  A model can improve RMSE while
  ranking quality regresses, hence the NDCG guardrail.

Metric keys are the canonical ones from :mod:`recsys_core.metrics`
(``HR@{k}``, ``NDCG@{k}``, ``MRR@{k}``); the legacy ``Hit Ratio@{k}`` alias is
accepted for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


class QualityGateError(AssertionError):
    """Raised when a CT run fails a promotion gate."""


@dataclass(frozen=True)
class QualityGateConfig:
    """Promotion thresholds for the CT pipeline."""

    # --- absolute thresholds -------------------------------------------------
    min_hr: float = 0.15
    min_ndcg: float = 0.12
    min_ild: float = 0.30
    max_latency_ms: float = 30.0
    # Optional tail-latency cap: a slow outlier request must not hide behind
    # a low mean — p95 is the number real users feel.
    max_latency_p95_ms: Optional[float] = None
    # Optional absolute cap on the rating-prediction error (regression models).
    max_rmse: Optional[float] = None

    # --- statistical sanity --------------------------------------------------
    # A gate must never pass on a handful of users.
    min_users_evaluated: int = 1

    # --- incumbent comparison ------------------------------------------------
    # Required relative RMSE improvement over the incumbent (0.005 == 0.5%).
    min_rmse_improvement_pct: Optional[float] = None
    # Maximum tolerated relative NDCG regression vs the incumbent (0.003 == 0.3%).
    max_ndcg_regression: float = 0.003
    # When True, the gate fails if no incumbent metrics are supplied.
    require_baseline: bool = False


def _get_metric(metrics: Mapping[str, float], key: str) -> Optional[float]:
    """Read a metric by canonical key, accepting the legacy alias."""
    if key in metrics:
        return float(metrics[key])
    if key.startswith("HR@"):
        legacy = f"Hit Ratio{key[2:]}"
        if legacy in metrics:
            return float(metrics[legacy])
    return None


def check_quality_gate(
    metrics: Mapping[str, float],
    config: QualityGateConfig | None = None,
    k: int = 10,
    baseline_metrics: Optional[Mapping[str, float]] = None,
) -> None:
    """Validate ``metrics`` against ``config``; raise ``QualityGateError`` on breach.

    Args:
        metrics: metrics of the *candidate* model.
        config: thresholds; defaults to :class:`QualityGateConfig`.
        k: ranking cutoff used by the metric keys.
        baseline_metrics: metrics of the currently promoted model.  When given,
            the candidate must beat it on RMSE and must not regress NDCG.
    """
    cfg = config or QualityGateConfig()
    failures: list[str] = []

    # --- statistical sanity --------------------------------------------------
    n_users = _get_metric(metrics, "n_users_evaluated")
    if n_users is not None and n_users < cfg.min_users_evaluated:
        failures.append(
            f"n_users_evaluated={int(n_users)} < min {cfg.min_users_evaluated} "
            "(metric is not statistically meaningful)"
        )

    # --- absolute thresholds -------------------------------------------------
    hr = _get_metric(metrics, f"HR@{k}")
    if hr is None:
        failures.append(f"metric HR@{k} missing from metrics dict")
    elif hr < cfg.min_hr:
        failures.append(f"HR@{k}={hr:.4f} < min {cfg.min_hr}")

    ndcg = _get_metric(metrics, f"NDCG@{k}")
    if ndcg is None:
        failures.append(f"metric NDCG@{k} missing from metrics dict")
    elif ndcg < cfg.min_ndcg:
        failures.append(f"NDCG@{k}={ndcg:.4f} < min {cfg.min_ndcg}")

    ild = _get_metric(metrics, f"ILD@{k}")
    if ild is None:
        ild = _get_metric(metrics, "ILD")
    # ILD is a beyond-accuracy metric: only enforced when the pipeline
    # actually computes it.
    if ild is not None and ild < cfg.min_ild:
        failures.append(f"ILD@{k}={ild:.4f} < min {cfg.min_ild}")

    rmse = metrics.get("rmse")
    if cfg.max_rmse is not None:
        if rmse is None:
            failures.append("metric rmse missing from metrics dict")
        elif float(rmse) > cfg.max_rmse:
            failures.append(f"rmse={float(rmse):.4f} > max {cfg.max_rmse}")

    latency = metrics.get("avg_latency_ms")
    if latency is None:
        failures.append("metric avg_latency_ms missing from metrics dict")
    elif float(latency) > cfg.max_latency_ms:
        failures.append(f"latency={float(latency):.2f}ms > max {cfg.max_latency_ms}ms")

    if cfg.max_latency_p95_ms is not None:
        latency_p95 = metrics.get("latency_p95_ms")
        if latency_p95 is None:
            failures.append("metric latency_p95_ms missing from metrics dict")
        elif float(latency_p95) > cfg.max_latency_p95_ms:
            failures.append(
                f"latency p95={float(latency_p95):.2f}ms > max {cfg.max_latency_p95_ms}ms")

    # --- incumbent comparison ------------------------------------------------
    if baseline_metrics is None:
        if cfg.require_baseline:
            failures.append("no incumbent metrics supplied but require_baseline=True")
    else:
        failures.extend(_compare_to_incumbent(metrics, baseline_metrics, cfg, k))

    if failures:
        raise QualityGateError("Quality gate FAILED: " + "; ".join(failures))


def _compare_to_incumbent(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    cfg: QualityGateConfig,
    k: int,
) -> list[str]:
    """Compare candidate metrics against the incumbent model."""
    failures: list[str] = []

    # RMSE: candidate must be at least X% better than the incumbent.
    if cfg.min_rmse_improvement_pct is not None:
        cand_rmse = metrics.get("rmse")
        base_rmse = baseline.get("rmse")
        if cand_rmse is None or base_rmse is None:
            failures.append(
                "rmse comparison against incumbent requires 'rmse' in both "
                f"candidate and baseline metrics (candidate={cand_rmse}, baseline={base_rmse})"
            )
        elif base_rmse > 0:
            improvement = (base_rmse - float(cand_rmse)) / base_rmse
            if improvement < cfg.min_rmse_improvement_pct:
                failures.append(
                    f"rmse improvement {improvement:.4%} < required "
                    f"{cfg.min_rmse_improvement_pct:.4%} vs incumbent "
                    f"(candidate={float(cand_rmse):.4f}, baseline={float(base_rmse):.4f})"
                )

    # NDCG: candidate must not regress beyond the tolerated margin.
    cand_ndcg = _get_metric(metrics, f"NDCG@{k}")
    base_ndcg = _get_metric(baseline, f"NDCG@{k}")
    if cand_ndcg is not None and base_ndcg is not None and base_ndcg > 0:
        regression = (base_ndcg - cand_ndcg) / base_ndcg
        if regression > cfg.max_ndcg_regression:
            failures.append(
                f"NDCG@{k} regression {regression:.4%} > tolerated "
                f"{cfg.max_ndcg_regression:.4%} vs incumbent "
                f"(candidate={cand_ndcg:.4f}, baseline={base_ndcg:.4f})"
            )

    return failures
