"""Fail-fast data validation for CT. Simple pandas checks, no new deps."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def validate_movies(df: pd.DataFrame) -> None:
    """Raise ValueError if catalog is unusable."""
    if df is None or df.empty:
        raise ValueError("movies is empty.")
    missing = {"movieId", "title", "genres"} - set(df.columns)
    if missing:
        raise ValueError(f"movies missing columns: {sorted(missing)}.")
    if df["movieId"].isna().any() or df["movieId"].duplicated().any():
        raise ValueError("movieId must be unique and non-null.")


def validate_ratings(df: pd.DataFrame) -> None:
    """Raise ValueError if interactions are unusable."""
    if df is None or df.empty:
        raise ValueError("ratings is empty.")
    missing = {"userId", "movieId", "rating"} - set(df.columns)
    if missing:
        raise ValueError(f"ratings missing columns: {sorted(missing)}.")
    if ((df["rating"] < 0.5) | (df["rating"] > 5.0)).any():
        raise ValueError("rating out of range [0.5, 5.0].")
    if df[["userId", "movieId"]].isna().any().any():
        raise ValueError("userId/movieId contain nulls.")
    if "timestamp" in df.columns and df["timestamp"].isna().all():
        raise ValueError("timestamp all null; LOO split needs ordering.")


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) between expected and actual distributions.
    PSI < 0.1: No significant drift.
    0.1 <= PSI <= 0.25: Moderate drift.
    PSI > 0.25: Significant drift requiring retraining or investigation.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    raw_bins = np.percentile(expected, percentiles)
    bins = np.unique(raw_bins)
    if len(bins) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / max(1, len(expected)) + epsilon
    act_pct = act_counts / max(1, len(actual)) + epsilon

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def check_data_drift(
    current_ratings: pd.DataFrame,
    reference_ratings: pd.DataFrame | None = None,
    null_threshold: float = 0.05,
    psi_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Audit data & feature distributions before model training.
    Logs warnings or alerts if rating distribution or null rates drift beyond tolerance.
    """
    import logging
    logger = logging.getLogger("DataDriftMonitor")

    report: dict[str, Any] = {
        "status": "PASS",
        "drift_detected": False,
        "null_rates": {},
        "rating_psi": 0.0,
        "mean_rating_current": float(current_ratings["rating"].mean()) if not current_ratings.empty else 0.0,
        "alerts": []
    }

    # 1. Null rate validation
    for col in current_ratings.columns:
        rate = float(current_ratings[col].isna().mean())
        report["null_rates"][col] = round(rate, 4)
        if rate > null_threshold:
            msg = f"Feature '{col}' null rate {rate:.2%} exceeds threshold {null_threshold:.2%}"
            report["alerts"].append(msg)
            report["drift_detected"] = True
            logger.warning("ALERT: %s", msg)

    # 2. Rating distribution drift (PSI)
    if reference_ratings is not None and not reference_ratings.empty:
        report["mean_rating_reference"] = float(reference_ratings["rating"].mean())
        psi = calculate_psi(
            expected=reference_ratings["rating"].to_numpy(),
            actual=current_ratings["rating"].to_numpy(),
        )
        report["rating_psi"] = round(psi, 4)
        if psi > psi_threshold:
            msg = f"Significant rating distribution drift detected! PSI={psi:.4f} > {psi_threshold}"
            report["alerts"].append(msg)
            report["drift_detected"] = True
            report["status"] = "DRIFT_ALERT"
            logger.warning("ALERT: %s", msg)
        elif psi >= 0.10:
            logger.info("Moderate rating distribution shift observed (PSI=%.4f).", psi)

    return report
