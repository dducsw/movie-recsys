"""
pipelines/training/tests/test_p1_fixes.py
-----------------------------------------
Regression tests for the Senior-MLE review P0/P1 fixes:

* canonical metric keys + bootstrap CI + paired delta (recsys_core.metrics)
* quality gate: incumbent comparison, min_users, alias fallback
* shared encoding helpers (recsys_core.encoding) round-trip
* versioning two-phase save/promote/rollback
* data validation hardening (NaN ratings, duplicate pairs, referential
  integrity, drift raise)
"""

import numpy as np
import pandas as pd
import pytest

from recsys_core.metrics import evaluate_ranking_metrics


# ── 1. Metrics: canonical keys + bootstrap ───────────────────────────────────

def _rng_eval_df(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, 21):  # 20 users
        n = 12
        ratings = rng.integers(1, 6, size=n).astype(float)
        preds = rng.random(n)
        for m in range(n):
            rows.append({"userId": u, "movieId": u * 100 + m,
                         "rating": float(ratings[m]), "pred": float(preds[m])})
    return pd.DataFrame(rows)


def test_evaluate_ranking_metrics_canonical_keys_and_counts():
    df = _rng_eval_df()
    res = evaluate_ranking_metrics(df, k=10, relevance_threshold=4.0)
    assert res["HR@10"] == pytest.approx(res["Hit Ratio@10"])
    assert "NDCG@10" in res and "MRR@10" in res
    assert res["n_users_evaluated"] == 20


def test_skip_users_without_relevant_items():
    # 1 user with a relevant item + 1 user with only low ratings (skipped,
    # but still counted in n_users_total via n_users_evaluated difference).
    df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0, "pred": 0.9},
        {"userId": 2, "movieId": 2, "rating": 1.0, "pred": 0.9},
    ])
    res = evaluate_ranking_metrics(df, k=10, relevance_threshold=4.0)
    assert res["n_users_evaluated"] == 1
    assert res["n_users_skipped"] == 1
    assert res["HR@10"] == 1.0  # perfect for the single evaluated user


def test_bootstrap_ranking_ci_contains_point_estimate():
    from recsys_core.metrics import bootstrap_ranking_ci, per_user_ranking_scores

    df = _rng_eval_df()
    scores = per_user_ranking_scores(df, k=10, relevance_threshold=4.0)
    ci = bootstrap_ranking_ci(scores, metric="ndcg", n_boot=300, seed=42)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["n"] == 20


def test_paired_bootstrap_delta_can_be_computed():
    from recsys_core.metrics import paired_bootstrap_delta, per_user_ranking_scores

    rng = np.random.default_rng(3)
    users = list(range(1, 31))
    base_df = pd.DataFrame({
        "userId": np.repeat(users, 3),
        "movieId": [u * 10 + j for u in users for j in range(3)],
        "rating": np.tile([5.0, 4.0, 1.0], len(users)),
        "pred": rng.random(len(users) * 3),
    })
    challenger_df = base_df.copy()
    challenger_df["pred"] = rng.random(len(base_df))

    scores_a = per_user_ranking_scores(base_df, k=3, relevance_threshold=4.0)
    scores_b = per_user_ranking_scores(challenger_df, k=3, relevance_threshold=4.0)
    delta = paired_bootstrap_delta(scores_a, scores_b, metric="ndcg", n_boot=200, seed=42)
    assert set(delta) >= {"delta", "ci_low", "ci_high", "significant"}
    assert delta["ci_low"] <= delta["ci_high"]
    assert delta["n_users"] == 30


# ── 2. Quality gate ──────────────────────────────────────────────────────────

def _gate_metrics(hr=0.3, ndcg=0.3, **extra):
    return {"HR@10": hr, "NDCG@10": ndcg, "avg_latency_ms": 5.0,
            "n_users_evaluated": 50, "rmse": 0.85, **extra}


def test_gate_passes_absolute_thresholds():
    from recsys_core.quality_gate import check_quality_gate

    check_quality_gate(_gate_metrics())  # clean pass on defaults


def test_gate_rejects_on_legacy_alias_and_canonical_equally():
    from recsys_core.quality_gate import check_quality_gate, QualityGateError

    legacy_pass = {"Hit Ratio@10": 0.9, "NDCG@10": 0.9, "avg_latency_ms": 5.0}
    check_quality_gate(legacy_pass)  # alias accepted
    legacy_fail = {"Hit Ratio@10": 0.1, "NDCG@10": 0.9, "avg_latency_ms": 5.0}
    with pytest.raises(QualityGateError, match="HR@10"):
        check_quality_gate(legacy_fail)


def test_gate_min_users_evaluated():
    from recsys_core.quality_gate import check_quality_gate, QualityGateConfig, QualityGateError

    metrics = _gate_metrics(n_users_evaluated=10)
    with pytest.raises(QualityGateError, match="statistically"):
        check_quality_gate(metrics, QualityGateConfig(min_users_evaluated=100))


def test_compare_rmse_improvement_and_ndcg_regression():
    from recsys_core.quality_gate import check_quality_gate, QualityGateConfig

    cfg = QualityGateConfig(max_latency_ms=30.0, min_rmse_improvement_pct=0.05,
                            max_ndcg_regression=0.05, min_hr=0.0, min_ndcg=0.0)
    baseline = {"rmse": 1.0, "NDCG@10": 0.20}
    # Candidate improves rmse 10% and does not regress NDCG -> passes.
    check_quality_gate(
        {"HR@10": 0.5, "NDCG@10": 0.21, "rmse": 0.90, "avg_latency_ms": 5.0},
        cfg, baseline_metrics=baseline)
    # NDCG regresses 10% -> fails even though RMSE improved.
    from recsys_core.quality_gate import QualityGateError

    with pytest.raises(QualityGateError, match="regression"):
        check_quality_gate(
            {"HR@10": 0.5, "NDCG@10": 0.18, "rmse": 0.90, "avg_latency_ms": 5.0},
            cfg, baseline_metrics=baseline)
    # RMSE improvement below the required 5% -> fails.
    with pytest.raises(QualityGateError, match="improvement"):
        check_quality_gate(
            {"HR@10": 0.5, "NDCG@10": 0.20, "rmse": 0.99, "avg_latency_ms": 5.0},
            cfg, baseline_metrics=baseline)


# ── 3. Encoding helpers ──────────────────────────────────────────────────────

def test_validate_category_maps():
    from recsys_core.encoding import validate_category_maps
    import pandas as pd

    assert validate_category_maps({"movieId": pd.Index([1, 2, 3])})
    assert not validate_category_maps({})
    assert not validate_category_maps(None)
    assert not validate_category_maps({"c": 3})  # scalars unusable


def test_best_iteration_of():
    from recsys_core.encoding import best_iteration_of

    class M:
        best_iteration = 7

    assert best_iteration_of(M()) == 7
    assert best_iteration_of(object) is None  # no attribute
    m0 = M()
    m0.best_iteration = 0
    assert best_iteration_of(m0) is None  # 0 means "not early stopped"


def test_prepare_lgb_prediction_data_roundtrip():
    from recsys_core.encoding import prepare_lgb_prediction_data

    df = pd.DataFrame({"movieId": [3, 1], "vote_average": [7.0, 6.0]})
    vocab = pd.Index([1, 2, 3])
    X = prepare_lgb_prediction_data(df, ["movieId", "vote_average"], {"movieId": vocab})
    assert list(X.columns) == ["movieId", "vote_average"]
    assert X["movieId"].dtype == "category"
    assert list(X["movieId"].cat.categories) == [1, 2, 3]
    # Unknown values become NaN without erroring (consistent with training).
    X_unseen = prepare_lgb_prediction_data(
        pd.DataFrame({"movieId": [9]}), ["movieId"], {"movieId": vocab})
    assert X_unseen["movieId"].isna().all()


# ── 4. Versioning: rollback ──────────────────────────────────────────────────

def test_versioning_rollback(tmp_path):
    from pipelines.training import versioning as ver

    v1 = ver.save_versioned({"m": 1}, str(tmp_path), "lgb_ranker")
    promoted1 = ver.promote(str(tmp_path), "lgb_ranker", v1)
    v2 = ver.save_versioned({"m": 2}, str(tmp_path), "lgb_ranker")
    promoted2 = ver.promote(str(tmp_path), "lgb_ranker", v2)
    assert ver.read_champion(str(tmp_path), "lgb_ranker") == promoted2

    restored = ver.rollback(str(tmp_path), "lgb_ranker")
    assert restored == promoted1
    assert ver.read_champion(str(tmp_path), "lgb_ranker") == promoted1

    # Metrics sidecar round-trip
    ver.save_metrics(str(tmp_path), "lgb_ranker", promoted1, {"rmse": 0.9})
    assert ver.load_champion_metrics(str(tmp_path), "lgb_ranker") == {"rmse": 0.9}


# ── 5. Data validation hardening ─────────────────────────────────────────────

def test_validate_ratings_rejects_nan_rating():
    from pipelines.training.validate import validate_ratings

    df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": np.nan, "timestamp": 1},
    ])
    with pytest.raises(ValueError, match="null/non-numeric"):
        validate_ratings(df)


def test_validate_ratings_rejects_duplicate_pairs():
    from pipelines.training.validate import validate_ratings

    df = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1},
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 2},
    ])
    with pytest.raises(ValueError, match="duplicated"):
        validate_ratings(df)


def test_validate_referential_integrity_catches_orphans():
    from pipelines.training.validate import validate_referential_integrity

    ratings = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 4.0},
        {"userId": 1, "movieId": 999, "rating": 4.0},
    ])
    movies = pd.DataFrame([{"movieId": 1, "title": "A", "genres": "Action"}])
    with pytest.raises(ValueError, match="missing from"):
        validate_referential_integrity(ratings, movies)


def test_check_data_drift_raises_on_drift_alert():
    from pipelines.training.validate import check_data_drift

    reference = pd.DataFrame({"rating": np.random.default_rng(1).normal(3.5, 0.5, 5000)})
    current = pd.DataFrame({"rating": np.random.default_rng(2).normal(2.0, 3.5, 5000)})
    with pytest.raises(ValueError, match="drift"):
        check_data_drift(current, reference, raise_on_drift=True)
    # Explicit opt-out keeps the old report-only behavior for callers that
    # want to alert instead of fail.
    report = check_data_drift(current, reference, raise_on_drift=False)
    assert report["status"] == "DRIFT_ALERT"

    # No drift vs same distribution -> PASS, does not raise.
    same = pd.DataFrame({"rating": np.random.default_rng(1).normal(3.5, 0.5, 5000)})
    assert check_data_drift(same, reference)["status"] == "PASS"
