"""
pipelines.training.ml_training.src.train
------------------------------------------
Offline A/B training for the rating-regression ensemble that serves Stage 2.

Pipeline:
    1. Time-based per-user split (leave-future-out).
    2. User features built on the **train split only**.
    3. LightGBM + CatBoost trained with early stopping.
    4. Offline evaluation on validation & test with **bootstrap CIs** and a
       **paired** model comparison.
    5. Quality gate against the currently published bundle (RMSE improvement +
       NDCG regression guardrail).
    6. Publish to ``models.joblib`` **only if the gate passes** (atomic swap).

Encoding contract (see ``models.py``):
    * LightGBM is predicted through ``prepare_lgb_prediction_data`` using the
      persisted ``category_maps``.
    * CatBoost is predicted through ``prepare_catboost_prediction_data`` so
      the frame is encoded exactly like the training frame.
"""

import json
import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from data_loader import load_all
from features import build_dataset, build_movie_features, build_user_features
from models import (
    prepare_catboost_prediction_data,
    prepare_lgb_prediction_data,
    train_catboost,
    train_lgbm,
)

try:
    from configs import CFG
except ImportError:
    from .configs import CFG


# ── Data split ───────────────────────────────────────────────────────────────

def time_split_per_user(ratings, n_test=5, n_val=5, min_ratings=15):
    """Leave-future-out split, per user, ordered by ``timestamp``.

    Deterministic by construction (pure sort), so no RNG seed is required.
    Users with fewer than ``min_ratings`` interactions stay entirely in train.
    """
    train_idx, val_idx, test_idx = [], [], []
    for uid, g in ratings.groupby("userId"):
        g = g.sort_values("timestamp")
        if len(g) < min_ratings:
            train_idx.extend(g.index.tolist())
            continue
        idx = g.index.tolist()
        test_idx.extend(idx[-n_test:])
        val_idx.extend(idx[-(n_test + n_val):-n_test])
        train_idx.extend(idx[:-(n_test + n_val)])
    return (ratings.loc[train_idx].reset_index(drop=True),
            ratings.loc[val_idx].reset_index(drop=True),
            ratings.loc[test_idx].reset_index(drop=True))


try:
    from recsys_core.envinfo import git_sha
    from recsys_core.metrics import (
        bootstrap_ranking_ci,
        evaluate_ranking_metrics,
        paired_bootstrap_delta,
        per_user_ranking_scores,
        rmse_mae,
    )
    from recsys_core.quality_gate import QualityGateConfig, QualityGateError, check_quality_gate
except ImportError:
    import sys
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    _core_path = os.path.join(_repo_root, "libs", "recsys_core", "src")
    if _core_path not in sys.path:
        sys.path.insert(0, _core_path)
    from recsys_core.envinfo import git_sha
    from recsys_core.metrics import (
        bootstrap_ranking_ci,
        evaluate_ranking_metrics,
        paired_bootstrap_delta,
        per_user_ranking_scores,
        rmse_mae,
    )
    from recsys_core.quality_gate import QualityGateConfig, QualityGateError, check_quality_gate


# ── Negative-sampling ranking evaluation (CT 99-neg protocol) ────────────────

def build_negative_pools(
    ratings: pd.DataFrame,
    target_df: pd.DataFrame,
    catalog_ids,
    n_negatives: int,
    seed: int,
) -> pd.DataFrame:
    """Per-user evaluation pool = ground-truth rows + ``n_negatives`` negatives.

    Ranking only the user's own holdout rows is degenerate — HR@k is
    identically 1.0 whenever k >= holdout size — so every ranking metric is
    computed over a pool that also contains sampled **unrated** catalog
    items, mirroring the CT 99-neg protocol.

    Pools are deterministic and shared: every model is scored on the exact
    same candidates for a split, which keeps the A/B comparison paired.
    """
    rng = np.random.default_rng(seed)
    rated = ratings.groupby("userId")["movieId"].agg(
        lambda s: set(s.astype(int))).to_dict()
    catalog = np.asarray(sorted({int(m) for m in catalog_ids}), dtype=int)

    frames = [target_df[["userId", "movieId", "rating", "timestamp"]]]
    for uid, _gt in target_df.groupby("userId"):
        avail = catalog[~np.isin(catalog, sorted(rated.get(uid, set())))]
        picked = rng.choice(avail, size=min(n_negatives, len(avail)), replace=False)
        frames.append(pd.DataFrame({
            "userId": uid, "movieId": picked, "rating": 0.0, "timestamp": 0}))
    return pd.concat(frames, ignore_index=True)


def score_pools(pool_ratings, movie_feats, user_feats, prepare, predict):
    """Score every user's pool separately; returns (eval_df, latency ms list).

    Latency is measured **per user** (feature encoding + predict), mirroring
    the serving hot path — a single batched predict divided by users hides
    exactly the tail a p95 gate has to see.
    """
    pool = build_dataset(pool_ratings, movie_feats, user_feats)
    eval_frames, latencies = [], []
    for _uid, rows in pool.groupby("userId"):
        features = rows.drop(
            columns=[c for c in ("rating", "timestamp") if c in rows]).copy()
        t0 = time.perf_counter()
        preds = np.asarray(predict(prepare(features)), dtype=float)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scored = rows[["userId", "movieId", "rating"]].copy()
        scored["pred"] = preds
        eval_frames.append(scored)
    return pd.concat(eval_frames, ignore_index=True), latencies


def latency_stats(latencies_ms) -> dict:
    """Mean / p50 / p95 of per-request scoring latencies (0.0 on empty input)."""
    if not latencies_ms:
        return {"avg_latency_ms": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0}
    return {
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    }


# ── Hyperparameters ──────────────────────────────────────────────────────────

def _best_params_path() -> str:
    """Absolute path of ``best_params.json`` (never CWD-relative)."""
    ml_training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ml_training_dir, "best_params.json")


def load_tuned_params(path: Optional[str] = None) -> dict:
    """Load tuned hyperparameters and **merge** them over the base config.

    The tuned file only carries the *searched* parameters; everything else
    (notably ``n_estimators`` / ``early_stopping_rounds`` / ``task_type`` /
    the RNG seeds) stays at the production value from ``configs.yaml``.  A
    tuning run must never silently downgrade the training budget.
    """
    path = path or _best_params_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f">>> Ignoring unreadable {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        print(f">>> Ignoring {path}: expected a JSON object.")
        return {}

    # Older tuning files inlined the boosting budget; drop it so the config
    # value wins.
    for key in ("lgbm", "catboost"):
        block = payload.get(key)
        if isinstance(block, dict):
            block.pop("n_estimators", None)
            block.pop("iterations", None)
            block.pop("early_stopping_rounds", None)
    return payload


def merge_params(base: dict, tuned: Optional[dict]) -> dict:
    """Merge a tuned block over the base config (tuned wins, base fills gaps)."""
    merged = dict(base)
    if tuned:
        merged.update(tuned)
    return merged


# ── Reporting ────────────────────────────────────────────────────────────────

def plot_and_save_comparison(
    metrics_lgb: dict,
    metrics_cb: dict,
    lgb_time: float,
    cb_time: float,
    lgb_inf_time: float,
    cb_inf_time: float,
    save_path: str = "model_comparison.png"
):
    """Plot a multi-panel comparison between LightGBM and CatBoost."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ["#2b5c8f", "#d95f02"]
    models = ["LightGBM", "CatBoost"]

    ranking_metrics = ["HR@10", "NDCG@10", "MRR@10"]
    lgb_scores = [metrics_lgb.get(m, 0.0) for m in ranking_metrics]
    cb_scores = [metrics_cb.get(m, 0.0) for m in ranking_metrics]

    x = np.arange(len(ranking_metrics))
    width = 0.35

    rects1 = axes[0].bar(x - width / 2, lgb_scores, width, label="LightGBM", color=colors[0], edgecolor="black", alpha=0.85)
    rects2 = axes[0].bar(x + width / 2, cb_scores, width, label="CatBoost", color=colors[1], edgecolor="black", alpha=0.85)

    axes[0].set_title("Ranking Quality (Higher is Better)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ranking_metrics, fontsize=10)
    axes[0].set_ylim(0, max(max(lgb_scores), max(cb_scores)) * 1.15)
    axes[0].legend(loc="upper left")
    axes[0].bar_label(rects1, fmt="%.4f", padding=3, fontsize=8.5)
    axes[0].bar_label(rects2, fmt="%.4f", padding=3, fontsize=8.5)

    axes[1].bar(models, [lgb_time, cb_time], color=colors, width=0.45, edgecolor="black", alpha=0.85)
    axes[1].set_title("Training Time (Lower is Better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Seconds (s)")
    axes[1].set_ylim(0, max(lgb_time, cb_time) * 1.18)
    axes[1].bar_label(axes[1].containers[0], fmt="%.2fs", padding=3, fontsize=9)

    axes[2].bar(models, [lgb_inf_time, cb_inf_time], color=colors, width=0.45, edgecolor="black", alpha=0.85)
    axes[2].set_title("Scoring Latency p95 (Lower is Better)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Milliseconds (ms)")
    axes[2].set_ylim(0, max(lgb_inf_time, cb_inf_time) * 1.18)
    axes[2].bar_label(axes[2].containers[0], fmt="%.1fms", padding=3, fontsize=9)

    plt.suptitle("Offline A/B Comparison: LightGBM vs. CatBoost", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved successfully to: {save_path}")
    plt.show()


# ── Publishing ───────────────────────────────────────────────────────────────

def _bundle_path() -> str:
    ml_training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ml_training_dir, "models.joblib")


def load_published_bundle(path: Optional[str] = None):
    """Load the currently published bundle, or ``None``."""
    path = path or _bundle_path()
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f">>> Could not read the published bundle: {exc}")
        return None


def publish_bundle(bundle: dict, path: Optional[str] = None) -> str:
    """Atomically publish the bundle (write temp file, then rename)."""
    path = path or _bundle_path()
    tmp_path = f"{path}.new"
    joblib.dump(bundle, tmp_path)
    os.replace(tmp_path, path)
    return path


def log_to_mlflow(bundle: dict, metrics: dict, params: dict) -> None:
    """Log the run to MLflow (params, metrics, env metadata, artifacts).

    Uses the correct flavors (``mlflow.lightgbm`` / ``mlflow.catboost``) so
    the registered weights are actually loadable from the registry.  Raises
    on failure: a registration that silently never happened must not pass as
    "tracked" (alert, never lose a production artifact reference quietly).
    """
    import mlflow
    import mlflow.catboost
    import mlflow.lightgbm

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("MovieRecSys-Rating")
    with mlflow.start_run(run_name=f"ml_training_{int(time.time())}"):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_params({f"env_{k}": str(v) for k, v in _environment_metadata().items()})
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.lightgbm.log_model(
            bundle["lgbm"], artifact_path="lgbm",
            registered_model_name="MovieRating-LGBM")
        mlflow.catboost.log_model(
            bundle["catboost"], artifact_path="catboost",
            registered_model_name="MovieRating-CatBoost")


def _environment_metadata() -> dict:
    import platform

    import sklearn

    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "git_sha": git_sha(),
    }
    return meta


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    eval_cfg = CFG.get("evaluation", {})
    k = int(eval_cfg.get("k", 10))
    relevance_threshold = float(eval_cfg.get("relevance_threshold", 4.0))
    n_boot = int(eval_cfg.get("bootstrap_iters", 1000))
    seed = int(CFG.get("split", {}).get("random_state", 42))

    print(">>> Loading data ...")
    crawled, links, movies, ratings, enriched = load_all()

    print(">>> Building movie features ...")
    movie_feats = build_movie_features(enriched)

    print(">>> Splitting ratings (time-based per user) ...")
    s = CFG["split"]
    train_r, val_r, test_r = time_split_per_user(
        ratings, n_test=s["n_holdout_per_user"], n_val=s["n_holdout_per_user"],
        min_ratings=2 * s["n_holdout_per_user"] + s["min_user_ratings"])

    print(">>> Building user features on train ...")
    user_feats = build_user_features(train_r, movie_feats)

    print(">>> Assembling datasets ...")
    train = build_dataset(train_r, movie_feats, user_feats)
    val = build_dataset(val_r, movie_feats, user_feats)
    test = build_dataset(test_r, movie_feats, user_feats)

    y_tr = train["rating"].astype(float).values
    y_val = val["rating"].astype(float).values
    y_te = test["rating"].astype(float).values

    tuned_params = load_tuned_params()
    if tuned_params:
        print(">>> Found tuned hyperparameters (merged over configs.yaml).")

    # -------- LightGBM --------
    print(">>> Training LightGBM ...")
    t0 = time.time()
    lgb_params = merge_params(CFG["lgbm"], tuned_params.get("lgbm"))

    lgb_model, lgb_cols, lgb_category_maps = train_lgbm(
        train.drop(columns=["rating", "timestamp"]).copy(),
        y_tr,
        val.drop(columns=["rating", "timestamp"]).copy(),
        y_val,
        params=lgb_params
    )
    lgb_time = time.time() - t0

    # LightGBM prediction frames use the *training* category vocabulary.
    X_val_lgb = prepare_lgb_prediction_data(val, lgb_cols, lgb_category_maps)
    X_test_lgb = prepare_lgb_prediction_data(test, lgb_cols, lgb_category_maps)

    pred_val_lgb = lgb_model.predict(X_val_lgb, num_iteration=lgb_model.best_iteration)
    pred_te_lgb = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)

    rmse_v_lgb, _ = rmse_mae(y_val, pred_val_lgb)
    rmse_t_lgb, mae_t_lgb = rmse_mae(y_te, pred_te_lgb)
    print(f"[LGBM] val RMSE={rmse_v_lgb:.4f} | test RMSE={rmse_t_lgb:.4f} MAE={mae_t_lgb:.4f} | time={lgb_time:.1f}s")

    # -------- CatBoost --------
    print(">>> Training CatBoost ...")
    t0 = time.time()
    cb_params = merge_params(CFG["catboost"], tuned_params.get("catboost"))

    cb_model, cb_cols = train_catboost(
        train.drop(columns=["rating", "timestamp"]).copy(),
        y_tr,
        val.drop(columns=["rating", "timestamp"]).copy(),
        y_val,
        params=cb_params
    )
    cb_time = time.time() - t0

    # CatBoost prediction frames must be encoded exactly like the training
    # frames (categoricals as str), otherwise the offline metrics are garbage.
    from models import feature_cols
    _, cb_cats = feature_cols(train)
    X_val_cb = prepare_catboost_prediction_data(val, cb_cols, cb_cats)
    X_test_cb = prepare_catboost_prediction_data(test, cb_cols, cb_cats)

    pred_val_cb = cb_model.predict(X_val_cb)
    pred_te_cb = cb_model.predict(X_test_cb)

    rmse_v_cb, _ = rmse_mae(y_val, pred_val_cb)
    rmse_t_cb, mae_t_cb = rmse_mae(y_te, pred_te_cb)
    print(f"[CB]   val RMSE={rmse_v_cb:.4f} | test RMSE={rmse_t_cb:.4f} MAE={mae_t_cb:.4f} | time={cb_time:.1f}s")

    # -------- Ranking metrics (negative-sampling protocol) --------
    # pred_te_* above only feed RMSE; ranking is scored on dedicated pools so
    # the metrics measure actual ranking skill against distractors.
    n_negatives = int(eval_cfg.get("ranking_negatives", 99))
    catalog_ids = movie_feats["movieId"].dropna().astype(int).unique()
    print(f">>> Sampling ranking pools ({n_negatives} negatives/user, seeded) ...")
    val_pool = build_negative_pools(ratings, val_r, catalog_ids, n_negatives, seed + 1)
    test_pool = build_negative_pools(ratings, test_r, catalog_ids, n_negatives, seed)

    def _prep_lgb(df):
        return prepare_lgb_prediction_data(df, lgb_cols, lgb_category_maps)

    def _prep_cb(df):
        return prepare_catboost_prediction_data(df, cb_cols, cb_cats)

    lgb_iter = lgb_model.best_iteration or None
    eval_val_lgb, _lat_val_lgb = score_pools(
        val_pool, movie_feats, user_feats, _prep_lgb,
        lambda X: lgb_model.predict(X, num_iteration=lgb_iter))
    eval_val_cb, _ = score_pools(
        val_pool, movie_feats, user_feats, _prep_cb, lambda X: cb_model.predict(X))
    eval_te_lgb, lat_te_lgb = score_pools(
        test_pool, movie_feats, user_feats, _prep_lgb,
        lambda X: lgb_model.predict(X, num_iteration=lgb_iter))
    eval_te_cb, lat_te_cb = score_pools(
        test_pool, movie_feats, user_feats, _prep_cb, lambda X: cb_model.predict(X))

    # Paired bootstrap on the *validation* pools: a point estimate alone is
    # not enough to declare a winner.
    per_user_lgb = per_user_ranking_scores(eval_val_lgb, k=k, relevance_threshold=relevance_threshold)
    per_user_cb = per_user_ranking_scores(eval_val_cb, k=k, relevance_threshold=relevance_threshold)
    delta = paired_bootstrap_delta(per_user_lgb, per_user_cb, metric="ndcg", n_boot=n_boot, seed=seed)

    lgb_wins = delta["delta"] > 0
    best_model_name = "LightGBM" if lgb_wins else "CatBoost"
    print(f">>> Paired NDCG@10 delta (LGBM - CB): {delta['delta']:+.4f} "
          f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] "
          f"significant={delta['significant']} (n={delta['n_users']})")
    if not delta["significant"]:
        print(">>> Delta is within bootstrap noise — treat the two models as tied "
              "and prefer the incumbent/cheaper one.")

    # -------- Test ranking metrics (negative-sampling pools) --------
    metrics_lgb = evaluate_ranking_metrics(eval_te_lgb, k=k, relevance_threshold=relevance_threshold)
    metrics_cb = evaluate_ranking_metrics(eval_te_cb, k=k, relevance_threshold=relevance_threshold)

    per_user_test_lgb = per_user_ranking_scores(eval_te_lgb, k=k, relevance_threshold=relevance_threshold)
    ci = bootstrap_ranking_ci(per_user_test_lgb, metric="ndcg", n_boot=n_boot, seed=seed)
    metrics_lgb[f"NDCG@{k}_ci_low"] = round(ci["ci_low"], 4)
    metrics_lgb[f"NDCG@{k}_ci_high"] = round(ci["ci_high"], 4)
    metrics_lgb["rmse"] = round(rmse_t_lgb, 4)
    metrics_lgb["mae"] = round(mae_t_lgb, 4)

    print("\n=== OFFLINE EVALUATION RESULTS (A/B COMPARISON) ===")
    print("| Metric | LightGBM | CatBoost |")
    print("| :--- | :--- | :--- |")
    print(f"| **HR@{k}** | {metrics_lgb[f'HR@{k}']:.4f} | {metrics_cb[f'HR@{k}']:.4f} |")
    print(f"| **NDCG@{k}** | {metrics_lgb[f'NDCG@{k}']:.4f} | {metrics_cb[f'NDCG@{k}']:.4f} |")
    print(f"| **MRR@{k}** | {metrics_lgb[f'MRR@{k}']:.4f} | {metrics_cb[f'MRR@{k}']:.4f} |")
    print(f"| **Test RMSE** | {rmse_t_lgb:.4f} | {rmse_t_cb:.4f} |")
    print(f"| **Test MAE** | {mae_t_lgb:.4f} | {mae_t_cb:.4f} |")
    print(f"| **Train Time (sec)** | {lgb_time:.2f}s | {cb_time:.2f}s |")
    print(f"| **Scoring latency p95** | {np.percentile(lat_te_lgb, 95):.1f}ms | {np.percentile(lat_te_cb, 95):.1f}ms |" if lat_te_lgb and lat_te_cb else "| **Scoring latency p95** | n/a | n/a |")
    print(f"| **Evaluable users** | {metrics_lgb['n_users_evaluated']:.0f} | {metrics_cb['n_users_evaluated']:.0f} |")

    print(f"\n>>> Best model based on Validation NDCG@10: {best_model_name}")

    plot_and_save_comparison(
        metrics_lgb=metrics_lgb,
        metrics_cb=metrics_cb,
        lgb_time=lgb_time,
        cb_time=cb_time,
        lgb_inf_time=float(np.percentile(lat_te_lgb, 95)) if lat_te_lgb else 0.0,
        cb_inf_time=float(np.percentile(lat_te_cb, 95)) if lat_te_cb else 0.0,
        save_path="ranking_comparison_results.png"
    )

    # -------- Quality gate against the published bundle --------
    candidate_metrics = {
        "rmse": round(rmse_t_lgb, 4),
        "mae": round(mae_t_lgb, 4),
        f"HR@{k}": metrics_lgb[f"HR@{k}"],
        f"NDCG@{k}": metrics_lgb[f"NDCG@{k}"],
        f"MRR@{k}": metrics_lgb[f"MRR@{k}"],
        "n_users_evaluated": metrics_lgb["n_users_evaluated"],
        **latency_stats(lat_te_lgb),
    }

    incumbent = load_published_bundle()
    baseline_metrics = incumbent.get("metrics") if incumbent else None
    if baseline_metrics:
        print(f">>> Incumbent baseline: {baseline_metrics}")
    else:
        print(">>> No incumbent baseline available; gate uses absolute thresholds only.")

    gate_cfg_dict = dict(CFG.get("quality_gates", {}))
    gate_cfg = QualityGateConfig(**{
        k_: v for k_, v in gate_cfg_dict.items()
        if k_ in QualityGateConfig.__dataclass_fields__
    })

    published = False
    try:
        check_quality_gate(candidate_metrics, gate_cfg, k=k, baseline_metrics=baseline_metrics)
        print(">>> Quality gate PASSED.")
        published = True
    except QualityGateError as exc:
        print(f">>> Quality gate FAILED: {exc}")
        print(">>> The published bundle is left untouched (no overwrite on failure).")

    ensemble_cfg = CFG.get("ensemble", {})
    bundle = {
        "lgbm": lgb_model,
        "lgbm_cols": lgb_cols,
        "lgb_category_maps": lgb_category_maps,
        "lgb_best_iteration": int(getattr(lgb_model, "best_iteration", 0) or 0),
        "catboost": cb_model,
        "cb_cols": cb_cols,
        "user_feats": user_feats,
        "movie_feats": movie_feats,
        "config": CFG,
        # Serving reads this first so blending follows training by default.
        "ensemble_weights": {
            "catboost": float(ensemble_cfg.get("catboost_weight", 0.6)),
            "lightgbm": float(ensemble_cfg.get("lightgbm_weight", 0.4)),
        },
        "best_model": best_model_name,
        "metrics": candidate_metrics,
        "seed": seed,
    }

    if published:
        target_path = publish_bundle(bundle)
        print(f">>> Published models.joblib to {target_path}")
    else:
        target_path = _bundle_path()
        print(f">>> Bundle NOT published; {target_path} still serves the previous model.")

    log_to_mlflow(
        bundle,
        candidate_metrics,
        {
            "lgbm": lgb_params,
            "catboost": cb_params,
            "seed": seed,
            "split_protocol": "time_by_user_leave_future_out",
            "best_model": best_model_name,
            "published": published,
            "n_users_evaluated": candidate_metrics["n_users_evaluated"],
        },
    )


if __name__ == "__main__":
    main()
