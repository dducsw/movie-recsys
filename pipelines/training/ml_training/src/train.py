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
    from recsys_core.metrics import (
        bootstrap_ranking_ci,
        evaluate_ranking_metrics,
        paired_bootstrap_delta,
        per_user_ranking_scores,
        rmse_mae,
    )
    from recsys_core.quality_gate import QualityGateConfig, QualityGateError, check_quality_gate


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
    axes[2].set_title("Inference Latency (Lower is Better)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Seconds per User (s/user)")
    axes[2].set_ylim(0, max(lgb_inf_time, cb_inf_time) * 1.18)
    axes[2].bar_label(axes[2].containers[0], fmt="%.4fs", padding=3, fontsize=9)

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
    """Log the run to MLflow (params, metrics, env metadata, artifacts)."""
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("MovieRecSys-Rating")
        with mlflow.start_run(run_name=f"ml_training_{int(time.time())}"):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_params({f"env_{k}": str(v) for k, v in _environment_metadata().items()})
            mlflow.log_dict(metrics, "metrics.json")
            mlflow.sklearn.log_model(
                bundle["lgbm"], artifact_path="lgbm",
                registered_model_name="MovieRating-LGBM")
            mlflow.sklearn.log_model(
                bundle["catboost"], artifact_path="catboost",
                registered_model_name="MovieRating-CatBoost")
    except Exception as exc:
        print(f">>> MLflow logging skipped ({exc}).")


def _environment_metadata() -> dict:
    import platform
    import subprocess

    import sklearn

    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5)
        meta["git_sha"] = out.decode().strip()
    except Exception:
        meta["git_sha"] = "nogit"
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

    t_inf_lgb = time.time()
    pred_te_lgb = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)
    lgb_inf_time = (time.time() - t_inf_lgb) / max(test["userId"].nunique(), 1)

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

    t_inf_cb = time.time()
    pred_te_cb = cb_model.predict(X_test_cb)
    cb_inf_time = (time.time() - t_inf_cb) / max(test["userId"].nunique(), 1)

    rmse_v_cb, _ = rmse_mae(y_val, pred_val_cb)
    rmse_t_cb, mae_t_cb = rmse_mae(y_te, pred_te_cb)
    print(f"[CB]   val RMSE={rmse_v_cb:.4f} | test RMSE={rmse_t_cb:.4f} MAE={mae_t_cb:.4f} | time={cb_time:.1f}s")

    # -------- Validation ranking metrics + paired model comparison --------
    eval_val_lgb = val[["userId", "movieId", "rating"]].copy()
    eval_val_lgb["pred"] = pred_val_lgb
    print(f"[VAL ranking] LGBM {evaluate_ranking_metrics(eval_val_lgb, k=k, relevance_threshold=relevance_threshold)[f'NDCG@{k}']:.4f}")

    eval_val_cb = val[["userId", "movieId", "rating"]].copy()
    eval_val_cb["pred"] = pred_val_cb

    # Paired bootstrap on the *validation* set: a point estimate alone is not
    # enough to declare a winner.
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

    # -------- Test ranking metrics --------
    eval_df_lgb = test[["userId", "movieId", "rating"]].copy()
    eval_df_lgb["pred"] = pred_te_lgb
    metrics_lgb = evaluate_ranking_metrics(eval_df_lgb, k=k, relevance_threshold=relevance_threshold)

    eval_df_cb = test[["userId", "movieId", "rating"]].copy()
    eval_df_cb["pred"] = pred_te_cb
    metrics_cb = evaluate_ranking_metrics(eval_df_cb, k=k, relevance_threshold=relevance_threshold)

    per_user_test_lgb = per_user_ranking_scores(eval_df_lgb, k=k, relevance_threshold=relevance_threshold)
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
    print(f"| **Inference Time (sec/user)** | {lgb_inf_time:.4f}s | {cb_inf_time:.4f}s |")
    print(f"| **Evaluable users** | {metrics_lgb['n_users_evaluated']:.0f} | {metrics_cb['n_users_evaluated']:.0f} |")

    print(f"\n>>> Best model based on Validation NDCG@10: {best_model_name}")

    plot_and_save_comparison(
        metrics_lgb=metrics_lgb,
        metrics_cb=metrics_cb,
        lgb_time=lgb_time,
        cb_time=cb_time,
        lgb_inf_time=lgb_inf_time,
        cb_inf_time=cb_inf_time,
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
        "avg_latency_ms": round(lgb_inf_time * 1000.0, 2),
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
