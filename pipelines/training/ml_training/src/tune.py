"""
pipelines.training.ml_training.src.tune
-----------------------------------------
Optuna hyperparameter search for the rating-regression ensemble.

Design notes
------------
* The sampler and the pruner are **seeded and pinned** so a tuning run is
  reproducible and two runs on different days are comparable.
* Every trial is logged to MLflow as a *nested* run under the parent study
  run, so the whole search is auditable — not just the winning config.
* The objective is validation **RMSE** (a fast proxy); final model selection
  happens in ``train.py`` on validation **NDCG@10** with a paired bootstrap.
* ``best_params.json`` contains **only the searched parameters** plus
  provenance.  It must never carry the boosting budget — that is owned by
  ``configs.yaml`` and must not be silently downgraded by a tuning run.

Usage::

    python src/tune.py --n-trials 100
"""

import json
import os
import platform
import time
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from data_loader import load_all
from features import build_dataset, build_movie_features, build_user_features
from models import feature_cols
from sklearn.metrics import mean_squared_error
from train import time_split_per_user

try:
    from configs import CFG
except ImportError:
    from .configs import CFG

try:
    from recsys_core.envinfo import git_sha
except ImportError:
    import sys as _sys
    import os as _os
    _repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
    _core_src = _os.path.join(_repo_root, "libs", "recsys_core", "src")
    if _core_src not in _sys.path:
        _sys.path.insert(0, _core_src)
    from recsys_core.envinfo import git_sha

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_CFG: Dict[str, Any] = CFG.get("tuning", {})
# Env override beats the config file so CI/operators can widen a search
# without editing YAML.
N_TRIALS = int(os.getenv("TUNE_N_TRIALS", TUNING_CFG.get("n_trials", 30)))
TUNE_SEED = int(os.getenv("TUNE_SEED", TUNING_CFG.get("seed", 42)))
STUDY_PREFIX = TUNING_CFG.get("study_prefix", "movie-recsys")
PRUNER = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)


def _sampler() -> optuna.samplers.BaseSampler:
    """TPESampler with a fixed seed — tuning runs must be replayable."""
    return optuna.samplers.TPESampler(seed=TUNE_SEED)


def _git_sha() -> str:
    """Git provenance — single shared implementation in recsys_core.envinfo."""
    return git_sha()


def _provenance() -> Dict[str, Any]:
    """Metadata recorded next to the tuned params so they can be traced back."""
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trials_per_model": N_TRIALS,
        "sampler": "TPESampler",
        "sampler_seed": TUNE_SEED,
        "pruner": "MedianPruner",
        "objective": "val_rmse",
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "train_rows": None,   # filled by main()
        "val_rows": None,     # filled by main()
    }


# ── Objectives ───────────────────────────────────────────────────────────────

def _prep_lgbm(X_tr, X_val):
    cols, cats = feature_cols(X_tr)
    for c in cats:
        X_tr[c] = X_tr[c].astype("category")
        X_val[c] = pd.Categorical(X_val[c], categories=X_tr[c].cat.categories)
    return cols, cats


def objective_lgbm(trial, X_tr, y_tr, X_val, y_val):
    cols, cats = _prep_lgbm(X_tr, X_val)

    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    dtr = lgb.Dataset(X_tr[cols], label=y_tr, categorical_feature=cats)
    dval = lgb.Dataset(X_val[cols], label=y_val, categorical_feature=cats, reference=dtr)

    model = lgb.train(
        param, dtr, num_boost_round=1000,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50), lgb.log_evaluation(0),
            # Report each iteration so MedianPruner can actually prune bad
            # trials mid-run (a declared pruner without reporting is dead).
            optuna.integration.LightGBMPruningCallback(
                trial, metric="rmse", valid_name="val"),
        ]
    )
    preds = model.predict(X_val[cols], num_iteration=model.best_iteration)
    trial.set_user_attr("best_iteration", int(model.best_iteration or 0))
    return float(np.sqrt(mean_squared_error(y_val, preds)))


def objective_catboost(trial, X_tr, y_tr, X_val, y_val):
    import catboost as cb

    cols, cats = feature_cols(X_tr)
    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    cat_idx = [cols.index(c) for c in cats]
    tr_pool = cb.Pool(X_tr[cols], y_tr, cat_features=cat_idx)
    val_pool = cb.Pool(X_val[cols], y_val, cat_features=cat_idx)

    param = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": "CPU",
        "random_seed": TUNE_SEED,
        "verbose": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
    }

    model = cb.CatBoostRegressor(iterations=1000, early_stopping_rounds=50, **param)
    prune_cb = optuna.integration.CatBoostPruningCallback(trial, "RMSE")
    model.fit(tr_pool, eval_set=val_pool, use_best_model=True, callbacks=[prune_cb])
    preds = model.predict(X_val[cols])
    trial.set_user_attr("best_iteration", int(model.get_best_iteration() or 0))
    return float(np.sqrt(mean_squared_error(y_val, preds)))


# ── MLflow integration ───────────────────────────────────────────────────────

def _mlflow_callback() -> Optional[Any]:
    """Log every trial as a nested MLflow run (whole search is auditable)."""
    try:
        import mlflow
    except Exception:
        return None


    class _TrialCallback:
        """Minimal Optuna callback that mirrors each trial into MLflow."""

        def __init__(self, study_name: str, parent_run_id: Optional[str]):
            self.study_name = study_name
            self.parent_run_id = parent_run_id
            self._by_number: Dict[int, Any] = {}

        def __call__(self, study, trial):
            try:
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
                with mlflow.start_run(
                    run_name=f"{self.study_name}_trial_{trial.number}",
                    tags={
                        "study_name": self.study_name,
                        "parent_run_id": self.parent_run_id or "",
                        "trial_state": str(trial.state),
                    },
                    nested=True,
                ) as run:
                    self._by_number[trial.number] = run.info.run_id
                    mlflow.log_params(trial.params)
                    if trial.value is not None:
                        mlflow.log_metric("rmse", float(trial.value))
                    if trial.user_attrs.get("best_iteration"):
                        mlflow.log_metric("best_iteration", float(trial.user_attrs["best_iteration"]))
            except Exception as exc:
                print(f">>> MLflow trial logging skipped ({exc}).")

    return _TrialCallback


def _start_parent_run(study_name: str) -> Optional[Any]:
    """Open the parent MLflow run that nests all trials."""
    try:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("MovieRecSys-Tuning")
        parent = mlflow.start_run(run_name=study_name)
        mlflow.log_params({
            "n_trials": N_TRIALS,
            "sampler": "TPESampler",
            "sampler_seed": TUNE_SEED,
            "pruner": "MedianPruner",
            "objective": "val_rmse",
            "git_sha": _git_sha(),
        })
        return parent
    except Exception as exc:
        print(f">>> MLflow tuning tracking skipped ({exc}).")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(">>> Loading data & building features ...")
    crawled, links, movies, ratings, enriched = load_all()
    movie_feats = build_movie_features(enriched)

    s = CFG["split"]
    train_r, val_r, test_r = time_split_per_user(
        ratings, n_test=s["n_holdout_per_user"],
        n_val=s["n_holdout_per_user"],
        min_ratings=2 * s["n_holdout_per_user"] + s["min_user_ratings"])

    user_feats = build_user_features(train_r, movie_feats)
    train = build_dataset(train_r, movie_feats, user_feats)
    val = build_dataset(val_r, movie_feats, user_feats)

    y_tr = train["rating"].astype(float).values
    y_val = val["rating"].astype(float).values

    X_tr = train.drop(columns=["rating", "timestamp"]).copy()
    X_val = val.drop(columns=["rating", "timestamp"]).copy()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    mlflow_parent = _start_parent_run(f"{STUDY_PREFIX}_tune_{stamp}")
    callback_cls = _mlflow_callback()
    provenance = _provenance()
    provenance["train_rows"] = int(len(train))
    provenance["val_rows"] = int(len(val))

    print(f">>> Tuning LightGBM ({N_TRIALS} trials, seed={TUNE_SEED}) ...")
    study_lgb = optuna.create_study(
        direction="minimize",
        study_name=f"{STUDY_PREFIX}_lgbm_{stamp}",
        sampler=_sampler(),
        pruner=PRUNER,
    )
    study_lgb.optimize(
        lambda t: objective_lgbm(t, X_tr, y_tr, X_val, y_val),
        n_trials=N_TRIALS,
        callbacks=[callback_cls(f"{STUDY_PREFIX}_lgbm", mlflow_parent.info.run_id)] if (mlflow_parent and callback_cls) else None,
    )
    print(f"Best LGBM RMSE: {study_lgb.best_value:.4f}")

    print(f">>> Tuning CatBoost ({N_TRIALS} trials, seed={TUNE_SEED}) ...")
    study_cb = optuna.create_study(
        direction="minimize",
        study_name=f"{STUDY_PREFIX}_catboost_{stamp}",
        sampler=_sampler(),
        pruner=PRUNER,
    )
    study_cb.optimize(
        lambda t: objective_catboost(t, X_tr, y_tr, X_val, y_val),
        n_trials=N_TRIALS,
        callbacks=[callback_cls(f"{STUDY_PREFIX}_catboost", mlflow_parent.info.run_id)] if (mlflow_parent and callback_cls) else None,
    )
    print(f"Best CatBoost RMSE: {study_cb.best_value:.4f}")

    # IMPORTANT: only the searched parameters are exported.  The boosting
    # budget (n_estimators / iterations / early_stopping_rounds) and the
    # hardware/runtime knobs stay owned by configs.yaml — a tuning run must
    # never silently downgrade it.
    payload = {
        "provenance": provenance,
        "lgbm_results": {
            "best_rmse": round(float(study_lgb.best_value), 4),
            "n_trials": N_TRIALS,
        },
        "catboost_results": {
            "best_rmse": round(float(study_cb.best_value), 4),
            "n_trials": N_TRIALS,
        },
        "lgbm": dict(study_lgb.best_params),
        "catboost": dict(study_cb.best_params),
    }

    out_path = _best_params_path()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f">>> Saved tuned hyperparameters to {out_path}")

    if mlflow_parent is not None:
        try:
            import mlflow

            mlflow.log_dict(payload, "best_params.json")
            mlflow.end_run()
        except Exception as exc:
            print(f">>> MLflow artifact logging skipped ({exc}).")


def _best_params_path() -> str:
    ml_training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ml_training_dir, "best_params.json")


if __name__ == "__main__":
    main()
