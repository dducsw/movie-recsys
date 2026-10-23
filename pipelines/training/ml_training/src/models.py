"""
pipelines.training.ml_training.src.models
------------------------------------------
Model training + *prediction-side* data preparation.

The categorical contract is the single most important invariant here:

* **LightGBM** — categorical columns are pandas ``category`` dtype whose
  vocabulary is captured from the **train split** and persisted as
  ``category_maps``.  At prediction time the *exact same* vocabulary must be
  applied via :func:`prepare_lgb_prediction_data`.  Values are **not**
  stringified: LightGBM matches categories by value, so training on ints and
  predicting on strings silently turns every row into "all categories
  missing".
* **CatBoost** — categorical columns are cast to ``str`` (train and predict)
  and the categorical positions are passed explicitly as ``cat_features``.

The prediction-prep helpers themselves live in ``recsys_core.encoding`` so the
training pipeline and the serving layer share one implementation.
"""

import os
import sys

import lightgbm as lgb

try:
    import catboost as cb
except ImportError:
    cb = None
import pandas as pd

try:
    from configs import CFG
except ImportError:
    from .configs import CFG

try:
    from recsys_core.encoding import (
        catboost_feature_positions,
        prepare_catboost_prediction_data,
        prepare_lgb_prediction_data,
    )
except ImportError:  # pragma: no cover - only hit when the package isn't installed
    _core_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        "libs", "recsys_core", "src",
    )
    if _core_path not in sys.path:
        sys.path.insert(0, _core_path)
    from recsys_core.encoding import (
        catboost_feature_positions,
        prepare_catboost_prediction_data,
        prepare_lgb_prediction_data,
    )


CAT_COLS = [
    "movieId",
    "userId",
    "primary_director",
    "primary_actor",
]

NUM_COLS = [
    "popularity",
    "vote_average",
    "vote_count",
    "adult",
    "release_year",
    "release_month",
    "release_decade",
    "n_genres",
    "n_cast",
    "n_keywords",
    "overview_len",
    "user_avg_rating",
    "user_std_rating",
    "user_n_ratings",
    "user_avg_pop",
    "user_avg_vote",
    "user_avg_year",
    "year_match",
    "genre_match_score",
]


def feature_cols(df):
    """Return ``(cols, cats)`` — the single source of truth for the schema."""
    cats = [c for c in CAT_COLS if c in df.columns]
    nums = [c for c in NUM_COLS if c in df.columns]
    # Only one-hot genre columns; genre_match_score is already in NUM_COLS.
    gens = [
        c for c in df.columns
        if c.startswith("genre_") and c not in NUM_COLS
    ]

    cols = cats + nums + gens
    duplicates = [c for c in set(cols) if cols.count(c) > 1]
    if duplicates:
        raise ValueError(f"Duplicate feature columns: {duplicates}")
    return cols, cats


def _deterministic_lgbm_params(params: dict) -> dict:
    """Make LightGBM bit-reproducible.

    With ``bagging_fraction``/``feature_fraction`` < 1 and a multi-threaded
    histogram builder, LightGBM is *not* reproducible unless the seeds are
    pinned and the row/column-wise construction is fixed.  Reproducibility
    also requires a **fixed thread count**, pinned via ``LGB_NUM_THREADS``.
    """
    params = dict(params)
    seed = int(params.pop("seed", CFG.get("split", {}).get("random_state", 42)))
    params.setdefault("seed", seed)
    params.setdefault("bagging_seed", seed)
    params.setdefault("feature_fraction_seed", seed)
    params.setdefault("data_random_seed", seed)
    params.setdefault("deterministic", True)
    params.setdefault("force_row_wise", True)
    params.setdefault(
        "num_threads",
        int(os.environ.get("LGB_NUM_THREADS", os.cpu_count() or 1)),
    )
    return params


def train_lgbm(X_tr, y_tr, X_val, y_val, params=None):
    cols, cats = feature_cols(X_tr)

    # ------------------------------------------------
    # Prepare categorical features
    # ------------------------------------------------
    category_maps = {}
    for c in cats:
        X_tr[c] = X_tr[c].astype("category")
        # IMPORTANT: save categories from TRAIN
        category_maps[c] = X_tr[c].cat.categories
        # Validation uses exactly the same categories
        X_val[c] = pd.Categorical(X_val[c], categories=category_maps[c])

    # ------------------------------------------------
    # Parameters
    # ------------------------------------------------
    params = dict(CFG["lgbm"] if params is None else params)
    n_estimators = params.pop("n_estimators", 1000)
    early_stopping = params.pop("early_stopping_rounds", 50)
    params = _deterministic_lgbm_params(params)
    params["objective"] = "regression"
    params["metric"] = ["rmse", "mae"]
    params["verbose"] = -1

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------
    dtr = lgb.Dataset(X_tr[cols], label=y_tr, categorical_feature=cats)
    dval = lgb.Dataset(X_val[cols], label=y_val, categorical_feature=cats, reference=dtr)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        valid_sets=[dtr, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(200)]
    )

    return model, cols, category_maps


def prepare_lgb_prediction_data(df, cols, category_maps):
    """Deprecated shim — use :func:`recsys_core.encoding.prepare_lgb_prediction_data`."""
    return prepare_lgb_prediction_data(df, cols, category_maps)


def prepare_catboost_prediction_data(df, cols, cats):
    """Deprecated shim — use :func:`recsys_core.encoding.prepare_catboost_prediction_data`."""
    return prepare_catboost_prediction_data(df, cols, cats)


def catboost_feature_positions(cols, cats):
    """Deprecated shim — use :func:`recsys_core.encoding.catboost_feature_positions`."""
    return catboost_feature_positions(cols, cats)


def train_catboost(X_tr, y_tr, X_val, y_val, params=None):
    if cb is None:
        raise ImportError("catboost is not installed. Please install with `pip install catboost`.")

    cols, cats = feature_cols(X_tr)

    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    cat_idx = catboost_feature_positions(cols, cats)

    tr_pool = cb.Pool(X_tr[cols], y_tr, cat_features=cat_idx)
    val_pool = cb.Pool(X_val[cols], y_val, cat_features=cat_idx)

    params = dict(CFG["catboost"] if params is None else params)
    iterations = params.pop("iterations", 1000)
    early_stopping = params.pop("early_stopping_rounds", 50)
    verbose = params.pop("verbose", 200)

    params["loss_function"] = "RMSE"
    params["eval_metric"] = "RMSE"
    # CPU is the default everywhere (Docker Compose has no verified GPU
    # passthrough).  "GPU" may be requested explicitly by an operator who has
    # verified the hardware.
    params.setdefault("task_type", "CPU")
    params.setdefault("random_seed", int(CFG.get("split", {}).get("random_state", 42)))

    model = cb.CatBoostRegressor(
        iterations=iterations,
        early_stopping_rounds=early_stopping,
        verbose=verbose,
        **params
    )

    model.fit(tr_pool, eval_set=val_pool, use_best_model=True)
    return model, cols
