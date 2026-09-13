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
    cats = [
        c for c in CAT_COLS
        if c in df.columns
    ]

    nums = [
        c for c in NUM_COLS
        if c in df.columns
    ]

    # Only one-hot genre columns.
    # genre_match_score is already in NUM_COLS.
    gens = [
        c for c in df.columns
        if c.startswith("genre_")
        and c not in NUM_COLS
    ]

    cols = cats + nums + gens

    duplicates = [
        c for c in set(cols)
        if cols.count(c) > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate feature columns: {duplicates}"
        )

    return cols, cats


def train_lgbm(X_tr, y_tr, X_val, y_val, params=None):

    cols, cats = feature_cols(X_tr)

    # ------------------------------------------------
    # Prepare categorical features
    # ------------------------------------------------

    category_maps = {}

    for c in cats:

        X_tr[c] = X_tr[c].astype("category")

        # IMPORTANT:
        # Save categories from TRAIN
        category_maps[c] = X_tr[c].cat.categories

        # Validation uses exactly the same categories
        X_val[c] = pd.Categorical(
            X_val[c],
            categories=category_maps[c]
        )

    # ------------------------------------------------
    # Parameters
    # ------------------------------------------------

    if params is None:
        params = dict(CFG["lgbm"])
    else:
        params = dict(params)

    n_estimators = params.pop(
        "n_estimators",
        1000
    )

    early_stopping = params.pop(
        "early_stopping_rounds",
        50
    )

    params["objective"] = "regression"
    params["metric"] = ["rmse", "mae"]
    params["verbose"] = -1

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    dtr = lgb.Dataset(
        X_tr[cols],
        label=y_tr,
        categorical_feature=cats
    )

    dval = lgb.Dataset(
        X_val[cols],
        label=y_val,
        categorical_feature=cats,
        reference=dtr
    )

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

    model = lgb.train(
        params,
        dtr,
        num_boost_round=n_estimators,
        valid_sets=[dtr, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping),
            lgb.log_evaluation(200)
        ]
    )

    return model, cols, category_maps


def prepare_lgb_prediction_data(
    df,
    cols,
    category_maps
):
    X = df[cols].copy()

    for c, categories in category_maps.items():
        X[c] = pd.Categorical(
            X[c],
            categories=categories
        )

    return X


def train_catboost(X_tr, y_tr, X_val, y_val, params=None):
    if cb is None:
        raise ImportError("catboost is not installed. Please install with `pip install catboost`.")

    cols, cats = feature_cols(X_tr)

    for c in cats:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    cat_idx = [
        cols.index(c)
        for c in cats
    ]

    tr_pool = cb.Pool(
        X_tr[cols],
        y_tr,
        cat_features=cat_idx
    )

    val_pool = cb.Pool(
        X_val[cols],
        y_val,
        cat_features=cat_idx
    )

    if params is None:
        params = dict(CFG["catboost"])
    else:
        params = dict(params)

    iterations = params.pop(
        "iterations",
        1000
    )

    early_stopping = params.pop(
        "early_stopping_rounds",
        50
    )

    verbose = params.pop(
        "verbose",
        200
    )

    params["loss_function"] = "RMSE"
    params["eval_metric"] = "RMSE"
    params["task_type"] = "CPU"
    params["random_seed"] = 42

    model = cb.CatBoostRegressor(
        iterations=iterations,
        early_stopping_rounds=early_stopping,
        verbose=verbose,
        **params
    )

    model.fit(
        tr_pool,
        eval_set=val_pool,
        use_best_model=True
    )

    return model, cols