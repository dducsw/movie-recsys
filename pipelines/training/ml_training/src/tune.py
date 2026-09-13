import json
import optuna
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from data_loader import load_all
from features import build_movie_features, build_user_features, build_dataset
from train import time_split_per_user
from configs import CFG
from models import feature_cols

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Note: RMSE is used as a fast proxy objective during hyperparameter tuning for computational speed;
# final model selection is performed using NDCG@10 on the validation set in train.py.
N_TRIALS = 30 # Number of tuning iterations per model

def objective_lgbm(trial, X_tr, y_tr, X_val, y_val):
    cols, cats = feature_cols(X_tr)
    cats = [c for c in cats if c in X_tr.columns]
    
    for c in cats:
        X_tr[c] = X_tr[c].astype("category")

        X_val[c] = pd.Categorical(
            X_val[c],
            categories=X_tr[c].cat.categories
        )   

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
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    preds = model.predict(X_val[cols], num_iteration=model.best_iteration)
    return np.sqrt(mean_squared_error(y_val, preds))

def objective_catboost(trial, X_tr, y_tr, X_val, y_val):
    cols, cats = feature_cols(X_tr)
    cats = [c for c in cats if c in X_tr.columns]
    
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
        "random_seed": 42,
        "verbose": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
    }
    
    model = cb.CatBoostRegressor(iterations=1000, early_stopping_rounds=50, **param)
    model.fit(tr_pool, eval_set=val_pool, use_best_model=True)
    preds = model.predict(X_val[cols])
    return np.sqrt(mean_squared_error(y_val, preds))

def main():
    print(">>> Loading data & building features ...")
    crawled, links, movies, ratings, enriched = load_all()
    movie_feats = build_movie_features(enriched)
    
    s = CFG["split"]
    train_r, val_r, test_r = time_split_per_user(
        ratings, n_test=s["n_holdout_per_user"],
        n_val=s["n_holdout_per_user"],
        min_ratings=2*s["n_holdout_per_user"] + s["min_user_ratings"],
        seed=s["random_state"])
    
    user_feats = build_user_features(train_r, movie_feats)
    train = build_dataset(train_r, movie_feats, user_feats)
    val   = build_dataset(val_r,   movie_feats, user_feats)
    
    y_tr  = train["rating"].astype(float).values
    y_val = val["rating"].astype(float).values

    X_tr = train.drop(columns=["rating","timestamp"]).copy()
    X_val = val.drop(columns=["rating","timestamp"]).copy()

    print(">>> Tuning LightGBM ...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lambda t: objective_lgbm(t, X_tr, y_tr, X_val, y_val), n_trials=N_TRIALS)
    print(f"Best LGBM RMSE: {study_lgb.best_value:.4f}")
    
    print(">>> Tuning CatBoost ...")
    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(lambda t: objective_catboost(t, X_tr, y_tr, X_val, y_val), n_trials=N_TRIALS)
    print(f"Best CatBoost RMSE: {study_cb.best_value:.4f}")

    best_params = {
        "lgbm": {**study_lgb.best_params, "n_estimators": 1000, "early_stopping_rounds": 50},
        "catboost": {**study_cb.best_params, "iterations": 1000, "early_stopping_rounds": 50}
    }
    
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print(">>> Saved best_params.json")

if __name__ == "__main__":
    main()