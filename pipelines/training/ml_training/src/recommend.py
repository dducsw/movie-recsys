import numpy as np
import pandas as pd
import joblib
import sys

def recommend_for_user(user_id: int, top_k: int = 10):
    bundle = joblib.load("models.joblib")
    lgb_model  = bundle["lgbm"]
    cb_model   = bundle["catboost"]
    user_feats = bundle["user_feats"]
    movie_feats= bundle["movie_feats"]
    cfg        = bundle["config"]

    # Check if user exists, handle cold start
    if user_id not in user_feats["userId"].values:
        uf_row = {c: 0 for c in user_feats.columns if c != "userId"}
    else:
        uf_row = user_feats[user_feats["userId"] == user_id].iloc[0].to_dict()

    # Candidate pool: all movies
    cand = movie_feats.copy()
    
    # Add user features to all candidate movies
    for c, v in uf_row.items():
        cand[c] = v

    cols_lgb = bundle["lgbm_cols"]
    cols_cb  = bundle["cb_cols"]
    
    # Fill any missing columns with 0
    for c in cols_lgb:
        if c not in cand.columns: cand[c] = 0
    for c in cols_cb:
        if c not in cand.columns: cand[c] = 0

    # --- Fix Datatypes for Categorical Features ---
    cat_cols_int = ["movieId", "userId"]
    cat_cols_str = ["primary_director", "primary_actor"]

    # For CatBoost: Must be string. Convert float/int to Int64 first to avoid "42.0", then to string
    for c in cat_cols_int:
        if c in cand.columns:
            cand[c] = cand[c].astype("Int64").astype(str).replace("<NA>", "nan")
    for c in cat_cols_str:
        if c in cand.columns:
            cand[c] = cand[c].astype(str)

    # Predict CatBoost
    pred_cb = cb_model.predict(cand[cols_cb])

    # For LightGBM: Safely cast to category
    for c in cat_cols_int + cat_cols_str:
        if c in cand.columns:
            # LightGBM handles missing categories natively if dtype is category
            cand[c] = cand[c].astype("category")

    # Predict LightGBM
    pred_lgb = lgb_model.predict(cand[cols_lgb], num_iteration=lgb_model.best_iteration)

    # Ensemble predictions (Average)
    cand["score"] = 0.5 * pred_lgb + 0.5 * pred_cb

    # Get Top K
    top = (cand.sort_values("score", ascending=False)
               .head(top_k)[["movieId", "title", "score"]])
    
    return top.reset_index(drop=True)

if __name__ == "__main__":
    uid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Top {10} recommendations for User {uid}:")
    print(recommend_for_user(uid, top_k=10))