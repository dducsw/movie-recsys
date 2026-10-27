"""
pipelines.training.ml_training.src.recommend
---------------------------------------------
Offline recommendation generation from the published ``models.joblib``.

Encoding contract (must match ``models.py``):
    * LightGBM is predicted through ``prepare_lgb_prediction_data`` using the
      persisted ``category_maps`` — the *training* vocabulary.  Casting to
      ``str`` and re-deriving categories from the candidate pool silently
      turns every row into "all categories missing".
    * CatBoost gets its own frame with categoricals cast to ``str``.

Usage::

    python src/recommend.py --user-id 1 --top-k 10
"""

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd
from models import (
    feature_cols,
    prepare_catboost_prediction_data,
    prepare_lgb_prediction_data,
)

try:
    from configs import CFG
except ImportError:
    from .configs import CFG


def ensemble_weights(bundle: Dict) -> Dict[str, float]:
    """Blending weights recorded in the bundle (also used by serving).

    Serving reads these first, so the offline CLI must respect the same
    weights — hardcoded 0.5/0.5 here would reproduce recommendations that
    diverge from production.
    """
    weights = bundle.get("ensemble_weights") or {}
    cb = float(weights.get("catboost", 0.6))
    lgb = float(weights.get("lightgbm", 0.4))
    total = cb + lgb or 1.0
    return {"catboost": cb / total, "lightgbm": lgb / total}


def _bundle_path() -> str:
    ml_training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ml_training_dir, "models.joblib")


def load_bundle(path: Optional[str] = None):
    import joblib

    path = path or _bundle_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")
    return joblib.load(path)


def build_candidate_frame(
    bundle: Dict,
    user_id: int,
    liked_movie_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Candidate pool = the full catalog, enriched with the user's profile."""
    user_feats = bundle["user_feats"]
    movie_feats = bundle["movie_feats"]

    if user_feats is not None and user_id in user_feats["userId"].values:
        uf_row = user_feats[user_feats["userId"] == user_id].iloc[0].to_dict()
    else:
        uf_row = {c: 0.0 for c in (user_feats.columns if user_feats is not None else []) if c != "userId"}
        uf_row["userId"] = user_id

    cand = movie_feats.copy()
    for c, v in uf_row.items():
        cand[c] = v
    return cand


def recommend_for_user(user_id: int, top_k: int = 10, bundle=None):
    bundle = bundle or load_bundle()
    top_k = top_k or int(CFG.get("recommend", {}).get("top_k", 10))

    lgb_model = bundle["lgbm"]
    cb_model = bundle["catboost"]
    lgb_cols = bundle["lgbm_cols"]
    cb_cols = bundle["cb_cols"]
    _, cb_cats = feature_cols(bundle.get("movie_feats"))

    cand = build_candidate_frame(bundle, user_id)
    w_cb = ensemble_weights(bundle)
    for c in set(cb_cols + lgb_cols):
        if c not in cand.columns:
            cand[c] = 0.0

    # --- CatBoost frame: categoricals as str (exactly like training) --------
    X_cb = prepare_catboost_prediction_data(cand, cb_cols, cb_cats)
    pred_cb = cb_model.predict(X_cb)

    # --- LightGBM frame: training vocabulary, values untouched --------------
    X_lgb = prepare_lgb_prediction_data(cand, lgb_cols, bundle["lgb_category_maps"])
    pred_lgb = lgb_model.predict(X_lgb, num_iteration=lgb_model.best_iteration)

    cand["score"] = w_cb["catboost"] * pred_cb + w_cb["lightgbm"] * pred_lgb

    top = (cand.sort_values("score", ascending=False)
               .head(top_k)[["movieId", "title", "score"]])
    return top.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Offline recommendations from models.joblib")
    parser.add_argument("--user-id", type=int, default=1, help="User to build recommendations for")
    parser.add_argument("--top-k", type=int, default=None, help="Number of recommendations")
    parser.add_argument("--bundle", type=str, default=None, help="Path to models.joblib")
    args = parser.parse_args()

    top_k = args.top_k or int(CFG.get("recommend", {}).get("top_k", 10))
    print(f"Top {top_k} recommendations for User {args.user_id}:")
    print(recommend_for_user(args.user_id, top_k=top_k, bundle=load_bundle(args.bundle)))


if __name__ == "__main__":
    main()
