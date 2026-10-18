"""
pipelines/training/full_eval.py
-------------------------------
Nightly exhaustive full-catalog evaluation (no negative sampling).

Complements fast CT 99-neg gate in ``train_pipeline.py`` with unbiased
HR@K / NDCG@K / ILD@K / Coverage over the entire catalog.

Usage:
    PYTHONPATH=apps/api:libs/recsys_core/src:pipelines/training \\
        python pipelines/training/full_eval.py [--k 10] [--max-users 0]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "libs", "recsys_core", "src"))

from recsys_core.features import (
    RANKING_FEATURES,
    build_movie_meta,
    build_user_profiles,
)

from pipelines.training.train_pipeline import (
    _default_meta,
    _jaccard,
    _ranking_feature_row,
    load_and_split_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FullEval")

_candidates = [
    os.path.join(REPO_ROOT, "notebooks", "04_pipeline_prototypes", "models"),
    os.path.join(REPO_ROOT, "pipelines", "training", "ml_pipeline", "models"),
    os.path.join(REPO_ROOT, "models"),
]
MODELS_DIR = next((p for p in _candidates if os.path.exists(p)), _candidates[0])


def load_ranker(path: str | None = None):
    """Load LightGBM ranker artifact."""
    path = path or os.path.join(MODELS_DIR, "lgb_ranker.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def full_catalog_eval(k: int = 10, max_users: int = 0) -> dict:
    """Score every catalog item per test user (exhaustive, slow)."""
    df_movies, train_ratings, test_ratings = load_and_split_data()
    ranker = load_ranker()
    with open(os.path.join(MODELS_DIR, "als_model.pkl"), "rb") as f:
        als_map = pickle.load(f)

    movie_meta = build_movie_meta(df_movies, als_map)
    user_profiles = build_user_profiles(train_ratings, movie_meta)
    genre_of = {int(mid): set(str(g).split("|")) if g else set()
                for mid, g in zip(df_movies["movieId"], df_movies["genres"])}
    catalog_ids = [int(m) for m in df_movies["movieId"].tolist()]

    if max_users > 0:
        test_ratings = test_ratings.head(max_users)

    hr, ndcg, ild, lat, recs = [], [], [], [], set()
    for _, row in test_ratings.iterrows():
        gt, uid = int(row["movieId"]), row["userId"]
        u_prof = user_profiles.get(
            uid, {"activity": 1.0, "bias": 0.0, "genres": set(), "als_vec": None})
        t0 = time.perf_counter()
        feats = [_ranking_feature_row(movie_meta.get(cid, _default_meta()), u_prof)
                 for cid in catalog_ids]
        scores = ranker.predict(
            pd.DataFrame(feats)[RANKING_FEATURES].astype(np.float32))
        lat.append((time.perf_counter() - t0) * 1000.0)
        top_k = [catalog_ids[i] for i in np.argsort(scores)[::-1][:k]]
        recs.update(top_k)
        hit = 1.0 if gt in top_k else 0.0
        hr.append(hit)
        ndcg.append(1.0 / np.log2(top_k.index(gt) + 2) if hit else 0.0)
        sims = [1.0 - _jaccard(genre_of.get(a, set()), genre_of.get(b, set()))
                for x, a in enumerate(top_k) for b in top_k[x + 1:]]
        ild.append(float(np.mean(sims)) if sims else 0.0)

    metrics = {
        f"HR@{k}": round(float(np.mean(hr)) if hr else 0.0, 4),
        f"NDCG@{k}": round(float(np.mean(ndcg)) if ndcg else 0.0, 4),
        f"ILD@{k}": round(float(np.mean(ild)) if ild else 0.0, 4),
        "coverage": round(len(recs) / max(len(catalog_ids), 1), 4),
        "avg_latency_ms": round(float(np.mean(lat)) if lat else 0.0, 2),
        "evaluated_users": len(hr),
        "protocol": "full-catalog",
    }
    logger.info("Full-catalog metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nightly full-catalog eval.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0,
                        help="Cap test users (0 = all).")
    args = parser.parse_args()
    full_catalog_eval(k=args.k, max_users=args.max_users)
