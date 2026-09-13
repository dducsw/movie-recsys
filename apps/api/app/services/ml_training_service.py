"""
apps/api/app/services/ml_training_service.py
--------------------------------------------
Service integrating the trained ML models from pipelines/training/ml_training/models.joblib
(CatBoost YetiRank + LightGBM Ranking Ensemble) with rich 47 movie features and 26 user features.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model_bundle = None
_bundle_loaded = False
_movie_feats_dict = {}
_user_feats_dict = {}

# Production Ensemble Hyperparameters:
# Blending weights between CatBoost (YetiRank) and LightGBM (LambdaRank).
# Validated via offline A/B NDCG@10 comparison where CatBoost achieved higher
# ranking discriminability on sparse interactions, complemented by LightGBM speed.
CATBOOST_ENSEMBLE_WEIGHT = float(os.getenv("RECSYS_CB_WEIGHT", "0.6"))
LIGHTGBM_ENSEMBLE_WEIGHT = float(os.getenv("RECSYS_LGB_WEIGHT", "0.4"))

# Cold-start heuristic user priors derived from population training set averages
COLD_START_USER_PROFILE = {
    "user_avg_rating": 3.8,
    "user_std_rating": 0.8,
    "user_avg_pop": 15.0,
    "user_avg_vote": 7.0,
    "user_avg_year": 2012.0,
}


def get_model_bundle() -> Optional[Dict[str, Any]]:
    """Lazy-load the trained models bundle singleton from evaluation/ml_training/models.joblib."""
    global _model_bundle, _bundle_loaded, _movie_feats_dict, _user_feats_dict
    if _bundle_loaded:
        return _model_bundle

    _bundle_loaded = True
    try:
        import joblib
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = curr_dir
        while repo_root and not (os.path.exists(os.path.join(repo_root, ".git")) or os.path.exists(os.path.join(repo_root, "docker-compose.yml"))):
            parent = os.path.dirname(repo_root)
            if parent == repo_root:
                break
            repo_root = parent

        candidates = [
            os.path.join(repo_root, "pipelines", "training", "ml_training", "models.joblib"),
            os.path.join(repo_root, "models", "models.joblib"),
            os.path.join(repo_root, "evaluation", "ml_training", "models.joblib"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        if not os.path.exists(model_path):
            logger.warning(f"ml_training model bundle not found at {model_path}.")
            _model_bundle = None
            return None

        t0 = time.time()
        logger.info(f"Loading ml_training model bundle from {model_path}...")
        bundle = joblib.load(model_path)
        _model_bundle = bundle

        # Pre-index movie features by movieId for O(1) candidate lookup
        if "movie_feats" in bundle and isinstance(bundle["movie_feats"], pd.DataFrame):
            mf_df = bundle["movie_feats"]
            _movie_feats_dict = {int(row["movieId"]): row.to_dict() for _, row in mf_df.iterrows()}
            logger.info(f"Pre-indexed {len(_movie_feats_dict):,} movie feature rows from ml_training bundle.")

        # Pre-index user features by userId
        if "user_feats" in bundle and isinstance(bundle["user_feats"], pd.DataFrame):
            uf_df = bundle["user_feats"]
            _user_feats_dict = {int(row["userId"]): row.to_dict() for _, row in uf_df.iterrows()}
            logger.info(f"Pre-indexed {len(_user_feats_dict):,} user profile feature rows from ml_training bundle.")

        duration = round(time.time() - t0, 2)
        logger.info(f"ml_training model bundle successfully loaded in {duration}s! Best model: {bundle.get('best_model')}")
        return _model_bundle
    except Exception as e:
        logger.warning(f"Failed to load ml_training model bundle: {e}. Fallback will be used.")
        _model_bundle = None
        return None


def get_user_bias(user_id: Optional[int], default_global_avg: float = 3.5) -> float:
    """Retrieve user rating bias (user_avg_rating - default_global_avg) if available, else 0.0."""
    if user_id is None:
        return 0.0
    if not _bundle_loaded:
        get_model_bundle()
    if user_id in _user_feats_dict:
        user_avg = float(_user_feats_dict[user_id].get("user_avg_rating", default_global_avg))
        return float(user_avg - default_global_avg)
    return 0.0


class MLTrainingModelService:
    """Provides ranking and inference using the CatBoost + LightGBM ensemble from ml_training."""

    @classmethod
    def score_candidates(
        cls,
        candidate_ids: List[int],
        user_id: Optional[int] = None,
        liked_movie_ids: Optional[List[int]] = None,
        target_genres: Optional[set] = None
    ) -> Dict[int, float]:
        """
        Score a list of candidate movie IDs using the CatBoost / LightGBM models from ml_training.
        Returns a dict mapping movieId -> rank_score.
        """
        if not candidate_ids:
            return {}

        bundle = get_model_bundle()
        if not bundle:
            return {}

        try:
            cb_model = bundle.get("catboost")
            lgb_model = bundle.get("lgbm")
            cb_cols = bundle.get("cb_cols", [])
            lgb_cols = bundle.get("lgbm_cols", [])

            # 1. Resolve User Feature Vector
            user_features = {}
            if user_id is not None and user_id in _user_feats_dict:
                user_features = _user_feats_dict[user_id].copy()
            else:
                user_features = {
                    "userId": user_id or 0,
                    "user_avg_rating": COLD_START_USER_PROFILE["user_avg_rating"],
                    "user_std_rating": COLD_START_USER_PROFILE["user_std_rating"],
                    "user_n_ratings": float(len(liked_movie_ids or [1])),
                    "user_avg_pop": COLD_START_USER_PROFILE["user_avg_pop"],
                    "user_avg_vote": COLD_START_USER_PROFILE["user_avg_vote"],
                    "user_avg_year": COLD_START_USER_PROFILE["user_avg_year"]
                }
                all_genres = [
                    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"
                ]
                active_genres = set(target_genres or [])
                for g in all_genres:
                    user_features[f"user_pref_genre_{g}"] = 1.0 if g in active_genres else 0.0

            # 2. Build Candidate Feature Matrix
            cand_rows = []
            valid_ids = []

            for mid in candidate_ids:
                if mid in _movie_feats_dict:
                    movie_row = dict(_movie_feats_dict[mid])
                else:
                    movie_row = {
                        "movieId": mid,
                        "primary_director": "Unknown",
                        "primary_actor": "Unknown",
                        "popularity": 5.0,
                        "vote_average": 6.5,
                        "vote_count": 50,
                        "adult": 0,
                        "release_year": 2015,
                        "release_month": 6,
                        "release_decade": 2010,
                        "n_genres": 2,
                        "n_cast": 5,
                        "n_keywords": 3,
                        "overview_len": 120
                    }

                combined = {**movie_row, **user_features}

                # Dynamic interaction features
                m_year = float(movie_row.get("release_year") or 2015)
                u_year = float(user_features.get("user_avg_year") or 2012)
                combined["year_match"] = float(1.0 / (1.0 + abs(m_year - u_year)))

                m_genres_str = str(movie_row.get("genres") or "")
                m_genres = set(m_genres_str.split("|")) if m_genres_str else set()
                u_active = set(target_genres or [])
                overlap = len(m_genres.intersection(u_active)) if u_active else 0
                combined["genre_match_score"] = float(overlap)

                cand_rows.append(combined)
                valid_ids.append(mid)

            if not cand_rows:
                return {}

            df_cand = pd.DataFrame(cand_rows)

            cat_cols_int = ["movieId", "userId"]
            cat_cols_str = ["primary_director", "primary_actor"]
            cat_all = set(cat_cols_int + cat_cols_str)

            # Ensure all numeric columns are clean float32 without NA
            for col in set(cb_cols + lgb_cols):
                if col not in df_cand.columns:
                    df_cand[col] = 0.0
                elif col not in cat_all:
                    df_cand[col] = pd.to_numeric(df_cand[col], errors="coerce").fillna(0.0).astype(np.float32)

            # Format categorical features for CatBoost
            for c in cat_cols_int:
                if c in df_cand.columns:
                    df_cand[c] = df_cand[c].fillna(0).astype(int).astype(str)
            for c in cat_cols_str:
                if c in df_cand.columns:
                    df_cand[c] = df_cand[c].fillna("Unknown").astype(str)

            # Predict CatBoost
            pred_cb = None
            if cb_model:
                try:
                    pred_cb = cb_model.predict(df_cand[cb_cols])
                except Exception as e:
                    logger.warning(f"CatBoost candidate scoring error: {e}")

            # Predict LightGBM
            pred_lgb = None
            if lgb_model:
                try:
                    df_lgb = df_cand.copy()
                    for c in cat_cols_int + cat_cols_str:
                        if c in df_lgb.columns:
                            df_lgb[c] = df_lgb[c].astype("category")
                    pred_lgb = lgb_model.predict(df_lgb[lgb_cols])
                except Exception as e:
                    logger.warning(f"LightGBM candidate scoring error: {e}")

            # Ensemble predictions
            scores = {}
            for idx, mid in enumerate(valid_ids):
                score_cb = float(pred_cb[idx]) if pred_cb is not None else None
                score_lgb = float(pred_lgb[idx]) if pred_lgb is not None else None

                if score_cb is not None and score_lgb is not None:
                    final_score = CATBOOST_ENSEMBLE_WEIGHT * score_cb + LIGHTGBM_ENSEMBLE_WEIGHT * score_lgb
                elif score_cb is not None:
                    final_score = score_cb
                elif score_lgb is not None:
                    final_score = score_lgb
                else:
                    final_score = float(cand_rows[idx].get("popularity", 1.0))

                scores[mid] = final_score

            return scores
        except Exception as err:
            logger.error(f"Error in MLTrainingModelService.score_candidates: {err}")
            return {}

    @classmethod
    def recommend_for_user(cls, user_id: int, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Direct full-catalog recommendation using the CatBoost + LightGBM model from ml_training.
        """
        bundle = get_model_bundle()
        if not bundle:
            return []

        try:
            user_feats = bundle.get("user_feats")
            movie_feats = bundle.get("movie_feats")
            cb_model = bundle.get("catboost")
            lgb_model = bundle.get("lgbm")
            cb_cols = bundle.get("cb_cols", [])
            lgb_cols = bundle.get("lgbm_cols", [])

            if movie_feats is None or movie_feats.empty:
                return []

            if user_feats is not None and user_id in user_feats["userId"].values:
                uf_row = user_feats[user_feats["userId"] == user_id].iloc[0].to_dict()
            else:
                uf_row = {c: 0 for c in (user_feats.columns if user_feats is not None else []) if c != "userId"}
                uf_row["userId"] = user_id

            cand = movie_feats.copy()
            for c, v in uf_row.items():
                cand[c] = v

            cat_cols_int = ["movieId", "userId"]
            cat_cols_str = ["primary_director", "primary_actor"]
            cat_all = set(cat_cols_int + cat_cols_str)

            for col in set(cb_cols + lgb_cols):
                if col not in cand.columns:
                    cand[col] = 0.0
                elif col not in cat_all:
                    cand[col] = pd.to_numeric(cand[col], errors="coerce").fillna(0.0).astype(np.float32)

            for c in cat_cols_int:
                if c in cand.columns:
                    cand[c] = cand[c].fillna(0).astype(int).astype(str)
            for c in cat_cols_str:
                if c in cand.columns:
                    cand[c] = cand[c].fillna("Unknown").astype(str)

            # Predict CatBoost
            pred_cb = cb_model.predict(cand[cb_cols]) if cb_model else np.zeros(len(cand))
            pred_lgb = pred_cb
            if lgb_model:
                try:
                    cand_lgb = cand.copy()
                    for c in cat_cols_int + cat_cols_str:
                        if c in cand_lgb.columns:
                            cand_lgb[c] = cand_lgb[c].astype("category")
                    pred_lgb = lgb_model.predict(cand_lgb[lgb_cols])
                except Exception:
                    pred_lgb = pred_cb

            cand["ml_score"] = CATBOOST_ENSEMBLE_WEIGHT * pred_cb + LIGHTGBM_ENSEMBLE_WEIGHT * pred_lgb
            top_movies = cand.sort_values("ml_score", ascending=False).head(top_k)

            # Try fetching live DB records; fallback to precomputed metadata if DB is offline
            db_dict = {}
            try:
                from app.models.movie import MovieModel
                movie_ids = top_movies["movieId"].astype(int).tolist()
                db_movies = MovieModel.get_by_ids(movie_ids)
                db_dict = {m["movieId"]: m for m in db_movies}
            except Exception:
                pass

            results = []
            for _, r in top_movies.iterrows():
                mid = int(r["movieId"])
                if mid in db_dict:
                    m_info = db_dict[mid]
                else:
                    m_info = {
                        "movieId": mid,
                        "title": r.get("title", f"Movie #{mid}"),
                        "release_date": str(r.get("release_date") or "2015"),
                        "genres": str(r.get("genres") or ""),
                        "popularity": float(r.get("popularity") or 1.0),
                        "vote_average": float(r.get("vote_average") or 7.0),
                        "poster_url": str(r.get("poster_url") or "")
                    }
                m_info["rank_score"] = float(r["ml_score"])
                results.append(m_info)

            return results
        except Exception as e:
            logger.error(f"Error in MLTrainingModelService.recommend_for_user: {e}")
            return []
