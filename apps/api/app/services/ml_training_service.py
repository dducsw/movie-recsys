"""
apps/api/app/services/ml_training_service.py
---------------------------------------------
Service integrating the trained ML models from
``pipelines/training/ml_training/models.joblib`` (CatBoost + LightGBM
ensemble) with the movie/user feature tables baked into the bundle.

Categorical contract
--------------------
The bundle persists ``lgb_category_maps`` — the categorical vocabulary
captured from the training split.  This service applies it verbatim via
:func:`recsys_core.encoding.prepare_lgb_prediction_data`.  ``movieId`` /
``userId`` are **never** stringified before LightGBM (LightGBM matches
categories by value, so casting ints to str would silently leave every row
with the categories missing).  CatBoost gets its own frame where the
categoricals are cast to ``str`` — exactly like its training frames.
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


try:
    from recsys_core.encoding import (
        best_iteration_of,
        prepare_catboost_prediction_data,
        prepare_lgb_prediction_data,
        validate_category_maps,
    )
    from recsys_core.features import ALL_GENRES
except ImportError:  # pragma: no cover - only hit outside the installed package
    _curr = os.path.dirname(os.path.abspath(__file__))
    _root = _curr
    while _root and not os.path.exists(os.path.join(_root, "libs")):
        _parent = os.path.dirname(_root)
        if _parent == _root:
            break
        _root = _parent
    _core_src = os.path.join(_root, "libs", "recsys_core", "src")
    if _core_src not in sys.path:
        sys.path.insert(0, _core_src)
    from recsys_core.encoding import (
        best_iteration_of,
        prepare_catboost_prediction_data,
        prepare_lgb_prediction_data,
        validate_category_maps,
    )
    from recsys_core.features import ALL_GENRES


_model_bundle = None
bundle_lock = threading.Lock()
_movie_feats_dict: Dict[int, Any] = {}
_user_feats_dict: Dict[int, Any] = {}


# Production Ensemble Hyperparameters:
# Blending weights between CatBoost and LightGBM.  The published bundle records
# the weights it was trained with (``ensemble_weights``); ``RECSYS_CB_WEIGHT`` /
# ``RECSYS_LGB_WEIGHT`` let ops override them per deployment without retraining.
DEFAULT_ENSEMBLE_WEIGHTS = {"catboost": 0.6, "lightgbm": 0.4}


def _ensemble_weights(bundle: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Resolve blending weights: bundle-recorded value -> env override -> default.

    Float conversion is validated for every source so a malformed bundle or a
    non-numeric env var fails fast to the next fallback instead of poisoning
    every request.
    """
    weights = dict(DEFAULT_ENSEMBLE_WEIGHTS)
    recorded = bundle.get("ensemble_weights") or {}
    try:
        weights["catboost"] = float(recorded.get("catboost", DEFAULT_ENSEMBLE_WEIGHTS["catboost"]))
        weights["lightgbm"] = float(recorded.get("lightgbm", DEFAULT_ENSEMBLE_WEIGHTS["lightgbm"]))
    except (TypeError, ValueError):
        logger.warning("Invalid bundle ensemble_weights=%r — falling back to defaults", recorded)
        weights = dict(DEFAULT_ENSEMBLE_WEIGHTS)
    env_cb = os.getenv("RECSYS_CB_WEIGHT")
    env_lgb = os.getenv("RECSYS_LGB_WEIGHT")
    try:
        if env_cb:
            weights["catboost"] = float(env_cb)
        if env_lgb:
            weights["lightgbm"] = float(env_lgb)
    except ValueError:
        logger.warning("Non-numeric ensemble weight env (RECSYS_CB_WEIGHT=%r, RECSYS_LGB_WEIGHT=%r) — keeping previous value",
                       env_cb, env_lgb)
    return weights

# Cold-start heuristic user priors derived from population training set averages
COLD_START_USER_PROFILE = {
    "user_avg_rating": 3.8,
    "user_std_rating": 0.8,
    "user_avg_pop": 15.0,
    "user_avg_vote": 7.0,
    "user_avg_year": 2012.0,
}

# Retry policy: a transient I/O error (or a bundle being swapped in) must not
# permanently disable the ensemble for the whole process.
_BUNDLE_RETRY_SECONDS = 60.0
_BUNDLE_LAST_FAILURE = 0.0
_BUNDLE_ABSENT_WARNED = False


def _repo_root() -> str:
    curr = os.path.dirname(os.path.abspath(__file__))
    repo_root = curr
    while repo_root and not (
        os.path.exists(os.path.join(repo_root, ".git"))
        or os.path.exists(os.path.join(repo_root, "docker-compose.yml"))
    ):
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            break
        repo_root = parent
    return repo_root


def _bundle_path() -> Optional[str]:
    """First existing candidate path for models.joblib."""
    repo_root = _repo_root()
    candidates = [
        os.path.join(repo_root, "pipelines", "training", "ml_training", "models.joblib"),
        os.path.join(repo_root, "models", "models.joblib"),
        os.path.join(repo_root, "evaluation", "ml_training", "models.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_model_bundle() -> Optional[Dict[str, Any]]:
    """Lazy-load the trained model bundle singleton.

    Two guarantees:

    * **Retries after failure.**  Unlike a "set the loaded flag first" pattern,
      a transient read error only disables the ensemble for
      ``_BUNDLE_RETRY_SECONDS`` — not forever.
    * **Locked.**  Two concurrent first requests cannot both ``joblib.load``
      the bundle.
    """
    global _model_bundle, _BUNDLE_LAST_FAILURE, _BUNDLE_ABSENT_WARNED, _movie_feats_dict, _user_feats_dict

    if _model_bundle is not None:
        return _model_bundle

    with bundle_lock:
        if _model_bundle is not None:
            return _model_bundle
        if _BUNDLE_LAST_FAILURE and (time.time() - _BUNDLE_LAST_FAILURE) < _BUNDLE_RETRY_SECONDS:
            return None

        model_path = _bundle_path()
        if model_path is None:
            # The file may appear later (CT publishes it); retry on TTL.
            if not _BUNDLE_ABSENT_WARNED:
                logger.warning("ml_training model bundle not found; will retry every %.0fs.",
                               _BUNDLE_RETRY_SECONDS)
                _BUNDLE_ABSENT_WARNED = True
            _BUNDLE_LAST_FAILURE = time.time()
            return None

        try:
            t0 = time.time()
            logger.info("Loading ml_training model bundle from %s...", model_path)
            import joblib

            bundle = joblib.load(model_path)
            if not validate_category_maps(bundle.get("lgb_category_maps")):
                logger.warning(
                    "Bundle is missing usable lgb_category_maps (stale bundle); "
                    "LightGBM will fall back to best-effort categorical inference."
                )

            _model_bundle = bundle
            _index_bundle_features(bundle)
            duration = round(time.time() - t0, 2)
            logger.info(
                "ml_training model bundle loaded in %ss. Best model: %s",
                duration, bundle.get("best_model"),
            )
            return _model_bundle
        except Exception as exc:
            logger.warning("Failed to load ml_training model bundle: %s. "
                           "Retrying in %.0fs.", exc, _BUNDLE_RETRY_SECONDS)
            _BUNDLE_LAST_FAILURE = time.time()
            return None


def _index_bundle_features(bundle: Dict[str, Any]) -> None:
    """Pre-index the bundle's feature tables for O(1) candidate lookup."""
    global _movie_feats_dict, _user_feats_dict

    mf_df = bundle.get("movie_feats")
    if isinstance(mf_df, pd.DataFrame) and not mf_df.empty:
        _movie_feats_dict = {int(r["movieId"]): r for r in mf_df.to_dict("records")}
        logger.info("Pre-indexed %s movie feature rows from ml_training bundle.",
                    f"{len(_movie_feats_dict):,}")

    uf_df = bundle.get("user_feats")
    if isinstance(uf_df, pd.DataFrame) and not uf_df.empty:
        _user_feats_dict = {int(r["userId"]): r for r in uf_df.to_dict("records")}
        logger.info("Pre-indexed %s user profile feature rows from ml_training bundle.",
                    f"{len(_user_feats_dict):,}")


def reset_bundle_state() -> None:
    """Test hook: clear the cached bundle so state is not shared across tests."""
    global _model_bundle, _BUNDLE_LAST_FAILURE, _BUNDLE_ABSENT_WARNED, _movie_feats_dict, _user_feats_dict
    with bundle_lock:
        _model_bundle = None
        _BUNDLE_LAST_FAILURE = 0.0
        _BUNDLE_ABSENT_WARNED = False
        _movie_feats_dict = {}
        _user_feats_dict = {}


def get_user_bias(user_id: Optional[int], default_global_avg: float = 3.5) -> float:
    """Retrieve user rating bias (user_avg_rating - default_global_avg) if available, else 0.0."""
    if user_id is None:
        return 0.0
    get_model_bundle()
    if user_id in _user_feats_dict:
        user_avg = float(_user_feats_dict[user_id].get("user_avg_rating", default_global_avg))
        return float(user_avg - default_global_avg)
    return 0.0


def _cold_start_user_features(
    user_id: Optional[int],
    liked_movie_ids: Optional[List[int]],
    target_genres: Optional[set],
) -> Dict[str, Any]:
    features: Dict[str, Any] = {
        "userId": user_id or 0,
        "user_avg_rating": COLD_START_USER_PROFILE["user_avg_rating"],
        "user_std_rating": COLD_START_USER_PROFILE["user_std_rating"],
        "user_n_ratings": float(len(liked_movie_ids or [1])),
        "user_avg_pop": COLD_START_USER_PROFILE["user_avg_pop"],
        "user_avg_vote": COLD_START_USER_PROFILE["user_avg_vote"],
        "user_avg_year": COLD_START_USER_PROFILE["user_avg_year"],
    }
    genres = set(target_genres or [])
    for g in ALL_GENRES:
        features[f"user_pref_genre_{g}"] = 1.0 if g in genres else 0.0
    return features


def _predict_lightgbm(bundle: Dict[str, Any], df_cand: pd.DataFrame, lgb_cols: List[str]) -> Optional[np.ndarray]:
    """Predict with LightGBM using the persisted training category vocabulary."""
    lgb_model = bundle.get("lgbm")
    if not lgb_model:
        return None
    try:
        category_maps = bundle.get("lgb_category_maps") or {}
        df_lgb = prepare_lgb_prediction_data(df_cand, lgb_cols, category_maps)
        return lgb_model.predict(df_lgb, num_iteration=best_iteration_of(lgb_model))
    except Exception as exc:
        logger.warning("LightGBM candidate scoring error: %s", exc)
        return None


def _predict_catboost(bundle: Dict[str, Any], df_cand: pd.DataFrame, cb_cols: List[str], cb_cats: List[str]) -> Optional[np.ndarray]:
    """Predict with CatBoost on a frame encoded exactly like training."""
    cb_model = bundle.get("catboost")
    if not cb_model:
        return None
    try:
        df_cb = prepare_catboost_prediction_data(df_cand, cb_cols, cb_cats)
        return cb_model.predict(df_cb)
    except Exception as exc:
        logger.warning("CatBoost candidate scoring error: %s", exc)
        return None


def _candidate_feature_matrix(
    bundle: Dict[str, Any],
    candidate_ids: List[int],
    user_features: Dict[str, Any],
    target_genres: Optional[set],
) -> pd.DataFrame:
    """Assemble the candidate feature frame (all needed columns, clean dtypes)."""
    cand_rows, valid_ids = [], []

    for mid in candidate_ids:
        if mid in _movie_feats_dict:
            movie_row = dict(_movie_feats_dict[mid])
        else:
            # Unseen candidate: use conservative population priors, never a
            # fabricated high-quality stub.
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
                "overview_len": 120,
            }
        cand_rows.append({**movie_row, **user_features})
        valid_ids.append(mid)

    df_cand = pd.DataFrame(cand_rows)

    # Interaction features are computed vectorised (parity with training).
    m_years = df_cand["release_year"].fillna(2015).astype(float)
    u_year = float(user_features.get("user_avg_year") or 2012)
    df_cand["year_match"] = 1.0 / (1.0 + (m_years - u_year).abs())

    m_genres = df_cand.get("genres", pd.Series("", index=df_cand.index)).fillna("").astype(str)
    target = set(target_genres or [])
    df_cand["genre_match_score"] = [
        float(len({g for g in s.split("|") if g} & target)) for s in m_genres
    ]

    cat_cols = set(bundle.get("lgb_category_maps") or {})
    int_cats = {"movieId", "userId"}
    str_cats = {"primary_director", "primary_actor"}
    need_columns = set(bundle.get("cb_cols", [])) | set(bundle.get("lgbm_cols", []))

    for col in need_columns:
        if col not in df_cand.columns:
            df_cand[col] = 0.0
        elif col not in (cat_cols | int_cats | str_cats):
            df_cand[col] = pd.to_numeric(
                df_cand[col], errors="coerce").fillna(0.0).astype(np.float32)

    # IDs may arrive as numpy scalars / floats — normalise before encoding.
    for c in int_cats:
        if c in df_cand.columns:
            df_cand[c] = df_cand[c].fillna(0).astype("int64")
    for c in str_cats:
        if c in df_cand.columns:
            df_cand[c] = df_cand[c].fillna("Unknown").astype(str)

    return df_cand


class MLTrainingModelService:
    """Ranking and inference using the CatBoost + LightGBM ensemble from ml_training."""

    @classmethod
    def score_candidates(
        cls,
        candidate_ids: List[int],
        user_id: Optional[int] = None,
        liked_movie_ids: Optional[List[int]] = None,
        target_genres: Optional[set] = None,
    ) -> Dict[int, float]:
        """Score candidate movie IDs using the CatBoost / LightGBM ensemble.

        Returns a dict mapping movieId -> rank_score.
        """
        if not candidate_ids:
            return {}

        bundle = get_model_bundle()
        if not bundle:
            return {}

        try:
            cb_cols = bundle.get("cb_cols", [])
            lgb_cols = bundle.get("lgbm_cols", [])
            cb_cats = [c for c in ("movieId", "userId", "primary_director", "primary_actor") if c in cb_cols]

            user_features = dict(_user_feats_dict.get(user_id, {})) if user_id is not None and user_id in _user_feats_dict \
                else _cold_start_user_features(user_id, liked_movie_ids, target_genres)

            df_cand = _candidate_feature_matrix(bundle, candidate_ids, user_features, target_genres)

            pred_cb = _predict_catboost(bundle, df_cand, cb_cols, cb_cats)
            pred_lgb = _predict_lightgbm(bundle, df_cand, lgb_cols)
            weights = _ensemble_weights(bundle)

            scores: Dict[int, float] = {}
            for idx, mid in enumerate(candidate_ids):
                score_cb = float(pred_cb[idx]) if pred_cb is not None else None
                score_lgb = float(pred_lgb[idx]) if pred_lgb is not None else None

                if score_cb is not None and score_lgb is not None:
                    final_score = weights["catboost"] * score_cb + weights["lightgbm"] * score_lgb
                elif score_cb is not None:
                    final_score = score_cb
                elif score_lgb is not None:
                    final_score = score_lgb
                else:
                    final_score = float(df_cand["popularity"].iloc[idx]) if "popularity" in df_cand else 0.0
                scores[mid] = final_score

            return scores
        except Exception as err:
            logger.error("Error in MLTrainingModelService.score_candidates: %s", err)
            return {}

    @classmethod
    def recommend_for_user(cls, user_id: int, top_k: int = 20) -> List[Dict[str, Any]]:
        """Direct full-catalog recommendation using the CatBoost + LightGBM ensemble."""
        bundle = get_model_bundle()
        if not bundle:
            return []

        try:
            movie_feats = bundle.get("movie_feats")
            if movie_feats is None or movie_feats.empty:
                return []

            if user_id not in _user_feats_dict:
                logger.debug("User %s unknown to the bundle; scoring with cold-start priors.", user_id)

            scores_by_id = cls.score_candidates(
                candidate_ids=[int(m) for m in movie_feats["movieId"].unique()],
                user_id=user_id,
                target_genres=None,
            )
            if not scores_by_id:
                return []

            top_ids = sorted(scores_by_id, key=lambda mid: scores_by_id[mid], reverse=True)[:top_k]
            top_scores = {mid: scores_by_id[mid] for mid in top_ids}

            # Prefer live DB records; fall back to precomputed bundle metadata.
            db_dict: Dict[int, Dict[str, Any]] = {}
            try:
                from app.models.movie import MovieModel

                db_movies = MovieModel.get_by_ids(top_ids)
                db_dict = {m["movieId"]: m for m in db_movies}
            except Exception as exc:
                logger.warning("DB enrichment unavailable for ml recommendations (%s).", exc)

            movie_feats_idx = movie_feats.set_index("movieId")
            results: List[Dict[str, Any]] = []
            for mid in top_ids:
                if mid in db_dict:
                    m_info = dict(db_dict[mid])
                else:
                    r = movie_feats_idx.loc[mid] if mid in movie_feats_idx.index else None
                    m_info = {
                        "movieId": int(mid),
                        "title": str(r.get("title", f"Movie #{mid}")) if r is not None else f"Movie #{mid}",
                        "release_date": str(r.get("release_date") or "2015") if r is not None else "2015",
                        "genres": str(r.get("genres") or "") if r is not None else "",
                        "popularity": float(r.get("popularity") or 1.0) if r is not None else 1.0,
                        "vote_average": float(r.get("vote_average") or 7.0) if r is not None else 7.0,
                        "poster_url": str(r.get("poster_url") or "") if r is not None else "",
                    }
                m_info["rank_score"] = float(top_scores[mid])
                results.append(m_info)

            return results
        except Exception as exc:
            logger.error("Error in MLTrainingModelService.recommend_for_user: %s", exc)
            return []
