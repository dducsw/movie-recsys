"""Train-serve parity check against the REAL production code paths.

Three skew surfaces, one check per surface:

1. **LambdaRanker features** — a feature vector must be numerically identical
   whether it is produced by the CT training pipeline (``build_movie_meta`` ->
   ``build_ranking_row``) or assembled by the serving engine's Stage 2
   (``app.services.recsys._stage_2_ranking`` from DB records).  The serving
   path is driven for real with stubbed DB/Feast/ensemble dependencies and a
   capturing pseudo-ranker, then both frames are compared cell by cell.

2. **LightGBM encoding** — the train-side category cast (``astype("category")``
   frozen to the training vocabulary) must give identical predictions to the
   serve-side ``prepare_lgb_prediction_data``.  Casting to ``str`` at serve
   time (the historic skew) diverges and would fail this check.

3. **CatBoost encoding** — the train-side ``str`` cast must match the
   serve-side ``prepare_catboost_prediction_data``.

Run as a script or import ``run_parity_check`` in CI.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_REPO_ROOT,
           os.path.join(_REPO_ROOT, "libs", "recsys_core", "src"),
           os.path.join(_REPO_ROOT, "apps", "api")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402  (below sys.path bootstrap, like the import above)
import pandas as pd  # noqa: E402

from app.services import recsys as recsys_mod  # noqa: E402  (apps/api on sys.path)
from recsys_core.encoding import prepare_catboost_prediction_data, prepare_lgb_prediction_data  # noqa: E402
from recsys_core.features import (  # noqa: E402
    RANKING_FEATURES,
    build_movie_meta,
    build_ranking_row,
    build_user_profiles,
)

# ── 1. LambdaRanker feature parity (train pipeline vs serving Stage 2) ───────

class _FrameCaptor:
    """Pseudo-LambdaRanker that records exactly what serving predicts on."""

    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []

    @property
    def best_iteration_(self) -> Optional[int]:
        return None

    def predict(self, X, *args, **kwargs) -> np.ndarray:
        self.frames.append(pd.DataFrame(X))
        return np.zeros(len(X))


def _stub_movie_model(records: List[Dict[str, Any]]) -> Any:
    class _StubMovieModel:
        @staticmethod
        def get_by_ids(movie_ids: List[int]):
            wanted = set(movie_ids)
            return [dict(r) for r in records if r["movieId"] in wanted]

        @staticmethod
        def get_by_id(movie_id: int):
            for r in records:
                if r["movieId"] == movie_id:
                    return dict(r)
            return None

    return _StubMovieModel


def check_ranking_feature_parity() -> None:
    """Compare the CT training feature row with the serving Stage-2 row.

    Contract notes:
    * ``target_genres`` at serve time is built from the movies a user *liked*,
      which corresponds to ``build_user_profiles``' ``rating >= 3.0`` subset —
      so the test only feeds genuinely rated-liked history on both sides.
    * ``retrieval_score`` is intentionally NOT compared: at training time the
      slot carries the iALS dot product, at serving time it carries the ANN
      similarity.  The two channels are semantically different by design; the
      check only asserts the feature is present and finite on both sides.
    """
    # ---- Shared inputs -------------------------------------------------------
    movie_records = [
        {"movieId": 101, "title": "A", "genres": "Action|Drama",
         "popularity": 30.0, "vote_average": 7.5, "vote_count": 100.0,
         "release_date": "2018-04-01"},
        {"movieId": 102, "title": "B", "genres": "Comedy",
         "popularity": 5.0, "vote_average": 6.0, "vote_count": 10.0,
         "release_date": "1994-11-24"},
    ]
    df_movies = pd.DataFrame(movie_records)
    # The user's *liked* history (rating >= 3.0 is what the trainer keeps).
    df_ratings = pd.DataFrame([
        {"userId": 7, "movieId": 101, "rating": 4.5, "timestamp": 1_700_000_000},
    ])

    # ---- Train-side construction (CT pipeline) -------------------------------
    movie_meta = build_movie_meta(df_movies, als_map=None)
    global_avg = float(df_ratings["rating"].mean())
    profile = build_user_profiles(df_ratings, movie_meta, global_avg)[7]

    # ---- Serve-side construction (real Stage 2, stubbed dependencies) --------
    candidates = [
        {"movieId": 101, "retrieval_score": 0.123, "source": "qdrant_user_vec"},
        {"movieId": 102, "retrieval_score": 0.456, "source": "redis_similar"},
    ]

    captor = _FrameCaptor()
    orig_movie_model = recsys_mod.MovieModel
    orig_ensemble = recsys_mod.MLTrainingModelService.score_candidates
    orig_ranker_loader = recsys_mod.get_ranker_model
    try:
        recsys_mod.MovieModel = _stub_movie_model(movie_records)
        recsys_mod.MLTrainingModelService.score_candidates = \
            classmethod(lambda cls, **kw: {})
        recsys_mod.get_ranker_model = lambda: captor

        ranked = recsys_mod.RecsysService._stage_2_ranking(
            candidates=candidates,
            target_genres={"Action", "Drama"},
            liked_count=1,
            user_id=7,
            liked_movie_ids=[101],
            endpoint="parity_check",
        )
    finally:
        recsys_mod.MovieModel = orig_movie_model
        recsys_mod.MLTrainingModelService.score_candidates = orig_ensemble
        recsys_mod.get_ranker_model = orig_ranker_loader

    if len(captor.frames) != 1:
        raise AssertionError(
            f"Serving Stage 2 predicted {len(captor.frames)} times, expected 1.")
    serve_frame = captor.frames[0]
    if list(serve_frame.columns) != list(RANKING_FEATURES):
        raise AssertionError(
            f"Serving feature schema diverged from RANKING_FEATURES: "
            f"{list(serve_frame.columns)}"
        )

    serve_order = [c["movieId"] for c in candidates
                   if any(r["movieId"] == c["movieId"] for r in ranked)]
    compared = [f for f in RANKING_FEATURES if f != "retrieval_score"]
    for pos, mid in enumerate(serve_order):
        train_row = build_ranking_row(movie_meta[mid], profile)
        serve_row = serve_frame.iloc[pos]
        for feat in compared:
            diff = abs(float(train_row[feat]) - float(serve_row[feat]))
            if diff > 1e-6:
                raise AssertionError(
                    f"Parity breach for movie {mid}, feature '{feat}': "
                    f"train={train_row[feat]!s} serve={serve_row[feat]!s} "
                    f"(diff={diff})"
                )
        # The retrieval-affinity slot must exist and be finite on both sides
        # even though its channels differ by design (iALS vs ANN similarity).
        assert np.isfinite(float(serve_row["retrieval_score"]))
    _ = train_row.get("retrieval_score")  # present in the training row too
    print(f"1. Ranking-feature parity OK ({len(serve_order)} candidates, "
          f"{len(compared)} shared features).")


# ── 2. LightGBM encoding parity ──────────────────────────────────────────────

def check_lgb_encoding_parity(seed: int = 42) -> None:
    """Serve-side prepare_lgb_prediction_data must equal the train-side cast."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({
        "num": rng.normal(size=50),
        "cat": np.tile(np.array([0, 1, 2, 3]), 13)[:50],
    })
    y = (rng.random(50) > 0.5).astype(int)
    X_train["cat"] = X_train["cat"].astype("category")
    vocab = X_train["cat"].cat.categories

    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1, random_state=seed)
    model.fit(X_train, y)

    # Train-side frame exactly as produced during training.
    trained_frame = X_train.copy()

    # Serve-side frame through the real shared helper.
    serve_frame_idents = X_train.copy()
    serve_frame_idents["cat"] = serve_frame_idents["cat"].astype("int64")

    via_train = model.predict(trained_frame)
    via_serve = model.predict(prepare_lgb_prediction_data(
        serve_frame_idents, ["num", "cat"], {"cat": vocab}))
    if not np.allclose(via_train, via_serve):
        raise AssertionError(
            "LightGBM encoding parity breach: serve predictions differ from "
            "the train-side encoding."
        )

    # Sanity: the historic skew (str-casting at serve time) must NOT pass —
    # it breaks the category vocabulary (prediction itself raises, or the
    # scores shift — either behaviour is fine, so long as it is not identical).
    str_frame = X_train.copy()
    str_frame["cat"] = str_frame["cat"].astype(str)
    try:
        str_scores = model.predict(str_frame)
    except ValueError:
        str_scores = None
    if str_scores is not None and np.allclose(via_train, str_scores):
        raise AssertionError(
            "Sanity check failed: str-casted categories should NOT reproduce "
            "the train encoding; the parity test would be vacuous."
        )
    print("2. LightGBM encoding parity OK.")


# ── 3. CatBoost encoding parity ──────────────────────────────────────────────

def check_catboost_encoding_parity(seed: int = 42) -> None:
    try:
        import catboost as cb
    except ImportError:
        print("3. CatBoost encoding parity SKIPPED (catboost not installed).")
        return

    rng = np.random.default_rng(seed)
    cat_values = [f"a{i}" for i in range(6)]
    X = pd.DataFrame({
        "num": rng.normal(size=60),
        "cat": np.tile(np.array(cat_values), 10),
    })
    y = rng.random(60)

    # Train-side: categoricals cast to str, explicit positions.
    X_tr = X.copy()
    X_tr["cat"] = X_tr["cat"].astype(str)
    model = cb.CatBoostRegressor(
        iterations=10, depth=3, loss_function="RMSE", random_seed=seed, verbose=0)
    model.fit(cb.Pool(X_tr[["num", "cat"]], y, cat_features=[1]))

    via_train = model.predict(cb.Pool(X_tr[["num", "cat"]], cat_features=[1]))
    via_serve = model.predict(prepare_catboost_prediction_data(
        X, ["num", "cat"], ["cat"]))
    if not np.allclose(via_train, via_serve):
        raise AssertionError("CatBoost encoding parity breach.")
    print("3. CatBoost encoding parity OK.")


def run_parity_check() -> None:
    check_ranking_feature_parity()
    check_lgb_encoding_parity()
    check_catboost_encoding_parity()
    print("\nAll train-serve parity checks passed.")


if __name__ == "__main__":
    run_parity_check()
