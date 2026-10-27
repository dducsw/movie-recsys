"""
pipelines.training.train_pipeline
----------------------------------
Automated 1-Click Continuous Training (CT) Pipeline for 3-Stage Movie RecSys.

Stages:
    1. Time-based Leave-One-Out split (no temporal leakage).  Catalog
       statistics are computed on the **train split only**.
    2. Retrieval: TF-IDF + true iALS (Hu-Koren-Volinsky confidence).
    3. Ranking: LightGBM LambdaRank on the shared 11-feature schema with
       mixed uniform + popularity-weighted (hard) negatives.
    4. Evaluation: 99-neg protocol (He et al. 2017) + ILD/coverage +
       bootstrap CI + latency gate via ``recsys_core.quality_gate``.
    5. **Promotion**: artifacts are only promoted to the serving path *after*
       the quality gate passes.
    6. MLflow: run logging (params, metrics, environment metadata, artifacts),
       Model Registry ``staging`` -> ``champion`` -> ``archived`` lifecycle.

Full-catalog exhaustive benchmarking lives in ``full_eval.py`` (nightly).
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

try:
    from recsys_core.envinfo import git_sha
    from recsys_core.features import (
        RANKING_FEATURES,
        build_movie_meta,
        build_user_profiles,
    )
    from recsys_core.metrics import bootstrap_ranking_ci
    from recsys_core.quality_gate import QualityGateConfig, check_quality_gate
    from recsys_core.retrieval import IALSConfig, train_ials_factors
    from recsys_core.sampling import sample_negatives
except ImportError:
    _core_path = os.path.join(REPO_ROOT, "libs", "recsys_core", "src")
    if _core_path not in sys.path:
        sys.path.insert(0, _core_path)
    from recsys_core.envinfo import git_sha
    from recsys_core.features import (
        RANKING_FEATURES,
        build_movie_meta,
        build_user_profiles,
    )
    from recsys_core.metrics import bootstrap_ranking_ci
    from recsys_core.quality_gate import QualityGateConfig, check_quality_gate
    from recsys_core.retrieval import IALSConfig, train_ials_factors
    from recsys_core.sampling import sample_negatives

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLOps-TrainPipeline")

_candidates = [
    os.path.join(REPO_ROOT, "pipelines", "training", "ml_pipeline", "models"),
    os.path.join(REPO_ROOT, "models"),
]
MODELS_DIR = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
os.makedirs(MODELS_DIR, exist_ok=True)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "MovieRecSys-Ranking"
REGISTERED_MODEL_NAME = "MovieRanker-LGBM"
CONFIG_PATH = os.path.join(REPO_ROOT, "pipelines", "training", "configs.yaml")

DEFAULT_RANKER_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10],
}


# ── Config ───────────────────────────────────────────────────────────────────

def load_configs() -> dict[str, Any]:
    """Load CT hyperparameters with sensible fallbacks."""
    cfg: dict[str, Any] = {
        "ranker": dict(DEFAULT_RANKER_PARAMS),
        "ials": {"factors": 32, "regularization": 0.01, "iterations": 20,
                 "alpha": 40.0, "random_state": 42},
        "sampling": {"n_neg_ratio": 0.5, "hard_ratio": 0.5, "seed": 42},
        "quality_gates": {"min_hr": 0.15, "min_ndcg": 0.12,
                          "min_ild": 0.30, "max_latency_ms": 30.0,
                          "min_users_evaluated": 1},
        "evaluation": {"ct_protocol_neg": 99, "ct_max_users": 200, "k": 10,
                       "bootstrap_iters": 1000},
        "mlflow": {"promote_to_production": True, "smoke_test": True},
    }
    if os.path.exists(CONFIG_PATH):
        try:
            import yaml

            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            for section in cfg:
                if isinstance(loaded.get(section), dict):
                    cfg[section].update(loaded[section])
        except Exception as exc:
            logger.warning("Could not load %s (%s), using defaults.", CONFIG_PATH, exc)
    return cfg


def _git_sha() -> str:
    """Current git commit SHA (or ``nogit``) — see :mod:`recsys_core.envinfo`."""
    return git_sha()


def _environment_metadata() -> dict[str, Any]:
    """Environment metadata logged with every run so a run can be replayed."""
    import sklearn

    meta: dict[str, Any] = {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
    }
    try:
        import mlflow

        meta["mlflow_version"] = mlflow.__version__
    except Exception:
        meta["mlflow_version"] = "not_installed"
    return meta


# ── Step 1: Data Loading & Time-based Split ──────────────────────────────────

def load_and_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load movies/ratings and split with time-based Leave-One-Out per user.

    IMPORTANT: the temporal split happens **before** any aggregate is computed,
    and the catalog statistics (``popularity`` / ``vote_average`` /
    ``vote_count``) are derived from the **train split only**.  Computing them
    on the full frame would leak the held-out rating into ``vote_average`` and
    inflate ``popularity``/``vote_count`` — the exact leakage the LOO protocol
    exists to prevent.
    """
    logger.info("Loading training datasets...")
    ratings_path = os.path.join(REPO_ROOT, "data", "ml-latest-small", "ratings.csv")
    if not os.path.exists(ratings_path):
        ratings_path = os.path.join(REPO_ROOT, "data", "simulator", "sim_ratings.csv")
    df_ratings = pd.read_csv(ratings_path)
    if "movieid" in df_ratings.columns:
        df_ratings.rename(columns={"movieid": "movieId"}, inplace=True)
    if "userid" in df_ratings.columns:
        df_ratings.rename(columns={"userid": "userId"}, inplace=True)

    from pipelines.training.validate import (
        check_data_drift,
        validate_movies,
        validate_ratings,
        validate_referential_integrity,
    )

    validate_ratings(df_ratings)

    # ---- Split FIRST (no aggregate may see the held-out rows) ----------------
    df_ratings = df_ratings.sort_values("timestamp")
    test_rows = df_ratings.groupby("userId").tail(1)
    train_rows = df_ratings.drop(test_rows.index)
    logger.info("Train: %s | Test LOO: %s", f"{len(train_rows):,}", f"{len(test_rows):,}")

    # Align movie catalog with ratings ID domain:
    # If ratings are from ml-latest-small, use ml-latest-small/movies.csv (9,742 movies with 100% ID coverage)
    # and enrich with aggregated ratings statistics or TMDB crawled metadata.
    if "ml-latest-small" in ratings_path and os.path.exists(os.path.join(REPO_ROOT, "data", "ml-latest-small", "movies.csv")):
        df_movies = pd.read_csv(os.path.join(REPO_ROOT, "data", "ml-latest-small", "movies.csv"))
        if "movieid" in df_movies.columns:
            df_movies.rename(columns={"movieid": "movieId"}, inplace=True)
        # Catalog statistics from the TRAIN domain only (no test leakage).
        pop_series = train_rows["movieId"].value_counts()
        avg_series = train_rows.groupby("movieId")["rating"].mean()
        cnt_series = train_rows.groupby("movieId")["rating"].count()
        global_mean = float(train_rows["rating"].mean())
        df_movies["popularity"] = df_movies["movieId"].map(pop_series).fillna(1.0).astype(float)
        df_movies["vote_average"] = df_movies["movieId"].map(avg_series).fillna(global_mean).astype(float)
        df_movies["vote_count"] = df_movies["movieId"].map(cnt_series).fillna(1.0).astype(float)
    else:
        movies_path = os.path.join(REPO_ROOT, "data", "crawler", "movies_crawled.csv")
        if not os.path.exists(movies_path):
            movies_path = os.path.join(REPO_ROOT, "data", "ml-latest-small", "movies.csv")
        df_movies = pd.read_csv(movies_path)
        if "movieid" in df_movies.columns:
            df_movies.rename(columns={"movieid": "movieId"}, inplace=True)
        df_movies["popularity"] = pd.to_numeric(df_movies.get("popularity", 1.0), errors="coerce").fillna(1.0)
        df_movies["vote_average"] = pd.to_numeric(df_movies.get("vote_average", 5.0), errors="coerce").fillna(5.0)

    df_movies["genres"] = df_movies["genres"].fillna("(no genres listed)")

    validate_movies(df_movies)
    validate_referential_integrity(df_ratings, df_movies)
    drift_report = check_data_drift(train_rows)
    logger.info("Data drift audit: %s (alerts=%d)", drift_report["status"], len(drift_report["alerts"]))

    logger.info("Loaded %s movies and %s ratings.", f"{len(df_movies):,}", f"{len(train_rows):,}")
    return df_movies, train_rows, test_rows


# ── Step 2: Retrieval Models Training ────────────────────────────────────────

def train_retrieval_models(
    df_movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    ials_config: IALSConfig | None = None,
) -> tuple[TfidfVectorizer, object, dict[int, np.ndarray]]:
    """Train TF-IDF content vectors + true iALS item factors.

    Artifacts are written as *candidates*; promotion happens after the quality
    gate (see :func:`run_pipeline`).
    """
    logger.info("--- [STAGE 1] Training Retrieval Models (TF-IDF + iALS) ---")

    from pipelines.training.versioning import save_versioned
    from recsys_core.features import TFIDF_MAX_FEATURES

    genres_text = df_movies["genres"].str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                 token_pattern=r"(?u)\b[\w-]+\b")
    tfidf_matrix = vectorizer.fit_transform(genres_text).astype(np.float32)
    save_versioned(vectorizer, MODELS_DIR, "tfidf_vectorizer")
    save_versioned(tfidf_matrix, MODELS_DIR, "tfidf_matrix")
    logger.info("Saved TF-IDF models.")

    cfg = load_configs()
    ials_cfg = ials_config or IALSConfig(**cfg.get("ials", {}))
    als_map = train_ials_factors(train_ratings, config=ials_cfg)
    save_versioned(als_map, MODELS_DIR, "als_model")
    logger.info("Saved iALS factors for %s movies.", f"{len(als_map):,}")

    return vectorizer, tfidf_matrix, als_map


# ── Step 3 helpers ───────────────────────────────────────────────────────────

def _ranking_feature_row(
    meta: dict[str, Any], user_profile: dict[str, Any]
) -> dict[str, float]:
    """Shared builder (train == serve). Delegates to recsys_core."""
    from recsys_core.features import build_ranking_row

    return build_ranking_row(meta, user_profile)


def _default_meta() -> dict[str, Any]:
    return {"popularity": 1.0, "vote_average": 5.0, "bayesian_rating": 5.0, "release_year": 2010.0,
            "genres": set(), "als_vec": None}


def train_ranking_model(
    df_movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    als_map: dict[int, np.ndarray] | None = None,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMRanker:
    """Train LightGBM LambdaRank with mixed hard-negative sampling.

    The artifact is saved as a *candidate*; it is promoted to the serving path
    only after the quality gate passes (see :func:`run_pipeline`).
    """
    logger.info("--- [STAGE 2] Training LightGBM LambdaRanker (11 Features) ---")
    full_cfg = load_configs()
    params = params or full_cfg.get("ranker", {})
    sampling_cfg = full_cfg.get("sampling", {})

    movie_meta = build_movie_meta(df_movies, als_map)
    global_avg = float(train_ratings["rating"].mean())
    user_profiles = build_user_profiles(train_ratings, movie_meta, global_avg)

    all_movie_ids = np.array([int(m) for m in df_movies["movieId"].unique()])
    popularity = train_ratings["movieId"].value_counts()
    rng = np.random.default_rng(sampling_cfg.get("seed", 42))

    feature_rows, y, groups = [], [], []
    for uid, g in train_ratings.groupby("userId"):
        if len(g) < 2:
            continue
        u_prof = user_profiles.get(
            uid, {"activity": 1.0, "bias": 0.0, "genres": set(), "als_vec": None})
        rated = set(int(m) for m in g["movieId"].values)
        n_neg = min(len(all_movie_ids) - len(rated),
                    max(1, int(len(g) * sampling_cfg.get("n_neg_ratio", 0.5))))
        sampled_negs = sample_negatives(
            rated, all_movie_ids, popularity=popularity, n_neg=n_neg,
            hard_ratio=sampling_cfg.get("hard_ratio", 0.5), rng=rng)
        groups.append(len(g) + len(sampled_negs))

        for _, row in g.iterrows():
            meta = movie_meta.get(int(row["movieId"]), _default_meta())
            feature_rows.append(_ranking_feature_row(meta, u_prof))
            y.append(int(min(max(round(float(row["rating"]) - 1), 0), 4)))
        for neg_id in sampled_negs.tolist():
            meta = movie_meta.get(int(neg_id), _default_meta())
            feature_rows.append(_ranking_feature_row(meta, u_prof))
            y.append(0)

    X = pd.DataFrame(feature_rows)[RANKING_FEATURES].astype(np.float32)
    y_arr = np.array(y, dtype=np.int32)
    seed = int(full_cfg.get("sampling", {}).get("seed", 42))
    ranker = lgb.LGBMRanker(
        objective=params.get("objective", "lambdarank"),
        metric=params.get("metric", "ndcg"),
        eval_at=params.get("eval_at", [5, 10]),
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 15),
        min_child_samples=params.get("min_child_samples", 5),
        random_state=seed, importance_type="gain", verbose=-1,
    )
    ranker.fit(X, y_arr, group=groups)

    from pipelines.training.versioning import save_versioned

    model_path = save_versioned(ranker, MODELS_DIR, "lgb_ranker")
    logger.info("Saved LightGBM Ranker candidate to %s (not promoted yet).", model_path)
    return ranker


# ── Step 4: Evaluation & Quality Gate ────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate_pipeline(
    ranker: lgb.LGBMRanker,
    df_movies: pd.DataFrame,
    test_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    als_map: dict[int, np.ndarray] | None = None,
    k: int = 10,
    enforce_gate: bool = False,
    gate_config: QualityGateConfig | None = None,
    baseline_metrics: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Evaluate with 99-neg protocol + ILD/coverage + bootstrap CI + latency.

    Set ``enforce_gate=True`` in CT (``run_pipeline``) to fail on breach;
    unit tests use the default ``False`` so tiny fixtures never raise.

    ``baseline_metrics`` are the metrics of the currently promoted model; when
    supplied the gate additionally requires that the candidate does not regress
    NDCG beyond the tolerated margin.
    """
    logger.info("--- [STAGE 3] Evaluating Metrics & Quality Gate ---")
    full_cfg = load_configs()
    eval_cfg = full_cfg.get("evaluation", {})
    n_neg_proto = eval_cfg.get("ct_protocol_neg", 99)
    max_users = eval_cfg.get("ct_max_users", 200)
    n_boot = int(eval_cfg.get("bootstrap_iters", 1000))
    seed = int(full_cfg.get("sampling", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    movie_meta = build_movie_meta(df_movies, als_map)
    user_profiles = build_user_profiles(train_ratings, movie_meta)
    genre_of = {int(mid): set(str(g).split("|")) if g else set()
                for mid, g in zip(df_movies["movieId"], df_movies["genres"])}
    catalog_ids = np.array([int(m) for m in df_movies["movieId"].values])

    hr_list, ndcg_list, ild_list, latencies = [], [], [], []
    recommended: set[int] = set()

    for _, test_row in test_ratings.head(max_users).iterrows():
        gt_movie, uid = int(test_row["movieId"]), test_row["userId"]
        u_prof = user_profiles.get(
            uid, {"activity": 1.0, "bias": 0.0, "genres": set(), "als_vec": None})
        pool = catalog_ids[catalog_ids != gt_movie]
        neg = rng.choice(pool, size=min(n_neg_proto, len(pool)), replace=False)
        candidates = [gt_movie, *neg.tolist()]

        t0 = time.perf_counter()
        c_features = [_ranking_feature_row(
            movie_meta.get(cid, _default_meta()), u_prof) for cid in candidates]
        scores = ranker.predict(
            pd.DataFrame(c_features)[RANKING_FEATURES].astype(np.float32))
        latencies.append((time.perf_counter() - t0) * 1000.0)

        ranked = np.argsort(scores)[::-1]
        top_k = [candidates[i] for i in ranked[:k]]
        recommended.update(top_k)

        hit = 1.0 if gt_movie in top_k else 0.0
        hr_list.append(hit)
        ndcg_list.append(
            1.0 / np.log2(top_k.index(gt_movie) + 2) if hit else 0.0)
        sims = [1.0 - _jaccard(genre_of.get(a, set()), genre_of.get(b, set()))
                for x, a in enumerate(top_k) for b in top_k[x + 1:]]
        ild_list.append(float(np.mean(sims)) if sims else 0.0)

    metrics = {
        f"HR@{k}": round(float(np.mean(hr_list)) if hr_list else 0.0, 4),
        f"NDCG@{k}": round(float(np.mean(ndcg_list)) if ndcg_list else 0.0, 4),
        f"ILD@{k}": round(float(np.mean(ild_list)) if ild_list else 0.0, 4),
        "coverage": round(len(recommended) / max(len(catalog_ids), 1), 4),
        "avg_latency_ms": round(float(np.mean(latencies)) if latencies else 0.0, 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "evaluated_users": len(hr_list),
    }

    # Bootstrap CI over users: a single point estimate is not enough to
    # declare a model "better" than the incumbent.
    if hr_list:
        per_user = pd.DataFrame({
            "userId": list(test_ratings.head(max_users)["userId"]),
            "hr": hr_list,
            "ndcg": ndcg_list,
        })
        ci = bootstrap_ranking_ci(per_user, metric="ndcg", n_boot=n_boot, seed=seed)
        metrics[f"NDCG@{k}_ci_low"] = round(ci["ci_low"], 4)
        metrics[f"NDCG@{k}_ci_high"] = round(ci["ci_high"], 4)
        metrics["NDCG_ci_width"] = round(ci["ci_high"] - ci["ci_low"], 4)

    logger.info("Eval: %s", metrics)

    if enforce_gate:
        gate = gate_config or QualityGateConfig(**full_cfg.get("quality_gates", {}))
        check_quality_gate(metrics, gate, k=k, baseline_metrics=baseline_metrics)
        logger.info("Quality gate PASSED.")
    return metrics


# ── Step 5: Promotion ────────────────────────────────────────────────────────

def promote_artifacts(
    metrics: dict[str, float],
    k: int = 10,
    models_dir: Optional[str] = None,
) -> str:
    """Promote the newest candidate artifacts to the serving path.

    Only called *after* the quality gate has passed.  The metrics of the
    promoted version are stored as a sidecar so the next CT run can compare
    the challenger against this incumbent.
    """
    from pipelines.training.versioning import list_versions, promote, save_metrics

    target_dir = models_dir or MODELS_DIR
    promoted = []
    for base_name in ("lgb_ranker", "tfidf_vectorizer", "tfidf_matrix", "als_model"):
        versions = list_versions(target_dir, base_name)
        if not versions:
            logger.warning("No candidate artifact found for %s; skipping promotion.", base_name)
            continue
        versioned_name = promote(target_dir, base_name, os.path.join(target_dir, versions[0]))
        if base_name == "lgb_ranker":
            save_metrics(target_dir, base_name, versioned_name, metrics)
        promoted.append(f"{base_name}->{versioned_name}")

    logger.info("Promoted artifacts: %s", ", ".join(promoted))
    return ",".join(promoted)


def rollback_artifacts(models_dir: Optional[str] = None) -> Optional[str]:
    """Roll the serving artifacts back to the previous version."""
    from pipelines.training.versioning import rollback

    target_dir = models_dir or MODELS_DIR
    return rollback(target_dir, "lgb_ranker")


# ── Step 6: MLflow ───────────────────────────────────────────────────────────

def smoke_test_promoted_model(
    ranker: lgb.LGBMRanker,
    models_dir: Optional[str] = None,
) -> bool:
    """Verify the artifact on disk reproduces the in-memory model's scores.

    This is the gate between the MLflow ``staging`` and ``champion`` aliases:
    it proves that what was written to disk is loadable and numerically
    identical to the model that was evaluated.
    """
    import pickle

    target_dir = models_dir or MODELS_DIR
    latest = os.path.join(target_dir, "lgb_ranker.pkl")
    if not os.path.exists(latest):
        logger.error("Smoke test FAILED: promoted artifact %s does not exist.", latest)
        return False

    try:
        with open(latest, "rb") as f:
            reloaded = pickle.load(f)

        rng = np.random.default_rng(0)
        probe = pd.DataFrame(
            rng.random((16, len(RANKING_FEATURES))),
            columns=list(RANKING_FEATURES),
        ).astype(np.float32)
        a = np.asarray(ranker.predict(probe), dtype=float)
        b = np.asarray(reloaded.predict(probe), dtype=float)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6):
            logger.error("Smoke test FAILED: reloaded model predictions differ from the evaluated model.")
            return False
        logger.info("Smoke test PASSED: promoted artifact reproduces the evaluated model.")
        return True
    except Exception as exc:
        logger.error("Smoke test FAILED: %s", exc)
        return False


def log_to_mlflow(
    ranker: lgb.LGBMRanker,
    metrics: dict[str, float],
    params: dict[str, Any],
    extra_params: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Log params, metrics, environment metadata and the model to MLflow.

    The model is registered under the ``staging`` alias.  Promotion to
    ``champion`` happens separately (and only after a smoke test).
    Returns the registered model version, or ``None`` if logging failed.
    """
    try:
        import mlflow
        import mlflow.lightgbm

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"lgb_ranker_{int(time.time())}"):
            logged = dict(params)
            logged.update(extra_params or {})
            mlflow.log_params(logged)
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})

            # Environment metadata so a production incident can be traced back
            # to an exact, reproducible training run.
            env = _environment_metadata()
            mlflow.log_params({f"env_{k}": str(v) for k, v in env.items()})
            mlflow.log_dict(env, "environment.json")

            if hasattr(ranker, "feature_importances_"):
                fi = dict(zip(RANKING_FEATURES,
                              [float(x) for x in ranker.feature_importances_]))
                logger.info("Feature importance: %s", fi)
                mlflow.log_dict(fi, "feature_importance.json")

            mlflow.lightgbm.log_model(
                ranker, artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME)
            logger.info("Logged run to MLflow at %s", MLFLOW_URI)

        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'")
        versions = sorted(versions, key=lambda v: int(v.version))
        if not versions:
            return None
        version = versions[-1]
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, "staging", version.version)
        logger.info("Registered %s version %s with alias 'staging'.",
                    REGISTERED_MODEL_NAME, version.version)
        return version.version
    except Exception as exc:
        logger.warning("MLflow logging skipped (%s).", exc)
        return None


def promote_in_mlflow(version: Optional[str], archive_previous: bool = True) -> bool:
    """Move the ``champion`` alias to ``version`` and archive the previous one.

    The previous champion is kept under the ``archived`` alias — it is never
    deleted, so rollback always has a target.
    """
    if not version:
        logger.warning("No MLflow version to promote.")
        return False
    try:
        import mlflow

        client = mlflow.MlflowClient()
        try:
            previous = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        except Exception:
            previous = None

        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
        logger.info("Promoted MLflow version %s to alias 'champion'.", version)

        if archive_previous and previous is not None and previous.version != version:
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, "archived", previous.version)
            logger.info("Archived previous champion %s (kept for rollback).",
                        previous.version)
        return True
    except Exception as exc:
        logger.warning("MLflow promotion skipped (%s).", exc)
        return False


# ── Orchestration ────────────────────────────────────────────────────────────

def run_pipeline() -> dict[str, float]:
    """Run full CT pipeline with enforced quality gate.

    Order of operations (a gate-failing model never reaches production):

        train -> evaluate -> gate -> promote -> MLflow -> smoke test -> champion
    """
    start = time.time()
    logger.info("=== STARTING CONTINUOUS TRAINING (CT) PIPELINE ===")
    df_movies, train_ratings, test_ratings = load_and_split_data()
    train_retrieval_models(df_movies, train_ratings)
    config = load_configs()
    ranker = train_ranking_model(
        df_movies, train_ratings, params=config["ranker"])

    k = config["evaluation"].get("k", 10)

    # Baseline = metrics of the currently promoted model (if any).
    from pipelines.training.versioning import load_champion_metrics

    baseline = load_champion_metrics(MODELS_DIR, "lgb_ranker")
    if baseline:
        logger.info("Incumbent baseline: %s", baseline)
    else:
        # Loud, not silent: until the first promotion writes
        # ``lgb_ranker.{champion}.metrics.json``, the incumbent comparison
        # (RMSE improvement + NDCG regression guardrail) is disabled and the
        # gate runs absolute thresholds only.
        logger.warning(
            "No incumbent baseline sidecar found in %s — incumbent comparison "
            "is DISABLED (absolute thresholds only). Bootstrap it by completing "
            "one promotion, or restore the metrics sidecar.", MODELS_DIR)

    metrics = evaluate_pipeline(
        ranker, df_movies, test_ratings, train_ratings,
        k=k, enforce_gate=True,
        gate_config=QualityGateConfig(**config["quality_gates"]),
        baseline_metrics=baseline)

    # ---- Gate passed -> promote ------------------------------------------
    promote_artifacts(metrics, k=k)

    version = log_to_mlflow(
        ranker, metrics, config["ranker"],
        extra_params={
            "seed": config.get("sampling", {}).get("seed", 42),
            "split_protocol": "time_based_loo",
            "n_features": len(RANKING_FEATURES),
            "data_rows_train": len(train_ratings),
            "data_rows_test": len(test_ratings),
            "movies": len(df_movies),
        },
    )

    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("promote_to_production", True):
        ok = True
        if mlflow_cfg.get("smoke_test", True):
            ok = smoke_test_promoted_model(ranker)
        if ok:
            promote_in_mlflow(version)
        else:
            logger.error("Smoke test failed — model kept at 'staging', champion untouched.")

    try:
        from pipelines.batch.upload_models_to_seaweedfs import upload_all_models

        upload_all_models()
    except Exception as exc:
        logger.warning("SeaweedFS sync skipped: %s", exc)
    logger.info("=== CT COMPLETED IN %ss ===", round(time.time() - start, 2))
    return metrics


if __name__ == "__main__":
    run_pipeline()
