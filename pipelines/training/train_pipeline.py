"""
pipelines/training/train_pipeline.py
------------------------------------
Automated 1-Click Continuous Training (CT) Pipeline for 3-Stage Movie RecSys.
Integrates MLflow Experiment Tracking, Metric Quality Gate, and Artifact Publishing.

Usage:
    python pipelines/training/train_pipeline.py
"""

import os
import sys
import time
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure repo root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

try:
    from recsys_core.features import build_movie_meta, build_user_profiles, RANKING_FEATURES
except ImportError:
    _core_path = os.path.join(REPO_ROOT, "libs", "recsys_core", "src")
    if _core_path not in sys.path:
        sys.path.insert(0, _core_path)
    from recsys_core.features import build_movie_meta, build_user_profiles, RANKING_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLOps-TrainPipeline")

_candidates = [
    os.path.join(REPO_ROOT, "notebooks", "04_pipeline_prototypes", "models"),
    os.path.join(REPO_ROOT, "pipelines", "training", "ml_pipeline", "models"),
    os.path.join(REPO_ROOT, "models"),
]
MODELS_DIR = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
os.makedirs(MODELS_DIR, exist_ok=True)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ── Step 1: Data Loading & Time-based Split ────────────────────────────────────

def load_and_split_data():
    """Load movies and ratings data, split using Time-based Leave-One-Out (LOO)."""
    logger.info("Loading training datasets...")
    movies_path = os.path.join(REPO_ROOT, "data", "crawler", "movies_crawled.csv")
    if not os.path.exists(movies_path):
        movies_path = os.path.join(REPO_ROOT, "data", "ml-latest-small", "movies.csv")

    df_movies = pd.read_csv(movies_path)
    if "movieid" in df_movies.columns:
        df_movies.rename(columns={"movieid": "movieId"}, inplace=True)
    df_movies["genres"] = df_movies["genres"].fillna("(no genres listed)")
    df_movies["popularity"] = pd.to_numeric(df_movies.get("popularity", 1.0), errors="coerce").fillna(1.0)
    df_movies["vote_average"] = pd.to_numeric(df_movies.get("vote_average", 5.0), errors="coerce").fillna(5.0)
    ratings_path = os.path.join(REPO_ROOT, "data", "ml-latest-small", "ratings.csv")
    if not os.path.exists(ratings_path):
        ratings_path = os.path.join(REPO_ROOT, "data", "simulator", "sim_ratings.csv")

    df_ratings = pd.read_csv(ratings_path)
    if "movieid" in df_ratings.columns:
        df_ratings.rename(columns={"movieid": "movieId"}, inplace=True)
    if "userid" in df_ratings.columns:
        df_ratings.rename(columns={"userid": "userId"}, inplace=True)

    logger.info(f"Loaded {len(df_movies):,} movies and {len(df_ratings):,} ratings.")

    # Time-based Leave-One-Out split per user
    df_ratings = df_ratings.sort_values("timestamp")
    test_rows = df_ratings.groupby("userId").tail(1)
    train_rows = df_ratings.drop(test_rows.index)

    logger.info(f"Train size: {len(train_rows):,} | Test size (LOO): {len(test_rows):,}")
    return df_movies, train_rows, test_rows


# ── Step 2: Retrieval Models Training ──────────────────────────────────────────

def train_retrieval_models(df_movies, train_ratings):
    """Train Content-Based TF-IDF and Collaborative Filtering (ALS/SVD) representations."""
    logger.info("--- [STAGE 1] Training Retrieval Models ---")

    # 1. Content-Based TF-IDF Matrix
    genres_text = df_movies["genres"].str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer(max_features=50, token_pattern=r"(?u)\b[\w-]+\b")
    tfidf_matrix = vectorizer.fit_transform(genres_text).astype(np.float32)

    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    logger.info("Saved Content-Based TF-IDF models.")

    # 2. Collaborative Filtering (iALS / Sparse SVD)
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    user_codes = train_ratings["userId"].astype("category").cat.codes
    movie_codes = train_ratings["movieId"].astype("category").cat.codes
    unique_movies = train_ratings["movieId"].astype("category").cat.categories

    interaction = csr_matrix((train_ratings["rating"].values, (user_codes, movie_codes)))
    n_comp = min(30, min(interaction.shape) - 1)
    if n_comp >= 2:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        item_factors = svd.fit_transform(interaction.T)
        als_map = {int(mid): item_factors[idx] for idx, mid in enumerate(unique_movies)}
    else:
        als_map = {}

    with open(os.path.join(MODELS_DIR, "als_model.pkl"), "wb") as f:
        pickle.dump(als_map, f)
    logger.info(f"Saved Collaborative Filtering factors for {len(als_map):,} movies.")

    return vectorizer, tfidf_matrix, als_map


CONFIG_PATH = os.path.join(REPO_ROOT, "pipelines", "training", "configs.yaml")


def load_configs():
    """Load hyperparameters from configs.yaml with sensible fallbacks."""
    default_cfg = {
        "ranker": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 5,
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5, 10],
        }
    }
    if os.path.exists(CONFIG_PATH):
        try:
            import yaml
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and "ranker" in loaded:
                    default_cfg["ranker"].update(loaded["ranker"])
        except Exception as e:
            logger.warning(f"Could not load {CONFIG_PATH} ({e}), using default config.")
    return default_cfg


# ── Step 3: Ranking Model (LightGBM LambdaRank) Training ───────────────────────

def train_ranking_model(df_movies, train_ratings, als_map=None, params=None):
    """Build pairwise/listwise ranking features with 8 standard columns matching serving schema."""
    logger.info("--- [STAGE 2] Training LightGBM LambdaRanker (8 Features) ---")

    if params is None:
        params = load_configs().get("ranker", {})

    movie_meta = build_movie_meta(df_movies, als_map)
    global_avg_rating = float(train_ratings["rating"].mean())
    user_profiles = build_user_profiles(train_ratings, movie_meta, global_avg_rating)

    feature_rows = []
    y = []
    groups = []

    for uid, g in train_ratings.groupby("userId"):
        g_len = len(g)
        if g_len < 2:
            continue
        groups.append(g_len)
        u_prof = user_profiles.get(uid, {"activity": 1.0, "bias": 0.0, "genres": set(), "als_vec": None})

        for _, row in g.iterrows():
            m_id = int(row["movieId"])
            meta = movie_meta.get(m_id, {
                "popularity": 1.0,
                "vote_average": 5.0,
                "release_year": 2010.0,
                "genres": set(),
                "als_vec": None
            })

            genre_overlap = float(len(u_prof["genres"].intersection(meta["genres"])))
            # Normalized genre overlap [0, 1] relative to user taste breadth
            cb_score = float(genre_overlap / max(len(u_prof["genres"]), 1))

            # ALS score as dot product between user ALS vector and movie ALS vector
            als_score = 0.0
            if u_prof["als_vec"] is not None and meta["als_vec"] is not None:
                try:
                    als_score = float(np.dot(u_prof["als_vec"], meta["als_vec"]))
                except Exception:
                    als_score = 0.0

            feat = {
                "popularity": meta["popularity"],
                "vote_average": meta["vote_average"],
                "genre_overlap": genre_overlap,
                "release_year": meta["release_year"],
                "user_activity": u_prof["activity"],
                "user_bias": u_prof["bias"],
                "als_score": als_score,
                "cb_score": cb_score
            }
            feature_rows.append(feat)

            rating = float(row["rating"])
            relevance = int(min(max(round(rating - 1), 0), 4))
            y.append(relevance)

    X = pd.DataFrame(feature_rows)[RANKING_FEATURES].astype(np.float32)
    y = np.array(y, dtype=np.int32)

    ranker = lgb.LGBMRanker(
        objective=params.get("objective", "lambdarank"),
        metric=params.get("metric", "ndcg"),
        eval_at=params.get("eval_at", [5, 10]),
        n_estimators=params.get("n_estimators", 100),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 15),
        min_child_samples=params.get("min_child_samples", 5),
        random_state=42,
        importance_type="gain"
    )

    ranker.fit(X, y, group=groups)

    model_path = os.path.join(MODELS_DIR, "lgb_ranker.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ranker, f)
    logger.info(f"Saved LightGBM Ranker (8 features) to {model_path}.")

    return ranker


# ── Step 4: Metric Evaluation & Quality Gate ───────────────────────────────────

def evaluate_pipeline(ranker, df_movies, test_ratings, train_ratings, als_map=None, k=10):
    """Evaluate Hit-Ratio@K, NDCG@K, Diversity, and inference latency using the 8-feature schema.

    Protocol Note:
        Uses a 99-negative sampling protocol (Koren 2008; He et al. 2017 NCF) for computational
        tractability during continuous training. For exhaustive full-catalog offline benchmarking,
        refer to `evaluation/recsys_utils.py`.
    """
    logger.info("--- [STAGE 3] Evaluating End-to-End Metrics & Quality Gate ---")

    movie_meta = build_movie_meta(df_movies, als_map)
    user_profiles = build_user_profiles(train_ratings, movie_meta)

    hr_list = []
    ndcg_list = []
    latencies = []

    max_eval_users = 200  # CT quality-gate speed cap
    for _, test_row in test_ratings.head(max_eval_users).iterrows():
        gt_movie = int(test_row["movieId"])
        uid = test_row["userId"]
        u_prof = user_profiles.get(uid, {"activity": 1.0, "bias": 0.0, "genres": set(), "als_vec": None})

        # 99 negative samples + 1 ground truth
        candidates = [gt_movie]
        all_other_movies = [int(m) for m in df_movies["movieId"].values if int(m) != gt_movie]
        neg_samples = np.random.choice(all_other_movies, size=min(99, len(all_other_movies)), replace=False)
        candidates.extend(neg_samples)

        # Build feature matrix
        t0 = time.perf_counter()
        c_features = []
        for cid in candidates:
            meta = movie_meta.get(cid, {
                "popularity": 1.0,
                "vote_average": 5.0,
                "release_year": 2010.0,
                "genres": set(),
                "als_vec": None
            })
            genre_overlap = float(len(u_prof["genres"].intersection(meta["genres"])))
            # Normalized genre overlap [0, 1] relative to user taste breadth
            cb_score = float(genre_overlap / max(len(u_prof["genres"]), 1))
            als_score = 0.0
            if u_prof["als_vec"] is not None and meta["als_vec"] is not None:
                try:
                    als_score = float(np.dot(u_prof["als_vec"], meta["als_vec"]))
                except Exception:
                    als_score = 0.0

            c_features.append({
                "popularity": meta["popularity"],
                "vote_average": meta["vote_average"],
                "genre_overlap": genre_overlap,
                "release_year": meta["release_year"],
                "user_activity": u_prof["activity"],
                "user_bias": u_prof["bias"],
                "als_score": als_score,
                "cb_score": cb_score
            })

        c_df = pd.DataFrame(c_features)[RANKING_FEATURES].astype(np.float32)
        scores = ranker.predict(c_df)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Rank candidates by score descending
        ranked_order = np.argsort(scores)[::-1]
        top_k_items = [candidates[i] for i in ranked_order[:k]]

        # HR@K
        hit = 1.0 if gt_movie in top_k_items else 0.0
        hr_list.append(hit)

        # NDCG@K
        if hit > 0:
            rank_pos = top_k_items.index(gt_movie) + 1
            ndcg = 1.0 / np.log2(rank_pos + 1)
        else:
            ndcg = 0.0
        ndcg_list.append(ndcg)

    mean_hr = float(np.mean(hr_list))
    mean_ndcg = float(np.mean(ndcg_list))
    avg_latency = float(np.mean(latencies))

    metrics = {
        f"HR@{k}": round(mean_hr, 4),
        f"NDCG@{k}": round(mean_ndcg, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "evaluated_users": len(hr_list)
    }

    logger.info(f"Evaluation Results: HR@{k}={metrics[f'HR@{k}']}, NDCG@{k}={metrics[f'NDCG@{k}']}, Latency={metrics['avg_latency_ms']}ms")

    # MLOps Metric Gate Assertion
    assert metrics["avg_latency_ms"] < 30.0, f"Quality Gate Failed: Latency {metrics['avg_latency_ms']}ms exceeds SLA 30ms!"
    logger.info("Quality Gate PASSED: Latency & Output schema verified.")

    return metrics


# ── Step 5: MLflow Logging ─────────────────────────────────────────────────────

def log_to_mlflow(ranker, metrics, params):
    """Log run params, metrics and model artifact to MLflow Tracking Server."""
    try:
        import mlflow
        import mlflow.lightgbm

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("MovieRecSys-Ranking")

        with mlflow.start_run(run_name=f"lgb_ranker_{int(time.time())}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if hasattr(ranker, "feature_importances_"):
                fi = dict(zip(RANKING_FEATURES, [float(x) for x in ranker.feature_importances_]))
                logger.info(f"Feature Importance (gain): {fi}")
                mlflow.log_dict(fi, "feature_importance.json")
            mlflow.lightgbm.log_model(ranker, artifact_path="model", registered_model_name="MovieRanker-LGBM")
            logger.info(f"Successfully logged experiment run to MLflow at {MLFLOW_URI}")
    except Exception as e:
        logger.warning(f"MLflow logging skipped/failed ({e}). Artifacts remain available locally.")


# ── Main Entrypoint ────────────────────────────────────────────────────────────

def run_pipeline():
    start_time = time.time()
    logger.info("==================================================")
    logger.info("STARTING CONTINUOUS TRAINING (CT) MLOPS PIPELINE")
    logger.info("==================================================")

    # 1. Data load & LOO split
    df_movies, train_ratings, test_ratings = load_and_split_data()

    # 2. Train Retrieval models
    vectorizer, tfidf_matrix, als_map = train_retrieval_models(df_movies, train_ratings)

    # 3. Train Ranking model
    config = load_configs()
    params = config.get("ranker", {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 5,
        "objective": "lambdarank"
    })
    ranker = train_ranking_model(df_movies, train_ratings, als_map=als_map, params=params)

    # 4. Evaluate & Quality Gate
    metrics = evaluate_pipeline(ranker, df_movies, test_ratings, train_ratings, als_map=als_map, k=10)

    # 5. MLflow Tracking
    log_to_mlflow(ranker, metrics, params)

    # 6. Optional Sync to SeaweedFS
    try:
        from pipeline.upload_models_to_seaweedfs import upload_all_models
        upload_all_models()
    except Exception:
        pass

    duration = round(time.time() - start_time, 2)
    logger.info("==================================================")
    logger.info(f"MLOPS TRAINING PIPELINE COMPLETED IN {duration}s")
    logger.info("==================================================")


if __name__ == "__main__":
    run_pipeline()
