"""
apps/api/app/services/recsys.py
-------------------------------
3-Stage Industrial Recommendation Engine.

    Stage 1  Candidate Retrieval  - Qdrant ANN + Redis precomputed similarities
                                    + realtime/DB trending
    Stage 2  Detailed Ranking     - ml_training CatBoost/LightGBM ensemble,
                                    then the CT LightGBM LambdaRanker as fallback
    Stage 3  Re-ranking           - Maximal Marginal Relevance over genre Jaccard

Resilience contract
-------------------
* No downstream failure (Qdrant, Redis, Feast, a model, **or the database**) is
  allowed to surface as a 5xx from a recommendation endpoint.
* Every degradation increments ``recsys_fallbacks_total{source, endpoint}``.
* Candidates keep their ``source`` tag through Stage 2 so an offline debugging
  pass can attribute a candidate to its retrieval channel.
* Recommendation cache keys embed a model fingerprint, so replacing an artifact
  invalidates the stale slates instead of serving them for the full TTL.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.models.movie import MovieModel
from app.services.cache import recsys_cache
from app.services.feature_store import FeatureStoreService
from app.services.metrics import (
    RECSYS_CACHE_HITS,
    RECSYS_CACHE_MISSES,
    RECSYS_CANDIDATES_COUNT,
    RECSYS_FALLBACKS_TOTAL,
    RECSYS_REQUESTS_TOTAL,
    track_stage_latency,
)
from app.services.ml_training_service import MLTrainingModelService, get_user_bias

try:
    from recsys_core.features import ALL_GENRES, RANKING_FEATURES, build_ranking_row
    from recsys_core.reranking import calculate_user_lambda, maximal_marginal_relevance
except ImportError:  # pragma: no cover - only hit when the package isn't installed
    import sys

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
    from recsys_core.features import ALL_GENRES, RANKING_FEATURES, build_ranking_row
    from recsys_core.reranking import calculate_user_lambda, maximal_marginal_relevance

logger = logging.getLogger(__name__)

# In-process singletons (lazy)
_qdrant_client = None
_ranker_model = None
_MODEL_VERSION: Optional[str] = None

# Never trust an unbounded network response: cap the artifact size we will
# download before handing it to pickle.
_MAX_MODEL_BYTES = 256 * 1024 * 1024


# ── Database guard ───────────────────────────────────────────────────────────

def _safe_db(fn: Callable[[], Any], endpoint: str, source: str, default: Any) -> Any:
    """Call a DB helper, falling back to ``default`` instead of raising."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("DB call degraded (%s): %s", source, exc)
        RECSYS_FALLBACKS_TOTAL.labels(source=source, endpoint=endpoint).inc()
        return default


def _safe_trending(limit: int, endpoint: str) -> List[Dict[str, Any]]:
    return _safe_db(lambda: MovieModel.get_trending(page=1, limit=limit),
                    endpoint=endpoint, source="db_trending", default=[])


def _safe_get_by_ids(ids: List[int], endpoint: str) -> Dict[int, Dict[str, Any]]:
    rows = _safe_db(lambda: MovieModel.get_by_ids(ids),
                    endpoint=endpoint, source="db_movies", default=[])
    return {m["movieId"]: m for m in rows}


def _safe_get_by_id(movie_id: int, endpoint: str) -> Optional[Dict[str, Any]]:
    return _safe_db(lambda: MovieModel.get_by_id(movie_id),
                    endpoint=endpoint, source="db_movie", default=None)


def _fallback(source: str, endpoint: str) -> None:
    """Record a degradation of ``source`` for ``endpoint``."""
    RECSYS_FALLBACKS_TOTAL.labels(source=source, endpoint=endpoint).inc()


# ── Client singletons ────────────────────────────────────────────────────────

def get_redis_client():
    """Online feature store (Redis db=0). Shared via redis_factory."""
    from app.services.redis_factory import get_redis

    return get_redis(db=0, timeout=0.5)


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        from qdrant_client import QdrantClient

        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=host, port=port, timeout=5.0, check_compatibility=False)
        _qdrant_client = client
        logger.info("Connected to Qdrant Vector Database.")
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")
        _qdrant_client = False
    return _qdrant_client if _qdrant_client is not False else None


# ── Ranker artifact loading ──────────────────────────────────────────────────

def _ranker_model_paths() -> List[str]:
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

    candidates = [
        os.path.join(repo_root, "pipelines", "training", "ml_pipeline", "models", "lgb_ranker.pkl"),
        os.path.join(repo_root, "models", "lgb_ranker.pkl"),
    ]
    pinned = os.getenv("RANKER_MODEL_FILE")
    if pinned:
        candidates = [pinned] + candidates
    return candidates


def _ranker_model_path() -> str:
    candidates = _ranker_model_paths()
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _download_ranker_model(model_path: str) -> bool:
    """Fetch the ranker artifact from SeaweedFS.

    Security: a pickle is code, therefore a network response is only trusted
    when its SHA-256 digest is verified.  The digest must come from the
    server's ``X-Checksum-SHA256`` header or from ``RANKER_MODEL_SHA256``.
    """
    import requests

    filer_url = os.getenv("SEAWEEDFS_FILER_URL", "http://localhost:8888")
    seaweed_url = f"{filer_url}/recsys-data/models/lgb_ranker.pkl"
    logger.info("Local ranker model not found. Fetching from SeaweedFS: %s...", seaweed_url)

    try:
        res = requests.get(seaweed_url, timeout=5.0)
    except Exception as exc:
        logger.warning("Could not reach SeaweedFS: %s", exc)
        return False

    if res.status_code != 200:
        logger.warning("SeaweedFS returned %s for the ranker model.", res.status_code)
        return False

    content = res.content
    if len(content) > _MAX_MODEL_BYTES:
        logger.error("Refusing ranker model fetch: %d bytes exceeds the %d byte limit.",
                     len(content), _MAX_MODEL_BYTES)
        return False

    expected_sha = res.headers.get("X-Checksum-SHA256") or os.getenv("RANKER_MODEL_SHA256")
    if not expected_sha:
        logger.error(
            "Refusing to trust a fetched pickle without an integrity check. "
            "Provide RANKER_MODEL_SHA256 (or have the filer send X-Checksum-SHA256)."
        )
        return False

    actual_sha = hashlib.sha256(content).hexdigest()
    if actual_sha.lower() != expected_sha.strip().lower():
        logger.error("Checksum mismatch for ranker model! Expected %s, got %s.",
                     expected_sha, actual_sha)
        return False

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(content)
    logger.info("Verified and downloaded ranker model to %s", model_path)
    return True


def get_ranker_model():
    """Lazy-load the CT LambdaRanker singleton. Returns ``None`` when absent."""
    global _ranker_model
    if _ranker_model is not None:
        return _ranker_model if _ranker_model is not False else None
    _ranker_model = _load_ranker_model()
    return _ranker_model if _ranker_model is not False else None


def _load_ranker_model():
    try:
        model_path = _ranker_model_path()
        if not os.path.exists(model_path) and not _download_ranker_model(model_path):
            logger.warning("Ranker model not found at %s and no verified source available.",
                           model_path)
            return False
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded LightGBM Ranker model from %s.", model_path)
        return model
    except Exception as e:
        logger.warning(f"Failed to load Ranker model: {e}")
        return False


def _bundle_path_if_present() -> Optional[str]:
    """Convenience accessor for cache fingerprinting (does not load the bundle)."""
    try:
        from app.services.ml_training_service import _bundle_path

        path = _bundle_path()
        return path if path and os.path.exists(path) else None
    except Exception:
        return None


def model_version() -> str:
    """Short fingerprint of the served model artifacts.

    Embedded in every recommendation cache key, so replacing an artifact
    invalidates the stale slates rather than serving them for the full TTL.
    The fingerprint uses the artifact size + mtime, which also catches a
    re-published file of identical size.
    """
    global _MODEL_VERSION
    if _MODEL_VERSION is None:
        digest = hashlib.md5()
        for path in _ranker_model_paths():
            if os.path.exists(path):
                st = os.stat(path)
                digest.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode())
        bundle_path = _bundle_path_if_present()
        if bundle_path:
            st = os.stat(bundle_path)
            digest.update(f"{bundle_path}|{st.st_size}|{int(st.st_mtime)}".encode())
        _MODEL_VERSION = digest.hexdigest()[:8]
        logger.info("Serving model fingerprint: %s", _MODEL_VERSION)
    return _MODEL_VERSION


def invalidate_model_caches() -> None:
    """Drop the in-process model fingerprint so new keys are generated."""
    global _MODEL_VERSION
    _MODEL_VERSION = None
    logger.info("Model fingerprint invalidated; new cache keys will be generated.")


def reset_model_state() -> None:  # pragma: no cover - test hook
    """Test hook: drop cached clients/models so state is not shared across tests."""
    global _qdrant_client, _ranker_model, _MODEL_VERSION
    _qdrant_client = None
    _ranker_model = None
    _MODEL_VERSION = None


def _cache_key(kind: str, *parts: Any) -> str:
    """Build a cache key versioned by the served model fingerprint.

    ``v0`` is the "model not loadable locally" case — the fingerprint is still
    deterministic and stable for that state.
    """
    return f"{kind}:v{model_version()}:" + ":".join(str(p) for p in parts)


def warmup_models() -> None:
    """Preload heavy models at startup. Fail-open, never crashes boot."""
    try:
        get_ranker_model()
    except Exception as e:
        logger.warning("Ranker warmup failed: %s", e)
    try:
        from app.services.ml_training_service import get_model_bundle

        get_model_bundle()
    except Exception as e:
        logger.warning("ML bundle warmup failed: %s", e)

    # Rate the artifact actually serving right now; caches built before a swap
    # must not be reused under the new fingerprint.
    invalidate_model_caches()
    _ = model_version()


# ── Heuristic fallback ───────────────────────────────────────────────────────

def _heuristic_rank(
    valid_candidates: List[Dict[str, Any]],
    feature_rows: List[Dict[str, float]],
    endpoint: str,
) -> None:
    """Genre overlap / retrieval affinity / quality heuristic when the ranker is down."""
    _fallback("heuristic_ranking", endpoint)
    for idx, feat in enumerate(feature_rows):
        valid_candidates[idx]["rank_score"] = (
            float(feat.get("genre_overlap", 0.0)) * 10.0
            + float(feat.get("retrieval_score", 0.0)) * 5.0
            + float(feat.get("vote_average", 0.0)) * 0.5
        )


# ── Service ──────────────────────────────────────────────────────────────────

class RecsysService:
    """
    3-Stage Industrial Recommendation Engine:
    - Stage 1: Candidate Retrieval (Qdrant Vector Search + Redis Cache + Realtime Trending)
    - Stage 2: Detailed Ranking (feather-fast ensemble scoring / LambdaRanker)
    - Stage 3: Re-ranking & Diversity (Maximal Marginal Relevance - MMR)
    """

    # ── Stage 1 ────────────────────────────────────────────────────────────

    @staticmethod
    def _stage_1_retrieval_similar(
        movie_id: int,
        limit: int = 80,
        endpoint: str = "similar_movies",
    ) -> List[Dict[str, Any]]:
        """Gather the candidate pool for a similar-movies request."""
        candidates: Dict[int, Dict[str, Any]] = {}

        # Source A: Redis Precomputed Similarities (batch pipeline)
        r = get_redis_client()
        if r:
            try:
                cache = r.get(f"movie:{movie_id}:similar")
                if cache:
                    for item in json.loads(cache):
                        m_id = item.get("movieId")
                        if m_id and m_id != movie_id:
                            candidates[m_id] = {
                                "movieId": m_id,
                                "retrieval_score": item.get("score", 0.5),
                                "source": "redis_similar",
                            }
            except Exception as e:
                logger.warning(f"Redis similar lookup error: {e}")

        # Source B: Qdrant Vector Nearest Neighbors
        qdrant = get_qdrant_client()
        if qdrant:
            try:
                target_vec = None
                if r:
                    try:
                        raw_vec = r.get(f"movie:{movie_id}:vec")
                        if raw_vec:
                            target_vec = json.loads(raw_vec)
                    except Exception:
                        pass

                if not target_vec:
                    res = qdrant.retrieve(collection_name="movies", ids=[movie_id], with_vectors=True)
                    if res and res[0].vector:
                        target_vec = res[0].vector
                        if r:
                            try:
                                r.setex(f"movie:{movie_id}:vec", 86400, json.dumps(target_vec))
                            except Exception:
                                pass

                if target_vec:
                    points = qdrant.query_points(
                        collection_name="movies",
                        query=target_vec,
                        limit=limit,
                        query_filter=None,
                    ).points
                    for pt in points:
                        p_id = int(pt.id)
                        if p_id == movie_id:
                            continue
                        score = float(pt.score)
                        if p_id not in candidates or score > candidates[p_id]["retrieval_score"]:
                            candidates[p_id] = {
                                "movieId": p_id,
                                "retrieval_score": score,
                                "source": "qdrant_vector",
                            }
            except Exception as e:
                logger.warning(f"Qdrant vector retrieval error: {e}")
                _fallback("qdrant", endpoint)

        # Source C: Realtime Trending from Redis
        if r:
            try:
                for tid_str in r.zrevrange("realtime:trending", 0, 30):
                    try:
                        tid = int(tid_str)
                    except ValueError:
                        continue
                    if tid != movie_id and tid not in candidates:
                        candidates[tid] = {
                            "movieId": tid,
                            "retrieval_score": 0.3,
                            "source": "redis_trending",
                        }
            except Exception as e:
                logger.warning(f"Redis trending lookup error: {e}")

        # Fallback: popular movies when the pool is thin.
        if len(candidates) < 20:
            trending = _safe_trending(30, endpoint)
            if trending:
                _fallback("db_trending", endpoint)
            for m in trending:
                mid = m["movieId"]
                if mid != movie_id and mid not in candidates:
                    candidates[mid] = {
                        "movieId": mid,
                        "retrieval_score": 0.2,
                        "source": "db_trending",
                    }

        return list(candidates.values())

    @staticmethod
    def _stage_1_retrieval_personalized(
        liked_movie_ids: List[int],
        limit: int = 100,
        endpoint: str = "personalized_recommendations",
    ) -> List[Dict[str, Any]]:
        """Gather the candidate pool for a personalized request."""
        candidates: Dict[int, Dict[str, Any]] = {}
        liked_set = set(liked_movie_ids)

        r = get_redis_client()
        qdrant = get_qdrant_client()

        # Source A: Qdrant vector search on a recency-weighted user vector.
        if qdrant and liked_movie_ids:
            try:
                # Cache hits follow the provided recency order; the misses
                # fetched from Qdrant are appended in the same order, so the
                # recency weights stay aligned with ``liked_movie_ids``.
                missing_ids: List[int] = []
                vectors: List[Any] = []
                if r:
                    try:
                        cached_raw = r.mget([f"movie:{mid}:vec" for mid in liked_movie_ids])
                    except Exception:
                        cached_raw = [None] * len(liked_movie_ids)
                    for mid, val in zip(liked_movie_ids, cached_raw):
                        if val:
                            vectors.append(json.loads(val))
                        else:
                            missing_ids.append(mid)
                else:
                    missing_ids = list(liked_movie_ids)

                if missing_ids:
                    liked_points = qdrant.retrieve(
                        collection_name="movies", ids=missing_ids, with_vectors=True)
                    for pt in liked_points:
                        if pt.vector:
                            vectors.append(pt.vector)
                            if r:
                                try:
                                    r.setex(f"movie:{pt.id}:vec", 86400, json.dumps(pt.vector))
                                except Exception:
                                    pass

                if vectors:
                    arr = np.array(vectors, dtype=np.float32)
                    weights = np.exp(np.linspace(-0.6, 0.0, len(arr)))
                    weights = weights / weights.sum()
                    weighted_vec = np.sum(arr * weights[:, np.newaxis], axis=0)
                    norm = float(np.linalg.norm(weighted_vec))
                    user_vec = (weighted_vec / norm).tolist() if norm > 1e-6 else np.mean(arr, axis=0).tolist()

                    points = qdrant.query_points(
                        collection_name="movies", query=user_vec, limit=limit).points
                    for pt in points:
                        p_id = int(pt.id)
                        if p_id in liked_set:
                            continue
                        candidates[p_id] = {
                            "movieId": p_id,
                            "retrieval_score": float(pt.score),
                            "source": "qdrant_user_vec",
                        }
            except Exception as e:
                logger.warning(f"Qdrant personalized vector search error: {e}")
                _fallback("qdrant", endpoint)

        # Source B: Redis precomputed similarities (single mget, no N+1).
        if r and liked_movie_ids:
            try:
                keys = [f"movie:{lmid}:similar" for lmid in liked_movie_ids[:5]]
                for cache in r.mget(keys):
                    if not cache:
                        continue
                    for item in json.loads(cache)[:20]:
                        m_id = item.get("movieId")
                        if m_id and m_id not in liked_set:
                            score = float(item.get("score", 0.4))
                            if m_id not in candidates or score > candidates[m_id]["retrieval_score"]:
                                candidates[m_id] = {
                                    "movieId": m_id,
                                    "retrieval_score": score,
                                    "source": "redis_similar",
                                }
            except Exception as e:
                logger.warning(f"Redis personalized lookup error: {e}")

        # Source C: Real-time Trending
        if r:
            try:
                for tid_str in r.zrevrange("realtime:trending", 0, 30):
                    try:
                        tid = int(tid_str)
                    except ValueError:
                        continue
                    if tid not in liked_set and tid not in candidates:
                        candidates[tid] = {
                            "movieId": tid,
                            "retrieval_score": 0.3,
                            "source": "redis_trending",
                        }
            except Exception as e:
                logger.warning(f"Redis trending lookup error: {e}")

        # Fallback to DB trending when the pool is thin.
        if len(candidates) < 30:
            trending = _safe_trending(50, endpoint)
            if trending:
                _fallback("db_trending", endpoint)
            for m in trending:
                mid = m["movieId"]
                if mid not in liked_set and mid not in candidates:
                    candidates[mid] = {
                        "movieId": mid,
                        "retrieval_score": 0.2,
                        "source": "db_trending",
                    }

        return list(candidates.values())

    # ── Stage 2 ────────────────────────────────────────────────────────────

    @staticmethod
    def _stage_2_ranking(
        candidates: List[Dict[str, Any]],
        target_genres: set,
        liked_count: int = 1,
        user_id: Optional[int] = None,
        liked_movie_ids: Optional[List[int]] = None,
        endpoint: str = "personalized_recommendations",
    ) -> List[Dict[str, Any]]:
        """Score candidates with the ml_training ensemble, LambdaRanker fallback.

        The candidate ``source`` tag is preserved as ``retrieval_source`` all
        the way through so downstream debugging can attribute each slate
        member to its retrieval channel.
        """
        if not candidates:
            return []

        cand_ids = [c["movieId"] for c in candidates]
        movie_dict = _safe_get_by_ids(cand_ids, endpoint=endpoint)
        if not movie_dict:
            # DB metadata is the backbone Stage 2 builds on; degrade gracefully.
            return []

        # 1. Primary: CatBoost + LightGBM ensemble from ml_training.
        try:
            ml_scores = MLTrainingModelService.score_candidates(
                candidate_ids=cand_ids,
                user_id=user_id,
                liked_movie_ids=liked_movie_ids,
                target_genres=target_genres,
            )
        except Exception as e:
            logger.warning("ml_training scoring raised: %s. Continuing with fallback...", e)
            _fallback("ml_training_ensemble", endpoint)
            ml_scores = {}

        if ml_scores:
            valid_candidates = []
            for cand in candidates:
                m_id = cand["movieId"]
                if m_id not in movie_dict:
                    continue
                valid_candidates.append({
                    **movie_dict[m_id],
                    "rank_score": float(ml_scores.get(m_id, 0.0)),
                    "retrieval_score": float(cand.get("retrieval_score", 0.0)),
                    "retrieval_source": cand.get("source"),
                })
            if valid_candidates:
                valid_candidates.sort(
                    key=lambda x: (x.get("rank_score", 0.0), float(x.get("popularity") or 0.0)),
                    reverse=True,
                )
                return valid_candidates

        # 2. Fallback: Feast online store + CT LightGBM LambdaRanker.
        feast_features = FeatureStoreService.get_online_movie_features(cand_ids)
        if not feast_features:
            _fallback("feast", endpoint)

        ranker = get_ranker_model()
        user_bias = get_user_bias(user_id)
        feature_rows: List[Dict[str, float]] = []
        valid_candidates = []

        for cand in candidates:
            m_id = cand["movieId"]
            if m_id not in movie_dict:
                continue

            movie = movie_dict[m_id]
            genres_str = movie.get("genres") or ""
            movie_genres = set(genres_str.split("|")) if genres_str else set()
            retrieval_score = float(cand.get("retrieval_score", 0.0))

            # Prefer Feast online store features, fallback to DB metadata.
            online_f = feast_features.get(m_id, {})
            pop = float(online_f.get("popularity") or movie.get("popularity", 0.0) or 0.0)
            vote_avg = float(online_f.get("vote_average") or movie.get("vote_average", 0.0) or 0.0)
            vote_cnt = float(online_f.get("vote_count") or movie.get("vote_count") or pop or 0.0)
            bayesian_rating = (
                float((vote_cnt / (vote_cnt + 25.0)) * vote_avg + (25.0 / (vote_cnt + 25.0)) * 5.0)
                if vote_cnt else vote_avg
            )

            if online_f.get("release_year"):
                release_year = int(online_f["release_year"])
            else:
                rel_date = str(movie.get("release_date") or "2010")
                try:
                    release_year = int(rel_date[:4])
                except ValueError:
                    release_year = 2010

            # Shared builder (train == serve). ``retrieval_score`` is the
            # collaborative/ANN affinity slot: training fills it with the iALS
            # dot product, serving fills it with the ANN similarity.
            meta = {
                "popularity": pop,
                "vote_average": vote_avg,
                "bayesian_rating": bayesian_rating,
                "release_year": float(release_year),
                "genres": movie_genres,
                "als_vec": None,
                "retrieval_score": retrieval_score,
            }
            profile = {
                "genres": set(target_genres),
                "activity": float(np.log1p(liked_count)),
                "bias": user_bias,
                "als_vec": None,
            }
            feature_rows.append(build_ranking_row(meta, profile))
            valid_candidates.append({
                **movie,
                "retrieval_score": retrieval_score,
                "retrieval_source": cand.get("source"),
            })

        if not valid_candidates:
            return []

        try:
            feat_df = pd.DataFrame(feature_rows)[list(RANKING_FEATURES)].astype(np.float32)
            if ranker is not None:
                kwargs: Dict[str, Any] = {}
                best_iter = getattr(ranker, "best_iteration_", None)
                if best_iter:
                    kwargs["num_iteration"] = int(best_iter)
                scores = ranker.predict(feat_df, **kwargs)
                for idx, score in enumerate(scores):
                    valid_candidates[idx]["rank_score"] = float(score)
            else:
                _heuristic_rank(valid_candidates, feature_rows, endpoint)
        except Exception as e:
            logger.warning("LightGBM ranking error (%s). Using heuristic scoring fallback...", e)
            _fallback("lgb_ranker", endpoint)
            _heuristic_rank(valid_candidates, feature_rows, endpoint)

        valid_candidates.sort(
            key=lambda x: (x.get("rank_score", 0.0), float(x.get("popularity") or 0.0)),
            reverse=True,
        )
        return valid_candidates

    # ── Stage 3 ────────────────────────────────────────────────────────────

    @staticmethod
    def _stage_3_mmr_reranking(
        ranked_movies: List[Dict[str, Any]],
        limit: int = 12,
        lmbda: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Stage 3: MMR diversification using recsys_core."""
        if not ranked_movies:
            return []
        return maximal_marginal_relevance(
            candidates=ranked_movies,
            limit=limit,
            lmbda=lmbda,
            relevance_key="rank_score",
        )

    # ── Orchestration ──────────────────────────────────────────────────────

    @staticmethod
    def _to_response(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            "movieId": m["movieId"],
            "title": m["title"],
            "release_date": m["release_date"],
            "genres": m["genres"],
            "popularity": m["popularity"],
            "vote_average": m.get("vote_average", 0.0),
            "poster_url": m["poster_url"],
        } for m in items]

    @classmethod
    def get_similar_movies(cls, movie_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        """Public API: 3-Stage Recommendation for Similar Movies."""
        key = _cache_key("similar", movie_id, limit)
        cached = recsys_cache.get(key)
        if cached is not None:
            RECSYS_CACHE_HITS.labels(cache_type="similar_l1").inc()
            return cached

        RECSYS_CACHE_MISSES.labels(cache_type="similar_l1").inc()
        POOL = "similar_movies"
        try:
            target_movie = _safe_get_by_id(movie_id, endpoint=POOL)
            if target_movie is None:
                RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="error").inc()
                return []
            genres_str = target_movie.get("genres") or ""
            target_genres = set(genres_str.split("|")) if genres_str else set()

            with track_stage_latency("stage_1_retrieval", endpoint=POOL):
                raw_candidates = cls._stage_1_retrieval_similar(movie_id, limit=80, endpoint=POOL)
            RECSYS_CANDIDATES_COUNT.labels(stage="stage_1_retrieval", endpoint=POOL).observe(len(raw_candidates))

            with track_stage_latency("stage_2_ranking", endpoint=POOL):
                ranked = cls._stage_2_ranking(raw_candidates, target_genres, liked_count=1, endpoint=POOL)
            RECSYS_CANDIDATES_COUNT.labels(stage="stage_2_ranking", endpoint=POOL).observe(len(ranked))

            ranked = [m for m in ranked if m["movieId"] != movie_id]

            dyn_lambda = calculate_user_lambda(list(target_genres), ALL_GENRES) if target_genres else 0.7
            with track_stage_latency("stage_3_mmr_reranking", endpoint=POOL):
                final = cls._stage_3_mmr_reranking(ranked, limit=limit, lmbda=dyn_lambda)
            RECSYS_CANDIDATES_COUNT.labels(stage="stage_3_reranking", endpoint=POOL).observe(len(final))

            res = cls._to_response(final)
            recsys_cache.set(key, res, ttl_seconds=300)
            RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="success").inc()
            return res
        except Exception as exc:
            logger.error("get_similar_movies failed for movie %s: %s", movie_id, exc)
            RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="error").inc()
            # Last-resort safety net: never let a 5xx reach the client.
            trending = _safe_trending(limit, POOL)
            if trending:
                _fallback("final_trending", POOL)
                return cls._to_response(trending[:limit])
            return []

    @classmethod
    def get_personalized_recommendations(
        cls,
        liked_movie_ids: List[int],
        limit: int = 20,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Public API: 3-Stage Recommendation for Personalized User Feed."""
        POOL = "personalized_recommendations"
        try:
            # Nearline session adaptation: blend fresh interactions when the
            # client only sent a sparse liked list.
            if not liked_movie_ids and session_id:
                try:
                    from app.services.event_producer import get_nearline_interactions

                    nearline_ids = get_nearline_interactions(session_id)
                    if nearline_ids:
                        logger.info("Blending %d nearline session interactions for session %s",
                                    len(nearline_ids), session_id)
                        liked_movie_ids = nearline_ids
                except Exception as exc:
                    logger.debug("Failed nearline interaction lookup: %s", exc)

            if not liked_movie_ids:
                liked_movie_ids = [int(mid) for mid in (liked_movie_ids or [])]

            if not liked_movie_ids:
                if user_id:
                    return cls.get_user_recommendations(user_id, limit=limit)
                trending = _safe_trending(limit, POOL)
                if trending:
                    _fallback("no_history_trending", POOL)
                RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="fallback").inc()
                return cls._to_response(trending[:limit])

            cache_key = _cache_key(
                "personalized", user_id or 0,
                ",".join(map(str, sorted(set(liked_movie_ids)))),
                limit,
            )
            cached = recsys_cache.get(cache_key)
            if cached is not None:
                RECSYS_CACHE_HITS.labels(cache_type="personalized_l1").inc()
                return cached

            RECSYS_CACHE_MISSES.labels(cache_type="personalized_l1").inc()

            liked_movies = list(_safe_get_by_ids(unique_in_order(liked_movie_ids), endpoint=POOL).values())
            target_genres = set()
            for m in liked_movies:
                if m.get("genres"):
                    target_genres.update(m["genres"].split("|"))

            with track_stage_latency("stage_1_retrieval", endpoint=POOL):
                raw_candidates = cls._stage_1_retrieval_personalized(
                    liked_movie_ids, limit=100, endpoint=POOL)
            RECSYS_CANDIDATES_COUNT.labels(stage="stage_1_retrieval", endpoint=POOL).observe(len(raw_candidates))

            with track_stage_latency("stage_2_ranking", endpoint=POOL):
                ranked = cls._stage_2_ranking(
                    raw_candidates,
                    target_genres,
                    liked_count=len(liked_movie_ids),
                    user_id=user_id,
                    liked_movie_ids=liked_movie_ids,
                    endpoint=POOL,
                )
            RECSYS_CANDIDATES_COUNT.labels(stage="stage_2_ranking", endpoint=POOL).observe(len(ranked))

            liked_set = set(liked_movie_ids)
            ranked = [m for m in ranked if m["movieId"] not in liked_set]

            dyn_lambda = calculate_user_lambda(list(target_genres), ALL_GENRES) if target_genres else 0.7
            with track_stage_latency("stage_3_mmr_reranking", endpoint=POOL):
                final = cls._stage_3_mmr_reranking(ranked, limit=limit, lmbda=dyn_lambda)
            RECSYS_CANDIDATES_COUNT.labels(stage="stage_3_reranking", endpoint=POOL).observe(len(final))

            res = cls._to_response(final)
            recsys_cache.set(cache_key, res, ttl_seconds=180)
            RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="success").inc()
            return res
        except Exception as exc:
            logger.error("get_personalized_recommendations failed: %s", exc)
            RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="error").inc()
            trending = _safe_trending(limit, POOL)
            if trending:
                _fallback("final_trending", POOL)
                return cls._to_response(trending[:limit])
            return []

    # ── Async wrappers (blocking inference must never block the loop) ──────

    @classmethod
    async def get_similar_movies_async(cls, movie_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        """Non-blocking async wrapper offloading heavy scoring to a worker thread."""
        return await asyncio.to_thread(cls.get_similar_movies, movie_id, limit)

    @classmethod
    async def get_personalized_recommendations_async(
        cls,
        liked_movie_ids: List[int],
        limit: int = 20,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Non-blocking async wrapper offloading heavy ranking and MMR."""
        return await asyncio.to_thread(
            cls.get_personalized_recommendations,
            liked_movie_ids=liked_movie_ids,
            limit=limit,
            user_id=user_id,
            session_id=session_id,
        )

    # ── Direct ML recommendations ──────────────────────────────────────────

    @classmethod
    def get_user_recommendations(cls, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Direct recommendations using the CatBoost + LightGBM ml_training ensemble."""
        POOL = "user_recommendations"
        cache_key = _cache_key("user_ml_recs", user_id, limit)
        cached = recsys_cache.get(cache_key)
        if cached is not None:
            RECSYS_CACHE_HITS.labels(cache_type="user_ml_recs").inc()
            return cached

        RECSYS_CACHE_MISSES.labels(cache_type="user_ml_recs").inc()
        try:
            recs = MLTrainingModelService.recommend_for_user(user_id=user_id, top_k=limit)
        except Exception as exc:
            logger.warning("recommend_for_user failed: %s", exc)
            _fallback("ml_training_ensemble", POOL)
            recs = []

        if not recs:
            trending = _safe_trending(limit, POOL)
            if trending:
                _fallback("final_trending", POOL)
            recs = trending
            status = "fallback"
        else:
            status = "success"

        recsys_cache.set(cache_key, recs, ttl_seconds=300)
        RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status=status).inc()
        return recs

    @classmethod
    def get_cold_start_recommendations(
        cls,
        preferred_genres: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Cold-start strategy: popular items matching onboarding genres."""
        POOL = "cold_start_recommendations"
        genre_key = "--".join(sorted(set(preferred_genres or [])))
        cache_key = _cache_key("cold_start", genre_key, limit)
        cached = recsys_cache.get(cache_key)
        if cached is not None:
            RECSYS_CACHE_HITS.labels(cache_type="cold_start").inc()
            return cached

        RECSYS_CACHE_MISSES.labels(cache_type="cold_start").inc()
        trending = _safe_trending(limit * 3, POOL)

        if not preferred_genres:
            res = trending[:limit]
        else:
            pref_set = {g.strip().lower() for g in preferred_genres}
            matched, unmatched = [], []
            for m in trending:
                m_genres = {g.strip().lower() for g in str(m.get("genres", "")).split("|")}
                (matched if pref_set.intersection(m_genres) else unmatched).append(m)
            res = (matched + unmatched)[:limit]

        recsys_cache.set(cache_key, res, ttl_seconds=300)
        RECSYS_REQUESTS_TOTAL.labels(endpoint=POOL, status="success" if res else "fallback").inc()
        return res


def unique_in_order(ids: List[int]) -> List[int]:
    """Deduplicate while preserving order (used to fetch liked-movie metadata)."""
    seen: set = set()
    out: List[int] = []
    for mid in ids:
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out
