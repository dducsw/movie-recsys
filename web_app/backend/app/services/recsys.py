import json
import logging
import math
import os
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from app.models.movie import MovieModel
from app.services.cache import recsys_cache
from app.services.metrics import (
    track_stage_latency,
    RECSYS_REQUESTS_TOTAL,
    RECSYS_CACHE_HITS,
    RECSYS_CACHE_MISSES,
    RECSYS_CANDIDATES_COUNT,
)
from app.services.feature_store import FeatureStoreService
from app.services.ml_training_service import MLTrainingModelService

logger = logging.getLogger(__name__)

# Singletons with lazy initialization
_redis_client = None
_qdrant_client = None
_ranker_model = None


def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        client = redis.Redis(host=host, port=port, db=0, decode_responses=True, socket_timeout=1.0)
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis online feature store.")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        _redis_client = False
    return _redis_client if _redis_client is not False else None


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


def get_ranker_model():
    global _ranker_model
    if _ranker_model is not None:
        return _ranker_model
    try:
        import requests
        curr = os.path.dirname(os.path.abspath(__file__))
        repo_root = curr
        while repo_root and not os.path.exists(os.path.join(repo_root, "evaluation")):
            parent = os.path.dirname(repo_root)
            if parent == repo_root:
                break
            repo_root = parent

        model_path = os.path.join(repo_root, "evaluation", "ml_pipeline", "models", "lgb_ranker.pkl")
        
        # Auto-download from SeaweedFS S3 storage if missing locally
        if not os.path.exists(model_path):
            filer_url = os.getenv("SEAWEEDFS_FILER_URL", "http://localhost:8888")
            seaweed_url = f"{filer_url}/recsys-data/models/lgb_ranker.pkl"
            logger.info(f"Local ranker model not found. Fetching from SeaweedFS: {seaweed_url}...")
            res = requests.get(seaweed_url, timeout=0.5)
            if res.status_code == 200:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, "wb") as f:
                    f.write(res.content)
                logger.info(f"Successfully downloaded ranker model from SeaweedFS to {model_path}")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                _ranker_model = pickle.load(f)
            logger.info(f"Successfully loaded LightGBM Ranker model from {model_path}.")
        else:
            logger.warning(f"Ranker model not found at {model_path}")
            _ranker_model = False
    except Exception as e:
        logger.warning(f"Failed to load Ranker model: {e}")
        _ranker_model = False
    return _ranker_model if _ranker_model is not False else None


class RecsysService:
    """
    3-Stage Industrial Recommendation Engine:
    - Stage 1: Candidate Retrieval (Qdrant Vector Search + Redis Cache + Realtime Trending)
    - Stage 2: Detailed Ranking (LightGBM LambdaRanker Feature Scoring)
    - Stage 3: Re-ranking & Diversity (Maximal Marginal Relevance - MMR)
    """

    @staticmethod
    def _stage_1_retrieval_similar(movie_id: int, limit: int = 80) -> List[Dict[str, Any]]:
        """Stage 1: Gather candidate pool for similar movie request."""
        candidates = {}

        # Source A: Redis Precomputed Similarities (from Spark Batch)
        r = get_redis_client()
        if r:
            try:
                cache = r.get(f"movie:{movie_id}:similar")
                if cache:
                    items = json.loads(cache)
                    for item in items:
                        m_id = item.get("movieId")
                        if m_id and m_id != movie_id:
                            candidates[m_id] = {"movieId": m_id, "retrieval_score": item.get("score", 0.5), "source": "redis_similar"}
            except Exception as e:
                logger.warning(f"Redis similar lookup error: {e}")

        # Source B: Qdrant Vector Nearest Neighbors
        qdrant = get_qdrant_client()
        if qdrant:
            try:
                # Retrieve vector for target movie
                res = qdrant.retrieve(collection_name="movies", ids=[movie_id], with_vectors=True)
                if res and res[0].vector:
                    target_vec = res[0].vector
                    points = qdrant.query_points(
                        collection_name="movies",
                        query=target_vec,
                        limit=limit,
                        query_filter=None
                    ).points
                    for pt in points:
                        p_id = int(pt.id)
                        if p_id != movie_id:
                            score = float(pt.score)
                            if p_id not in candidates or score > candidates[p_id]["retrieval_score"]:
                                candidates[p_id] = {"movieId": p_id, "retrieval_score": score, "source": "qdrant_vector"}
            except Exception as e:
                logger.warning(f"Qdrant vector retrieval error: {e}")

        # Source C: Realtime Trending from Redis
        if r:
            try:
                trending_ids = r.zrevrange("realtime:trending", 0, 30)
                for tid_str in trending_ids:
                    try:
                        tid = int(tid_str)
                        if tid != movie_id and tid not in candidates:
                            candidates[tid] = {"movieId": tid, "retrieval_score": 0.3, "source": "redis_trending"}
                    except ValueError:
                        pass
            except Exception as e:
                logger.warning(f"Redis trending lookup error: {e}")

        # Fallback: Popular movies if candidate pool is small
        if len(candidates) < 20:
            trending = MovieModel.get_trending(page=1, limit=30)
            for m in trending:
                mid = m["movieId"]
                if mid != movie_id and mid not in candidates:
                    candidates[mid] = {"movieId": mid, "retrieval_score": 0.2, "source": "db_trending"}

        return list(candidates.values())

    @staticmethod
    def _stage_1_retrieval_personalized(liked_movie_ids: List[int], limit: int = 100) -> List[Dict[str, Any]]:
        """Stage 1: Gather candidate pool for personalized recommendation request."""
        candidates = {}
        liked_set = set(liked_movie_ids)

        r = get_redis_client()
        qdrant = get_qdrant_client()

        # Source A: Qdrant Vector Search using averaged user vector
        if qdrant and liked_movie_ids:
            try:
                liked_points = qdrant.retrieve(collection_name="movies", ids=liked_movie_ids[:10], with_vectors=True)
                vectors = [pt.vector for pt in liked_points if pt.vector]
                if vectors:
                    avg_vec = np.mean(vectors, axis=0).tolist()
                    points = qdrant.query_points(
                        collection_name="movies",
                        query=avg_vec,
                        limit=limit
                    ).points
                    for pt in points:
                        p_id = int(pt.id)
                        if p_id not in liked_set:
                            candidates[p_id] = {"movieId": p_id, "retrieval_score": float(pt.score), "source": "qdrant_user_vec"}
            except Exception as e:
                logger.warning(f"Qdrant personalized vector search error: {e}")

        # Source B: Redis Precomputed Similarities for each liked movie
        if r and liked_movie_ids:
            try:
                for lmid in liked_movie_ids[:5]:
                    cache = r.get(f"movie:{lmid}:similar")
                    if cache:
                        items = json.loads(cache)[:20]
                        for item in items:
                            m_id = item.get("movieId")
                            if m_id and m_id not in liked_set:
                                score = float(item.get("score", 0.4))
                                if m_id not in candidates or score > candidates[m_id]["retrieval_score"]:
                                    candidates[m_id] = {"movieId": m_id, "retrieval_score": score, "source": "redis_similar"}
            except Exception as e:
                logger.warning(f"Redis personalized lookup error: {e}")

        # Source C: Real-time Trending
        if r:
            try:
                trending_ids = r.zrevrange("realtime:trending", 0, 30)
                for tid_str in trending_ids:
                    try:
                        tid = int(tid_str)
                        if tid not in liked_set and tid not in candidates:
                            candidates[tid] = {"movieId": tid, "retrieval_score": 0.3, "source": "redis_trending"}
                    except ValueError:
                        pass
            except Exception as e:
                logger.warning(f"Redis trending lookup error: {e}")

        # Fallback to Database Trending
        if len(candidates) < 30:
            trending = MovieModel.get_trending(page=1, limit=50)
            for m in trending:
                mid = m["movieId"]
                if mid not in liked_set and mid not in candidates:
                    candidates[mid] = {"movieId": mid, "retrieval_score": 0.2, "source": "db_trending"}

        return list(candidates.values())

    @staticmethod
    def _stage_2_ranking(
        candidates: List[Dict[str, Any]],
        target_genres: set,
        liked_count: int = 1,
        user_id: Optional[int] = None,
        liked_movie_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Stage 2: Score candidates using ml_training CatBoost/LightGBM model or LambdaRanker fallback."""
        if not candidates:
            return []

        cand_ids = [c["movieId"] for c in candidates]
        movies_data = MovieModel.get_by_ids(cand_ids)
        movie_dict = {m["movieId"]: m for m in movies_data}

        # 1. Primary: Score candidates using CatBoost + LightGBM ensemble from ml_training
        try:
            ml_scores = MLTrainingModelService.score_candidates(
                candidate_ids=cand_ids,
                user_id=user_id,
                liked_movie_ids=liked_movie_ids,
                target_genres=target_genres
            )
            if ml_scores:
                valid_candidates = []
                for cand in candidates:
                    m_id = cand["movieId"]
                    if m_id in movie_dict:
                        movie = dict(movie_dict[m_id])
                        movie["rank_score"] = float(ml_scores.get(m_id, 0.0))
                        valid_candidates.append(movie)
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: (x.get("rank_score", 0.0), x.get("popularity", 0.0) or 0.0), reverse=True)
                    return valid_candidates
        except Exception as e:
            logger.warning(f"ml_training scoring encountered error: {e}. Continuing with fallback...")

        # 2. Fallback: Retrieve online features from Feast Feature Store (Redis) + LightGBM ranker
        feast_features = FeatureStoreService.get_online_movie_features(cand_ids)

        ranker = get_ranker_model()
        feature_rows = []
        valid_candidates = []

        for cand in candidates:
            m_id = cand["movieId"]
            if m_id not in movie_dict:
                continue

            movie = movie_dict[m_id]
            genres_str = movie.get("genres") or ""
            movie_genres = set(genres_str.split("|")) if genres_str else set()
            genre_overlap = len(target_genres.intersection(movie_genres))

            # Prefer Feast online store features, fallback to DB metadata
            online_f = feast_features.get(m_id, {})
            pop = float(online_f.get("popularity") or movie.get("popularity", 0.0) or 0.0)
            vote_avg = float(online_f.get("vote_average") or movie.get("vote_average", 0.0) or 0.0)
            
            if "release_year" in online_f and online_f["release_year"]:
                release_year = int(online_f["release_year"])
            else:
                rel_date = str(movie.get("release_date") or "2010")
                try:
                    release_year = int(rel_date[:4])
                except ValueError:
                    release_year = 2010

            retrieval_score = float(cand.get("retrieval_score", 0.0))

            feat = {
                "popularity": pop,
                "vote_average": vote_avg,
                "genre_overlap": genre_overlap,
                "release_year": release_year,
                "user_activity": float(min(liked_count, 20)),
                "user_bias": 0.0,
                "als_score": retrieval_score,
                "cb_score": float(genre_overlap * 2.0)
            }

            feature_rows.append(feat)
            valid_candidates.append(movie)

        if not valid_candidates:
            return []

        if ranker is not False and len(feature_rows) > 0:
            try:
                feat_df = pd.DataFrame(feature_rows)
                cols = ['popularity', 'vote_average', 'genre_overlap', 'release_year', 'user_activity', 'user_bias', 'als_score', 'cb_score']
                feat_df = feat_df[cols]
                scores = ranker.predict(feat_df)
                for idx, score in enumerate(scores):
                    valid_candidates[idx]["rank_score"] = float(score)
            except Exception as e:
                logger.warning(f"LightGBM ranking error ({e}). Using heuristic scoring fallback...")
                for idx, feat in enumerate(feature_rows):
                    score = (feat["genre_overlap"] * 10.0) + (feat["als_score"] * 5.0) + (feat["vote_average"] * 0.5)
                    valid_candidates[idx]["rank_score"] = score
        else:
            for idx, feat in enumerate(feature_rows):
                score = (feat["genre_overlap"] * 10.0) + (feat["als_score"] * 5.0) + (feat["vote_average"] * 0.5)
                valid_candidates[idx]["rank_score"] = score

        valid_candidates.sort(key=lambda x: (x.get("rank_score", 0.0), x.get("popularity", 0.0) or 0.0), reverse=True)
        return valid_candidates

    @staticmethod
    def _stage_3_mmr_reranking(ranked_movies: List[Dict[str, Any]], limit: int = 12, lmbda: float = 0.7) -> List[Dict[str, Any]]:
        """Stage 3: Maximal Marginal Relevance (MMR) for candidate diversification."""
        if not ranked_movies:
            return []

        if len(ranked_movies) <= limit:
            return ranked_movies

        selected = [ranked_movies[0]]
        candidates = ranked_movies[1:].copy()

        def genre_jaccard_sim(m1, m2):
            g1 = set(m1["genres"].split("|")) if m1.get("genres") else set()
            g2 = set(m2["genres"].split("|")) if m2.get("genres") else set()
            if not g1 or not g2:
                return 0.0
            union = g1.union(g2)
            return len(g1.intersection(g2)) / len(union)

        while len(selected) < limit and candidates:
            best_score = -float("inf")
            best_idx = 0

            for idx, cand in enumerate(candidates):
                rel_score = cand.get("rank_score", 0.0)
                max_sim = max(genre_jaccard_sim(cand, s) for s in selected)
                mmr_score = (lmbda * rel_score) - ((1.0 - lmbda) * max_sim * 10.0)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(candidates.pop(best_idx))

        return selected

    @classmethod
    def get_similar_movies(cls, movie_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        """Public API: 3-Stage Recommendation for Similar Movies."""
        cache_key = f"similar:{movie_id}:{limit}"
        cached = recsys_cache.get(cache_key)
        if cached is not None:
            RECSYS_CACHE_HITS.labels(cache_type="similar_l1").inc()
            return cached

        RECSYS_CACHE_MISSES.labels(cache_type="similar_l1").inc()
        target_movie = MovieModel.get_by_id(movie_id)
        target_genres = set(target_movie["genres"].split("|")) if target_movie and target_movie.get("genres") else set()

        # Stage 1: Retrieval
        with track_stage_latency("stage_1_retrieval_similar"):
            raw_candidates = cls._stage_1_retrieval_similar(movie_id, limit=80)
        RECSYS_CANDIDATES_COUNT.labels(stage="stage_1_retrieval").observe(len(raw_candidates))

        # Stage 2: Ranking
        with track_stage_latency("stage_2_ranking"):
            ranked_candidates = cls._stage_2_ranking(raw_candidates, target_genres, liked_count=1)
        RECSYS_CANDIDATES_COUNT.labels(stage="stage_2_ranking").observe(len(ranked_candidates))

        # Remove target movie if present
        filtered_ranked = [m for m in ranked_candidates if m["movieId"] != movie_id]

        # Stage 3: Re-ranking (MMR)
        with track_stage_latency("stage_3_mmr_reranking"):
            final_results = cls._stage_3_mmr_reranking(filtered_ranked, limit=limit, lmbda=0.75)
        RECSYS_CANDIDATES_COUNT.labels(stage="stage_3_reranking").observe(len(final_results))

        res = [{
            "movieId": m["movieId"],
            "title": m["title"],
            "release_date": m["release_date"],
            "genres": m["genres"],
            "popularity": m["popularity"],
            "vote_average": m.get("vote_average", 0.0),
            "poster_url": m["poster_url"]
        } for m in final_results]
        recsys_cache.set(cache_key, res, ttl_seconds=300)
        RECSYS_REQUESTS_TOTAL.labels(endpoint="similar_movies", status="success").inc()
        return res

    @classmethod
    def get_personalized_recommendations(
        cls,
        liked_movie_ids: List[int],
        limit: int = 20,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Public API: 3-Stage Recommendation for Personalized User Feed."""
        if not liked_movie_ids:
            if user_id:
                return cls.get_user_recommendations(user_id, limit=limit)
            return MovieModel.get_trending(page=1, limit=limit)

        sorted_ids = sorted(liked_movie_ids)
        cache_key = f"personalized:{user_id or 0}:{','.join(map(str, sorted_ids))}:{limit}"
        cached = recsys_cache.get(cache_key)
        if cached is not None:
            RECSYS_CACHE_HITS.labels(cache_type="personalized_l1").inc()
            return cached

        RECSYS_CACHE_MISSES.labels(cache_type="personalized_l1").inc()
        liked_movies = MovieModel.get_by_ids(liked_movie_ids)
        target_genres = set()
        for m in liked_movies:
            if m.get("genres"):
                target_genres.update(m["genres"].split("|"))

        # Stage 1: Retrieval
        with track_stage_latency("stage_1_retrieval_personalized"):
            raw_candidates = cls._stage_1_retrieval_personalized(liked_movie_ids, limit=100)
        RECSYS_CANDIDATES_COUNT.labels(stage="stage_1_retrieval").observe(len(raw_candidates))

        # Stage 2: Ranking with ML Training Model
        with track_stage_latency("stage_2_ranking"):
            ranked_candidates = cls._stage_2_ranking(
                raw_candidates,
                target_genres,
                liked_count=len(liked_movie_ids),
                user_id=user_id,
                liked_movie_ids=liked_movie_ids
            )
        RECSYS_CANDIDATES_COUNT.labels(stage="stage_2_ranking").observe(len(ranked_candidates))

        # Filter out already liked movies
        liked_set = set(liked_movie_ids)
        filtered_ranked = [m for m in ranked_candidates if m["movieId"] not in liked_set]

        # Stage 3: Re-ranking (MMR)
        with track_stage_latency("stage_3_mmr_reranking"):
            final_results = cls._stage_3_mmr_reranking(filtered_ranked, limit=limit, lmbda=0.7)
        RECSYS_CANDIDATES_COUNT.labels(stage="stage_3_reranking").observe(len(final_results))

        res = [{
            "movieId": m["movieId"],
            "title": m["title"],
            "release_date": m["release_date"],
            "genres": m["genres"],
            "popularity": m["popularity"],
            "vote_average": m.get("vote_average", 0.0),
            "poster_url": m["poster_url"]
        } for m in final_results]
        recsys_cache.set(cache_key, res, ttl_seconds=180)
        RECSYS_REQUESTS_TOTAL.labels(endpoint="personalized_recommendations", status="success").inc()
        return res

    @classmethod
    def get_user_recommendations(cls, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Direct recommendations using CatBoost + LightGBM model from ml_training."""
        cache_key = f"user_ml_recs:{user_id}:{limit}"
        cached = recsys_cache.get(cache_key)
        if cached is not None:
            RECSYS_CACHE_HITS.labels(cache_type="user_ml_recs").inc()
            return cached

        RECSYS_CACHE_MISSES.labels(cache_type="user_ml_recs").inc()
        recs = MLTrainingModelService.recommend_for_user(user_id=user_id, top_k=limit)
        if not recs:
            recs = MovieModel.get_trending(page=1, limit=limit)

        recsys_cache.set(cache_key, recs, ttl_seconds=300)
        RECSYS_REQUESTS_TOTAL.labels(endpoint="user_recommendations", status="success").inc()
        return recs

