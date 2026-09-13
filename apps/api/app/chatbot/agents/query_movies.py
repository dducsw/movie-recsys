import logging
from typing import List, Dict, Any, Optional
from app.chatbot.state import GraphState
from app.models.movie import MovieModel
from app.services.recsys import RecsysService
from app.services.ml_training_service import MLTrainingModelService

logger = logging.getLogger(__name__)


def query_movies_node(state: GraphState) -> dict:
    """
    Retrieve candidate movies from the catalog and 3-stage recommendation engine
    using the detected intent and user context.
    """
    intent_output = state.get("intent_output")
    intent = intent_output.intent if intent_output else "fallback"
    target = intent_output.target_title.strip() if intent_output and intent_output.target_title else ""
    genres = intent_output.genres or [] if intent_output else []

    user_id = state.get("user_id")
    user_liked_ids = state.get("user_liked_ids") or []

    candidates: List[Dict[str, Any]] = []
    matched: Optional[Dict[str, Any]] = None

    # 1. Similar Movie Intent
    if (intent == "similar" or "similar" in intent) and target:
        results = MovieModel.search_with_filters(query_str=target, limit=5)
        if results:
            matched = results[0]
            movie_id = matched.get("movieId") or matched.get("movieid")
            if movie_id:
                similar = RecsysService.get_similar_movies(int(movie_id), limit=15)
                if similar:
                    candidates = similar

    # 2. Personalized Recommendation Intent
    if (intent == "personalized" or "personal" in intent or "recommend" in intent) and not candidates:
        if user_id:
            user_recs = RecsysService.get_user_recommendations(user_id=user_id, limit=12)
            if user_recs:
                candidates = user_recs
        elif user_liked_ids:
            personalized = RecsysService.get_personalized_recommendations(user_liked_ids, limit=12)
            if personalized:
                candidates = personalized

    # 3. Genre Intent
    if not candidates and genres:
        seen_ids = set()
        for g in genres:
            g_clean = g.strip()
            results = MovieModel.get_movie_by_genre(g_clean.lower(), page=1, limit=8)
            for m in results:
                mid = m.get("movieId") or m.get("movieid")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    candidates.append(m)

    # 4. Search by full user message
    if not candidates:
        last_user_msg = state["messages"][-1].content if state.get("messages") else ""
        if last_user_msg and len(last_user_msg.strip()) > 2:
            search_res = MovieModel.search_with_filters(query_str=last_user_msg, limit=10)
            if search_res:
                candidates = search_res

    # 5. Fallback to Trending Movies (Ensure carousel is never empty)
    if not candidates:
        candidates = MovieModel.get_trending(page=1, limit=10)

    # Re-score candidates using ml_training model if applicable
    if candidates:
        cand_ids = [m.get("movieId") or m.get("movieid") for m in candidates if (m.get("movieId") or m.get("movieid"))]
        try:
            target_g_set = set(genres)
            ml_scores = MLTrainingModelService.score_candidates(
                candidate_ids=cand_ids,
                user_id=user_id,
                liked_movie_ids=user_liked_ids,
                target_genres=target_g_set
            )
            if ml_scores:
                for cand in candidates:
                    mid = cand.get("movieId") or cand.get("movieid")
                    if mid in ml_scores:
                        cand["rank_score"] = ml_scores[mid]
                candidates.sort(key=lambda x: (x.get("rank_score", 0.0), x.get("popularity", 0.0) or 0.0), reverse=True)
        except Exception as e:
            logger.warning(f"Could not re-score candidates in chatbot query: {e}")

    logger.info(f"Retrieved {len(candidates)} candidates for intent '{intent}'")
    return {"matched_movie": matched, "candidate_movies": candidates}