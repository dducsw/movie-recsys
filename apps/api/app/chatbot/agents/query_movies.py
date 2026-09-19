import logging
from typing import Any

from app.chatbot.sql_retriever import SQLRetriever
from app.chatbot.state import GraphState
from app.models.movie import MovieModel
from app.services.recsys import RecsysService

logger = logging.getLogger(__name__)


def query_movies_node(state: GraphState) -> dict:
    """
    Retrieve candidate movies and facts from either:
    1. SQL Database Retrieval (direct facts, cast, directors, counts, user watchlist)
    2. 3-Stage RecSys Engine (ALS + Vector similarity + Genre matching)
    3. Lexical / Keyword matching for open-ended queries
    """
    intent_output = state.get("intent_output")
    intent = intent_output.intent if intent_output else "fallback"
    target = intent_output.target_title.strip() if intent_output and intent_output.target_title else ""
    genres = intent_output.genres or [] if intent_output else []

    user_id = state.get("user_id")
    user_liked_ids = state.get("user_liked_ids") or []
    messages = state.get("messages", [])
    last_user_msg = messages[-1].content if messages else ""

    candidates: list[dict[str, Any]] = []
    matched: dict[str, Any] | None = None
    sql_query_text: str | None = None
    sql_results_data: list[dict[str, Any]] | None = None
    sql_message_text: str | None = None

    # 1. SQL Query Intent (Factual inquiries, directors, actors, statistics, watchlist)
    if intent == "sql_query":
        conv_summary = " ".join([m.content for m in messages[-4:]]) if len(messages) > 1 else ""
        sql_res = SQLRetriever.query(
            user_query=last_user_msg,
            conversation_summary=conv_summary,
            user_id=user_id
        )
        sql_query_text = sql_res.get("sql")
        sql_results_data = sql_res.get("rows")
        sql_message_text = sql_res.get("message")

        if sql_res.get("movie_cards"):
            candidates = sql_res["movie_cards"]

    # 2. Similar Movie Intent
    elif (intent == "similar" or "similar" in intent) and target:
        try:
            results = MovieModel.search_with_filters(query_str=target, limit=5)
            if results:
                matched = results[0]
                movie_id = matched.get("movieId") or matched.get("movieid")
                if movie_id:
                    similar = RecsysService.get_similar_movies(int(movie_id), limit=15)
                    if similar:
                        candidates = similar
        except Exception as e:
            logger.warning(f"Similar movies search failed (DB offline?): {e}")

    # 3. Personalized Recommendation Intent
    elif (intent == "personalized" or "personal" in intent or "recommend" in intent) and not candidates:
        try:
            if user_id:
                user_recs = RecsysService.get_user_recommendations(user_id=user_id, limit=12)
                if user_recs:
                    candidates = user_recs
            elif user_liked_ids:
                personalized = RecsysService.get_personalized_recommendations(user_liked_ids, limit=12)
                if personalized:
                    candidates = personalized
        except Exception as e:
            logger.warning(f"Personalized recommendation failed (DB offline?): {e}")

    # 4. Genre Intent
    elif intent == "genre" and genres:
        try:
            seen_ids = set()
            for g in genres:
                g_clean = g.strip()
                results = MovieModel.get_movie_by_genre(g_clean.lower(), page=1, limit=8)
                for m in results:
                    mid = m.get("movieId") or m.get("movieid")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        candidates.append(m)
        except Exception as e:
            logger.warning(f"Genre movie query failed (DB offline?): {e}")

    # 5. General QA / Plot / Trivia or Search by keywords
    if not candidates and last_user_msg and len(last_user_msg.strip()) > 2:
        try:
            # If user mentioned a specific movie title in question, match it for context
            search_res = MovieModel.search_with_filters(query_str=last_user_msg, limit=5)
            if search_res:
                matched = search_res[0]
                # If query is not general_qa, provide search results as candidate cards
                if intent != "general_qa":
                    candidates = search_res
        except Exception as e:
            logger.debug(f"Keyword search bypassed (DB offline?): {e}")

    # 6. Fallback to Trending Movies only for pure recommendation inquiries
    if not candidates and intent in ("similar", "genre", "personalized"):
        try:
            trending = MovieModel.get_trending(page=1, limit=6)
            if trending:
                candidates = trending
        except Exception as e:
            logger.warning(f"Trending fallback bypassed (DB offline?): {e}")

    return {
        "candidate_movies": candidates,
        "matched_movie": matched,
        "sql_query": sql_query_text,
        "sql_results": sql_results_data,
        "sql_message": sql_message_text
    }