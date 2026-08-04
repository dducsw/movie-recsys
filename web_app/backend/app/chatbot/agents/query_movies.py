from typing import List, Dict, Any, Optional
from app.chatbot.state import GraphState
from app.models.movie import MovieModel
from app.services.recsys import RecsysService


def query_movies_node(state: GraphState) -> dict:
    """Retrieve candidate movies from the database via MovieModel."""
    intent = state["intent_output"].intent
    target = state.get("intent_output").target_title.strip() if state.get("intent_output").target_title else ""
    genres = state.get("intent_output").genres or []

    candidates: List[Dict[str, Any]] = []
    matched: Optional[Dict[str, Any]] = None

    if intent == "similar" and target:
        # Fuzzy search for the referenced title
        results = MovieModel.search(target, limit=5)
        if results:
            matched = results[0]
            movie_id = matched.get("movieId") or matched.get("movieid")
            if movie_id:
                similar = RecsysService.get_similar_movies(int(movie_id), limit=15)
                if similar:
                    candidates = similar

    if not candidates and genres:
        seen_ids: set = set()
        for g in genres:
            g_lower = g.strip().lower()
            if not g_lower:
                continue
            results = MovieModel.get_movie_by_genre(g_lower, page=1, limit=6)
            for m in results:
                mid = m.get("movieId") or m.get("movieid")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    candidates.append(m)

    # General text search or keyword fallback if no candidates retrieved yet
    if not candidates:
        last_user_msg = state["messages"][-1].content if state.get("messages") else ""
        if last_user_msg:
            # Try search using the user message query
            search_res = MovieModel.search(last_user_msg, limit=10)
            if search_res:
                candidates = search_res

    # Final fallback to top trending movies so recommendations panel is NEVER empty on recommendation turns
    if not candidates:
        candidates = MovieModel.get_trending(page=1, limit=10)

    return {"matched_movie": matched, "candidate_movies": candidates}