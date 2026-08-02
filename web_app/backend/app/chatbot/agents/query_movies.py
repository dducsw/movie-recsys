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
                similar_ids = RecsysService.get_similar_movies(int(movie_id), top_n=15)
                if similar_ids:
                    candidates = MovieModel.get_by_ids(similar_ids)

    elif intent == "genre" and genres:
        # Fetch movies for each requested genre, page 1, up to 5 per genre
        seen_ids: set = set()
        for g in genres:
            g_lower = g.strip().lower()
            if not g_lower:
                continue
            results = MovieModel.get_movie_by_genre(g_lower, page=1, limit=5)
            for m in results:
                mid = m.get("movieId") or m.get("movieid")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    candidates.append(m)
        # Sort by popularity descending and cap at 15
        candidates.sort(key=lambda x: x.get("popularity", 0) or 0, reverse=True)
        candidates = candidates[:15]

    elif intent == "followup":
        # Pass through — generate_answer will use enriched_movies from prior turns
        pass

    return {"matched_movie": matched, "candidate_movies": candidates}