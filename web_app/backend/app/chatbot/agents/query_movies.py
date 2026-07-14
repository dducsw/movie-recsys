from typing import List, Dict, Any, Optional
from app.chatbot.state import GraphState
from app import MovieModel, RecsysService

def query_movies_node(state: GraphState) -> dict:
    """Retrieve candidate movies from local CSV data."""
    intent = state.get("intent_output").intent
    target = state.get("intent_output").target_title.strip()
    genres = state.get("intent_output").genres

    candidates: List[Dict[str, Any]] = []
    matched: Optional[Dict[str, Any]] = None

    if intent == "similar" and target:
        # Fuzzy search for the refernced title
        results = MovieModel.search(target, limit=5)
        if results:
            matched = results[0]
            movie_id = matched.get("movieid")
            if movie_id:
                similar_ids = RecsysService.get_similar_movies(int(movie_id), top_n=15)
                candidates = MovieModel.get_by_id(similar_ids)

    elif intent == "genre" and genres:
        genre_pop = MovieModel.get_all_genres_and_popularity()
        print(genre_pop)
        for g in genres:
            g_lower = g.lower()
            if g_lower in genre_pop:
                candidates.extent(MovieModel.get_movie_by_genre(g_lower, limit=5))
        # Deduplitcate by movieId
        seen = set()
        unique = []
        for m in candidates:
            mid = m.get("movieid")
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(m)

        candidates = unique[:15]

    elif intent == "followup":
        # Look at enriched_movies from previous turns if available
        # For followup, we'll pass through to generate_answer with exisiting context
        pass

    return {
        "matched_movie": matched,
        "candidate_movies": candidates
    }

