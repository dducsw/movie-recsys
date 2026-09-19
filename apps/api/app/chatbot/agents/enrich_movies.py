from typing import Any

from app.chatbot.state import GraphState


def enrich_movies_node(state: GraphState) -> dict:
    """Maps MovieModel fields to the structure expected by generate_answer_node:
      - overview  → description
      - poster_url → image_url
    """
    candidates = state.get("candidate_movies", [])
    if not candidates:
        return {"enriched_movies": []}

    enriched: list[dict[str, Any]] = []
    for movie in candidates:
        enriched.append({
            **movie,
            "image_url": movie.get("poster_url") or "",
            "description": movie.get("overview") or "",
        })

    return {"enriched_movies": enriched}