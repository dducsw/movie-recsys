import logging
import os
import requests
from typing import Any

from app.chatbot.state import GraphState

logger = logging.getLogger(__name__)


def search_tavily_for_movie(title: str, year: str = "") -> str:
    """
    Search Tavily API for real-world movie reviews, audience reception, and trivia.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ""

    query = f"{title} {year} movie review audience reception summary".strip()
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 2,
                "search_depth": "basic",
                "include_answer": True
            },
            timeout=2.5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("answer"):
                return data["answer"]
            results = data.get("results", [])
            snippets = [r.get("content", "") for r in results if r.get("content")]
            if snippets:
                return " ".join(snippets)[:350]
    except Exception as e:
        logger.debug(f"Tavily search for '{title}' bypassed: {e}")
    return ""


def enrich_movies_node(state: GraphState) -> dict:
    """
    Enriches candidate movies using:
    1. Tavily Search for real-time reviews, audience consensus, and critical reception.
    2. Field normalization for frontend presentation (poster_url -> image_url, overview -> description).
    """
    candidates = state.get("candidate_movies", [])
    if not candidates:
        return {"enriched_movies": []}

    enriched: list[dict[str, Any]] = []
    # Enrich top candidates with Tavily while keeping conversational latency fast
    for i, movie in enumerate(candidates):
        title = movie.get("title", "")
        rel_date = movie.get("release_date", "")
        year = rel_date[:4] if rel_date and len(rel_date) >= 4 else ""

        tavily_info = ""
        if i == 0 and title:
            tavily_info = search_tavily_for_movie(title, year)

        enriched.append({
            **movie,
            "image_url": movie.get("poster_url") or "",
            "description": movie.get("overview") or "",
            "tavily_context": tavily_info,
        })

    return {"enriched_movies": enriched}