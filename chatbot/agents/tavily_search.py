from chatbot.state import GraphState
from typing import List, Dict, Any
from chatbot.agents import get_tavily

def tavily_search_node(state: GraphState) -> dict:
    """Search Tavily for poster images and descriptions."""
    candidates = state["candidate_movies"]
    if not candidates:
        return {
            "enriched_movies": []
        }
    
    enriched: List[Dict[str, Any]] = []
    
    for movie in candidates:
        title = movie.get("title", "")
        release_date = movie.get("release_date", "")
        query = f"{title} {release_date} movie poster synopsis" if release_date \
            else f"{title} movie poster synopsis."
        
        try:
            tavily_client = get_tavily()
            result = tavily_client.search(query)

            images = result.get("images", [])
            if images:
                image_url = images[0]

            description = result.get("answer", "")
            
            if not description:
                search_results = result.get("results", [])
                if search_results:
                    description = search_results[0].get("content", "")

            enriched.append({
                **movie,
                "image_url": image_url,
                "description": description
            })
        except Exception:
            enriched.append({
                **movie,
                "image_url": "",
                "descriptioon": ""
            })
    return {
        "enriched_movies": enriched
    }