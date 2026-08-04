from app.chatbot.state import GraphState

def route_after_intent(state: GraphState) -> str:
    """Always route to query_movies so candidate movies are retrieved for horizontal cards display."""
    return "query_movies"
    
def route_after_query(state: GraphState) -> str:
    """Skip enrich movie if no candidates found."""
    return "enrich_movies" if state["candidate_movies"] \
        else "generate_answer"