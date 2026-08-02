from app.chatbot.state import GraphState

def route_after_intent(state: GraphState) -> str:
    """Determine text node based on intent."""
    intent = state["intent_output"].intent
    
    if intent in ("similar", "genre"):
        return "query_movies"
    else:
        return "generate_answer"
    
def route_after_query(state: GraphState) -> str:
    """Skip enrich movie if no candidates found."""
    return "enrich_movies" if state["candidate_movies"] \
        else "generate_answer"