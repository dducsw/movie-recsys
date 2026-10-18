from app.chatbot.state import GraphState

# Intents that need no catalog retrieval (pure chit-chat / explanation).
_NO_RETRIEVAL = {"general_qa", "fallback"}


def route_after_intent(state: GraphState) -> str:
    """Dialogue policy: skip retrieval for pure QA/chit-chat without a target."""
    intent_out = state.get("intent_output")
    intent = intent_out.intent if intent_out else "fallback"
    target = (intent_out.target_title if intent_out else "") or ""
    if intent in _NO_RETRIEVAL and not target.strip():
        return "generate_answer"
    return "query_movies"

def route_after_query(state: GraphState) -> str:
    """Skip enrich movie if no candidates found."""
    return "enrich_movies" if state["candidate_movies"] \
        else "generate_answer"
