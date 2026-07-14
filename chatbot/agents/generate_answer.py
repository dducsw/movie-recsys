import json
from langchain_core.messages import AIMessage

from chatbot.state import GraphState
from chatbot.prompts import ANSWER_PROMPT
from agents import get_llm

def generate_answer_node(state: GraphState) -> dict:
    """Generate contextual response using conversation history."""
    messages = state["messages"]
    original_query = state["original_query"]
    enriched = state.get("enriched_movies", [])
    intent = state["intent_output"].intent
    matched = state.get("matched_movie")

    # Build movie context string
    movie_context = ""
    if matched and intent == "similar":
        movie_context += f"Referenced original movie: {matched.get('title', 'Unknown')}\n\n"
    
    if enriched:
        movie_list_str = json.dumps(
            [{
                "title": m.get("title"),
                "release_date": m.get("release_date"),
                "description": m.get("description", "")
            } for m in enriched],
            ensure_ascii=False, indent=2
        )
        movie_context += f"List of recommended movies:\n{movie_list_str}"
    elif intent == "followup":
        movie_context = "The user is asking follow-up questions about movies previously suggested in the conversation."
    elif intent == "fallback":
        movie_context = "No movies are recommended for this question."

    llm = get_llm()
    prompt = ANSWER_PROMPT
    chain = prompt | llm

    response = chain.invoke({
        "history": messages,
        "movie_context": movie_context,
    })
    final_text = response.content.strip()

    return {
        "final_text": final_text,
        "messages": [AIMessage(content=final_text)]
    }