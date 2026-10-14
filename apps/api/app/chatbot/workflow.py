from langgraph.graph import END, START, StateGraph

from app.chatbot.agents import (
    detect_intent_node,
    enrich_movies_node,
    generate_answer_node,
    query_movies_node,
)
from app.chatbot.route import route_after_intent, route_after_query
from app.chatbot.state import GraphState


def build_chatbot_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("query_movies", query_movies_node)
    workflow.add_node("enrich_movies", enrich_movies_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # Add edges
    workflow.add_edge(START, "detect_intent")
    workflow.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "query_movies": "query_movies",
            "generate_answer": "generate_answer"
        }
    )
    workflow.add_conditional_edges(
        "query_movies",
        route_after_query,
        {
            "enrich_movies": "enrich_movies",
            "generate_answer": "generate_answer"
        }
    )
    workflow.add_edge("enrich_movies", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()