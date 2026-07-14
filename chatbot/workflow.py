from langgraph.graph import StateGraph, START, END
from chatbot.state import GraphState
from langgraph.checkpoint.memory import MemorySaver
from chatbot.route import route_after_intent, route_after_query
from chatbot.agents import (
    detect_intent_node,
    query_movies_node,
    tavily_search_node,
    generate_answer_node
)

def build_chatbot_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("query_movies", query_movies_node)
    workflow.add_node("tavily_search", tavily_search_node)
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
            "tavilly_search": "tavily_search",
            "generate_answer": "generate_answer"
        }
    )
    workflow.add_edge("tavily_search", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Use MemorySaver for checkpointing (persists state between invocations)
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)