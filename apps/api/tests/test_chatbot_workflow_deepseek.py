import os
import pytest
from dotenv import load_dotenv

# Ensure environment variables are loaded for tests
load_dotenv(".env")
load_dotenv("../../.env")

from langchain_core.messages import HumanMessage
from app.chatbot import build_chatbot_graph
from app.chatbot.agents import get_deepseek_llm
from app.chatbot.agents.detect_intent import detect_intent_node
from app.chatbot.agents.generate_answer import generate_answer_node
from app.chatbot.state import GraphState, YearQueryModel, IntentOutput
from app.services.chatbot import ChatbotService


def test_deepseek_llm_client_initialization():
    """Verify that get_deepseek_llm correctly initializes an active DeepSeek LLM instance."""
    llm = get_deepseek_llm()
    assert llm is not None
    # Model name should be deepseek-chat, deepseek/deepseek-chat, or deepseek-v4.1-flash:free
    assert "deepseek" in llm.model_name.lower() or "deepseek" in str(getattr(llm, "openai_api_base", "")).lower()


def test_detect_intent_deepseek_year_case():
    """Verify DeepSeek extracts year intent and parameter into specialized YearQueryModel."""
    state = {
        "messages": [HumanMessage(content="tìm phim năm 2022")],
        "session_id": "test_ds_year",
        "intent_output": None,
        "detected_language": None,
        "matched_movie": None,
        "candidate_movies": [],
        "enriched_movies": [],
        "user_id": None,
        "user_liked_ids": None,
        "sql_query": None,
        "sql_results": None,
        "sql_message": None,
        "final_text": ""
    }
    result = detect_intent_node(state)
    intent_output = result.get("intent_output")

    assert intent_output is not None
    assert isinstance(intent_output, IntentOutput)
    assert intent_output.intent == "year"
    assert "2022" in intent_output.year


def test_detect_intent_deepseek_director_case():
    """Verify DeepSeek extracts director intent and director name."""
    state = {
        "messages": [HumanMessage(content="phim của đạo diễn Christopher Nolan")],
        "session_id": "test_ds_dir",
        "intent_output": None,
        "detected_language": None,
        "matched_movie": None,
        "candidate_movies": [],
        "enriched_movies": [],
        "user_id": None,
        "user_liked_ids": None,
        "sql_query": None,
        "sql_results": None,
        "sql_message": None,
        "final_text": ""
    }
    result = detect_intent_node(state)
    intent_output = result.get("intent_output")

    assert intent_output is not None
    assert intent_output.intent == "director"
    assert "nolan" in intent_output.director.lower()


def test_generate_answer_deepseek_node():
    """Verify generate_answer_node invokes DeepSeek to synthesize response text with context."""
    test_movie = {
        "movieId": 361743,
        "title": "Top Gun: Maverick",
        "release_date": "2022-05-21",
        "vote_average": 8.2,
        "director": "Joseph Kosinski",
        "genres": "Action|Drama",
        "description": "After thirty years, Maverick trains graduates for a specialized mission.",
        "tavily_context": "Widely acclaimed aerial action blockbuster."
    }

    state = {
        "messages": [HumanMessage(content="tìm phim năm 2022")],
        "session_id": "test_gen_ans",
        "intent_output": IntentOutput(intent="year", year="2022", language="vi"),
        "detected_language": "vi",
        "matched_movie": None,
        "candidate_movies": [test_movie],
        "enriched_movies": [test_movie],
        "user_id": None,
        "user_liked_ids": None,
        "sql_query": None,
        "sql_results": None,
        "sql_message": None,
        "final_text": ""
    }

    result = generate_answer_node(state)
    final_text = result.get("final_text", "")

    assert final_text is not None
    assert len(final_text) > 20
    # Should contain references to the movie or year
    assert "Top Gun" in final_text or "Maverick" in final_text or "2022" in final_text or "phim" in final_text.lower()


def test_end_to_end_chatbot_workflow_deepseek():
    """Verify the compiled LangGraph workflow end-to-end with DeepSeek."""
    graph = build_chatbot_graph()
    assert graph is not None

    initial_state = {
        "messages": [HumanMessage(content="gợi ý phim hoạt hình hay")],
        "session_id": "test_e2e_deepseek",
        "intent_output": None,
        "detected_language": None,
        "matched_movie": None,
        "candidate_movies": [],
        "enriched_movies": [],
        "user_id": None,
        "user_liked_ids": None,
        "sql_query": None,
        "sql_results": None,
        "sql_message": None,
        "final_text": ""
    }

    final_state = graph.invoke(initial_state)

    # 1. Intent node output
    assert final_state.get("intent_output") is not None
    assert final_state["intent_output"].intent in ("genre", "search_filter")
    assert "Animation" in final_state["intent_output"].genres or "hoạt hình" in str(final_state["intent_output"].genres).lower()

    # 2. Query & Enrich node outputs
    assert len(final_state.get("candidate_movies", [])) > 0

    # 3. Final answer generated by DeepSeek
    final_text = final_state.get("final_text", "")
    assert len(final_text) > 20


def test_chatbot_service_integration():
    """Verify ChatbotService integration using DeepSeek under the hood."""
    result = ChatbotService.get_reply(
        message="tìm phim năm 2022",
        session_id="test_service_deepseek_sess"
    )

    assert isinstance(result, dict)
    assert "text" in result
    assert "movies" in result
    assert len(result["text"]) > 20
    # Returned movies should all be from 2022
    for m in result["movies"]:
        if m.get("release_date"):
            assert "2022" in m["release_date"]
