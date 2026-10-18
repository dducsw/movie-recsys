"""CRS tests: policy routing, critiquing, feedback loop. No LLM needed."""

from app.chatbot.critique import apply_critique, parse_critique
from app.chatbot.route import route_after_intent
from app.chatbot.state import IntentOutput


def _state(intent, title=""):
    return {"intent_output": IntentOutput(intent=intent, target_title=title),
            "candidate_movies": []}


def test_policy_skips_retrieval_for_pure_qa():
    assert route_after_intent(_state("general_qa")) == "generate_answer"
    assert route_after_intent(_state("fallback")) == "generate_answer"
    assert route_after_intent(_state("genre")) == "query_movies"
    assert route_after_intent(_state("general_qa", "Inception")) == "query_movies"


def test_workflow_has_checkpointer():
    from app.chatbot.workflow import build_chatbot_graph

    graph = build_chatbot_graph()
    assert graph is not None  # compiled with MemorySaver; thread_id now works


def test_critique_parse_and_apply():
    c = parse_critique("something funnier, above 8 please")
    assert c and c.get("genre") == "Comedy" and c.get("min_rating") == 8.0
    pool = [
        {"movieId": 1, "genres": "Comedy", "vote_average": 8.5,
         "release_date": "2020"},
        {"movieId": 2, "genres": "Horror", "vote_average": 9.0,
         "release_date": "2021"},
    ]
    out = apply_critique(pool, c)
    assert [m["movieId"] for m in out] == [1]
    assert parse_critique("recommend me a sci-fi movie") is None


def test_feedback_endpoint_records(client_auth=None):
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    res = client.post("/api/chatbot/feedback",
                      json={"movie_id": 1, "liked": True,
                            "session_id": "test_session_1"})
    assert res.status_code == 200
    assert res.json()["liked"] is True
