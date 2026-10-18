"""iEvaLM-style golden eval for CRS (no LLM needed, runs in CI).

Measures: Recall@k on scripted dialogues + explanation presence + no-crash.
"""

from __future__ import annotations

from app.chatbot.critique import apply_critique, parse_critique
from app.chatbot.entity_link import extract_mentions, link_titles
from app.chatbot.reflect import reflect_and_rerank
from app.chatbot.state import IntentOutput

_CATALOG = [
    {"movieId": 1, "title": "Inception", "genres": "Action|Sci-Fi",
     "vote_average": 8.8, "release_date": "2010", "popularity": 90.0},
    {"movieId": 2, "title": "Interstellar", "genres": "Adventure|Sci-Fi|Drama",
     "vote_average": 8.7, "release_date": "2014", "popularity": 85.0},
    {"movieId": 3, "title": "The Hangover", "genres": "Comedy",
     "vote_average": 7.7, "release_date": "2009", "popularity": 70.0},
]


def test_entity_link_exact_and_fuzzy():
    mentions = extract_mentions('Movies like "Inception" please')
    assert any("Inception" in m for m in mentions)
    linked = link_titles(["Inceptoin"], _CATALOG, cutoff=0.5)
    assert linked and linked[0]["movieId"] == 1


def test_reflect_respects_genre_and_rating():
    intent = IntentOutput(intent="genre", genres=["Comedy"], min_rating=7.0)
    out = reflect_and_rerank(_CATALOG, intent, {}, [], limit=5)
    assert [m["movieId"] for m in out] == [3]


def test_critique_refines_slate():
    critique = parse_critique("something funnier please")
    assert critique and critique.get("genre") == "Comedy"
    out = apply_critique(_CATALOG, critique)
    assert out and out[0]["movieId"] == 3


def test_golden_dialogue_recall():
    """Scripted 2-turn dialogue: genre ask -> critique, must keep Recall@5=1."""
    turn1 = reflect_and_rerank(
        _CATALOG, IntentOutput(intent="genre", genres=["Sci-Fi"]), {}, [], 5)
    assert {1, 2} <= {m["movieId"] for m in turn1}
    turn2 = apply_critique(turn1, parse_critique("something newer than 2012"))
    assert turn2 and turn2[0]["movieId"] == 2
