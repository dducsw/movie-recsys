import pytest
from app.chatbot.state import (
    YearQueryModel,
    DirectorQueryModel,
    ActorQueryModel,
    GenreQueryModel,
    RatingQueryModel,
    SimilarQueryModel,
    SearchFilterQueryModel,
    SQLStatsQueryModel,
    MoreRecommendationsQueryModel,
    PersonalizedQueryModel,
    GeneralQAQueryModel,
    IntentOutput,
    parse_intent_to_output,
)
from app.chatbot.agents.detect_intent import extract_json_from_llm_response, heuristic_detect_intent


def test_year_query_model():
    payload = {"intent": "year", "year": "2024", "language": "vi"}
    m = YearQueryModel(**payload)
    assert m.intent == "year"
    assert m.year == "2024"

    out = parse_intent_to_output(payload)
    assert isinstance(out, IntentOutput)
    assert out.intent == "year"
    assert out.year == "2024"


def test_director_query_model():
    payload = {"intent": "director", "director": "Christopher Nolan", "language": "vi"}
    m = DirectorQueryModel(**payload)
    assert m.intent == "director"
    assert m.director == "Christopher Nolan"

    out = parse_intent_to_output(payload)
    assert out.intent == "director"
    assert out.director == "Christopher Nolan"


def test_actor_query_model():
    payload = {"intent": "actor", "actors": ["Leonardo DiCaprio", "Brad Pitt"]}
    m = ActorQueryModel(**payload)
    assert m.intent == "actor"
    assert len(m.actors) == 2

    out = parse_intent_to_output(payload)
    assert out.intent == "actor"
    assert "Leonardo DiCaprio" in out.actors


def test_genre_query_model():
    payload = {"intent": "genre", "genres": ["Action", "Sci-Fi"]}
    m = GenreQueryModel(**payload)
    assert m.intent == "genre"
    assert "Action" in m.genres

    out = parse_intent_to_output(payload)
    assert out.intent == "genre"
    assert "Sci-Fi" in out.genres


def test_rating_query_model():
    payload = {"intent": "rating", "min_rating": 8.5}
    m = RatingQueryModel(**payload)
    assert m.intent == "rating"
    assert m.min_rating == 8.5

    out = parse_intent_to_output(payload)
    assert out.intent == "rating"
    assert out.min_rating == 8.5


def test_similar_query_model():
    payload = {"intent": "similar", "target_title": "Interstellar"}
    m = SimilarQueryModel(**payload)
    assert m.intent == "similar"
    assert m.target_title == "Interstellar"

    out = parse_intent_to_output(payload)
    assert out.intent == "similar"
    assert out.target_title == "Interstellar"


def test_search_filter_query_model():
    payload = {
        "intent": "search_filter",
        "year": "2020",
        "director": "Christopher Nolan",
        "actors": ["John David Washington"],
        "genres": ["Action", "Sci-Fi"],
        "min_rating": 7.5
    }
    m = SearchFilterQueryModel(**payload)
    assert m.intent == "search_filter"
    assert m.year == "2020"
    assert m.director == "Christopher Nolan"

    out = parse_intent_to_output(payload)
    assert out.intent == "search_filter"
    assert out.year == "2020"
    assert out.min_rating == 7.5


def test_extract_json_from_llm_markdown_fences():
    # Markdown json code block
    text = """Here is the extracted intent:
```json
{
  "intent": "year",
  "year": "2022",
  "language": "vi"
}
```
"""
    data = extract_json_from_llm_response(text)
    assert data is not None
    assert data["intent"] == "year"
    assert data["year"] == "2022"

    # Raw braces without code block
    raw = '{"intent": "director", "director": "Quentin Tarantino"}'
    data2 = extract_json_from_llm_response(raw)
    assert data2 is not None
    assert data2["intent"] == "director"
    assert data2["director"] == "Quentin Tarantino"


def test_heuristic_fallback_intents():
    # Year
    res_year = heuristic_detect_intent("tìm phim năm 2022")
    assert res_year.intent == "year"
    assert res_year.year == "2022"

    # Director
    res_dir = heuristic_detect_intent("phim của đạo diễn Christopher Nolan")
    assert res_dir.intent == "director"
    assert "christopher nolan" in res_dir.director.lower()

    # Rating
    res_rat = heuristic_detect_intent("phim có điểm trên 8.0")
    assert res_rat.intent == "rating"
    assert res_rat.min_rating == 8.0

    # Genre
    res_gen = heuristic_detect_intent("gợi ý phim hoạt hình hay")
    assert res_gen.intent == "genre"
    assert "Animation" in res_gen.genres
