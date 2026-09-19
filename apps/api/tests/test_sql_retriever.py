import pytest
from app.chatbot.sql_retriever import (
    validate_and_sanitize_sql,
    heuristic_generate_sql,
    SQLRetriever
)
from app.chatbot.agents.detect_intent import heuristic_detect_intent
from app.chatbot.agents.generate_answer import build_fallback_response


def test_sql_validation_allows_valid_select():
    # Basic SELECT with WHERE
    valid, sql, err = validate_and_sanitize_sql("SELECT * FROM movies WHERE title ILIKE '%Inception%'")
    assert valid is True
    assert "LIMIT 20" in sql
    assert err is None

    # Enforce existing LIMIT clamp
    valid, sql, err = validate_and_sanitize_sql("SELECT * FROM movies LIMIT 100", max_limit=15)
    assert valid is True
    assert "LIMIT 15" in sql


def test_sql_validation_blocks_harmful_queries():
    # Disallow DROP
    valid, _, err = validate_and_sanitize_sql("DROP TABLE users;")
    assert valid is False

    # Disallow DELETE
    valid, _, err = validate_and_sanitize_sql("DELETE FROM movies WHERE movieid = 1")
    assert valid is False

    # Disallow INSERT
    valid, _, err = validate_and_sanitize_sql("INSERT INTO users (username) VALUES ('hacker')")
    assert valid is False

    # Disallow UPDATE
    valid, _, err = validate_and_sanitize_sql("UPDATE users SET password_hash = '123'")
    assert valid is False

    # Disallow SQL injection with multi-statements
    valid, _, err = validate_and_sanitize_sql("SELECT * FROM movies; DROP TABLE users;")
    assert valid is False
    assert "Multi-statement" in err


def test_heuristic_sql_generation():
    # Director query
    sql_dir = heuristic_generate_sql("Phim của đạo diễn Christopher Nolan")
    assert sql_dir is not None
    assert "directors" in sql_dir
    assert "christopher nolan" in sql_dir.lower()

    # Actor query
    sql_actor = heuristic_generate_sql("Những phim có Leonardo DiCaprio đóng")
    assert sql_actor is not None
    assert "actors" in sql_actor
    assert "leonardo dicaprio" in sql_actor.lower()

    # Count statistics
    sql_count = heuristic_generate_sql("Hệ thống có bao nhiêu bộ phim?")
    assert sql_count is not None
    assert "COUNT(*)" in sql_count

    # User watchlist query
    sql_wl = heuristic_generate_sql("phim tôi đã lưu trong danh sách theo dõi", user_id=42)
    assert sql_wl is not None
    assert "user_watchlist" in sql_wl
    assert "42" in sql_wl


def test_new_intent_detection_heuristics():
    # SQL Intent: Director inquiry
    res_dir = heuristic_detect_intent("Các tác phẩm của đạo diễn James Cameron")
    assert res_dir.intent == "sql_query"

    # SQL Intent: Total movies count
    res_count = heuristic_detect_intent("Cho tôi biết có bao nhiêu phim trong hệ thống")
    assert res_count.intent == "sql_query"

    # General QA Intent: Plot analysis
    res_plot = heuristic_detect_intent("Giải thích ý nghĩa cái kết của phim Shutter Island")
    assert res_plot.intent == "general_qa"

    # General QA Intent: Greeting
    res_greet = heuristic_detect_intent("Chào bạn, bạn có thể giúp gì cho tôi?")
    assert res_greet.intent == "general_qa"


def test_build_fallback_response_with_sql_data():
    # Test count aggregation formatting
    count_rows = [{"total_movies": 1250}]
    resp_count = build_fallback_response(
        user_query="Có bao nhiêu phim?",
        intent="sql_query",
        matched_movie=None,
        enriched=[],
        sql_results=count_rows,
        lang="vi"
    )
    assert "1,250" in resp_count or "1250" in resp_count

    # Test movie records formatting
    movie_rows = [
        {"movieId": 101, "title": "Oppenheimer", "release_date": "2023-07-21", "vote_average": 8.2, "director": "Christopher Nolan"}
    ]
    resp_movies = build_fallback_response(
        user_query="Phim của Christopher Nolan",
        intent="sql_query",
        matched_movie=None,
        enriched=[],
        sql_results=movie_rows,
        lang="vi"
    )
    assert "Oppenheimer" in resp_movies
    assert "Christopher Nolan" in resp_movies
    assert "8.2" in resp_movies

    # Test empty SQL result
    resp_empty = build_fallback_response(
        user_query="Phim lạ",
        intent="sql_query",
        matched_movie=None,
        enriched=[],
        sql_results=[],
        lang="vi"
    )
    assert "không tìm thấy" in resp_empty.lower()
