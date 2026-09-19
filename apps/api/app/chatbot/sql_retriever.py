import logging
import re
from typing import Any

from app.chatbot.agents import get_llm
from app.config.db import get_db_connection

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# POSTGRESQL SCHEMA SPECIFICATION FOR LLM
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_DOCUMENTATION = """
You have access to a PostgreSQL database with normalized movie recommendation tables:

1. movies:
   - movieid (INTEGER, PRIMARY KEY) -- Always alias as "movieId"
   - title (VARCHAR(500))
   - release_date (VARCHAR(50), e.g. '2022-03-04' or '2010')
   - popularity (DOUBLE PRECISION)
   - adult (BOOLEAN)
   - overview (TEXT)
   - vote_average (DOUBLE PRECISION, scale 0-10)
   - vote_count (INT)
   - poster_url (VARCHAR(500))
   - director_id (INT, REFERENCES directors(id))
   - trailer_url (VARCHAR(500))

2. directors:
   - id (SERIAL PRIMARY KEY)
   - name (VARCHAR(255))

3. actors:
   - id (SERIAL PRIMARY KEY)
   - name (VARCHAR(255))

4. movie_actors:
   - movie_id (INT, REFERENCES movies(movieid))
   - actor_id (INT, REFERENCES actors(id))
   - cast_order (INT, 0 = lead role)

5. genres:
   - id (SERIAL PRIMARY KEY)
   - name (VARCHAR(100), e.g. 'Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Animation', 'Romance', 'Thriller', 'Crime', 'Adventure', 'Fantasy')

6. movie_genres:
   - movie_id (INT, REFERENCES movies(movieid))
   - genre_id (INT, REFERENCES genres(id))

7. tags:
   - id (SERIAL PRIMARY KEY)
   - name (VARCHAR(255))

8. movie_tags:
   - movie_id (INT, REFERENCES movies(movieid))
   - tag_id (INT, REFERENCES tags(id))

9. user_ratings:
   - id (SERIAL PRIMARY KEY)
   - user_id (INT, REFERENCES users(id))
   - movie_id (INT)
   - rating (FLOAT, scale 0.5 - 5.0)
   - updated_at (TIMESTAMP)

10. user_watchlist:
    - id (SERIAL PRIMARY KEY)
    - user_id (INT, REFERENCES users(id))
    - movie_id (INT)
    - created_at (TIMESTAMP)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY & VALIDATION GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN_KEYWORDS = [
    r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b",
    r"\balter\b", r"\btruncate\b", r"\bcreate\b", r"\bgrant\b",
    r"\brevoke\b", r"\bexecute\b", r"\bexec\b", r"\bcall\b",
    r"\bpg_sleep\b", r"\bcopy\b", r"\bshutdown\b", r"\bvacuum\b"
]


def validate_and_sanitize_sql(query: str, max_limit: int = 20) -> tuple[bool, str, str | None]:
    """
    Validates that a SQL query is strictly read-only, harmless, and enforces LIMIT.
    Returns: (is_valid, sanitized_sql, error_message)
    """
    if not query or not query.strip():
        return False, "", "Empty SQL query."

    clean_sql = query.strip()
    # Strip markdown SQL fences if LLM produced them
    clean_sql = re.sub(r"^```(?:sql)?\s*", "", clean_sql, flags=re.IGNORECASE)
    clean_sql = re.sub(r"\s*```$", "", clean_sql)
    clean_sql = clean_sql.strip()

    # Disallow multiple statements separated by semicolon
    statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "", "Multi-statement queries are forbidden."
    
    clean_sql = statements[0] if statements else clean_sql

    # Must start with SELECT or WITH
    if not re.match(r"^(?:SELECT|WITH)\b", clean_sql, re.IGNORECASE):
        return False, "", "Only SELECT or WITH queries are permitted."

    # Check forbidden mutating keywords
    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, clean_sql, re.IGNORECASE):
            return False, "", f"Query contains forbidden keyword: {pattern}"

    # Ensure LIMIT clause exists and is clamped
    limit_match = re.search(r"\bLIMIT\s+(\d+)", clean_sql, re.IGNORECASE)
    if limit_match:
        current_limit = int(limit_match.group(1))
        if current_limit > max_limit or current_limit <= 0:
            clean_sql = re.sub(r"\bLIMIT\s+\d+", f"LIMIT {max_limit}", clean_sql, flags=re.IGNORECASE)
    else:
        # Append default limit
        clean_sql = f"{clean_sql} LIMIT {max_limit}"

    return True, clean_sql, None


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC SQL GENERATOR (OFFLINE / FAST FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_generate_sql(query_text: str, user_id: int | None = None) -> str | None:
    """
    Fast, deterministic rule-based Text-to-SQL builder for common questions
    when LLM is offline or query is straightforward.
    """
    q = query_text.lower().strip()

    # 1. Total movie count / Statistics
    if any(k in q for k in ["bao nhiêu phim", "tổng số phim", "có bao nhiêu bộ phim", "how many movies", "total movies"]):
        return "SELECT COUNT(*) AS total_movies FROM movies;"

    # 2. Watchlist inquiry for authenticated user
    if any(k in q for k in ["danh sách theo dõi", "phim tôi đã lưu", "watchlist", "phim đã lưu"]) and user_id:
        return f"""
            SELECT m.movieid AS "movieId", m.title, m.release_date, m.vote_average, m.popularity, m.poster_url, d.name AS director
            FROM movies m
            JOIN user_watchlist w ON m.movieid = w.movie_id
            LEFT JOIN directors d ON m.director_id = d.id
            WHERE w.user_id = {int(user_id)}
            ORDER BY w.created_at DESC
            LIMIT 10;
        """

    # 3. Director queries:
    # 3a. Who directed movie [Title]: "ai là đạo diễn của phim [Title]", "who directed [Title]"
    who_directed_match = re.search(r"(?:ai là đạo diễn của(?: phim)?|who directed|đạo diễn của phim)\s+['\"]?([a-zA-Z0-9\s\u00C0-\u1EF9]+)['\"]?", q)
    if who_directed_match:
        movie_title = who_directed_match.group(1).strip()
        movie_title = re.sub(r"[?.,!]+$", "", movie_title).strip()
        if len(movie_title) >= 2:
            return f"""
                SELECT m.movieid AS "movieId", m.title, m.release_date, m.vote_average, m.popularity, m.poster_url, d.name AS director
                FROM movies m
                LEFT JOIN directors d ON m.director_id = d.id
                WHERE m.title ILIKE '%{movie_title}%'
                LIMIT 5;
            """

    # 3b. Movies by director: "đạo diễn [X]", "phim của đạo diễn [X]", "directed by [X]"
    dir_match = re.search(r"(?:đạo diễn|directed by|director|phim của đạo diễn)\s+['\"]?([a-zA-Z0-9\s\u00C0-\u1EF9]+)['\"]?", q)
    if dir_match:
        director_name = dir_match.group(1).strip()
        director_name = re.sub(r"^(là\s+|ai\s+|nào\s+|của\s+|phim\s+)", "", director_name).strip()
        director_name = re.sub(r"[?.,!]+$", "", director_name).strip()
        if len(director_name) >= 3 and director_name not in ["nào", "ai", "gì", "nhất"]:
            return f"""
                SELECT m.movieid AS "movieId", m.title, m.release_date, m.vote_average, m.popularity, m.poster_url, d.name AS director
                FROM movies m
                JOIN directors d ON m.director_id = d.id
                WHERE d.name ILIKE '%{director_name}%'
                ORDER BY m.popularity DESC, m.vote_average DESC
                LIMIT 10;
            """

    # 4. Actor queries: "diễn viên [X]", "phim có [X] đóng", "phim của [X]", "starring [X]"
    act_match = re.search(r"(?:diễn viên|starring|đóng bởi|phim có)\s+['\"]?([a-zA-Z0-9\s\u00C0-\u1EF9]+)['\"]?", q)
    if act_match:
        actor_name = act_match.group(1).strip()
        actor_name = re.sub(r"^(là\s+|ai\s+|nào\s+)", "", actor_name).strip()
        if len(actor_name) >= 3 and actor_name not in ["nào", "ai", "gì", "nhất"]:
            return f"""
                SELECT m.movieid AS "movieId", m.title, m.release_date, m.vote_average, m.popularity, m.poster_url, a.name AS actor
                FROM movies m
                JOIN movie_actors ma ON m.movieid = ma.movie_id
                JOIN actors a ON ma.actor_id = a.id
                WHERE a.name ILIKE '%{actor_name}%'
                ORDER BY m.popularity DESC
                LIMIT 10;
            """

    # 5. Top rated movies in specific year: "phim hay nhất năm [YYYY]" or "top movies in [YYYY]"
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", q)
    if year_match and any(k in q for k in ["top", "hay nhất", "điểm cao", "best", "đánh giá cao"]):
        year = year_match.group(1)
        return f"""
            SELECT m.movieid AS "movieId", m.title, m.release_date, m.vote_average, m.vote_count, m.popularity, m.poster_url, d.name AS director
            FROM movies m
            LEFT JOIN directors d ON m.director_id = d.id
            WHERE m.release_date LIKE '{year}%' AND m.vote_count >= 10
            ORDER BY m.vote_average DESC, m.vote_count DESC
            LIMIT 10;
        """

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM-POWERED TEXT-TO-SQL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def llm_generate_sql(query_text: str, conversation_summary: str = "", user_id: int | None = None) -> str | None:
    """
    Use LLM to generate a safe, syntactically correct PostgreSQL query
    based on natural language intent.
    """
    llm = get_llm(temperature=0.1)
    if llm is None:
        return None

    system_prompt = f"""You are a senior PostgreSQL database engineer for a movie recommendation platform.
Your task is to convert the user's natural language question into a single, valid PostgreSQL SELECT statement.

{SCHEMA_DOCUMENTATION}

CRITICAL RULES:
1. Generate ONLY valid SQL. Output nothing else — no explanation, no markdown backticks.
2. ONLY SELECT or WITH statements are allowed. NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE.
3. Use case-insensitive matching `ILIKE '%keyword%'` for text filters on titles, directors, actors, genres.
4. When selecting movies, ALWAYS include `m.movieid AS "movieId"`, `m.title`, `m.release_date`, `m.vote_average`, `m.popularity`, and `m.poster_url` so the frontend UI can display interactive movie cards.
5. If the user mentions their watchlist, favorites, or saved movies, use `user_watchlist` with `user_id = {user_id or 0}`.
6. Always append `LIMIT 10` unless the user specifies a specific count (e.g. "top 5" -> LIMIT 5). Maximum LIMIT is 20.
7. If the question cannot be answered by SQL (e.g. pure philosophical question or movie plot synopsis request), respond with: NONE
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {conversation_summary}\nQuestion: {query_text}\nPostgreSQL Query:"}
    ]

    try:
        response = llm.invoke(messages)
        sql = response.content.strip()
        if not sql or sql.upper() == "NONE":
            return None
        return sql
    except Exception as e:
        logger.warning(f"LLM SQL generation error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SQL RETRIEVAL ENGINE & EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class SQLRetriever:
    """
    Standardized SQL Retrieval Engine for Chatbot:
    Validates, executes queries safely with timeout, and extracts movie entities for UI rendering.
    """

    @staticmethod
    def query(
        user_query: str,
        conversation_summary: str = "",
        user_id: int | None = None
    ) -> dict[str, Any]:
        """
        Main entrypoint: Generates, validates, executes SQL, and extracts movie cards.
        """
        # 1. Try fast deterministic heuristic first
        sql_candidate = heuristic_generate_sql(user_query, user_id=user_id)

        # 2. If heuristic didn't match, ask LLM
        if not sql_candidate:
            sql_candidate = llm_generate_sql(user_query, conversation_summary, user_id=user_id)

        if not sql_candidate:
            return {
                "success": False,
                "executed": False,
                "sql": None,
                "rows": [],
                "movie_cards": [],
                "message": "No structured SQL query generated."
            }

        # 3. Validate and sanitize SQL
        is_valid, sanitized_sql, err = validate_and_sanitize_sql(sql_candidate)
        if not is_valid:
            logger.warning(f"SQL validation rejected query: {sql_candidate} | Reason: {err}")
            return {
                "success": False,
                "executed": False,
                "sql": sql_candidate,
                "rows": [],
                "movie_cards": [],
                "message": f"Query validation failed: {err}"
            }

        # 4. Execute safely against PostgreSQL
        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            # Set statement timeout to 3 seconds to avoid blocking
            cur.execute("SET LOCAL statement_timeout = '3000ms';")
            cur.execute(sanitized_sql)
            raw_rows = cur.fetchall()
            rows = [dict(r) for r in raw_rows]

            # 5. Extract movie cards if rows represent movies
            movie_cards: list[dict[str, Any]] = []
            seen_ids = set()
            for r in rows:
                mid = r.get("movieId") or r.get("movieid")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    movie_cards.append({
                        "movieId": int(mid),
                        "title": r.get("title", "Untitled"),
                        "release_date": r.get("release_date", ""),
                        "vote_average": r.get("vote_average", 0.0),
                        "popularity": r.get("popularity", 0.0),
                        "poster_url": r.get("poster_url") or r.get("image_url") or "",
                        "director": r.get("director", "")
                    })

            return {
                "success": True,
                "executed": True,
                "sql": sanitized_sql,
                "rows": rows,
                "movie_cards": movie_cards,
                "message": f"Found {len(rows)} record(s)."
            }
        except Exception as e:
            logger.error(f"SQL execution error on [{sanitized_sql}]: {e}")
            return {
                "success": False,
                "executed": True,
                "sql": sanitized_sql,
                "rows": [],
                "movie_cards": [],
                "message": f"Execution error: {e}"
            }
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
