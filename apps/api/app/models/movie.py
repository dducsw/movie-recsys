from app.config.db import get_db_connection
from app.services.cache import cache_get, cache_set
from fastapi import HTTPException
from typing import List, Dict, Any
import os
import re
import requests

class MovieModel:
    @staticmethod
    def get_trending(page: int, limit: int) -> List[Dict[str, Any]]:
        cache_key = f"movies:trending:{page}:{limit}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        conn = get_db_connection()
        cur = conn.cursor()
        offset = (page - 1) * limit
        try:
            query = """
                WITH paginated_movies AS (
                    SELECT * 
                    FROM movies 
                    ORDER BY popularity DESC 
                    LIMIT %s OFFSET %s
                )
                SELECT 
                    m.movieid AS "movieId", 
                    m.title, 
                    m.release_date, 
                    m.popularity, 
                    m.adult, 
                    m.overview, 
                    m.vote_average, 
                    m.vote_count, 
                    m.poster_url,
                    d.name AS director,
                    COALESCE(STRING_AGG(DISTINCT g.name, '|'), '') AS genres,
                    COALESCE(STRING_AGG(DISTINCT a.name, '|'), '') AS cast,
                    COALESCE(STRING_AGG(DISTINCT t.name, '|'), '') AS keywords
                FROM paginated_movies m
                LEFT JOIN directors d ON m.director_id = d.id
                LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                LEFT JOIN movie_actors ma ON m.movieid = ma.movie_id
                LEFT JOIN actors a ON ma.actor_id = a.id
                LEFT JOIN movie_tags mt ON m.movieid = mt.movie_id
                LEFT JOIN tags t ON mt.tag_id = t.id
                GROUP BY m.movieid, m.title, m.release_date, m.popularity, m.adult, m.overview, m.vote_average, m.vote_count, m.poster_url, d.name
                ORDER BY m.popularity DESC
            """
            cur.execute(query, (limit, offset))
            rows = [dict(r) for r in cur.fetchall()]
            cache_set(cache_key, rows, ttl_seconds=300)
            return rows
        except Exception as e:
            print(f"[Error] Failed to fetch trending movies: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch data from movies table.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_latest(page: int, limit: int) -> List[Dict[str, Any]]:
        cache_key = f"movies:latest:{page}:{limit}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        conn = get_db_connection()
        cur = conn.cursor()
        offset = (page - 1) * limit
        try:
            query = """
                WITH paginated_movies AS (
                    SELECT * 
                    FROM movies 
                    WHERE release_date != '' AND release_date IS NOT NULL
                    ORDER BY release_date DESC, popularity DESC
                    LIMIT %s OFFSET %s
                )
                SELECT 
                    m.movieid AS "movieId", 
                    m.title, 
                    m.release_date, 
                    m.popularity, 
                    m.adult, 
                    m.overview, 
                    m.vote_average, 
                    m.vote_count, 
                    m.poster_url,
                    d.name AS director,
                    COALESCE(STRING_AGG(DISTINCT g.name, '|'), '') AS genres,
                    COALESCE(STRING_AGG(DISTINCT a.name, '|'), '') AS cast,
                    COALESCE(STRING_AGG(DISTINCT t.name, '|'), '') AS keywords
                FROM paginated_movies m
                LEFT JOIN directors d ON m.director_id = d.id
                LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                LEFT JOIN movie_actors ma ON m.movieid = ma.movie_id
                LEFT JOIN actors a ON ma.actor_id = a.id
                LEFT JOIN movie_tags mt ON m.movieid = mt.movie_id
                LEFT JOIN tags t ON mt.tag_id = t.id
                GROUP BY m.movieid, m.title, m.release_date, m.popularity, m.adult, m.overview, m.vote_average, m.vote_count, m.poster_url, d.name
                ORDER BY m.release_date DESC, m.popularity DESC
            """
            cur.execute(query, (limit, offset))
            rows = [dict(r) for r in cur.fetchall()]
            cache_set(cache_key, rows, ttl_seconds=300)
            return rows
        except Exception as e:
            print(f"[Error] Failed to fetch latest movies: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch data from movies table.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_total_count() -> int:
        cache_key = "movies:total_count"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute('SELECT COUNT(*) FROM movies')
            count = cur.fetchone()['count']
            cache_set(cache_key, count, ttl_seconds=3600)
            return count
        except Exception as e:
            print(f"[Error] Failed to get movies count: {e}")
            raise HTTPException(status_code=500, detail="Failed to count movies table.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def search(query_str: str, limit: int) -> List[Dict[str, Any]]:
        return MovieModel.search_with_filters(query_str=query_str, limit=limit)

    @staticmethod
    def search_with_filters(
        query_str: str = "",
        genre: str = None,
        year: str = None,
        status: str = None,
        limit: int = 30
    ) -> List[Dict[str, Any]]:
        cache_key = f"movies:search:{query_str.strip().lower()}:{genre}:{year}:{status}:{limit}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            where_clauses = []
            params = []

            # 1. Text Query Filter
            if query_str and query_str.strip():
                q = f"%{query_str.strip()}%"
                where_clauses.append("""
                    (m.title ILIKE %s 
                     OR d.name ILIKE %s 
                     OR g.name ILIKE %s 
                     OR a.name ILIKE %s 
                     OR t.name ILIKE %s)
                """)
                params.extend([q, q, q, q, q])

            # 2. Genre Filter
            if genre and genre.strip() and genre.strip().lower() != 'all':
                g_param = f"%{genre.strip()}%"
                where_clauses.append("g.name ILIKE %s")
                params.append(g_param)

            # 3. Year Filter
            if year and year.strip() and year.strip().lower() != 'all':
                y = year.strip()
                if y == '2026':
                    where_clauses.append("m.release_date LIKE '2026%'")
                elif y == '2025':
                    where_clauses.append("m.release_date LIKE '2025%'")
                elif y == '2024':
                    where_clauses.append("m.release_date LIKE '2024%'")
                elif y == '2020-2023':
                    where_clauses.append("m.release_date >= '2020' AND m.release_date <= '2023-12-31'")
                elif y == '2010s':
                    where_clauses.append("m.release_date >= '2010' AND m.release_date <= '2019-12-31'")
                elif y == '2000s':
                    where_clauses.append("m.release_date >= '2000' AND m.release_date <= '2009-12-31'")
                elif y == 'classic':
                    where_clauses.append("m.release_date < '2000' AND m.release_date != ''")
                elif len(y) == 4 and y.isdigit():
                    where_clauses.append("m.release_date LIKE %s")
                    params.append(f"{y}%")

            # 4. Status Filter ('released' / 'upcoming')
            if status and status.strip() and status.strip().lower() != 'all':
                st = status.strip().lower()
                if st == 'released':
                    where_clauses.append("m.release_date <= TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD') AND m.release_date != ''")
                elif st == 'upcoming':
                    where_clauses.append("m.release_date > TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD')")

            where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            params.append(limit)

            sql_query = f"""
                WITH matched_movies AS (
                    SELECT m.movieid
                    FROM movies m
                    LEFT JOIN directors d ON m.director_id = d.id
                    LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                    LEFT JOIN genres g ON mg.genre_id = g.id
                    LEFT JOIN movie_actors ma ON m.movieid = ma.movie_id
                    LEFT JOIN actors a ON ma.actor_id = a.id
                    LEFT JOIN movie_tags mt ON m.movieid = mt.movie_id
                    LEFT JOIN tags t ON mt.tag_id = t.id
                    {where_sql}
                    GROUP BY m.movieid
                ),
                paginated_movies AS (
                    SELECT m.*
                    FROM movies m
                    JOIN matched_movies mm ON m.movieid = mm.movieid
                    ORDER BY m.popularity DESC
                    LIMIT %s
                )
                SELECT 
                    m.movieid AS "movieId", 
                    m.title, 
                    m.release_date, 
                    m.popularity, 
                    m.adult, 
                    m.overview, 
                    m.vote_average, 
                    m.vote_count, 
                    m.poster_url,
                    d.name AS director,
                    COALESCE(STRING_AGG(DISTINCT g.name, '|'), '') AS genres,
                    COALESCE(STRING_AGG(DISTINCT a.name, '|'), '') AS cast,
                    COALESCE(STRING_AGG(DISTINCT t.name, '|'), '') AS keywords
                FROM paginated_movies m
                LEFT JOIN directors d ON m.director_id = d.id
                LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                LEFT JOIN movie_actors ma ON m.movieid = ma.movie_id
                LEFT JOIN actors a ON ma.actor_id = a.id
                LEFT JOIN movie_tags mt ON m.movieid = mt.movie_id
                LEFT JOIN tags t ON mt.tag_id = t.id
                GROUP BY m.movieid, m.title, m.release_date, m.popularity, m.adult, m.overview, m.vote_average, m.vote_count, m.poster_url, d.name
                ORDER BY m.popularity DESC
            """
            cur.execute(sql_query, tuple(params))
            rows = [dict(r) for r in cur.fetchall()]
            cache_set(cache_key, rows, ttl_seconds=180)
            return rows
        except Exception as e:
            print(f"[Error] Search with filters failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to execute search query with filters.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_id(movie_id: int) -> Dict[str, Any]:
        cache_key = f"movie:detail:{movie_id}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT 
                    m.movieid AS "movieId", 
                    m.title, 
                    m.release_date, 
                    m.popularity, 
                    m.adult, 
                    m.overview, 
                    m.vote_average, 
                    m.vote_count, 
                    m.poster_url,
                    m.trailer_url,
                    d.name AS director,
                    COALESCE(STRING_AGG(DISTINCT g.name, '|'), '') AS genres,
                    COALESCE(STRING_AGG(DISTINCT a.name, '|'), '') AS cast,
                    COALESCE(STRING_AGG(DISTINCT t.name, '|'), '') AS keywords
                FROM movies m
                LEFT JOIN directors d ON m.director_id = d.id
                LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                LEFT JOIN movie_actors ma ON m.movieid = ma.movie_id
                LEFT JOIN actors a ON ma.actor_id = a.id
                LEFT JOIN movie_tags mt ON m.movieid = mt.movie_id
                LEFT JOIN tags t ON mt.tag_id = t.id
                WHERE m.movieid = %s
                GROUP BY m.movieid, m.trailer_url, d.name
            """
            cur.execute(query, (movie_id,))
            movie = cur.fetchone()
            if not movie:
                raise HTTPException(status_code=404, detail="Movie not found.")
            movie = dict(movie)
            
            # Fast non-blocking trailer resolution
            trailer_url = movie.get("trailer_url")
            if not trailer_url:
                title = movie.get("title", "")
                year = ""
                r_date = str(movie.get("release_date") or "")
                if r_date:
                    year = r_date.split("-")[0]
                search_q = f"{title} {year} official trailer".replace(" ", "+")
                resolved_url = f"https://www.youtube.com/results?search_query={search_q}"
                
                api_key = os.getenv("TMDB_API_KEY")
                if api_key:
                    try:
                        res = requests.get(
                            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                            params={"api_key": api_key},
                            timeout=0.8
                        )
                        if res.status_code == 200:
                            for video in res.json().get("results", []):
                                if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                                    resolved_url = f"https://www.youtube.com/embed/{video.get('key')}"
                                    break
                    except Exception:
                        pass

                movie["trailer_url"] = resolved_url
                try:
                    cur.execute("UPDATE movies SET trailer_url = %s WHERE movieid = %s", (resolved_url, movie_id))
                    conn.commit()
                except Exception:
                    pass

            cache_set(cache_key, movie, ttl_seconds=3600)
            return movie
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Error] Failed to fetch movie detail: {e}")
            raise HTTPException(status_code=500, detail="Failed to query movie detail.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_ids(movie_ids: List[int]) -> List[Dict[str, Any]]:
        if not movie_ids:
            return []
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT 
                    m.movieid AS "movieId", 
                    m.title, 
                    m.release_date, 
                    m.popularity, 
                    m.adult, 
                    m.overview, 
                    m.vote_average, 
                    m.vote_count, 
                    m.poster_url,
                    d.name AS director,
                    COALESCE(STRING_AGG(DISTINCT g.name, '|'), '') AS genres,
                    COALESCE(STRING_AGG(DISTINCT a.name, '|'), '') AS cast,
                    COALESCE(STRING_AGG(DISTINCT t.name, '|'), '') AS keywords
                FROM movies m
                LEFT JOIN directors d ON m.director_id = d.id
                LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                LEFT JOIN movie_actors ma ON m.movieid = ma.movie_id
                LEFT JOIN actors a ON ma.actor_id = a.id
                LEFT JOIN movie_tags mt ON m.movieid = mt.movie_id
                LEFT JOIN tags t ON mt.tag_id = t.id
                WHERE m.movieid IN %s
                GROUP BY m.movieid, d.name
            """
            cur.execute(query, (tuple(movie_ids),))
            return cur.fetchall()
        except Exception as e:
            print(f"[Error] Failed to fetch movies by IDs: {e}")
            raise HTTPException(status_code=500, detail="Failed to query movies by list of IDs.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_all_genres_and_popularity() -> List[Dict[str, Any]]:
        """
        Retrieves movie ids, genres, and popularity of all movies.
        Used by the recommendation service to compute similarities.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT 
                    m.movieid AS "movieId", 
                    m.title, 
                    COALESCE(STRING_AGG(DISTINCT g.name, '|'), '') AS genres, 
                    m.popularity, 
                    m.vote_average, 
                    m.poster_url, 
                    m.release_date
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movieid = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                GROUP BY m.movieid
            """
            cur.execute(query)
            return cur.fetchall()
        except Exception as e:
            print(f"[Error] Failed to fetch all movie genres: {e}")
            raise HTTPException(status_code=500, detail="Failed to query genres from movies.")
        finally:
            cur.close()
            conn.close()
    
    @staticmethod
    def get_movie_by_genre(genre: str, page: int, limit: int) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        offset = (page - 1) * limit

        try:
            query = """
                SELECT
                    m.movieid AS "movieId",
                    m.title,
                    m.release_date,
                    m.popularity,
                    m.adult,
                    m.overview,
                    m.vote_average,
                    m.vote_count,
                    m.poster_url,
                    d.name AS director,
                    COALESCE(STRING_AGG(DISTINCT g2.name, '|'), '') AS genres,
                    COALESCE(STRING_AGG(DISTINCT a.name, '|'), '') AS cast,
                    COALESCE(STRING_AGG(DISTINCT t.name, '|'), '') AS keywords
                FROM movies m

                JOIN movie_genres mg_filter
                    ON m.movieid = mg_filter.movie_id
                JOIN genres g_filter
                    ON mg_filter.genre_id = g_filter.id

                LEFT JOIN directors d
                    ON m.director_id = d.id

                LEFT JOIN movie_genres mg
                    ON m.movieid = mg.movie_id
                LEFT JOIN genres g2
                    ON mg.genre_id = g2.id

                LEFT JOIN movie_actors ma
                    ON ma.movie_id = m.movieid
                LEFT JOIN actors a
                    ON a.id = ma.actor_id

                LEFT JOIN movie_tags mt
                    ON mt.movie_id = m.movieid
                LEFT JOIN tags t
                    ON t.id = mt.tag_id

                WHERE g_filter.name ILIKE %s

                GROUP BY
                    m.movieid,
                    m.title,
                    m.release_date,
                    m.popularity,
                    m.adult,
                    m.overview,
                    m.vote_average,
                    m.vote_count,
                    m.poster_url,
                    d.name

                ORDER BY m.popularity DESC

                LIMIT %s OFFSET %s;
                """
            cur.execute(query, (f"%{genre}%", limit, offset))
            return cur.fetchall()

        except Exception as e:
            print(f"[Error] Failed to fetch movies by genre: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch movies by genre."
            )
        finally:
            cur.close()
            conn.close()