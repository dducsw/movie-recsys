from app.config.db import get_db_connection
from fastapi import HTTPException
from typing import List, Dict, Any
import os
import requests

class MovieModel:
    @staticmethod
    def get_trending(page: int, limit: int) -> List[Dict[str, Any]]:
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
            return cur.fetchall()
        except Exception as e:
            print(f"[Error] Failed to fetch trending movies: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch data from movies table.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_latest(page: int, limit: int) -> List[Dict[str, Any]]:
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
            return cur.fetchall()
        except Exception as e:
            print(f"[Error] Failed to fetch latest movies: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch data from movies table.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_total_count() -> int:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute('SELECT COUNT(*) FROM movies')
            return cur.fetchone()['count']
        except Exception as e:
            print(f"[Error] Failed to get movies count: {e}")
            raise HTTPException(status_code=500, detail="Failed to count movies table.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def search(query_str: str, limit: int) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            search_query = f"%{query_str}%"
            sql_query = """
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
                    WHERE m.title ILIKE %s 
                       OR d.name ILIKE %s 
                       OR g.name ILIKE %s 
                       OR a.name ILIKE %s 
                       OR t.name ILIKE %s
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
            cur.execute(sql_query, (search_query, search_query, search_query, search_query, search_query, limit))
            return cur.fetchall()
        except Exception as e:
            print(f"[Error] Search failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to execute search query.")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_id(movie_id: int) -> Dict[str, Any]:
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
                GROUP BY m.movieid, d.name
            """
            cur.execute(query, (movie_id,))
            movie = cur.fetchone()
            if not movie:
                raise HTTPException(status_code=404, detail="Movie not found.")
            
            # If trailer_url is not in database, attempt to fetch it from TMDB
            trailer_url = movie.get("trailer_url")
            if not trailer_url:
                try:
                    api_key = os.getenv("TMDB_API_KEY")
                    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
                    res = requests.get(url, params={"api_key": api_key}, timeout=2.0)
                    
                    youtube_key = None
                    if res.status_code == 200:
                        videos_data = res.json().get("results", [])
                        # 1. Search for official trailers on YouTube
                        for video in videos_data:
                            if video.get("site") == "YouTube" and video.get("type") == "Trailer" and video.get("official"):
                                youtube_key = video.get("key")
                                break
                        # 2. Fallback to any trailer on YouTube
                        if not youtube_key:
                            for video in videos_data:
                                if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                                    youtube_key = video.get("key")
                                    break
                        # 3. Fallback to any video on YouTube
                        if not youtube_key:
                            for video in videos_data:
                                if video.get("site") == "YouTube":
                                    youtube_key = video.get("key")
                                    break
                                    
                    if youtube_key:
                        trailer_url = f"https://www.youtube.com/embed/{youtube_key}"
                    else:
                        # Fallback to YouTube search link
                        title = movie.get("title", "")
                        year = ""
                        r_date = movie.get("release_date", "")
                        if r_date:
                            year = r_date.split("-")[0]
                        search_q = f"{title} {year} official trailer".replace(" ", "+")
                        trailer_url = f"https://www.youtube.com/results?search_query={search_q}"
                        
                    # Save to database cache
                    conn2 = get_db_connection()
                    cur2 = conn2.cursor()
                    cur2.execute("UPDATE movies SET trailer_url = %s WHERE movieid = %s", (trailer_url, movie_id))
                    conn2.commit()
                    cur2.close()
                    conn2.close()
                    
                    # Update local dictionary
                    movie["trailer_url"] = trailer_url
                except Exception as ex:
                    print(f"[Warning] Failed to fetch or cache trailer URL: {ex}")
                    # Fallback to YouTube search link in case of error
                    title = movie.get("title", "")
                    year = ""
                    r_date = movie.get("release_date", "")
                    if r_date:
                        year = r_date.split("-")[0]
                    search_q = f"{title} {year} official trailer".replace(" ", "+")
                    movie["trailer_url"] = f"https://www.youtube.com/results?search_query={search_q}"
                    
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
