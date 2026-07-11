from app.config.db import get_db_connection
from fastapi import HTTPException
from typing import List, Dict, Any

class MovieModel:
    @staticmethod
    def get_trending(page: int, limit: int) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        offset = (page - 1) * limit
        try:
            query = """
                SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
                FROM movies 
                ORDER BY popularity DESC 
                LIMIT %s OFFSET %s
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
                SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
                FROM movies 
                WHERE release_date != '' AND release_date IS NOT NULL
                ORDER BY release_date DESC, popularity DESC
                LIMIT %s OFFSET %s
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
                SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
                FROM movies 
                WHERE title ILIKE %s 
                ORDER BY popularity DESC 
                LIMIT %s
            """
            cur.execute(sql_query, (search_query, limit))
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
                SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
                FROM movies 
                WHERE movieid = %s
            """
            cur.execute(query, (movie_id,))
            movie = cur.fetchone()
            if not movie:
                raise HTTPException(status_code=404, detail="Movie not found.")
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
            # Prepare query to fetch movies by multiple IDs
            query = """
                SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
                FROM movies 
                WHERE movieid IN %s
            """
            # psycopg2 needs a tuple for IN clause
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
                SELECT movieid AS "movieId", title, genres, popularity, vote_average, poster_url, release_date
                FROM movies
            """
            cur.execute(query)
            return cur.fetchall()
        except Exception as e:
            print(f"[Error] Failed to fetch all movie genres: {e}")
            raise HTTPException(status_code=500, detail="Failed to query genres from movies.")
        finally:
            cur.close()
            conn.close()
