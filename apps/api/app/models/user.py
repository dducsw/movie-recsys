import logging
from typing import List, Dict, Any, Optional
from app.config.db import get_db_connection

logger = logging.getLogger(__name__)


def init_db_tables():
    """Tự động tạo các bảng liên quan đến User nếu chưa tồn tại trong PostgreSQL."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_ratings (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                movie_id INTEGER NOT NULL,
                rating FLOAT NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, movie_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_watchlist (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                movie_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, movie_id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                favorite_genres TEXT,
                favorite_movie_ids TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_comments (
                id SERIAL PRIMARY KEY,
                movie_id INTEGER NOT NULL,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                parent_id INTEGER REFERENCES movie_comments(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_movie_comments_movie_id ON movie_comments(movie_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_movie_comments_parent_id ON movie_comments(parent_id);")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS comment_likes (
                id SERIAL PRIMARY KEY,
                comment_id INTEGER REFERENCES movie_comments(id) ON DELETE CASCADE,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(comment_id, user_id)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comment_likes_comment_id ON comment_likes(comment_id);")

        conn.commit()
        cur.close()
        logger.info("Successfully initialized PostgreSQL User and Comment tables.")
    except Exception as e:
        logger.error(f"Failed to initialize user DB tables: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


class UserModel:
    @staticmethod
    def create_user(username: str, email: str, password_hash: str) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                INSERT INTO users (username, email, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id, username, email, created_at;
            """
            cur.execute(query, (username, email, password_hash))
            user = cur.fetchone()
            conn.commit()
            return dict(user)
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_email(email: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = "SELECT * FROM users WHERE LOWER(email) = LOWER(%s);"
            cur.execute(query, (email.strip(),))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_username(username: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = "SELECT * FROM users WHERE LOWER(username) = LOWER(%s);"
            cur.execute(query, (username.strip(),))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_identifier(identifier: str) -> Optional[Dict[str, Any]]:
        """Find user by either email or username."""
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            ident = identifier.strip().lower()
            query = "SELECT * FROM users WHERE LOWER(email) = %s OR LOWER(username) = %s LIMIT 1;"
            cur.execute(query, (ident, ident))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_by_id(user_id: int) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = "SELECT id, username, email, created_at FROM users WHERE id = %s;"
            cur.execute(query, (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def save_rating(user_id: int, movie_id: int, rating: float) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = """
                INSERT INTO user_ratings (user_id, movie_id, rating, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, movie_id) 
                DO UPDATE SET rating = EXCLUDED.rating, updated_at = CURRENT_TIMESTAMP
                RETURNING user_id, movie_id, rating, updated_at;
            """
            cur.execute(query, (user_id, movie_id, rating))
            result = cur.fetchone()
            conn.commit()
            return dict(result)
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_user_ratings(user_id: int) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = "SELECT movie_id, rating, updated_at FROM user_ratings WHERE user_id = %s ORDER BY updated_at DESC;"
            cur.execute(query, (user_id,))
            return [dict(r) for r in cur.fetchall()]
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def toggle_watchlist(user_id: int, movie_id: int) -> bool:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM user_watchlist WHERE user_id = %s AND movie_id = %s;", (user_id, movie_id))
            row = cur.fetchone()
            if row:
                cur.execute("DELETE FROM user_watchlist WHERE user_id = %s AND movie_id = %s;", (user_id, movie_id))
                is_in = False
            else:
                cur.execute("INSERT INTO user_watchlist (user_id, movie_id) VALUES (%s, %s);", (user_id, movie_id))
                is_in = True
            conn.commit()
            return is_in
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_user_watchlist(user_id: int) -> List[int]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = "SELECT movie_id FROM user_watchlist WHERE user_id = %s ORDER BY created_at DESC;"
            cur.execute(query, (user_id,))
            return [r["movie_id"] for r in cur.fetchall()]
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def save_preferences(user_id: int, favorite_genres: List[str], favorite_movie_ids: List[int]) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            genres_str = "|".join(favorite_genres)
            movies_str = ",".join(str(m) for m in favorite_movie_ids)
            query = """
                INSERT INTO user_preferences (user_id, favorite_genres, favorite_movie_ids, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id)
                DO UPDATE SET favorite_genres = EXCLUDED.favorite_genres,
                              favorite_movie_ids = EXCLUDED.favorite_movie_ids,
                              updated_at = CURRENT_TIMESTAMP
                RETURNING user_id, favorite_genres, favorite_movie_ids;
            """
            cur.execute(query, (user_id, genres_str, movies_str))
            res = cur.fetchone()
            conn.commit()
            return dict(res)
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_user_preferences(user_id: int) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            query = "SELECT favorite_genres, favorite_movie_ids FROM user_preferences WHERE user_id = %s;"
            cur.execute(query, (user_id,))
            row = cur.fetchone()
            if not row:
                return {"favorite_genres": [], "favorite_movie_ids": []}
            
            genres = row["favorite_genres"].split("|") if row.get("favorite_genres") else []
            movie_ids = [int(x) for x in row["favorite_movie_ids"].split(",") if x.strip()] if row.get("favorite_movie_ids") else []
            return {"favorite_genres": genres, "favorite_movie_ids": movie_ids}
        finally:
            cur.close()
            conn.close()
