import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5435")
DB_NAME = os.getenv("DB_NAME", "movie_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")

def get_db_connection():
    """
    Creates and returns a connection to the PostgreSQL database.
    Uses RealDictCursor to return dictionary values.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"[Error] Failed to connect to PostgreSQL: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database connection failure. Make sure the PostgreSQL container is running."
        )
