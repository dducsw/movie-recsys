import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5435")
DB_NAME = os.getenv("DB_NAME", "movie_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")

_db_pool = None


def get_db_pool() -> ThreadedConnectionPool:
    """Initialize or return the singleton ThreadedConnectionPool."""
    global _db_pool
    if _db_pool is None or _db_pool.closed:
        try:
            _db_pool = ThreadedConnectionPool(
                minconn=int(os.getenv("DB_POOL_MIN", "2")),
                maxconn=int(os.getenv("DB_POOL_MAX", "20")),
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                cursor_factory=RealDictCursor
            )
            logger.info("PostgreSQL ThreadedConnectionPool initialized (2-20 connections).")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Connection Pool: {e}")
            raise HTTPException(
                status_code=500,
                detail="Database connection failure. Make sure PostgreSQL container is running."
            )
    return _db_pool


class PooledConnectionProxy:
    """Connection proxy that returns connection to pool upon close() instead of dropping it."""
    def __init__(self, pool: ThreadedConnectionPool, conn):
        self._pool = pool
        self._conn = conn
        self._closed = False

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        if not self._closed and self._pool and not self._pool.closed:
            self._closed = True
            try:
                self._pool.putconn(self._conn)
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


def get_db_connection():
    """
    Retrieves a pooled connection from the PostgreSQL Connection Pool.
    Calling conn.close() safely returns the connection back to the pool.
    """
    pool = get_db_pool()
    try:
        raw_conn = pool.getconn()
        return PooledConnectionProxy(pool, raw_conn)
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        raise HTTPException(
            status_code=500,
            detail="Database connection pool exhausted or unreachable."
        )


@contextmanager
def get_db_cursor(commit_on_exit: bool = False):
    """
    Context manager for database operations with automatic connection release.
    Usage:
        with get_db_cursor() as cur:
            cur.execute("SELECT ...")
            results = cur.fetchall()
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        yield cur
        if commit_on_exit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
