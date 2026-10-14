import logging
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import HTTPException

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

DB_HOST = settings.DB_HOST
DB_PORT = str(settings.DB_PORT)
DB_NAME = settings.DB_NAME
DB_USER = settings.DB_USER
DB_PASSWORD = settings.DB_PASSWORD

# SQLAlchemy Engine configured with safe, production-grade connection pooling
# pool_pre_ping=True tests connections prior to checkout to discard stale/dropped connections
# Explicit pool_size, max_overflow, and pool_timeout prevent connection exhaustion DoS
engine = create_engine(
    settings.sqlalchemy_database_url,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides an isolated SQLAlchemy session per request
    with automatic commit/rollback and cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


_db_pool = None


def get_db_pool() -> ThreadedConnectionPool:
    """Initialize or return the singleton ThreadedConnectionPool for psycopg2."""
    global _db_pool
    if _db_pool is None or _db_pool.closed:
        try:
            _db_pool = ThreadedConnectionPool(
                minconn=max(1, settings.DB_POOL_SIZE // 2),
                maxconn=settings.DB_POOL_SIZE + settings.DB_MAX_OVERFLOW,
                host=DB_HOST,
                port=int(DB_PORT),
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                cursor_factory=RealDictCursor,
            )
            logger.info("PostgreSQL ThreadedConnectionPool initialized safely with pooled bounds.")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Connection Pool: {e}")
            raise HTTPException(
                status_code=500,
                detail="Database connection failure. Make sure PostgreSQL service is running."
            )
    return _db_pool


class PooledConnectionProxy:
    """Connection proxy that safely returns connection to pool upon close() instead of dropping it."""
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
