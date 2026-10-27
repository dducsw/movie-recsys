import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized, strongly typed application configuration loaded strictly
    from environment variables and .env file.
    """
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")

    # JWT Authentication
    JWT_SECRET_KEY: str = Field(
        default="movienex_super_secret_jwt_key_development_only_replace_in_prod_2026",
        description="Cryptographic secret key for signing JWT tokens"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, description="Access token lifespan in minutes")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token lifespan in days")

    # Database Configuration (PostgreSQL)
    DB_HOST: str = Field(default="127.0.0.1", alias="POSTGRES_HOST")
    DB_PORT: int = Field(default=5435, alias="POSTGRES_PORT")
    DB_NAME: str = Field(default="movie_db", alias="POSTGRES_DB")
    DB_USER: str = Field(default="postgres", alias="POSTGRES_USER")
    DB_PASSWORD: str = Field(default="", alias="POSTGRES_PASSWORD")
    DB_POOL_SIZE: int = Field(default=10, description="SQLAlchemy connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=10, description="SQLAlchemy connection pool max overflow")
    DB_POOL_TIMEOUT: int = Field(default=30, description="SQLAlchemy connection pool timeout in seconds")
    DB_POOL_RECYCLE: int = Field(default=1800, description="SQLAlchemy pool recycle in seconds")

    # In-memory Cache & Feature Store (Redis)
    REDIS_HOST: str = Field(default="127.0.0.1")
    REDIS_PORT: int = Field(default=6379)

    # Vector DB (Qdrant)
    QDRANT_HOST: str = Field(default="127.0.0.1")
    QDRANT_PORT: int = Field(default=6333)

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
        description="Comma-separated list of allowed origins"
    )

    # External APIs
    TMDB_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, alias="OPEN_ROUTER_API_KEY")
    OPENROUTER_MODEL: str = Field(default="google/gemini-2.0-flash-exp:free")
    ZHIPUAI_API_KEY: Optional[str] = Field(default=None)
    TAVILY_API_KEY: Optional[str] = Field(default=None)
    TOKENHARBOR_API_KEY: Optional[str] = Field(default=None, alias="TOKEN_HABOR_API_KEY")
    TOKENHARBOR_BASE_URL: str = Field(default="https://tokenharbor.ai/v1")
    TOKENHARBOR_MODEL: str = Field(default="deepseek-v4.1-flash:free")

    # Rate Limiting
    RATE_LIMIT_LOGIN: str = Field(default="10/minute")
    RATE_LIMIT_REGISTER: str = Field(default="5/minute")
    RATE_LIMIT_REFRESH: str = Field(default="20/minute")
    RATE_LIMIT_CHATBOT: str = Field(default="30/minute")

    # 3-Stage Recsys tuning — defaults mirror the shipped behaviour; overrides
    # are env-only so the values train/serve on stay in one auditable place.
    # Thin-pool floors: DB trending tops the candidate pool up below these.
    RECSYS_POOL_FLOOR_SIMILAR: int = Field(default=20, description="Min candidates before DB trending fallback (similar endpoint)")
    RECSYS_POOL_FLOOR_PERSONALIZED: int = Field(default=30, description="Min candidates before DB trending fallback (personalized endpoint)")
    # zrevrange stop is INCLUSIVE — default 30 reproduces the shipped 31-item window.
    RECSYS_TRENDING_ZSTOP: int = Field(default=30, description="Inclusive zrevrange stop for realtime:trending")
    # Conservative retrieval-score floors for sources without a real score.
    RECSYS_SCORE_REDIS_SIMILAR: float = Field(default=0.5, description="Score floor when a batch similarity entry has none")
    RECSYS_SCORE_PERSONAL_SIMILAR: float = Field(default=0.4, description="Score floor for personalized redis_similar entries")
    RECSYS_SCORE_TRENDING: float = Field(default=0.3, description="Fixed score of realtime trending candidates")
    RECSYS_SCORE_DB_FALLBACK: float = Field(default=0.2, description="Fixed score of DB trending fallback candidates")
    # Heuristic ranker weights (genre proximity, retrieval affinity, quality).
    RECSYS_HEUR_W_GENRE: float = Field(default=10.0, description="Heuristic weight: genre overlap")
    RECSYS_HEUR_W_RETRIEVAL: float = Field(default=5.0, description="Heuristic weight: retrieval affinity")
    RECSYS_HEUR_W_QUALITY: float = Field(default=0.5, description="Heuristic weight: vote_average")
    # Recommendation cache TTLs (Redis-backed recsys_cache).
    RECSYS_CACHE_TTL_SIMILAR: int = Field(default=300, description="Cache TTL (s) for similar-movies slates")
    RECSYS_CACHE_TTL_PERSONALIZED: int = Field(default=180, description="Cache TTL (s) for personalized slates")
    RECSYS_CACHE_TTL_USER_RECS: int = Field(default=300, description="Cache TTL (s) for direct ML user recommendations")

    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        # If in production, prevent default insecure keys
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            insecure_defaults = [
                "movienex_super_secret_jwt_key_development_only_replace_in_prod_2026",
                "movienex_default_secret_key_2026_dev_mode",
                "generate-a-strong-secret-key-for-jwt-signing"
            ]
            if not v or v in insecure_defaults or len(v) < 32:
                raise ValueError("In production, JWT_SECRET_KEY must be set to a secure string with at least 32 characters.")
        return v

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def sqlalchemy_database_url(self) -> str:
        from urllib.parse import quote_plus
        escaped_password = quote_plus(self.DB_PASSWORD) if self.DB_PASSWORD else ""
        return f"postgresql://{self.DB_USER}:{escaped_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


@lru_cache()
def get_settings() -> Settings:
    """Returns singleton instance of Settings."""
    return Settings()
