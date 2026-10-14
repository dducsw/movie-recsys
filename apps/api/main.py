import os
import logging
from dotenv import load_dotenv

# Ensure .env is loaded from current, parent, or project root directory
load_dotenv(".env")
load_dotenv("../../.env")
load_dotenv("../.env")

import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config.settings import get_settings
from app.config.security import limiter, SecurityHeadersMiddleware
from app import (
    movie_router,
    recsys_router,
    chatbot_router,
    events_router,
    auth_router,
    onboarding_router,
    comment_router,
    init_db_tables
)

logger = logging.getLogger("movienex.api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Automatically check and initialize PostgreSQL DB tables on startup."""
    init_db_tables()
    yield


# Configure FastAPI app with production hardening toggles for API docs
app = FastAPI(
    title="MovieNex Recommendation System API (MVC)",
    description="Backend API for MovieNex Recsys Demo structured in MVC architecture",
    version="1.2.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Attach rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Attach custom security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Enable CORS with explicit origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-Id", "X-Requested-With", "Accept"],
    max_age=600,
)


# Global Exception Handler to mask internal stack traces and server internals
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Pass through standard HTTPException so appropriate status codes and details are preserved
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers
        )
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()}
        )
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include application routers
app.include_router(movie_router)
app.include_router(recsys_router)
app.include_router(chatbot_router)
app.include_router(events_router)
app.include_router(auth_router)
app.include_router(onboarding_router)
app.include_router(comment_router)


# Prometheus Telemetry & Metrics Instrumentation
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/docs", "/openapi.json"]
    ).instrument(app)
except Exception as e:
    logger.warning(f"Instrumentator setup encounter: {e}")

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus telemetry & metrics exposition endpoint."""
    if generate_latest is not None:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return Response(content="# Prometheus telemetry disabled\n", media_type="text/plain")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Movie Recommender System API!",
        "docs": "API documentation is available at /docs in development." if not settings.is_production else "Documentation disabled in production."
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
