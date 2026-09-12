import os
import uvicorn
from dotenv import load_dotenv, find_dotenv

from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv(find_dotenv())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers and DB initializer
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tự động kiểm tra và khởi tạo các bảng DB người dùng khi khởi chạy API."""
    init_db_tables()
    yield

app = FastAPI(
    title="MovieNex Recommendation System API (MVC)",
    description="Backend API for MovieNex Recsys Demo structured in MVC architecture",
    version="1.2.0",
    lifespan=lifespan
)

# Enable CORS with explicit origins
raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
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
    print(f"[Warning] Instrumentator setup encounter: {e}")

from fastapi import Response
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
    return {"message": "Welcome to the Movie Recommender System API! Go to /docs for API documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
