from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, Field
from app.models.movie import MovieModel
from app.views.schemas import PaginatedMovieResponse, SearchMovieResponse, MovieResponse
from app.services.event_producer import emit, EventType
from app.services.session import get_session_id
from app.deps import decode_cursor

router = APIRouter(prefix="/api/movies", tags=["Movies"])

@router.get("/trending", response_model=PaginatedMovieResponse)
def get_trending_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: Optional[str] = Query(default=None, description="Opaque cursor for keyset pagination"),
    session_id: str = Depends(get_session_id),
):
    """
    Trending movies. Use cursor for deep pagination, page for simple browsing.
    """
    if cursor:
        keyset = decode_cursor(cursor)
        movies, next_cursor = MovieModel.get_trending_cursor(limit, keyset)
        emit(EventType.TRENDING_BROWSE, user_id=session_id, extra={"cursor": True, "limit": limit})
        return {"page": page, "limit": limit, "total_movies": 0,
                "results": movies, "next_cursor": next_cursor}
    movies = MovieModel.get_trending(page, limit)
    total_count = MovieModel.get_total_count()
    emit(EventType.TRENDING_BROWSE, user_id=session_id, extra={"page": page, "limit": limit})
    return {
        "page": page,
        "limit": limit,
        "total_movies": total_count,
        "results": movies,
        "next_cursor": None,
    }

@router.get("/latest", response_model=PaginatedMovieResponse)
def get_latest_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    session_id: str = Depends(get_session_id),
):
    """
    Get the latest movies paginated list, sorted by release_date.
    """
    movies = MovieModel.get_latest(page, limit)
    total_count = MovieModel.get_total_count()
    emit(EventType.LATEST_BROWSE, user_id=session_id, extra={"page": page, "limit": limit})
    return {
        "page": page,
        "limit": limit,
        "total_movies": total_count,
        "results": movies
    }

@router.get("/search", response_model=SearchMovieResponse)
def search_movies(
    query: str = Query(default=""),
    genre: Optional[str] = Query(default=None),
    year: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    session_id: str = Depends(get_session_id),
):
    """
    Search movies with optional genre, release year, and status filters.
    """
    results = MovieModel.search_with_filters(
        query_str=query,
        genre=genre,
        year=year,
        status=status,
        limit=limit
    )
    emit(EventType.MOVIE_SEARCH, user_id=session_id, extra={"query": query, "genre": genre, "year": year, "status": status, "result_count": len(results)})
    return {
        "query": query or genre or year or status or "all",
        "count": len(results),
        "results": results
    }

class MovieBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movie_ids: List[int] = Field(default_factory=list, max_length=100)

@router.post("/batch", response_model=List[MovieResponse])
def get_movies_batch(body: MovieBatchRequest):
    """
    Get movie details for a batch of movie IDs in a single query (max 100).
    """
    if not body.movie_ids:
        return []
    return MovieModel.get_by_ids(body.movie_ids[:100])

@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie_detail(movie_id: int, session_id: str = Depends(get_session_id)):
    """
    Get the details of a specific movie by movieId.
    """
    movie = MovieModel.get_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found.")
    emit(
        EventType.DETAIL_VIEW,
        user_id=session_id,
        movie_id=movie_id,
        extra={"title": movie.get("title"), "genres": movie.get("genres")},
    )
    return movie
