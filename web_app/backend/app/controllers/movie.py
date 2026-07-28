from fastapi import APIRouter, Query, HTTPException, Depends
from app.models.movie import MovieModel
from app.views.schemas import PaginatedMovieResponse, SearchMovieResponse, MovieResponse
from app.services.event_producer import emit, EventType
from app.services.session import get_session_id

router = APIRouter(prefix="/api/movies", tags=["Movies"])

@router.get("/trending", response_model=PaginatedMovieResponse)
def get_trending_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    session_id: str = Depends(get_session_id),
):
    """
    Get the trending movies paginated list, sorted by popularity.
    """
    movies = MovieModel.get_trending(page, limit)
    total_count = MovieModel.get_total_count()
    emit(EventType.TRENDING_BROWSE, user_id=session_id, extra={"page": page, "limit": limit})
    return {
        "page": page,
        "limit": limit,
        "total_movies": total_count,
        "results": movies
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
    query: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=50),
    session_id: str = Depends(get_session_id),
):
    """
    Search movies by title (case-insensitive).
    """
    results = MovieModel.search(query, limit)
    emit(EventType.MOVIE_SEARCH, user_id=session_id, extra={"query": query, "result_count": len(results)})
    return {
        "query": query,
        "count": len(results),
        "results": results
    }

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
