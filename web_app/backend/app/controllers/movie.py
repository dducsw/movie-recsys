from fastapi import APIRouter, Query, HTTPException
from app.models.movie import MovieModel
from app.views.schemas import PaginatedMovieResponse, SearchMovieResponse, MovieResponse

router = APIRouter(prefix="/api/movies", tags=["Movies"])

@router.get("/trending", response_model=PaginatedMovieResponse)
def get_trending_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get the trending movies paginated list, sorted by popularity.
    """
    movies = MovieModel.get_trending(page, limit)
    total_count = MovieModel.get_total_count()
    return {
        "page": page,
        "limit": limit,
        "total_movies": total_count,
        "results": movies
    }

@router.get("/latest", response_model=PaginatedMovieResponse)
def get_latest_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get the latest movies paginated list, sorted by release_date.
    """
    movies = MovieModel.get_latest(page, limit)
    total_count = MovieModel.get_total_count()
    return {
        "page": page,
        "limit": limit,
        "total_movies": total_count,
        "results": movies
    }

@router.get("/search", response_model=SearchMovieResponse)
def search_movies(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=50)
):
    """
    Search movies by title (case-insensitive).
    """
    results = MovieModel.search(query, limit)
    return {
        "query": query,
        "count": len(results),
        "results": results
    }

@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie_detail(movie_id: int):
    """
    Get the details of a specific movie by movieId.
    """
    movie = MovieModel.get_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found.")
    return movie
