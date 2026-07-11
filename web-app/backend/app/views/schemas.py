from pydantic import BaseModel
from typing import List, Optional

class MovieResponse(BaseModel):
    movieId: int
    title: str
    release_date: Optional[str] = ""
    genres: Optional[str] = ""
    popularity: Optional[float] = 0.0
    adult: Optional[bool] = False
    overview: Optional[str] = ""
    vote_average: Optional[float] = 0.0
    vote_count: Optional[int] = 0
    poster_url: Optional[str] = ""

    class Config:
        from_attributes = True

class PaginatedMovieResponse(BaseModel):
    page: int
    limit: int
    total_movies: int
    results: List[MovieResponse]

class SearchMovieResponse(BaseModel):
    query: str
    count: int
    results: List[MovieResponse]

class RecommendationResponse(BaseModel):
    results: List[MovieResponse]
