from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class MovieResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    director: Optional[str] = ""
    cast: Optional[str] = ""
    keywords: Optional[str] = ""
    trailer_url: Optional[str] = ""

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
