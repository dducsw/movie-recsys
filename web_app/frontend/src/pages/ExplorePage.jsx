import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search, Sparkles, ChevronLeft, ChevronRight } from 'lucide-react';
import MovieCard from '../components/MovieCard/MovieCard';
import { API_BASE_URL } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import './ExplorePage.css';

export default function ExplorePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { likedMovies } = useWatchlist();

  const searchQuery = searchParams.get('search') || '';
  const filterType = searchParams.get('filter') || 'trending';
  const genreParam = searchParams.get('genre') || 'all';
  const yearParam = searchParams.get('year') || 'all';
  const statusParam = searchParams.get('status') || 'all';
  const pageParam = parseInt(searchParams.get('page') || '1', 10);

  const [movies, setMovies] = useState([]);
  const [totalMovies, setTotalMovies] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let url = `${API_BASE_URL}/movies/trending?page=${pageParam}&limit=24`;

    if (searchQuery.trim()) {
      url = `${API_BASE_URL}/movies/search?query=${encodeURIComponent(searchQuery.trim())}&limit=40`;
      if (genreParam !== 'all') url += `&genre=${encodeURIComponent(genreParam)}`;
      if (yearParam !== 'all') url += `&year=${encodeURIComponent(yearParam)}`;
      if (statusParam !== 'all') url += `&status=${encodeURIComponent(statusParam)}`;
    } else if (genreParam !== 'all') {
      url = `${API_BASE_URL}/movies/search?genre=${encodeURIComponent(genreParam)}&limit=40`;
    } else if (filterType === 'latest') {
      url = `${API_BASE_URL}/movies/latest?page=${pageParam}&limit=24`;
    } else if (filterType === 'recommendations') {
      const q = likedMovies.length > 0 ? `?movie_ids=${likedMovies.join(',')}&limit=40` : '?limit=40';
      url = `${API_BASE_URL}/recommendations${q}`;
    }

    fetch(url, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : { results: [], total_movies: 0 }))
      .then(data => {
        setMovies(data.results || []);
        setTotalMovies(data.total_movies || data.results?.length || 0);
      })
      .catch(() => {
        setMovies([]);
      })
      .finally(() => setLoading(false));
  }, [searchQuery, filterType, genreParam, yearParam, statusParam, pageParam, likedMovies]);

  const handlePageChange = (newPage) => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set('page', newPage.toString());
    setSearchParams(nextParams);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const getPageTitle = () => {
    if (searchQuery) return `Search results for "${searchQuery}"`;
    if (genreParam !== 'all') return `${genreParam} Movies`;
    if (filterType === 'recommendations') return 'AI Personalized Recommendations';
    if (filterType === 'latest') return 'New & Upcoming Releases';
    return 'Explore Trending Movies';
  };

  return (
    <div className="explore-page-container">
      {/* Header */}
      <div className="explore-page-header">
        <div className="explore-title-group">
          <div className="explore-title-row">
            {filterType === 'recommendations' && <Sparkles className="w-5 h-5 text-yellow-400" />}
            <h1 className="explore-main-title">{getPageTitle()}</h1>
          </div>
          <p className="explore-sub-count">
            Showing {movies.length} {movies.length === 1 ? 'movie' : 'movies'}
          </p>
        </div>
      </div>

      {/* Movies Grid */}
      {loading ? (
        <div className="spinner-container">
          <div className="spinner" />
          <p className="explore-sub-count">Loading catalog...</p>
        </div>
      ) : movies.length === 0 ? (
        <div className="explore-empty-state">
          <Search className="explore-empty-icon" />
          <h3 className="explore-empty-title">No movies found</h3>
          <p className="explore-empty-desc">
            Try adjusting your search query, genre tags, or explore our trending catalog.
          </p>
        </div>
      ) : (
        <div className="explore-movies-grid">
          {movies.map((movie, idx) => (
            <MovieCard
              key={movie.movieId || movie.id || idx}
              movie={movie}
              source="explore_grid"
              position={idx}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {movies.length >= 20 && (
        <div className="explore-pagination">
          <button
            className="btn-explore-page"
            onClick={() => handlePageChange(Math.max(1, pageParam - 1))}
            disabled={pageParam <= 1}
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Previous</span>
          </button>
          
          <span className="explore-page-tag">
            Page {pageParam}
          </span>

          <button
            className="btn-explore-page"
            onClick={() => handlePageChange(pageParam + 1)}
          >
            <span>Next</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
