import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search, Sparkles, ChevronLeft, ChevronRight } from 'lucide-react';
import MovieCard from '../components/MovieCard/MovieCard';
import { apiFetch } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import './ExplorePage.css';

const PAGE_SIZE = 24;

export default function ExplorePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { likedMovies } = useWatchlist();

  // TopNav ghi ?query=, link cũ dùng ?search= — nhận cả hai.
  const searchQuery = searchParams.get('search') || searchParams.get('query') || '';
  const filterType = searchParams.get('filter') || 'trending';
  const genreParam = searchParams.get('genre') || 'all';
  const yearParam = searchParams.get('year') || 'all';
  const statusParam = searchParams.get('status') || 'all';
  const pageParam = parseInt(searchParams.get('page') || '1', 10);

  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  useEffect(() => {
    setLoading(true);
    setVisibleCount(PAGE_SIZE);
    const controller = new AbortController();
    const timer = setTimeout(() => {
      let path = `/movies/trending?page=${pageParam}&limit=24`;

      if (searchQuery.trim()) {
        path = `/movies/search?query=${encodeURIComponent(searchQuery.trim())}&limit=40`;
        if (genreParam !== 'all') path += `&genre=${encodeURIComponent(genreParam)}`;
        if (yearParam !== 'all') path += `&year=${encodeURIComponent(yearParam)}`;
        if (statusParam !== 'all') path += `&status=${encodeURIComponent(statusParam)}`;
      } else if (genreParam !== 'all') {
        path = `/movies/search?genre=${encodeURIComponent(genreParam)}&limit=40`;
      } else if (filterType === 'latest') {
        path = `/movies/latest?page=${pageParam}&limit=24`;
      } else if (filterType === 'recommendations') {
        const ids = likedMovies.slice(0, 50);
        path = `/recommendations${ids.length > 0 ? `?movie_ids=${ids.join(',')}&limit=40` : '?limit=40'}`;
      }

      apiFetch(path, { signal: controller.signal })
        .then((data) => {
          setMovies(data.results || []);
        })
        .catch((err) => {
          if (err?.name === 'AbortError') return;
          setMovies([]);
        })
        .finally(() => setLoading(false));
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
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
    if (filterType === 'recommendations') return 'Recommendations For You';
    if (filterType === 'latest') return 'New & Upcoming Releases';
    return 'Trending Movies';
  };

  return (
    <div className="explore-page-container">
      <div className="explore-page-header">
        <div className="explore-title-group">
          <div className="explore-title-row">
            {filterType === 'recommendations' && <Sparkles className="explore-recommend-icon" />}
            <h1 className="explore-main-title">{getPageTitle()}</h1>
          </div>
          <p className="explore-sub-count">
            {loading ? 'Loading…' : `${movies.length} movies`}
          </p>
        </div>
      </div>

      {loading ? (
        <div className="spinner-container">
          <div className="spinner" />
        </div>
      ) : movies.length === 0 ? (
        <div className="explore-empty-state">
          <Search className="explore-empty-icon" />
          <h3 className="explore-empty-title">No movies found</h3>
          <p className="explore-empty-desc">
            Try a different keyword, genre, or browse trending movies.
          </p>
        </div>
      ) : (
        <>
          <div className="explore-movies-grid">
            {movies.slice(0, visibleCount).map((movie, idx) => (
              <MovieCard
                key={movie.movieId || movie.id || idx}
                movie={movie}
                source="explore_grid"
                position={idx}
              />
            ))}
          </div>
          {visibleCount < movies.length && (
            <div className="explore-show-more">
              <button className="explore-nav-btn" onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}>
                Show more ({movies.length - visibleCount})
              </button>
            </div>
          )}
        </>
      )}

      {movies.length >= 20 && (
        <div className="explore-pagination-wrapper">
          <div className="explore-pagination-dock">
            <button
              className="explore-nav-btn"
              onClick={() => handlePageChange(Math.max(1, pageParam - 1))}
              disabled={pageParam <= 1}
              title="Previous Page"
            >
              <ChevronLeft />
              <span>Prev</span>
            </button>

            <span className="explore-page-indicator">Page {pageParam}</span>

            <button
              className="explore-nav-btn"
              onClick={() => handlePageChange(pageParam + 1)}
              title="Next Page"
            >
              <span>Next</span>
              <ChevronRight />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
