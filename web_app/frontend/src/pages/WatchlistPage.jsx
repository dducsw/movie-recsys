import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bookmark, Film, ArrowRight } from 'lucide-react';
import MovieCard from '../components/MovieCard/MovieCard';
import { API_BASE_URL } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import './WatchlistPage.css';
import './ExplorePage.css';

export default function WatchlistPage() {
  const navigate = useNavigate();
  const { likedMovies } = useWatchlist();

  const [movieDetails, setMovieDetails] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (likedMovies.length === 0) {
      setMovieDetails([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    // Fetch movies in liked list
    Promise.all(
      likedMovies.map(id =>
        fetch(`${API_BASE_URL}/movies/${id}`, { credentials: 'include' })
          .then(r => (r.ok ? r.json() : null))
          .then(d => (d?.movie || d))
          .catch(() => null)
      )
    ).then(results => {
      setMovieDetails(results.filter(Boolean));
      setLoading(false);
    });
  }, [likedMovies]);

  return (
    <div className="watchlist-page-container">
      {/* Header */}
      <div className="watchlist-page-header">
        <div>
          <h1 className="watchlist-header-title">
            <Bookmark className="w-6 h-6 text-red-500" fill="#ef4444" />
            <span>My Favourites</span>
          </h1>
          <p className="explore-sub-count">
            {likedMovies.length} saved {likedMovies.length === 1 ? 'movie' : 'movies'} in your personal library
          </p>
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div className="spinner-container">
          <div className="spinner" />
          <p className="explore-sub-count">Loading your favorites...</p>
        </div>
      ) : movieDetails.length === 0 ? (
        <div className="explore-empty-state">
          <Film className="explore-empty-icon" />
          <h3 className="explore-empty-title">Your Watchlist is Empty</h3>
          <p className="explore-empty-desc">
            Explore our movie catalog and tap the heart icon to save movies you love. Your selections will directly tune your personalized AI recommendation feed.
          </p>
          <button
            className="btn-play-trailer"
            style={{ marginTop: '12px' }}
            onClick={() => navigate('/explore')}
          >
            <span>Explore Movies</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      ) : (
        <div className="explore-movies-grid">
          {movieDetails.map((movie, idx) => (
            <MovieCard
              key={movie.movieId || movie.id || idx}
              movie={movie}
              source="watchlist"
              position={idx}
            />
          ))}
        </div>
      )}
    </div>
  );
}
