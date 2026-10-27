import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bookmark, Film, ArrowRight } from 'lucide-react';
import MovieCard from '../components/MovieCard/MovieCard';
import { apiFetch } from '../api/client';
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
    // Fetch movies in liked list via single batch request
    apiFetch('/movies/batch', {
      method: 'POST',
      body: JSON.stringify({ movie_ids: likedMovies })
    })
      .then(results => {
        setMovieDetails(Array.isArray(results) ? results : []);
        setLoading(false);
      })
      .catch(() => {
        setMovieDetails([]);
        setLoading(false);
      });
  }, [likedMovies]);

  return (
    <div className="watchlist-page-container">
      {/* Header */}
      <div className="watchlist-page-header">
        <div>
          <h1 className="watchlist-header-title">
            <Bookmark className="watchlist-title-icon" />
            <span>My Watchlist</span>
          </h1>
          <p className="explore-sub-count">
            {likedMovies.length} {likedMovies.length === 1 ? 'movie' : 'movies'} saved
          </p>
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div className="spinner-container">
          <div className="spinner" />
          <p className="explore-sub-count">Loading your watchlist…</p>
        </div>
      ) : movieDetails.length === 0 ? (
        <div className="explore-empty-state">
          <Film className="explore-empty-icon" />
          <h3 className="explore-empty-title">Your watchlist is empty</h3>
          <p className="explore-empty-desc">
            Browse the catalog and tap the heart icon on any movie to save it. Your likes also improve your recommendations.
          </p>
          <button className="btn-play-trailer" onClick={() => navigate('/explore')}>
            <span>Browse Movies</span>
            <ArrowRight className="watchlist-cta-icon" />
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
