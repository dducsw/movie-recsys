import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Play, 
  Heart, 
  Star, 
  Calendar, 
  Film, 
  ArrowLeft,
  X,
  User,
  Clapperboard
} from 'lucide-react';
import MovieRow from '../components/MovieRow/MovieRow';
import { API_BASE_URL } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import './MovieDetailPage.css';

export default function MovieDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { isLiked, toggleLike, getRating, setRating } = useWatchlist();

  const [movie, setMovie] = useState(null);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isPlayingTrailer, setIsPlayingTrailer] = useState(false);
  const [hoverRating, setHoverRating] = useState(0);

  const movieId = Number(id);
  const liked = isLiked(movieId);
  const currentRating = getRating(movieId);

  useEffect(() => {
    setLoading(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // 1. Fetch Movie Detail
    fetch(`${API_BASE_URL}/movies/${movieId}`, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : null))
      .then(data => {
        if (data && (data.movieId || data.id || data.title)) {
          setMovie(data.movie || data);
        } else {
          setMovie(null);
        }
      })
      .catch(() => setMovie(null))
      .finally(() => setLoading(false));

    // 2. Fetch 3-Stage Similar Movies
    fetch(`${API_BASE_URL}/movies/${movieId}/recommendations?limit=18`, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : { results: [] }))
      .then(data => {
        setSimilarMovies(data.results || []);
      })
      .catch(() => setSimilarMovies([]));
  }, [movieId]);

  if (loading) {
    return (
      <div className="spinner-container">
        <div className="spinner" />
        <p className="explore-sub-count">Loading movie details...</p>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="explore-empty-state">
        <Film className="explore-empty-icon" />
        <h2 className="explore-empty-title">Movie Not Found</h2>
        <p className="explore-empty-desc">The requested movie could not be found in our catalog.</p>
        <button
          className="btn-play-trailer"
          style={{ marginTop: '16px' }}
          onClick={() => navigate('/')}
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Home</span>
        </button>
      </div>
    );
  }

  const year = movie.release_date ? movie.release_date.split('-')[0] : '2025';
  const voteScore = movie.vote_average ? (Number(movie.vote_average) / 2).toFixed(1) : '4.0';
  const genres = movie.genres ? movie.genres.split('|') : [];
  const castList = movie.cast ? movie.cast.split('|').slice(0, 8) : [];
  const backdrop = movie.poster_url && !movie.poster_url.includes('placeholder')
    ? movie.poster_url
    : 'https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1200';

  const getEmbedUrl = () => {
    if (movie.trailer_url) {
      if (movie.trailer_url.includes('/embed/')) {
        const parts = movie.trailer_url.split('/embed/');
        const videoId = parts[1]?.split('?')[0];
        if (videoId && !videoId.startsWith('?')) {
          return `https://www.youtube-nocookie.com/embed/${videoId}`;
        }
      } else if (movie.trailer_url.includes('watch?v=')) {
        const videoId = movie.trailer_url.split('watch?v=')[1]?.split('&')[0];
        if (videoId) return `https://www.youtube-nocookie.com/embed/${videoId}`;
      } else if (movie.trailer_url.includes('youtu.be/')) {
        const videoId = movie.trailer_url.split('youtu.be/')[1]?.split('?')[0];
        if (videoId) return `https://www.youtube-nocookie.com/embed/${videoId}`;
      }
    }
    const cleanTitle = movie.title.replace(/\s*\(\d{4}\)/, '').trim();
    return `https://www.youtube-nocookie.com/embed?listType=search&list=${encodeURIComponent(cleanTitle + ' official trailer')}`;
  };

  return (
    <div className="movie-detail-page">
      {/* Back Button */}
      <button className="btn-back-nav" onClick={() => navigate(-1)}>
        <ArrowLeft className="w-4 h-4" />
        <span>Back to browse</span>
      </button>

      {/* Hero Movie Presentation Header */}
      <div className="detail-hero-card">
        <div 
          className="detail-backdrop-blur"
          style={{ backgroundImage: `url(${backdrop})` }}
        />
        <div className="detail-backdrop-gradient" />

        <div className="detail-content-layout">
          {/* Movie Poster */}
          <div className="detail-poster-wrapper">
            <img
              src={backdrop}
              alt={movie.title}
              className="detail-poster-img"
            />
            <button
              className="detail-poster-overlay"
              onClick={() => setIsPlayingTrailer(true)}
            >
              <div className="streamix-play-badge">
                <Play className="w-6 h-6 fill-current ml-0.5" />
              </div>
            </button>
          </div>

          {/* Details Column */}
          <div className="detail-info-pane">
            {/* Genre Chips & Metadata */}
            <div className="detail-tags-row">
              {genres.map(g => (
                <span key={g} className="detail-genre-chip">
                  {g}
                </span>
              ))}
              <span className="detail-rating-chip">
                <Star className="w-3.5 h-3.5 fill-amber-400" />
                <span>{voteScore} Rating</span>
              </span>
              <span className="detail-genre-chip">
                <Calendar className="w-3.5 h-3.5 text-neutral-400" />
                <span>{year}</span>
              </span>
            </div>

            {/* Main Title */}
            <h1 className="detail-title">{movie.title}</h1>

            {/* Overview */}
            <p className="detail-overview">
              {movie.overview || "No synopsis available for this movie."}
            </p>

            {/* Director & Cast Tags Section */}
            <div className="detail-meta-group">
              {movie.director && (
                <div className="detail-director-box">
                  <span className="detail-meta-label">Director:</span>
                  <div className="detail-director-pill">
                    <Clapperboard className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{movie.director}</span>
                  </div>
                </div>
              )}

              {castList.length > 0 && (
                <div className="detail-cast-section">
                  <span className="detail-meta-label">Cast & Starring</span>
                  <div className="detail-cast-tags">
                    {castList.map((actor, idx) => (
                      <span key={idx} className="detail-cast-pill">
                        <User className="w-3 h-3 text-neutral-400" />
                        <span>{actor.trim()}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Action Bar */}
            <div className="detail-actions-row">
              <button
                className="btn-play-trailer"
                onClick={() => setIsPlayingTrailer(true)}
              >
                <Play className="w-4 h-4 fill-current" />
                <span>Play Trailer</span>
              </button>

              <button
                className={`btn-detail-fav ${liked ? 'liked' : ''}`}
                onClick={() => toggleLike(movieId, movie)}
                title={liked ? "Remove from Favorites" : "Add to Favorites"}
              >
                <Heart 
                  className="w-4 h-4" 
                  fill={liked ? "#ef4444" : "none"}
                  color={liked ? "#ef4444" : "currentColor"}
                />
                <span>{liked ? 'Favorited' : 'Add to Favorites'}</span>
              </button>

              {/* Star Rating Controls */}
              <div className="detail-rating-box">
                <span className="detail-meta-label">Your Rating:</span>
                <div className="detail-rating-stars">
                  {[1, 2, 3, 4, 5].map((star) => {
                    const active = (hoverRating || currentRating) >= star;
                    return (
                      <button
                        key={star}
                        type="button"
                        className={`star-btn ${active ? 'active' : ''}`}
                        onMouseEnter={() => setHoverRating(star)}
                        onMouseLeave={() => setHoverRating(0)}
                        onClick={() => setRating(movieId, star)}
                        title={`Rate ${star} stars`}
                      >
                        <Star 
                          className="w-4 h-4" 
                          fill={active ? "#fbbf24" : "none"}
                          color={active ? "#fbbf24" : "var(--text-muted)"}
                        />
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trailer Modal Player */}
      {isPlayingTrailer && (
        <div className="trailer-modal-overlay" onClick={() => setIsPlayingTrailer(false)}>
          <div className="trailer-modal-card" onClick={e => e.stopPropagation()}>
            <div className="trailer-modal-header">
              <h3 className="trailer-modal-title">{movie.title} — Official Trailer</h3>
              <button 
                className="btn-close-modal"
                onClick={() => setIsPlayingTrailer(false)}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="trailer-iframe-box">
              <iframe
                className="trailer-iframe"
                src={`${getEmbedUrl()}?autoplay=1`}
                title={`${movie.title} Trailer`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          </div>
        </div>
      )}

      {/* Similar Movies Recommendation Row */}
      {similarMovies.length > 0 && (
        <MovieRow
          title="More Like This"
          badge="Top Recommendations"
          movies={similarMovies}
          source="similar_movies"
        />
      )}
    </div>
  );
}
