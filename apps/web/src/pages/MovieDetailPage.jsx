import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Play, Heart, Star, Calendar, Film, ArrowLeft, X, User, Clapperboard } from 'lucide-react';
import MovieRow from '../components/MovieRow/MovieRow';
import CommentSection from '../components/CommentSection/CommentSection';
import { apiFetch } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import { formatRating, releaseYear, getEmbedUrl, posterUrl } from '../utils/format';
import './MovieDetailPage.css';

export default function MovieDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { isLiked, toggleLike, getRating, setRating } = useWatchlist();

  // undefined = đang load, null = không tìm thấy, loadError = lỗi mạng/server
  const [movie, setMovie] = useState(undefined);
  const [loadError, setLoadError] = useState(false);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [isPlayingTrailer, setIsPlayingTrailer] = useState(false);
  const [hoverRating, setHoverRating] = useState(0);

  const movieId = Number(id);
  const liked = isLiked(movieId);
  const currentRating = getRating(movieId);

  const fetchMovie = () => {
    setMovie(undefined);
    setLoadError(false);
    window.scrollTo({ top: 0 });

    // Parallel: detail + similar.
    Promise.all([
      apiFetch(`/movies/${movieId}`).catch((err) => ({ __error: err })),
      apiFetch(`/movies/${movieId}/recommendations?limit=18`).catch(() => ({ results: [] }))
    ]).then(([detail, similar]) => {
      const data = detail?.__error ? null : detail?.movie || detail;
      if (detail?.__error) setLoadError(true);
      setMovie(
        data && (data.movieId || data.id || data.title) ? data : null
      );
      setSimilarMovies(similar?.results || []);
    });
  };

  useEffect(() => {
    fetchMovie();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [movieId]);

  if (movie === undefined) {
    return (
      <div className="spinner-container">
        <div className="spinner" />
        <p className="explore-sub-count">Loading movie details...</p>
      </div>
    );
  }

  if (loadError || !movie) {
    return (
      <div className="explore-empty-state">
        <Film className="explore-empty-icon" />
        <h2 className="explore-empty-title">
          {loadError ? 'Failed to load movie' : 'Movie not found'}
        </h2>
        <p className="explore-empty-desc">
          {loadError
            ? 'A connection error occurred or the server is not ready. Please try again.'
            : 'The requested movie could not be found in our catalog.'}
        </p>
        <div className="detail-error-actions">
          {loadError && (
            <button className="btn-play-trailer" onClick={fetchMovie}>
              <span>Retry</span>
            </button>
          )}
          <button className="btn-play-trailer" onClick={() => navigate('/')}>
            <ArrowLeft className="detail-error-back-icon" />
            <span>Back to Home</span>
          </button>
        </div>
      </div>
    );
  }

  const year = releaseYear(movie.release_date);
  const rating = formatRating(movie.vote_average);
  const genres = movie.genres ? movie.genres.split('|').filter(Boolean) : [];
  const castList = movie.cast ? movie.cast.split('|').slice(0, 8) : [];
  const poster = posterUrl(movie.poster_url);

  // Embed an toàn; không parse được link thì fallback tìm kiếm theo tên phim.
  const embedUrl = getEmbedUrl(movie.trailer_url)
    || `https://www.youtube-nocookie.com/embed?listType=search&list=${encodeURIComponent(
      (movie.title || '').replace(/\s*\(\d{4}\)/, '').trim() + ' official trailer'
    )}`;

  return (
    <div className="movie-detail-page">
      <button className="btn-back-nav" onClick={() => navigate(-1)}>
        <ArrowLeft className="detail-back-icon" />
        <span>Back</span>
      </button>

      <div className="detail-hero-card">
        <div
          className="detail-backdrop-blur"
          style={{ backgroundImage: `url(${poster})` }}
        />

        <div className="detail-content-layout">
          {/* Poster */}
          <div className="detail-poster-wrapper">
            <img src={poster} alt={movie.title} className="detail-poster-img" />
            <button
              className="detail-poster-overlay"
              onClick={() => setIsPlayingTrailer(true)}
              aria-label="Play trailer"
            >
              <div className="streamix-play-badge">
                <Play className="detail-play-icon" />
              </div>
            </button>
          </div>

          {/* Info */}
          <div className="detail-info-pane">
            <div className="detail-tags-row">
              {genres.map((g) => (
                <span key={g} className="detail-genre-chip">{g}</span>
              ))}
              {rating && (
                <span className="detail-rating-chip">
                  <Star className="detail-star-icon" />
                  <span>{rating} Rating</span>
                </span>
              )}
              {year && (
                <span className="detail-genre-chip">
                  <Calendar className="detail-tag-icon" />
                  <span>{year}</span>
                </span>
              )}
            </div>

            <h1 className="detail-title">{movie.title}</h1>
            <p className="detail-overview">
              {movie.overview || 'No synopsis available for this movie.'}
            </p>

            <div className="detail-meta-group">
              {movie.director && (
                <div className="detail-director-box">
                  <span className="detail-meta-label">Director:</span>
                  <div className="detail-director-pill">
                    <Clapperboard className="detail-tag-icon" />
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
                        <User className="detail-tag-icon" />
                        <span>{actor.trim()}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="detail-actions-row">
              <button className="btn-play-trailer" onClick={() => setIsPlayingTrailer(true)}>
                <Play className="detail-play-icon" />
                <span>Play Trailer</span>
              </button>

              <button
                className={`btn-detail-fav ${liked ? 'liked' : ''}`}
                onClick={() => toggleLike(movieId, movie)}
                title={liked ? 'Remove from Favourites' : 'Add to Favourites'}
              >
                <Heart className="detail-action-icon" />
                <span>{liked ? 'Favorited' : 'Add to Favourites'}</span>
              </button>

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
                        <Star />
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trailer Modal */}
      {isPlayingTrailer && (
        <div className="trailer-modal-overlay" onClick={() => setIsPlayingTrailer(false)}>
          <div className="trailer-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="trailer-modal-header">
              <h3 className="trailer-modal-title">{movie.title} — Official Trailer</h3>
              <button
                className="btn-close-modal"
                onClick={() => setIsPlayingTrailer(false)}
                aria-label="Close"
              >
                <X />
              </button>
            </div>
            <div className="trailer-iframe-box">
              <iframe
                className="trailer-iframe"
                src={`${embedUrl}?autoplay=1`}
                title={`${movie.title} Trailer`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          </div>
        </div>
      )}

      {similarMovies.length > 0 && (
        <MovieRow
          title="More Like This"
          movies={similarMovies}
          source="similar_movies"
        />
      )}

      <CommentSection movieId={movieId} movieTitle={movie.title} />
    </div>
  );
}
