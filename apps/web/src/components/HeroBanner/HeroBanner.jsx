import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Plus, Check, Star } from 'lucide-react';
import { useWatchlist } from '../../context/WatchlistContext';
import { formatRating, releaseYear } from '../../utils/format';
import './HeroBanner.css';

const FADE_COPY_MS = 8000;

export default function HeroBanner({
  movie,
  currentIndex = 0,
  totalSlides = 5,
  onSelectSlide
}) {
  const navigate = useNavigate();
  const { isLiked, toggleLike } = useWatchlist();

  // Auto slide; reset chu kỳ khi user chuyển slide.
  useEffect(() => {
    if (!onSelectSlide) return;
    const interval = setInterval(() => {
      onSelectSlide((currentIndex + 1) % totalSlides);
    }, FADE_COPY_MS);
    return () => clearInterval(interval);
  }, [onSelectSlide, currentIndex, totalSlides]);

  if (!movie) return null;

  const movieId = movie.movieId || movie.id;
  const liked = isLiked(movieId);

  const genres = movie.genres ? movie.genres.split('|').slice(0, 3) : [];
  const rating = formatRating(movie.vote_average);
  const year = releaseYear(movie.release_date);
  const backdrop = movie.poster_url || '';

  return (
    <div className="billboard-wrapper">
      <div
        className="billboard-media-bg"
        style={{ backgroundImage: `url(${backdrop})` }}
      />

      <div className="billboard-gradient-left" />
      <div className="billboard-gradient-bottom" />

      <div className="billboard-content-container">
        <div className="billboard-info-block">
          <h1 className="billboard-title">{movie.title}</h1>

          <div className="billboard-meta-row">
            {rating && (
              <span className="billboard-rating-pill">
                <Star />
                <span>{rating}</span>
              </span>
            )}
            {year && <span>{year}</span>}
            {genres.map((g) => (
              <span key={g} className="billboard-genre-tag">{g}</span>
            ))}
          </div>

          <p className="billboard-overview">{movie.overview}</p>

          <div className="billboard-actions-row">
            <button
              className="btn-billboard-play"
              onClick={() => navigate(`/movie/${movieId}`)}
            >
              <Play className="billboard-play-icon" />
              <span>View Details</span>
            </button>

            <button
              className={`btn-billboard-glass ${liked ? 'active' : ''}`}
              onClick={() => toggleLike(movieId, movie)}
              title={liked ? 'In Your Watchlist' : 'Add to Watchlist'}
            >
              {liked ? <Check className="billboard-glass-icon" /> : <Plus className="billboard-glass-icon" />}
              <span>{liked ? 'In Watchlist' : 'Add to Watchlist'}</span>
            </button>
          </div>
        </div>

        <div className="billboard-carousel-dock">
          <div className="billboard-slide-dots">
            {Array.from({ length: totalSlides }).map((_, idx) => (
              <button
                key={idx}
                type="button"
                className={`slide-dot ${idx === currentIndex ? 'active' : ''}`}
                onClick={(e) => { e.stopPropagation(); onSelectSlide && onSelectSlide(idx); }}
                aria-label={`Slide ${idx + 1}`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
