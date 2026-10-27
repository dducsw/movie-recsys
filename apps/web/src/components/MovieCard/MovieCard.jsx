import { memo, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Heart, Play, Star } from 'lucide-react';
import { queueImpression, trackClick, trackDwell } from '../../api/client';
import { formatRating, posterUrl, releaseYear } from '../../utils/format';
import { useWatchlist } from '../../context/WatchlistContext';
import './MovieCard.css';

function MovieCard({
  movie,
  onClick,
  source = 'listing',
  position = null
}) {
  const cardRef = useRef(null);
  const hoverStartRef = useRef(null);
  const navigate = useNavigate();
  const { isLiked, toggleLike } = useWatchlist();

  const movieId = movie?.movieId || movie?.id;
  const liked = isLiked(movieId);

  // Impression: 1 lần khi card hiển thị >50%.
  useEffect(() => {
    const el = cardRef.current;
    if (!el || !movieId) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          queueImpression({ movieId, source, position });
          observer.disconnect();
        }
      },
      { threshold: 0.5 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [movieId, source, position]);

  const year = releaseYear(movie?.release_date);
  const rating = formatRating(movie?.vote_average);

  const handleMouseEnter = () => {
    hoverStartRef.current = Date.now();
  };

  const handleMouseLeave = () => {
    if (!hoverStartRef.current) return;
    const elapsed = Date.now() - hoverStartRef.current;
    hoverStartRef.current = null;
    if (elapsed >= 1000) {
      trackDwell(movieId, elapsed, source, position);
    }
  };

  const handleCardClick = () => {
    trackClick(movieId, source, position);
    if (onClick) {
      onClick(movie);
    } else {
      navigate(`/movie/${movieId}`);
    }
  };

  return (
    <div
      ref={cardRef}
      className="streamix-movie-card"
      onClick={handleCardClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') handleCardClick(); }}
    >
      <div className="streamix-poster-box">
        <img
          className="streamix-poster-img"
          src={posterUrl(movie?.poster_url)}
          alt={movie.title}
          loading="lazy"
          decoding="async"
          onError={(e) => {
            e.currentTarget.onerror = null; // tránh loop nếu placeholder lỗi
            e.currentTarget.src = posterUrl('');
          }}
        />

        <button
          className={`streamix-heart-btn ${liked ? 'liked' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            toggleLike(movieId, movie);
          }}
          title={liked ? 'Remove from Favourites' : 'Add to Favourites'}
          aria-label={liked ? 'Remove from Favourites' : 'Add to Favourites'}
        >
          <Heart />
        </button>

        <div className="streamix-card-hover">
          <div className="streamix-play-badge">
            <Play />
          </div>
        </div>
      </div>

      <div className="streamix-card-info">
        <h4 className="streamix-card-title" title={movie.title}>
          {movie.title}
        </h4>

        <div className="streamix-card-subline">
          <span>{year || '—'}</span>
          {rating && (
            <>
              <span>•</span>
              <div className="card-rating-badge">
                <Star />
                <span>{rating}</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(MovieCard);
