import React, { useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Heart, Play, Star } from 'lucide-react';
import { trackImpressions, trackClick } from '../../api/client';
import { useWatchlist } from '../../context/WatchlistContext';
import './MovieCard.css';

function MovieCard({ 
  movie, 
  onClick,
  source = 'listing',
  position = null
}) {
  const cardRef = useRef(null);
  const navigate = useNavigate();
  const { isLiked, toggleLike } = useWatchlist();

  const movieId = movie?.movieId || movie?.id;
  const liked = isLiked(movieId);

  useEffect(() => {
    if (!cardRef.current || !movieId) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            trackImpressions([{ movieId, source: movie?.source || source, position }]);
            observer.disconnect();
          }
        });
      },
      { threshold: 0.5 }
    );
    observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, [movieId, source, position, movie?.source]);

  const year = movie?.release_date ? movie.release_date.split('-')[0] : '2025';
  const voteScore = movie?.vote_average ? (Number(movie.vote_average) / 2).toFixed(1) : '4.3';
  const posterSrc = movie?.poster_url && !movie.poster_url.includes('placeholder.com')
    ? movie.poster_url
    : 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300';

  const handleCardClick = () => {
    trackClick(movieId, movie?.source || source, position);
    if (onClick) {
      onClick(movie);
    } else {
      navigate(`/movie/${movieId}`);
    }
  };

  const handleLikeClick = (e) => {
    e.stopPropagation();
    toggleLike(movieId, movie);
  };

  return (
    <div ref={cardRef} className="streamix-movie-card" onClick={handleCardClick}>
      {/* Poster Box */}
      <div className="streamix-poster-box">
        <img
          className="streamix-poster-img"
          src={posterSrc}
          alt={movie.title}
          loading="lazy"
          decoding="async"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300';
          }}
        />

        {/* Favorite Heart Button */}
        <button 
          className={`streamix-heart-btn ${liked ? 'liked' : ''}`}
          onClick={handleLikeClick}
          title={liked ? "Remove from Favorites" : "Add to Favorites"}
        >
          <Heart 
            className="w-4 h-4" 
            fill={liked ? "#ef4444" : "none"}
            color={liked ? "#ef4444" : "#ffffff"}
          />
        </button>

        {/* Hover Play Badge */}
        <div className="streamix-card-hover">
          <div className="streamix-play-badge">
            <Play className="w-5 h-5 fill-current ml-0.5" />
          </div>
        </div>
      </div>

      {/* Metadata (Title & Year / Rating) */}
      <div className="streamix-card-info">
        <h4 className="streamix-card-title" title={movie.title}>
          {movie.title}
        </h4>
        
        <div className="streamix-card-subline">
          <span>{year}</span>
          <span>•</span>
          <div className="card-rating-badge">
            <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
            <span>{voteScore}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MovieCard;
