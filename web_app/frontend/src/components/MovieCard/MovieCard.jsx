import React, { useRef, useEffect, useState } from 'react';
import { trackImpressions } from '../../api/client';
import './MovieCard.css';

function MovieCard({ 
  movie, 
  onClick, 
  isLiked = false,
  onToggleLike,
  userRating = 0,
  onRateMovie
}) {
  const cardRef = useRef(null);
  const [hoverRating, setHoverRating] = useState(0);

  useEffect(() => {
    if (!cardRef.current || !movie || !movie.movieId) return;
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            trackImpressions([{ movieId: movie.movieId, source: movie.source || 'listing', position: movie.position ?? null }]);
            observer.disconnect();
          }
        });
      },
      { threshold: 0.5 }
    );
    observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, [movie]);

  const year = movie?.release_date ? movie.release_date.split('-')[0] : '2025';
  const voteScore = movie?.vote_average ? Number(movie.vote_average).toFixed(1) : '8.5';
  const posterSrc = movie?.poster_url && !movie.poster_url.includes('placeholder.com')
    ? movie.poster_url
    : 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300';

  const handleLikeClick = (e) => {
    e.stopPropagation();
    if (onToggleLike) {
      onToggleLike(movie.movieId || movie.id);
    }
  };

  const handleStarClick = (e, star) => {
    e.stopPropagation();
    if (onRateMovie) {
      onRateMovie(movie.movieId || movie.id, star);
    }
  };

  return (
    <div ref={cardRef} className="streamix-movie-card" onClick={onClick}>
      {/* Poster Container */}
      <div className="streamix-poster-box">
        <img
          className="streamix-poster-img"
          src={posterSrc}
          alt={movie.title}
          loading="lazy"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300';
          }}
        />

        {/* Minimalist Transparent Heart Button with Colored Border */}
        <button 
          className={`streamix-heart-btn ${isLiked ? 'liked' : ''}`}
          onClick={handleLikeClick}
          title={isLiked ? "Remove from Favorites" : "Add to Favorites"}
        >
          <svg 
            viewBox="0 0 24 24" 
            fill={isLiked ? "#ef4444" : "none"} 
            stroke={isLiked ? "#ef4444" : "currentColor"} 
            strokeWidth="2.2" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            className="card-heart-svg"
          >
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
          </svg>
        </button>

        {/* Hover Play Overlay with Optically Centered Play Icon */}
        <div className="streamix-card-hover">
          <div className="streamix-play-badge">
            <svg viewBox="0 0 24 24" fill="currentColor" className="card-play-svg">
              <polygon points="6 4 20 12 6 20 6 4" />
            </svg>
          </div>
        </div>
      </div>

      {/* Metadata (Clean 2-line layout: Title / Year | ★ Rating) */}
      <div className="streamix-card-info">
        <h4 className="streamix-card-title" title={movie.title}>{movie.title}</h4>
        
        <div className="streamix-card-subline">
          <span className="card-year-text">{year}</span>
          <span className="card-sep">|</span>
          <span className="card-star-score">★ {voteScore}</span>
        </div>

        {/* Interactive 5-Star Rating Row */}
        <div className="streamix-rating-stars" onClick={(e) => e.stopPropagation()}>
          {[1, 2, 3, 4, 5].map((star) => (
            <span
              key={star}
              className={`mini-star ${(hoverRating || userRating) >= star ? 'active' : ''}`}
              onMouseEnter={() => setHoverRating(star)}
              onMouseLeave={() => setHoverRating(0)}
              onClick={(e) => handleStarClick(e, star)}
              title={`Rate ${star} stars`}
            >
              ★
            </span>
          ))}
          {userRating > 0 && <span className="mini-rated-tag">{userRating}★</span>}
        </div>
      </div>
    </div>
  );
}

export default MovieCard;