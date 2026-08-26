import React, { useRef } from 'react';
import MovieCard from '../MovieCard/MovieCard';
import './MovieRow.css';

function MovieRow({ 
  title, 
  movies = [], 
  onMovieClick, 
  onSeeAll, 
  badge = null,
  likedMovies = [],
  movieRatings = {},
  onToggleLike,
  onRateMovie
}) {
  const rowRef = useRef(null);

  const scroll = (direction) => {
    if (!rowRef.current) return;
    const { scrollLeft, clientWidth } = rowRef.current;
    const scrollAmount = clientWidth * 0.75;
    rowRef.current.scrollTo({
      left: direction === 'left' ? scrollLeft - scrollAmount : scrollLeft + scrollAmount,
      behavior: 'smooth',
    });
  };

  if (!movies || movies.length === 0) return null;

  return (
    <section className="streamix-row-section">
      <div className="streamix-row-header">
        <div className="streamix-title-group">
          <h2 className="streamix-row-title">{title}</h2>
          {badge && <span className="streamix-row-badge">{badge}</span>}
        </div>
        
        <div className="streamix-row-controls">
          {onSeeAll && (
            <button className="streamix-see-all-btn" onClick={onSeeAll}>
              See All
            </button>
          )}
          <div className="streamix-arrow-group">
            <button className="streamix-arrow-btn" onClick={() => scroll('left')} title="Scroll left">‹</button>
            <button className="streamix-arrow-btn" onClick={() => scroll('right')} title="Scroll right">›</button>
          </div>
        </div>
      </div>

      <div className="streamix-cards-track" ref={rowRef}>
        {movies.map((movie) => {
          const mid = movie.movieId || movie.id;
          return (
            <div key={mid} className="streamix-card-wrapper">
              <MovieCard
                movie={movie}
                onClick={() => onMovieClick && onMovieClick(mid)}
                isLiked={likedMovies.includes(mid)}
                onToggleLike={onToggleLike}
                userRating={movieRatings[mid] || 0}
                onRateMovie={onRateMovie}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default MovieRow;