import React from 'react';
import MovieCard from '../MovieCard/MovieCard';
import './MovieRow.css';

function MovieRow({ 
  title, 
  movies, 
  loading, 
  scrollRef, 
  onScroll, 
  onMovieClick, 
  headerExtra, 
  onSeeAll,
  fallbackMessage,
  source = 'unknown',   // listing nguồn: 'trending', 'latest', 'recommendations', 'similar', 'search'
}) {
  return (
    <div className="section-wrapper">
      <div className="section-header">
        <h2>{title}</h2>
        {headerExtra}
        {onSeeAll && movies && movies.length > 0 && (
          <span className="see-all-link" onClick={onSeeAll}>See All</span>
        )}
      </div>

      {loading ? (
        <div className="scroll-row-wrapper">
          <div className="horizontal-scroll">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="skeleton-card" />
            ))}
          </div>
        </div>
      ) : !movies || movies.length === 0 ? (
        fallbackMessage ? (
          <div className="no-results" style={{ padding: '30px', background: 'rgba(0,0,0,0.01)', borderRadius: '10px' }}>
            {fallbackMessage}
          </div>
        ) : null
      ) : (
        <div className="scroll-row-wrapper">
          <button className="scroll-arrow-btn left" onClick={() => onScroll('left')}>‹</button>
          
          <div className="horizontal-scroll" ref={scrollRef}>
            {movies.map((movie, idx) => (
              <MovieCard 
                key={movie.movieId} 
                movie={movie} 
                onClick={() => onMovieClick(movie.movieId, source, idx)} 
              />
            ))}
          </div>

          <button className="scroll-arrow-btn right" onClick={() => onScroll('right')}>›</button>
        </div>
      )}
    </div>
  );
}

export default MovieRow;