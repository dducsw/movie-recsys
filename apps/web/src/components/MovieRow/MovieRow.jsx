import { useRef } from 'react';
import { ChevronLeft, ChevronRight, ArrowRight } from 'lucide-react';
import MovieCard from '../MovieCard/MovieCard';
import './MovieRow.css';

export default function MovieRow({
  title,
  movies = [],
  onMovieClick,
  onSeeAll,
  source = 'row'
}) {
  const rowRef = useRef(null);

  const scroll = (direction) => {
    if (!rowRef.current) return;
    const { scrollLeft, clientWidth } = rowRef.current;
    const scrollAmount = clientWidth * 0.75;
    rowRef.current.scrollTo({
      left: direction === 'left' ? scrollLeft - scrollAmount : scrollLeft + scrollAmount,
      behavior: 'smooth'
    });
  };

  if (movies.length === 0) return null;

  return (
    <section className="streamix-row-section">
      <div className="streamix-row-header">
        <h2 className="streamix-row-title">{title}</h2>

        <div className="streamix-row-controls">
          {onSeeAll && (
            <button className="streamix-see-all-btn" onClick={onSeeAll}>
              <span>See All</span>
              <ArrowRight />
            </button>
          )}
          <div className="streamix-arrow-group">
            <button className="streamix-arrow-btn" onClick={() => scroll('left')} title="Scroll left">
              <ChevronLeft />
            </button>
            <button className="streamix-arrow-btn" onClick={() => scroll('right')} title="Scroll right">
              <ChevronRight />
            </button>
          </div>
        </div>
      </div>

      <div className="streamix-cards-track" ref={rowRef}>
        {movies.map((movie, idx) => {
          const mid = movie.movieId || movie.id;
          return (
            <div key={mid || idx} className="streamix-card-wrapper">
              <MovieCard
                movie={movie}
                source={source}
                position={idx}
                onClick={onMovieClick ? () => onMovieClick(mid) : null}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}
