import { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import './GenreFilterBar.css';

const FILTERS = [
  'Action',
  'Adventure',
  'Animation',
  'Comedy',
  'Crime',
  'Drama',
  'Fantasy',
  'Horror',
  'Science Fiction',
  'Thriller',
  'Romance',
  'Mystery'
];

export default function GenreFilterBar({ onSelectFilter }) {
  const scrollRef = useRef(null);

  const scroll = (direction) => {
    scrollRef.current?.scrollBy({
      left: direction === 'left' ? -240 : 240,
      behavior: 'smooth'
    });
  };

  return (
    <div className="genre-filter-container">
      <div className="genre-filter-scroll" ref={scrollRef}>
        {FILTERS.map((filter) => (
          <button
            key={filter}
            className="filter-pill-btn"
            onClick={() => onSelectFilter && onSelectFilter(filter)}
          >
            {filter}
          </button>
        ))}
      </div>

      <div className="filter-nav-arrows">
        <button className="filter-arrow-btn" onClick={() => scroll('left')} title="Scroll left">
          <ChevronLeft />
        </button>
        <button className="filter-arrow-btn" onClick={() => scroll('right')} title="Scroll right">
          <ChevronRight />
        </button>
      </div>
    </div>
  );
}
