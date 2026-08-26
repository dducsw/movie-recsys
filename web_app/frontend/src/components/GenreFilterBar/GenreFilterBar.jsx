import React, { useRef } from 'react';
import './GenreFilterBar.css';

function GenreFilterBar({ activeFilter, onSelectFilter }) {
  const scrollRef = useRef(null);

  const filters = [
    'Trending',
    'Action',
    'Adventure',
    'Comedy',
    'Crime',
    'Drama',
    'Fantasy',
    'Horror',
    'Sci-Fi',
    'Animation',
    'Thriller',
    'Romance',
    'Mystery'
  ];

  const scroll = (direction) => {
    if (!scrollRef.current) return;
    const scrollAmount = 240;
    scrollRef.current.scrollBy({
      left: direction === 'left' ? -scrollAmount : scrollAmount,
      behavior: 'smooth'
    });
  };

  return (
    <div className="genre-filter-container">
      <div className="genre-filter-scroll" ref={scrollRef}>
        {filters.map((filter) => (
          <button
            key={filter}
            className={`filter-pill-btn ${activeFilter === filter ? 'active' : ''}`}
            onClick={() => onSelectFilter && onSelectFilter(filter)}
          >
            {filter}
          </button>
        ))}
      </div>

      <div className="filter-nav-arrows">
        <button className="filter-arrow-btn" onClick={() => scroll('left')} title="Scroll left">
          ‹
        </button>
        <button className="filter-arrow-btn" onClick={() => scroll('right')} title="Scroll right">
          ›
        </button>
      </div>
    </div>
  );
}

export default GenreFilterBar;
