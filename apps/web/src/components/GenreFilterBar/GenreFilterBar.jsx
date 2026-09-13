import React, { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
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
    'Science Fiction',
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
    <div className="genre-filter-container flex items-center justify-between gap-2 py-2">
      <div className="genre-filter-scroll flex items-center gap-2 overflow-x-auto scrollbar-none py-1" ref={scrollRef}>
        {filters.map((filter) => (
          <button
            key={filter}
            className={`filter-pill-btn shrink-0 text-xs px-4 py-2 rounded-full font-medium transition-all ${
              activeFilter === filter 
                ? 'active bg-red-600 text-white shadow-md shadow-red-600/30' 
                : 'bg-neutral-800/80 text-neutral-300 hover:text-white hover:bg-neutral-700/80'
            }`}
            onClick={() => onSelectFilter && onSelectFilter(filter)}
          >
            {filter}
          </button>
        ))}
      </div>

      <div className="filter-nav-arrows flex items-center gap-1 shrink-0">
        <button 
          className="filter-arrow-btn p-1.5 rounded-full bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors" 
          onClick={() => scroll('left')} 
          title="Scroll left"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
        <button 
          className="filter-arrow-btn p-1.5 rounded-full bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors" 
          onClick={() => scroll('right')} 
          title="Scroll right"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

export default GenreFilterBar;
