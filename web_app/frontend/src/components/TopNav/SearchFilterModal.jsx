import React from 'react';
import './SearchFilterModal.css';

const STATUS_OPTIONS = [
  { id: 'all', label: 'All Status' },
  { id: 'released', label: 'Released' },
  { id: 'upcoming', label: 'Upcoming' }
];

const YEAR_OPTIONS = [
  { id: 'all', label: 'All Years' },
  { id: '2026', label: '2026' },
  { id: '2025', label: '2025' },
  { id: '2024', label: '2024' },
  { id: '2020-2023', label: '2020 – 2023' },
  { id: '2010s', label: '2010s' },
  { id: '2000s', label: '2000s' },
  { id: 'classic', label: 'Classic (<2000)' }
];

const GENRE_OPTIONS = [
  'All Genres',
  'Action', 'Adventure', 'Comedy', 'Crime',
  'Drama', 'Fantasy', 'Horror', 'Mystery',
  'Romance', 'Science Fiction', 'Thriller'
];

export default function SearchFilterModal({
  isOpen,
  onClose,
  selectedStatus,
  setSelectedStatus,
  selectedYear,
  setSelectedYear,
  selectedGenre,
  setSelectedGenre,
  onApplyFilters,
  onResetFilters
}) {
  if (!isOpen) return null;

  return (
    <div className="filter-modal-overlay" onClick={onClose}>
      <div className="filter-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="filter-modal-header">
          <div className="filter-header-left">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="filter-header-icon">
              <line x1="4" y1="21" x2="4" y2="14" />
              <line x1="4" y1="10" x2="4" y2="3" />
              <line x1="12" y1="21" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12" y2="3" />
              <line x1="20" y1="21" x2="20" y2="16" />
              <line x1="20" y1="12" x2="20" y2="3" />
              <line x1="1" y1="14" x2="7" y2="14" />
              <line x1="9" y1="8" x2="15" y2="8" />
              <line x1="17" y1="16" x2="23" y2="16" />
            </svg>
            <h3 className="filter-header-title">Filter Movies</h3>
          </div>
          <div className="filter-header-actions">
            <button className="btn-filter-reset" onClick={onResetFilters}>
              Reset
            </button>
            <button className="btn-filter-close" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>

        {/* Filter Body */}
        <div className="filter-modal-body">
          {/* 1. Release Status */}
          <div className="filter-group">
            <label className="filter-group-label">Release Status</label>
            <div className="filter-chips-row">
              {STATUS_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  className={`filter-chip ${selectedStatus === opt.id ? 'active' : ''}`}
                  onClick={() => setSelectedStatus(opt.id)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 2. Release Year */}
          <div className="filter-group">
            <label className="filter-group-label">Release Year</label>
            <div className="filter-chips-row wrap">
              {YEAR_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  className={`filter-chip ${selectedYear === opt.id ? 'active' : ''}`}
                  onClick={() => setSelectedYear(opt.id)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 3. Movie Genre */}
          <div className="filter-group">
            <label className="filter-group-label">Movie Genre</label>
            <div className="filter-chips-row wrap">
              {GENRE_OPTIONS.map((g) => {
                const genreId = g === 'All Genres' ? 'all' : g;
                return (
                  <button
                    key={g}
                    className={`filter-chip ${selectedGenre === genreId ? 'active' : ''}`}
                    onClick={() => setSelectedGenre(genreId)}
                  >
                    {g}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Footer Apply Button */}
        <div className="filter-modal-footer">
          <button className="btn-apply-filters" onClick={onApplyFilters}>
            Apply Filters
          </button>
        </div>
      </div>
    </div>
  );
}
