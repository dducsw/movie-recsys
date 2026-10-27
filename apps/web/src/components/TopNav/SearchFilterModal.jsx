import { useEffect, useState } from 'react';
import { SlidersHorizontal } from 'lucide-react';
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
  { id: '2000s', label: '2000s' },
  { id: 'classic', label: 'Classic (<2000)' }
];

const GENRE_OPTIONS = [
  'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
  'Drama', 'Fantasy', 'Horror', 'Mystery',
  'Romance', 'Science Fiction', 'Thriller'
];

const DEFAULTS = { status: 'all', year: 'all', genre: 'all' };

export default function SearchFilterModal({ isOpen, onClose, initialFilters, onApply }) {
  const [draft, setDraft] = useState(initialFilters || DEFAULTS);

  // Mỗi lần mở modal, reset draft về filter đang áp dụng hiện tại.
  useEffect(() => {
    if (isOpen) setDraft({ ...DEFAULTS, ...initialFilters });
  }, [isOpen, initialFilters]);

  if (!isOpen) return null;

  return (
    <div className="filter-modal-overlay" onClick={onClose}>
      <div className="filter-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="filter-modal-header">
          <div className="filter-header-left">
            <SlidersHorizontal className="filter-header-icon" />
            <h3 className="filter-header-title">Filter Movies</h3>
          </div>
          <button className="btn-filter-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="filter-modal-body">
          <div className="filter-group">
            <label className="filter-group-label">Release Status</label>
            <div className="filter-chips-row">
              {STATUS_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  className={`filter-chip ${draft.status === opt.id ? 'active' : ''}`}
                  onClick={() => setDraft((d) => ({ ...d, status: opt.id }))}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label className="filter-group-label">Release Year</label>
            <div className="filter-chips-row wrap">
              {YEAR_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  className={`filter-chip ${draft.year === opt.id ? 'active' : ''}`}
                  onClick={() => setDraft((d) => ({ ...d, year: opt.id }))}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group">
            <label className="filter-group-label">Movie Genre</label>
            <div className="filter-chips-row wrap">
              {GENRE_OPTIONS.map((g) => (
                <button
                  key={g}
                  className={`filter-chip ${draft.genre === g ? 'active' : ''}`}
                  onClick={() => setDraft((d) => ({ ...d, genre: g }))}
                >
                  {g}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="filter-modal-footer">
          <button
            className="btn-filter-reset"
            onClick={() => setDraft(DEFAULTS)}
          >
            Reset
          </button>
          <button className="btn-apply-filters" onClick={() => onApply(draft)}>
            Apply Filters
          </button>
        </div>
      </div>
    </div>
  );
}
