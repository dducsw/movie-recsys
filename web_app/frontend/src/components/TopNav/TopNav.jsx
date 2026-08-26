import React, { useState } from 'react';
import SearchFilterModal from './SearchFilterModal';
import './TopNav.css';

function TopNav({ 
  searchQuery, 
  setSearchQuery, 
  handleSearch, 
  handleClearSearch, 
  user, 
  onLogout, 
  onOpenAuthModal,
  setView,
  selectedStatus = 'all',
  setSelectedStatus,
  selectedYear = 'all',
  setSelectedYear,
  selectedGenre = 'all',
  setSelectedGenre,
  onApplyFilters,
  onResetFilters
}) {
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const [isFilterModalOpen, setIsFilterModalOpen] = useState(false);

  const hasActiveFilters = 
    selectedStatus !== 'all' || 
    selectedYear !== 'all' || 
    selectedGenre !== 'all';

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleModalApply = () => {
    setIsFilterModalOpen(false);
    if (onApplyFilters) {
      onApplyFilters();
    }
  };

  const handleModalReset = () => {
    if (onResetFilters) {
      onResetFilters();
    }
  };

  return (
    <header className="streamix-topnav">
      {/* 1. Sleek Search Box with Embedded Filter Icon Button */}
      <div className="streamix-search-dock">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="search-dock-icon">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <input
          type="text"
          className="search-dock-input"
          placeholder="Search movies by title, genre, or keyword..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        
        {searchQuery && (
          <button className="search-dock-clear" onClick={handleClearSearch} title="Clear search">
            ✕
          </button>
        )}

        {/* Filter Toggle Button on Right Corner of Search Bar */}
        <button
          className={`search-filter-btn ${hasActiveFilters ? 'active' : ''}`}
          onClick={() => setIsFilterModalOpen(true)}
          title="Filter by Year, Genre, Status"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="filter-icon-svg">
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
          {hasActiveFilters && <span className="filter-active-dot" />}
        </button>
      </div>

      {/* 2. Right Icons: Notification & Profile */}
      <div className="streamix-nav-right">
        <button 
          className="nav-round-icon-btn" 
          onClick={() => alert("Notification: 3 new personalized movie recommendations ready for you!")}
          title="Notifications"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 0 1-3.46 0" />
          </svg>
          <span className="round-badge-dot" />
        </button>

        {user ? (
          <div className="nav-user-dock" onClick={() => setShowUserDropdown(!showUserDropdown)}>
            <div className="user-round-avatar">
              {(user.username || 'U')[0].toUpperCase()}
            </div>

            {showUserDropdown && (
              <div className="user-popup-card">
                <div className="user-popup-info">
                  <span className="popup-name">{user.username}</span>
                  <span className="popup-email">{user.email || 'Free Member'}</span>
                </div>
                <hr className="popup-divider" />
                <button className="popup-item" onClick={() => setView('watchlist')}>My Favourites</button>
                <button className="popup-item" onClick={() => setView('chatbot')}>AI Assistant</button>
                <hr className="popup-divider" />
                <button className="popup-item logout" onClick={onLogout}>Sign Out</button>
              </div>
            )}
          </div>
        ) : (
          <button className="btn-signin-round" onClick={onOpenAuthModal}>
            Sign In
          </button>
        )}
      </div>

      {/* Filter Modal Overlay */}
      <SearchFilterModal
        isOpen={isFilterModalOpen}
        onClose={() => setIsFilterModalOpen(false)}
        selectedStatus={selectedStatus}
        setSelectedStatus={setSelectedStatus}
        selectedYear={selectedYear}
        setSelectedYear={setSelectedYear}
        selectedGenre={selectedGenre}
        setSelectedGenre={setSelectedGenre}
        onApplyFilters={handleModalApply}
        onResetFilters={handleModalReset}
      />
    </header>
  );
}

export default TopNav;
