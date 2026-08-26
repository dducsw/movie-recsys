import React from 'react';
import './Sidebar.css';

function Sidebar({ 
  view, 
  setView, 
  allType, 
  handleSeeAll, 
  setSelectedMovieId, 
  handleClearSearch, 
  darkMode, 
  setDarkMode,
  user,
  onOpenAuthModal
}) {
  return (
    <aside className="cinemax-sidebar">
      {/* Brand Logo with Play Icon */}
      <div 
        className="sidebar-brand-box" 
        onClick={() => { setView('home'); handleClearSearch(); setSelectedMovieId(null); }}
      >
        <div className="brand-logo-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="6 4 20 12 6 20 6 4" />
          </svg>
        </div>
        <span className="brand-name">Movie<span className="brand-accent">Nex</span></span>
      </div>

      <nav className="sidebar-nav-container">
        {/* Main Group */}
        <div className="nav-group">
          <button 
            className={`nav-item-btn ${view === 'home' && !allType ? 'active' : ''}`}
            onClick={() => { setView('home'); handleClearSearch(); setSelectedMovieId(null); }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
            <span>Home</span>
          </button>

          <button 
            className={`nav-item-btn ${view === 'all' && allType === 'trending' ? 'active' : ''}`}
            onClick={() => handleSeeAll('trending')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <circle cx="12" cy="12" r="10" />
              <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
            </svg>
            <span>Explore</span>
          </button>

          <button 
            className={`nav-item-btn ${view === 'all' && allType === 'genre' ? 'active' : ''}`}
            onClick={() => {
              setView('home');
              const el = document.getElementById('genres-section');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
            <span>Genres</span>
          </button>

          <button 
            className={`nav-item-btn ${view === 'watchlist' ? 'active' : ''}`}
            onClick={() => {
              if (!user) {
                onOpenAuthModal();
              } else {
                setView('watchlist');
                handleClearSearch();
                setSelectedMovieId(null);
              }
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
            </svg>
            <span>Favourites</span>
          </button>
        </div>

        <div className="sidebar-divider" />

        {/* Secondary Group */}
        <div className="nav-group">
          <button 
            className={`nav-item-btn ${view === 'all' && allType === 'recs' ? 'active' : ''}`}
            onClick={() => handleSeeAll('recs')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            <span>For You (AI)</span>
          </button>

          <button 
            className={`nav-item-btn ${view === 'all' && allType === 'latest' ? 'active' : ''}`}
            onClick={() => handleSeeAll('latest')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <span>Recently Added</span>
          </button>

          <button 
            className={`nav-item-btn ${view === 'chatbot' ? 'active' : ''}`}
            onClick={() => { setView('chatbot'); handleClearSearch(); setSelectedMovieId(null); }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <span>AI Assistant</span>
          </button>
        </div>

        <div className="sidebar-divider" />

        {/* Bottom Preferences */}
        <div className="nav-group bottom-pref">
          <div className="theme-toggle-row" onClick={() => setDarkMode(!darkMode)}>
            <div className="toggle-left">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="nav-icon">
                {darkMode ? (
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                ) : (
                  <circle cx="12" cy="12" r="5" />
                )}
              </svg>
              <span>Dark Mode</span>
            </div>
            <div className={`switch-pill ${darkMode ? 'on' : 'off'}`}>
              <div className="switch-dot" />
            </div>
          </div>
        </div>
      </nav>
    </aside>
  );
}

export default Sidebar;
