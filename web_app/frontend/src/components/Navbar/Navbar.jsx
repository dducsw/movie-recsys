import React from 'react';
import './Navbar.css';

function Navbar({ view, allType, setView, handleClearSearch, setSelectedMovieId, handleSeeAll, isSearching, user, onLogout }) {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <div 
          className="navbar-logo" 
          onClick={() => { setView('home'); handleClearSearch(); setSelectedMovieId(null); }}
        >
          MovieNex <span>RecSys</span>
        </div>
        <ul className="navbar-menu">
          <li 
            style={view === 'home' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} 
            onClick={() => { setView('home'); handleClearSearch(); setSelectedMovieId(null); }}
          >
            Home
          </li>
          <li 
            style={view === 'all' && allType === 'trending' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} 
            onClick={() => handleSeeAll('trending')}
          >
            Trending
          </li>
          <li 
            style={view === 'chatbot' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} 
            onClick={() => { setView('chatbot'); handleClearSearch(); setSelectedMovieId(null); }}
          >
            AI Chatbot
          </li>
          <li 
            style={view === 'all' && allType === 'recs' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} 
            onClick={() => handleSeeAll('recs')}
          >
            RecSys
          </li>
        </ul>
      </div>
      <div className="navbar-right">
        <span 
          className="nav-icon" 
          style={{ display: 'flex', alignItems: 'center' }} 
          onClick={() => alert("System Status: Recommendation System is active and running!")}
        >
          <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: '20px', height: '20px', color: 'white' }}>
            <path d="M12 22c1.1 0 2-.9 2-2h-4c0 1.1.89 2 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/>
          </svg>
        </span>
        <span className="nav-lang">EN</span>
        
        {user ? (
          <div className="nav-user-menu" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="nav-avatar" title={`Logged in as ${user}`}>
              <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: '100%', height: '100%', opacity: 0.9 }}>
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
            <span className="nav-user-greeting" style={{ fontSize: '13.5px', fontWeight: 600, color: 'rgba(255,255,255,0.9)' }}>
              Hi, {user}
            </span>
            <button className="nav-auth-btn signout" onClick={onLogout} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.3)', color: 'white', padding: '6px 14px', borderRadius: '20px', cursor: 'pointer', fontSize: '12.5px', fontWeight: 600, transition: 'all 0.2s ease' }}>
              Sign Out
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="nav-avatar">
              <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: '100%', height: '100%', opacity: 0.5 }}>
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
            <button className="nav-auth-btn signin" onClick={() => setView('auth')} style={{ background: 'white', border: 'none', color: '#032541', padding: '6px 16px', borderRadius: '20px', cursor: 'pointer', fontSize: '12.5px', fontWeight: 700, boxShadow: '0 2px 5px rgba(0,0,0,0.15)', transition: 'all 0.2s ease' }}>
              Sign In
            </button>
          </div>
        )}

        {isSearching && (
          <button 
            className="search-clear-btn" 
            style={{ padding: '2px 10px', fontSize: '12px' }} 
            onClick={handleClearSearch}
          >
            Close Search
          </button>
        )}
      </div>
    </nav>
  );
}

export default Navbar;