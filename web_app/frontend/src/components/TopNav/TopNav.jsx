import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { 
  Search, 
  SlidersHorizontal, 
  X, 
  User, 
  LogIn, 
  LogOut, 
  Bookmark, 
  Sparkles, 
  Play, 
  Sun, 
  Moon,
  Compass,
  Film,
  Menu
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';
import SearchFilterModal from './SearchFilterModal';
import './TopNav.css';

export default function TopNav({
  searchQuery = '',
  setSearchQuery,
  onOpenAuth,
  activeFilters = {},
  onApplyFilters,
  onResetFilters
}) {
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  // Detect scroll to transition from transparent to frosted glass
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 40) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/explore?query=${encodeURIComponent(searchQuery.trim())}`);
      setMobileMenuOpen(false);
    }
  };

  const handleGenresClick = (e) => {
    e.preventDefault();
    setMobileMenuOpen(false);
    if (location.pathname !== '/') {
      navigate('/explore');
    } else {
      const el = document.getElementById('genres-section');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const hasActiveFilters = Boolean(
    activeFilters.genre || activeFilters.year || activeFilters.status
  );

  return (
    <>
      <header className={`streamix-topnav ${scrolled ? 'scrolled' : 'transparent'}`}>
        {/* Left: Brand Logo & Nav Links */}
        <div className="streamix-nav-left">
          <div className="streamix-brand" onClick={() => navigate('/')}>
            <div className="brand-logo-icon">
              <Play className="w-4 h-4 fill-current" />
            </div>
            <span className="brand-name">
              Movie<span className="brand-accent">Nex</span>
            </span>
          </div>

          {/* Desktop Navigation Links */}
          <nav className="streamix-nav-links">
            <NavLink 
              to="/" 
              className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
            >
              Home
            </NavLink>

            <NavLink 
              to="/explore" 
              className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
            >
              Explore
            </NavLink>

            <a 
              href="#genres" 
              onClick={handleGenresClick} 
              className="nav-link-item"
            >
              Genres
            </a>

            <NavLink 
              to="/watchlist" 
              className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
            >
              Favourites
            </NavLink>

            <NavLink 
              to="/chatbot" 
              className={({ isActive }) => `nav-link-item special-ai ${isActive ? 'active' : ''}`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>AI Assistant</span>
            </NavLink>
          </nav>
        </div>

        {/* Right: Search Dock, Theme, Auth */}
        <div className="streamix-nav-right">
          {/* Search Dock */}
          <form className="streamix-search-dock" onSubmit={handleSearchSubmit}>
            <Search className="search-dock-icon" />
            <input
              type="text"
              placeholder="Search movies, cast, genres..."
              value={searchQuery}
              onChange={(e) => setSearchQuery && setSearchQuery(e.target.value)}
              className="search-dock-input"
            />

            {searchQuery && (
              <button
                type="button"
                className="search-dock-clear"
                onClick={() => setSearchQuery && setSearchQuery('')}
              >
                <X className="w-3 h-3" />
              </button>
            )}

            <button
              type="button"
              className={`search-filter-btn ${hasActiveFilters ? 'active' : ''}`}
              onClick={() => setFilterModalOpen(true)}
              title="Search Filters"
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              {hasActiveFilters && <span className="filter-active-dot" />}
            </button>
          </form>

          {/* Theme Toggle Button */}
          <button 
            className="nav-round-icon-btn" 
            onClick={toggleTheme} 
            title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-500" />}
          </button>

          {/* User Profile / Auth */}
          {user ? (
            <div className="nav-user-dock">
              <div 
                className="user-round-avatar"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                title={user.username}
              >
                {user.username ? user.username[0].toUpperCase() : 'U'}
              </div>

              {userMenuOpen && (
                <div className="user-popup-card" onClick={() => setUserMenuOpen(false)}>
                  <div className="user-popup-info">
                    <span className="popup-name">{user.username}</span>
                    <span className="popup-email">{user.email}</span>
                  </div>
                  <div className="popup-divider" />
                  
                  <button className="popup-item" onClick={() => navigate('/watchlist')}>
                    <Bookmark className="w-4 h-4" />
                    <span>My Watchlist</span>
                  </button>

                  <button className="popup-item logout" onClick={logout}>
                    <LogOut className="w-4 h-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              )}
            </div>
          ) : (
            <button className="btn-signin-round" onClick={onOpenAuth}>
              <LogIn className="w-4 h-4" />
              <span>Sign In</span>
            </button>
          )}

          {/* Mobile Hamburger Toggle */}
          <button 
            className="mobile-hamburger-btn" 
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            title="Menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer" onClick={() => setMobileMenuOpen(false)}>
          <div className="mobile-drawer-card" onClick={(e) => e.stopPropagation()}>
            <NavLink to="/" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <Film className="w-4 h-4" />
              <span>Home</span>
            </NavLink>
            <NavLink to="/explore" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <Compass className="w-4 h-4" />
              <span>Explore</span>
            </NavLink>
            <a href="#genres" className="mobile-nav-item" onClick={handleGenresClick}>
              <SlidersHorizontal className="w-4 h-4" />
              <span>Genres</span>
            </a>
            <NavLink to="/watchlist" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <Bookmark className="w-4 h-4" />
              <span>Favourites</span>
            </NavLink>
            <NavLink to="/chatbot" className="mobile-nav-item special-ai" onClick={() => setMobileMenuOpen(false)}>
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>AI Assistant</span>
            </NavLink>
          </div>
        </div>
      )}

      {/* Multi-Criteria Filter Modal */}
      <SearchFilterModal
        isOpen={filterModalOpen}
        onClose={() => setFilterModalOpen(false)}
        activeFilters={activeFilters}
        onApply={(filters) => {
          if (onApplyFilters) onApplyFilters(filters);
          setFilterModalOpen(false);
          navigate('/explore');
        }}
        onReset={() => {
          if (onResetFilters) onResetFilters();
          setFilterModalOpen(false);
        }}
      />
    </>
  );
}
