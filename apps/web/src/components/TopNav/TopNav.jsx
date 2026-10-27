import { useEffect, useState } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Search, SlidersHorizontal, LogIn, LogOut, Bookmark, Sparkles, Play, Sun, Moon, X, Menu } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';
import SearchFilterModal from './SearchFilterModal';
import './TopNav.css';

export default function TopNav({ searchQuery = '', setSearchQuery, onOpenAuth }) {
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [filters, setFilters] = useState({ genre: 'all', year: 'all', status: 'all' });
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Đóng mọi menu khi đổi trang.
  useEffect(() => {
    setUserMenuOpen(false);
    setMobileMenuOpen(false);
  }, [location]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/explore?query=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const applyFilters = (next) => {
    setFilters(next);
    setFilterModalOpen(false);
    const params = new URLSearchParams();
    if (searchQuery.trim()) params.set('query', searchQuery.trim());
    if (next.genre !== 'all') params.set('genre', next.genre);
    if (next.year !== 'all') params.set('year', next.year);
    if (next.status !== 'all') params.set('status', next.status);
    navigate(`/explore${params.toString() ? `?${params}` : ''}`);
  };

  const hasActiveFilters = filters.genre !== 'all' || filters.year !== 'all' || filters.status !== 'all';

  const navLinkClass = ({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`;

  return (
    <header className={`streamix-topnav ${scrolled ? 'scrolled' : 'transparent'}`}>
      {/* Logo + Nav */}
      <div className="streamix-nav-left">
        <div className="streamix-brand" onClick={() => navigate('/')}>
          <div className="brand-logo-icon">
            <Play />
          </div>
          <span className="brand-name">
            Movie<span className="brand-accent">Nex</span>
          </span>
        </div>

        <nav className="streamix-nav-links">
          <NavLink to="/" className={navLinkClass}>Home</NavLink>
          <NavLink to="/explore" className={navLinkClass}>Explore</NavLink>
          <NavLink to="/watchlist" className={navLinkClass}>Watchlist</NavLink>
          <NavLink to="/chatbot" className={navLinkClass}>
            <Sparkles className="nav-ai-icon" />
            <span>AI Chat</span>
          </NavLink>
        </nav>
      </div>

      {/* Search, Theme, Auth */}
      <div className="streamix-nav-right">
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
              aria-label="Clear search"
            >
              ✕
            </button>
          )}

          <button
            type="button"
            className={`search-filter-btn ${hasActiveFilters ? 'active' : ''}`}
            onClick={() => setFilterModalOpen(true)}
            title="Search Filters"
          >
            <SlidersHorizontal />
            {hasActiveFilters && <span className="filter-active-dot" />}
          </button>
        </form>

        <button
          className="nav-round-icon-btn"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {theme === 'dark' ? <Sun /> : <Moon />}
        </button>

        {user ? (
          <div className="nav-user-dock">
            <button
              className="user-round-avatar"
              onClick={() => setUserMenuOpen(!userMenuOpen)}
              title={user.username}
            >
              {user.username ? user.username[0].toUpperCase() : 'U'}
            </button>

            {userMenuOpen && (
              <div className="user-popup-card" onClick={() => setUserMenuOpen(false)}>
                <div className="user-popup-info">
                  <span className="popup-name">{user.username}</span>
                  <span className="popup-email">{user.email}</span>
                </div>
                <div className="popup-divider" />

                <button className="popup-item" onClick={() => navigate('/watchlist')}>
                  <Bookmark className="popup-item-icon" />
                  <span>My Watchlist</span>
                </button>

                <button className="popup-item logout" onClick={logout}>
                  <LogOut className="popup-item-icon" />
                  <span>Sign Out</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <button className="btn-signin-round" onClick={onOpenAuth}>
            <LogIn />
            <span>Sign In</span>
          </button>
        )}

        <button
          className="mobile-hamburger-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          title="Menu"
        >
          {mobileMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer" onClick={() => setMobileMenuOpen(false)}>
          <div className="mobile-drawer-card" onClick={(e) => e.stopPropagation()}>
            <NavLink to="/" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <span>Home</span>
            </NavLink>
            <NavLink to="/explore" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <span>Explore</span>
            </NavLink>
            <NavLink to="/watchlist" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <span>Watchlist</span>
            </NavLink>
            <NavLink to="/chatbot" className="mobile-nav-item" onClick={() => setMobileMenuOpen(false)}>
              <span>AI Chat</span>
            </NavLink>
          </div>
        </div>
      )}

      <SearchFilterModal
        isOpen={filterModalOpen}
        onClose={() => setFilterModalOpen(false)}
        initialFilters={filters}
        onApply={applyFilters}
      />
    </header>
  );
}
