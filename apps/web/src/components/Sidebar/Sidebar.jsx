import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { 
  Play, 
  Home, 
  Compass, 
  Layers, 
  Bookmark, 
  Sparkles, 
  Bot, 
  Moon, 
  Sun, 
  LogIn, 
  LogOut, 
  User 
} from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import { useAuth } from '../../context/AuthContext';
import './Sidebar.css';

function Sidebar() {
  const { darkMode, toggleTheme } = useTheme();
  const { user, openAuthModal, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <aside className="cinemax-sidebar">
      {/* Brand Logo */}
      <div 
        className="sidebar-brand-box cursor-pointer" 
        onClick={() => navigate('/')}
      >
        <div className="brand-logo-icon">
          <Play className="w-5 h-5 fill-current" />
        </div>
        <span className="brand-name">Movie<span className="brand-accent">Nex</span></span>
      </div>

      <nav className="sidebar-nav-container">
        {/* Main Navigation */}
        <div className="nav-group">
          <NavLink 
            to="/" 
            end
            className={({ isActive }) => `nav-item-btn ${isActive ? 'active' : ''}`}
          >
            <Home className="nav-icon w-5 h-5" />
            <span>Home</span>
          </NavLink>

          <NavLink 
            to="/explore" 
            className={({ isActive }) => `nav-item-btn ${isActive ? 'active' : ''}`}
          >
            <Compass className="nav-icon w-5 h-5" />
            <span>Explore</span>
          </NavLink>

          <NavLink 
            to="/genres" 
            className={({ isActive }) => `nav-item-btn ${isActive ? 'active' : ''}`}
          >
            <Layers className="nav-icon w-5 h-5" />
            <span>Genres</span>
          </NavLink>

          <NavLink 
            to="/watchlist" 
            className={({ isActive }) => `nav-item-btn ${isActive ? 'active' : ''}`}
            onClick={(e) => {
              if (!user) {
                e.preventDefault();
                openAuthModal();
              }
            }}
          >
            <Bookmark className="nav-icon w-5 h-5" />
            <span>Favourites</span>
          </NavLink>
        </div>

        <div className="sidebar-divider" />

        {/* AI & Features Navigation */}
        <div className="nav-group">
          <NavLink 
            to="/ai-assistant" 
            className={({ isActive }) => `nav-item-btn special-ai-btn ${isActive ? 'active' : ''}`}
          >
            <Bot className="nav-icon w-5 h-5 text-indigo-400" />
            <div className="flex items-center gap-1.5">
              <span>AI Assistant</span>
              <Sparkles className="w-3.5 h-3.5 text-yellow-400 animate-pulse" />
            </div>
          </NavLink>
        </div>
      </nav>

      {/* Footer / Account & Theme Toggle */}
      <div className="sidebar-footer">
        {/* Theme Toggle Button */}
        <button 
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {darkMode ? (
            <>
              <Sun className="nav-icon w-5 h-5 text-amber-400" />
              <span>Light Mode</span>
            </>
          ) : (
            <>
              <Moon className="nav-icon w-5 h-5 text-indigo-400" />
              <span>Dark Mode</span>
            </>
          )}
        </button>

        <div className="sidebar-divider" />

        {/* Auth / Profile Area */}
        {user ? (
          <div className="user-profile-widget">
            <div className="flex items-center gap-3">
              <div className="user-avatar-circle">
                {user.avatar ? (
                  <img src={user.avatar} alt={user.name} className="w-full h-full rounded-full object-cover" />
                ) : (
                  <User className="w-5 h-5" />
                )}
              </div>
              <div className="user-text-info overflow-hidden">
                <span className="user-display-name truncate block">{user.name || user.username}</span>
                <span className="user-role-badge">Member</span>
              </div>
            </div>
            <button 
              className="logout-icon-btn hover:text-red-400 transition-colors mt-2" 
              onClick={logout} 
              title="Sign Out"
            >
              <LogOut className="w-4 h-4" />
              <span className="text-xs">Sign Out</span>
            </button>
          </div>
        ) : (
          <button 
            className="login-action-btn flex items-center justify-center gap-2"
            onClick={openAuthModal}
          >
            <LogIn className="w-4 h-4" />
            <span>Sign In</span>
          </button>
        )}
      </div>
    </aside>
  );
}

export default Sidebar;
