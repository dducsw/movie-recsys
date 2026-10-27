import React, { useState } from 'react';
import { X, Lock, Mail, User, LogIn, UserPlus } from 'lucide-react';
import { API_BASE_URL, setAccessToken } from '../api/client';
import './AuthModal.css';

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [identifier, setIdentifier] = useState(''); // Email or Username for login
  const [email, setEmail] = useState('');           // Email for register
  const [username, setUsername] = useState('');     // Optional username for register
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const endpoint = isLogin ? '/auth/login' : '/auth/register';
    const payload = isLogin
      ? { identifier: identifier.trim(), password }
      : { email: email.trim().toLowerCase(), password, username: username.trim() || undefined };

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest'
        },
        credentials: 'include',
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed. Please try again.');
      }

      // Save token in memory (never in localStorage for XSS protection)
      setAccessToken(data.access_token);

      if (onAuthSuccess) {
        onAuthSuccess(data.user, !isLogin);
      }
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal-card" onClick={(e) => e.stopPropagation()}>
        <button onClick={onClose} className="auth-modal-close" aria-label="Close">
          <X />
        </button>

        <div className="auth-modal-header">
          <h2 className="auth-modal-title">
            {isLogin ? 'Sign in to MovieNex' : 'Create an Account'}
          </h2>
          <p className="auth-modal-subtitle">
            {isLogin
              ? 'Sign in to get personalized movie recommendations'
              : 'Sign up with your email to save movies & get AI recommendations'}
          </p>
        </div>

        {error && (
          <div className="auth-modal-error">
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-modal-form">
          {isLogin ? (
            /* Login Mode: Email or Username */
            <div className="auth-input-group">
              <Mail className="auth-input-icon" />
              <input
                type="text"
                placeholder="Email or username"
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                required
                className="auth-input"
                autoFocus
              />
            </div>
          ) : (
            /* Register Mode: Email required, Username optional */
            <>
              <div className="auth-input-group">
                <Mail className="auth-input-icon" />
                <input
                  type="email"
                  placeholder="Email address (required)"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="auth-input"
                  autoFocus
                />
              </div>

              <div className="auth-input-group">
                <User className="auth-input-icon" />
                <input
                  type="text"
                  placeholder="Display name (optional)"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="auth-input"
                />
              </div>
            </>
          )}

          <div className="auth-input-group">
            <Lock className="auth-input-icon" />
            <input
              type="password"
              placeholder="Password (min. 6 characters)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
              className="auth-input"
            />
          </div>

          <button 
            type="submit" 
            className="auth-submit-btn" 
            disabled={loading}
          >
            {isLogin ? <LogIn className="auth-submit-icon" /> : <UserPlus className="auth-submit-icon" />}
            <span>{loading ? 'Please wait…' : isLogin ? 'Sign In' : 'Sign Up with Email'}</span>
          </button>
        </form>

        <div className="auth-toggle-mode">
          {isLogin ? "Don't have an account? " : 'Already have an account? '}
          <button
            type="button"
            className="auth-toggle-btn"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
            }}
          >
            {isLogin ? 'Sign Up' : 'Sign In'}
          </button>
        </div>
      </div>
    </div>
  );
}
