import React, { useState } from 'react';
import { X, Lock, Mail, User, LogIn, UserPlus } from 'lucide-react';
import { API_BASE_URL } from '../api/client';
import './AuthModal.css';

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
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
      ? { email, password }
      : { username, email, password };

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      // Save token
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('user_info', JSON.stringify(data.user));

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
        <button onClick={onClose} className="auth-modal-close">
          <X className="w-4 h-4" />
        </button>

        <h2 className="auth-modal-title">
          {isLogin ? 'Sign In to MovieNex' : 'Create a MovieNex Account'}
        </h2>
        <p className="auth-modal-subtitle">
          {isLogin
            ? 'Sign in to access personalized movie recommendations in real-time'
            : 'Join to track watched movies and receive AI recommendations'}
        </p>

        {error && <div className="auth-modal-error">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-modal-form">
          {!isLogin && (
            <div className="auth-input-group">
              <User className="auth-input-icon" />
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="auth-input"
              />
            </div>
          )}

          <div className="auth-input-group">
            <Mail className="auth-input-icon" />
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="auth-input"
            />
          </div>

          <div className="auth-input-group">
            <Lock className="auth-input-icon" />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="auth-input"
            />
          </div>

          <button 
            type="submit" 
            className="auth-submit-btn" 
            disabled={loading}
          >
            {isLogin ? <LogIn className="w-4 h-4" /> : <UserPlus className="w-4 h-4" />}
            <span>{loading ? 'Processing...' : isLogin ? 'Sign In' : 'Register Account'}</span>
          </button>
        </form>

        <div className="auth-toggle-mode">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
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
