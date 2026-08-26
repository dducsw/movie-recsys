import React, { useState } from 'react';
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
        throw new Error(data.detail || 'Thao tác thất bại');
      }

      // Save token
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('user_info', JSON.stringify(data.user));

      onAuthSuccess(data.user, !isLogin);
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
        <button onClick={onClose} className="auth-modal-close">✕</button>

        <h2 className="auth-modal-title">
          {isLogin ? 'Sign In to MovieNex' : 'Create a MovieNex Account'}
        </h2>
        <p className="auth-modal-subtitle">
          {isLogin
            ? 'Sign in to access personalized movie recommendations in real-time'
            : 'Join MovieNex to discover curated movies with AI recommendations'}
        </p>

        {error && <div className="auth-error-box">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          {!isLogin && (
            <div className="auth-form-group">
              <label className="auth-form-label">Username</label>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. alex123"
                className="auth-form-input"
              />
            </div>
          )}

          <div className="auth-form-group">
            <label className="auth-form-label">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@example.com"
              className="auth-form-input"
            />
          </div>

          <div className="auth-form-group">
            <label className="auth-form-label">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="auth-form-input"
            />
          </div>

          <button type="submit" disabled={loading} className="btn-auth-submit">
            {loading ? 'Processing...' : isLogin ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <div className="auth-switch-prompt">
          {isLogin ? (
            <p>
              Don't have an account?{' '}
              <button
                onClick={() => { setIsLogin(false); setError(''); }}
                className="auth-switch-btn"
              >
                Sign up now
              </button>
            </p>
          ) : (
            <p>
              Already have an account?{' '}
              <button
                onClick={() => { setIsLogin(true); setError(''); }}
                className="auth-switch-btn"
              >
                Sign in
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
