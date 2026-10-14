import React, { useState } from 'react';
import { API_BASE_URL, setAccessToken } from '../../api/client';
import './AuthView.css';

function AuthView({ setView, onLoginSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');

    if (!username.trim() || !password) {
      setErrorMsg('Please fill in all required fields.');
      return;
    }

    if (isRegister) {
      if (!email.trim()) {
        setErrorMsg('Please enter your email.');
        return;
      }
      if (password !== confirmPassword) {
        setErrorMsg('Passwords do not match.');
        return;
      }
      if (password.length < 6) {
        setErrorMsg('Password should be at least 6 characters.');
        return;
      }

      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
          },
          credentials: 'include',
          body: JSON.stringify({
            email: email.trim().toLowerCase(),
            password,
            username: username.trim(),
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Registration failed');
        }

        setSuccessMsg('Account created successfully! Switching to Login...');
        setTimeout(() => {
          setIsRegister(false);
          setPassword('');
          setConfirmPassword('');
          setSuccessMsg('');
        }, 1500);
      } catch (err) {
        setErrorMsg(err.message);
      } finally {
        setLoading(false);
      }
    } else {
      // Login via secure API
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
          },
          credentials: 'include',
          body: JSON.stringify({
            identifier: username.trim(),
            password,
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Invalid username or password.');
        }

        if (data.access_token) {
          setAccessToken(data.access_token);
        }

        if (onLoginSuccess) {
          onLoginSuccess(data.user?.username || username);
        }
        setSuccessMsg('Logged in successfully!');
        setTimeout(() => {
          setView('home');
        }, 800);
      } catch (err) {
        setErrorMsg(err.message);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="auth-view-container">
      <div className="auth-back-btn" onClick={() => setView('home')}>
        ← Back to Homepage
      </div>
      
      <div className="auth-card">
        <div className="auth-logo">
          MovieNex <span>RecSys</span>
        </div>
        
        <h2>{isRegister ? 'Create an Account' : 'Welcome Back'}</h2>
        <p className="auth-subtitle">
          {isRegister 
            ? 'Sign up to get personalized movie recommendations' 
            : 'Log in to access your customized watchlist and AI assistant'}
        </p>

        {errorMsg && <div className="auth-error-banner">{errorMsg}</div>}
        {successMsg && <div className="auth-success-banner">{successMsg}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label>Username / Account</label>
            <input 
              type="text" 
              placeholder="e.g. alex_movie" 
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          {isRegister && (
            <div className="form-group">
              <label>Email Address</label>
              <input 
                type="email" 
                placeholder="alex@example.com" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          )}

          <div className="form-group">
            <label>Password</label>
            <input 
              type="password" 
              placeholder="••••••••" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {isRegister && (
            <div className="form-group">
              <label>Confirm Password</label>
              <input 
                type="password" 
                placeholder="••••••••" 
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
          )}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? 'Processing...' : (isRegister ? 'Register Now' : 'Sign In')}
          </button>
        </form>

        <div className="auth-switch">
          {isRegister ? (
            <span>
              Already have an account?{' '}
              <a href="#login" onClick={(e) => { e.preventDefault(); setIsRegister(false); setErrorMsg(''); }}>
                Sign In
              </a>
            </span>
          ) : (
            <span>
              Don't have an account?{' '}
              <a href="#register" onClick={(e) => { e.preventDefault(); setIsRegister(true); setErrorMsg(''); }}>
                Create one
              </a>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default AuthView;