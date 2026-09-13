import React, { useState } from 'react';
import './AuthView.css';

function AuthView({ setView, onLoginSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');

    if (!username || !password) {
      setErrorMsg('Please fill in all required fields.');
      return;
    }

    if (isRegister) {
      if (!email) {
        setErrorMsg('Please enter your email.');
        return;
      }
      if (password !== confirmPassword) {
        setErrorMsg('Passwords do not match.');
        return;
      }
      if (password.length < 4) {
        setErrorMsg('Password should be at least 4 characters.');
        return;
      }

      // Check if user exists
      const existingUsers = JSON.parse(localStorage.getItem('movienex_users') || '[]');
      const userExists = existingUsers.some(u => u.username.toLowerCase() === username.toLowerCase());
      
      if (userExists) {
        setErrorMsg('Username is already taken.');
        return;
      }

      // Save new user
      const newUser = { username, email, password };
      existingUsers.push(newUser);
      localStorage.setItem('movienex_users', JSON.stringify(existingUsers));
      
      setSuccessMsg('Account created successfully! Switching to Login...');
      setTimeout(() => {
        setIsRegister(false);
        setPassword('');
        setConfirmPassword('');
        setSuccessMsg('');
      }, 1500);

    } else {
      // Login
      const existingUsers = JSON.parse(localStorage.getItem('movienex_users') || '[]');
      
      // Seed a default demo account if none exists
      if (existingUsers.length === 0) {
        const demoUser = { username: 'demo', email: 'demo@movienex.com', password: 'demo' };
        existingUsers.push(demoUser);
        localStorage.setItem('movienex_users', JSON.stringify(existingUsers));
      }

      const matchedUser = existingUsers.find(
        u => u.username.toLowerCase() === username.toLowerCase() && u.password === password
      );

      if (!matchedUser) {
        setErrorMsg('Invalid username or password.');
        return;
      }

      // Log in
      localStorage.setItem('movienex_active_user', JSON.stringify({ username: matchedUser.username }));
      onLoginSuccess(matchedUser.username);
      setSuccessMsg('Logged in successfully!');
      setTimeout(() => {
        setView('home');
      }, 800);
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
        
        <h2>{isRegister ? 'Create Account' : 'Welcome Back'}</h2>
        <p className="auth-subtitle">
          {isRegister ? 'Sign up to unlock simulated movie watching and watchlist syncing.' : 'Sign in to access personalized recommendations.'}
        </p>

        {errorMsg && <div className="auth-alert error">{errorMsg}</div>}
        {successMsg && <div className="auth-alert success">{successMsg}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username (e.g. demo)"
              required
            />
          </div>

          {isRegister && (
            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter email address"
                required
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={isRegister ? "Create password" : "Enter password (e.g. demo)"}
              required
            />
          </div>

          {isRegister && (
            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                type="password"
                id="confirmPassword"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Repeat password"
                required
              />
            </div>
          )}

          <button type="submit" className="auth-submit-btn">
            {isRegister ? 'Register' : 'Sign In'}
          </button>
        </form>

        <div className="auth-toggle">
          {isRegister ? (
            <span>
              Already have an account?{' '}
              <button className="toggle-mode-btn" onClick={() => { setIsRegister(false); setErrorMsg(''); }}>Sign In</button>
            </span>
          ) : (
            <span>
              Don't have an account?{' '}
              <button className="toggle-mode-btn" onClick={() => { setIsRegister(true); setErrorMsg(''); }}>Register here</button>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default AuthView;