import React, { useState } from 'react';
import { X, Lock, Mail, User, LogIn, UserPlus, Sparkles } from 'lucide-react';
import { API_BASE_URL } from '../api/client';
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Xác thực thất bại. Vui lòng thử lại.');
      }

      // Save token and user info
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
        <button onClick={onClose} className="auth-modal-close" aria-label="Đóng">
          <X className="w-4 h-4" />
        </button>

        <div className="auth-modal-header">
          <h2 className="auth-modal-title">
            {isLogin ? 'Đăng Nhập MovieNex' : 'Đăng Ký Tài Khoản'}
          </h2>
          <p className="auth-modal-subtitle">
            {isLogin
              ? 'Đăng nhập để trải nghiệm hệ thống gợi ý phim cá nhân hóa thời gian thực'
              : 'Đăng ký nhanh chóng bằng Email để lưu phim yêu thích & nhận gợi ý AI'}
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
                placeholder="Email hoặc Tên người dùng"
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
                  placeholder="Địa chỉ Email (Bắt buộc)"
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
                  placeholder="Tên hiển thị (Tùy chọn, để trống sẽ tự tạo)"
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
              placeholder="Mật khẩu (Tối thiểu 6 ký tự)"
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
            {isLogin ? <LogIn className="w-4 h-4" /> : <UserPlus className="w-4 h-4" />}
            <span>{loading ? 'Đang xử lý...' : isLogin ? 'Đăng Nhập' : 'Đăng Ký Bằng Email'}</span>
          </button>
        </form>

        <div className="auth-toggle-mode">
          {isLogin ? 'Chưa có tài khoản? ' : 'Đã có tài khoản? '}
          <button
            type="button"
            className="auth-toggle-btn"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
            }}
          >
            {isLogin ? 'Đăng Ký Ngay' : 'Đăng Nhập'}
          </button>
        </div>
      </div>
    </div>
  );
}
