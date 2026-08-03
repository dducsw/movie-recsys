import { useState, useEffect } from 'react';
import { apiFetch } from '../api/client';

export function useAuth() {
  const [user, setUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      apiFetch('/auth/me')
        .then((data) => {
          if (data && data.user) {
            setUser(data.user);
            setUserProfile(data);
          } else {
            localStorage.removeItem('auth_token');
            setUser(null);
          }
        })
        .catch(() => {
          localStorage.removeItem('auth_token');
          setUser(null);
        });
    }
  }, []);

  const handleLogout = (onSuccess) => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_info');
    setUser(null);
    setUserProfile(null);
    if (onSuccess) onSuccess();
  };

  const handleAuthSuccess = (userData, isNewUser) => {
    setUser(userData);
    if (isNewUser) {
      setIsOnboardingOpen(true);
    }
  };

  return {
    user,
    setUser,
    userProfile,
    setUserProfile,
    isAuthModalOpen,
    setIsAuthModalOpen,
    isOnboardingOpen,
    setIsOnboardingOpen,
    handleLogout,
    handleAuthSuccess
  };
}
