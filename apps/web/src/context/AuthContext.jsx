import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_BASE_URL, setAccessToken, refreshAccessToken } from '../api/client';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);

  // Initialize session from secure HttpOnly cookie on page load
  useEffect(() => {
    let isMounted = true;
    refreshAccessToken()
      .then((data) => {
        if (!isMounted) return;
        setUser(data?.user || null);
      })
      .catch(() => {
        if (isMounted) setUser(null);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const logout = async () => {
    try {
      await fetch(`${API_BASE_URL}/auth/logout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
        },
        credentials: 'include',
      });
    } catch {
      // Ignored
    } finally {
      setAccessToken(null);
      setUser(null);
    }
  };

  const handleAuthSuccess = (userData, isNewUser) => {
    setUser(userData);
    setIsAuthModalOpen(false);
    if (isNewUser) {
      setIsOnboardingOpen(true);
    }
  };

  return (
    <AuthContext.Provider value={{
      user,
      setUser,
      logout,
      isAuthModalOpen,
      setIsAuthModalOpen,
      openAuthModal: () => setIsAuthModalOpen(true),
      closeAuthModal: () => setIsAuthModalOpen(false),
      isOnboardingOpen,
      setIsOnboardingOpen,
      handleAuthSuccess
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
