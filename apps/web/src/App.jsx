import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import TopNav from './components/TopNav/TopNav';
import AuthModal from './components/AuthModal';
import OnboardingModal from './components/OnboardingModal';

// Context Providers
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import { WatchlistProvider } from './context/WatchlistContext';

// Pages
import HomePage from './pages/HomePage';
import ExplorePage from './pages/ExplorePage';
import MovieDetailPage from './pages/MovieDetailPage';
import WatchlistPage from './pages/WatchlistPage';
import ChatbotPage from './pages/ChatbotPage';

import './App.css';

function MainLayout() {
  const { isAuthModalOpen, openAuthModal, closeAuthModal, isOnboardingOpen, setIsOnboardingOpen, handleAuthSuccess } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <div className="streamix-app-layout">
      {/* Floating Transparent Streaming TopNav */}
      <TopNav
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        onOpenAuth={openAuthModal}
      />

      {/* Full-Bleed Content Viewport */}
      <main className="streamix-main-container">
        <div className="streamix-content-body">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/explore" element={<ExplorePage />} />
            <Route path="/genres" element={<ExplorePage />} />
            <Route path="/movie/:id" element={<MovieDetailPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/chatbot" element={<ChatbotPage />} />
            <Route path="/ai-assistant" element={<ChatbotPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>

      {/* Global Auth Modals */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={closeAuthModal}
        onAuthSuccess={handleAuthSuccess}
      />

      <OnboardingModal
        isOpen={isOnboardingOpen}
        onClose={() => setIsOnboardingOpen(false)}
      />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <WatchlistProvider>
            <MainLayout />
          </WatchlistProvider>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}
