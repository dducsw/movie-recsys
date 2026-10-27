import React, { Suspense, lazy, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import TopNav from './components/TopNav/TopNav';
import AuthModal from './components/AuthModal';
import OnboardingModal from './components/OnboardingModal';
import ErrorBoundary from './components/ErrorBoundary';

// Context Providers
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import { WatchlistProvider } from './context/WatchlistContext';

// Lazy pages: smaller initial bundle, faster first paint.
const HomePage = lazy(() => import('./pages/HomePage'));
const ExplorePage = lazy(() => import('./pages/ExplorePage'));
const MovieDetailPage = lazy(() => import('./pages/MovieDetailPage'));
const WatchlistPage = lazy(() => import('./pages/WatchlistPage'));
const ChatbotPage = lazy(() => import('./pages/ChatbotPage'));

import './App.css';

const PageFallback = () => <div style={{ padding: 32, color: 'var(--text-muted)' }}>Loading…</div>;

function MainLayout() {
  const { isAuthModalOpen, openAuthModal, closeAuthModal, isOnboardingOpen, setIsOnboardingOpen, handleAuthSuccess } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <div className="streamix-app-layout">
      <TopNav
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        onOpenAuth={openAuthModal}
      />

      <main className="streamix-main-container">
        <div className="streamix-content-body">
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/explore" element={<ExplorePage />} />
              <Route path="/movie/:id" element={<MovieDetailPage />} />
              <Route path="/watchlist" element={<WatchlistPage />} />
              <Route path="/chatbot" element={<ChatbotPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </div>
      </main>

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
            <ErrorBoundary>
              <MainLayout />
            </ErrorBoundary>
          </WatchlistProvider>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}
