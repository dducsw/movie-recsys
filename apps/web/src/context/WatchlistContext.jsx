import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiFetch } from '../api/client';
import { useAuth } from './AuthContext';

const WatchlistContext = createContext();

const LIKED_KEY = 'mn_liked';
const RATINGS_KEY = 'mn_ratings';

// localStorage có thể bị hỏng/thay bởi extension — fail về rỗng, không crash app.
const readJson = (key, fallback) => {
  try {
    const saved = localStorage.getItem(key);
    return saved ? JSON.parse(saved) : fallback;
  } catch {
    return fallback;
  }
};

export function WatchlistProvider({ children }) {
  const { user } = useAuth();

  const [likedMovies, setLikedMovies] = useState(() => readJson(LIKED_KEY, []));
  const [userRatings, setUserRatings] = useState(() => readJson(RATINGS_KEY, {}));

  useEffect(() => {
    localStorage.setItem(LIKED_KEY, JSON.stringify(likedMovies));
  }, [likedMovies]);

  useEffect(() => {
    localStorage.setItem(RATINGS_KEY, JSON.stringify(userRatings));
  }, [userRatings]);

  // Gộp watchlist/rating từ server vào local state (dùng khi login & sau onboarding).
  const reloadPreferences = async () => {
    if (!user) return;
    try {
      const data = await apiFetch('/auth/me');
      if (Array.isArray(data?.watchlist) && data.watchlist.length > 0) {
        const movieIds = data.watchlist.map((w) => w.movie_id || w.movieId).filter(Boolean);
        setLikedMovies((prev) => Array.from(new Set([...prev, ...movieIds])));
      }
      if (Array.isArray(data?.ratings) && data.ratings.length > 0) {
        const ratingsMap = {};
        data.ratings.forEach((r) => {
          if (r.movie_id) ratingsMap[r.movie_id] = r.rating;
        });
        setUserRatings((prev) => ({ ...prev, ...ratingsMap }));
      }
    } catch {
      // Server chưa sẵn sàng — giữ state local.
    }
  };

  // Sync 1 lần khi có user (login/refresh).
  useEffect(() => {
    reloadPreferences();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const toggleLike = (movieId) => {
    const id = Number(movieId);

    setLikedMovies(prev => {
      const isAlreadyLiked = prev.includes(id);
      const next = isAlreadyLiked ? prev.filter(mId => mId !== id) : [...prev, id];

      // Emit telemetry to backend via secure apiFetch
      apiFetch('/events/watchlist', {
        method: 'POST',
        body: JSON.stringify({ movie_id: id })
      }).catch(() => {});

      return next;
    });
  };

  const setRating = (movieId, rating) => {
    const id = Number(movieId);
    const score = Number(rating);

    setUserRatings(prev => ({
      ...prev,
      [id]: score
    }));

    apiFetch('/events/rating', {
      method: 'POST',
      body: JSON.stringify({ movie_id: id, rating: score })
    }).catch(() => {});
  };

  const isLiked = (movieId) => likedMovies.includes(Number(movieId));
  const getRating = (movieId) => userRatings[Number(movieId)] || 0;

  return (
    <WatchlistContext.Provider value={{
      likedMovies,
      userRatings,
      toggleLike,
      setRating,
      isLiked,
      getRating,
      reloadPreferences
    }}>
      {children}
    </WatchlistContext.Provider>
  );
}

export const useWatchlist = () => useContext(WatchlistContext);
