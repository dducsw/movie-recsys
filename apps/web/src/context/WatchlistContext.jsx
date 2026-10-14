import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiFetch } from '../api/client';
import { useAuth } from './AuthContext';

const WatchlistContext = createContext();

export function WatchlistProvider({ children }) {
  const { user } = useAuth();

  const [likedMovies, setLikedMovies] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_liked');
    return saved ? JSON.parse(saved) : [];
  });

  const [userRatings, setUserRatings] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_ratings');
    return saved ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem('tmdb_recsys_liked', JSON.stringify(likedMovies));
  }, [likedMovies]);

  useEffect(() => {
    localStorage.setItem('tmdb_recsys_ratings', JSON.stringify(userRatings));
  }, [userRatings]);

  // Sync with backend if logged in
  useEffect(() => {
    if (user) {
      apiFetch('/auth/me')
        .then(data => {
          if (data && data.watchlist && Array.isArray(data.watchlist)) {
            const movieIds = data.watchlist.map(w => w.movie_id || w.movieId).filter(Boolean);
            if (movieIds.length > 0) {
              setLikedMovies(prev => Array.from(new Set([...prev, ...movieIds])));
            }
          }
          if (data && data.ratings && Array.isArray(data.ratings)) {
            const ratingsMap = {};
            data.ratings.forEach(r => {
              if (r.movie_id) ratingsMap[r.movie_id] = r.rating;
            });
            setUserRatings(prev => ({ ...prev, ...ratingsMap }));
          }
        })
        .catch(() => {});
    }
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

  const reloadPreferences = async () => {
    if (user) {
      try {
        const data = await apiFetch('/auth/me');
        if (data && data.watchlist && Array.isArray(data.watchlist)) {
          const movieIds = data.watchlist.map(w => w.movie_id || w.movieId).filter(Boolean);
          if (movieIds.length > 0) {
            setLikedMovies(prev => Array.from(new Set([...prev, ...movieIds])));
          }
        }
      } catch {
        // Ignored
      }
    }
  };

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
