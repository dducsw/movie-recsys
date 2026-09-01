import React, { createContext, useContext, useState, useEffect } from 'react';
import { API_BASE_URL, getOrCreateSessionId } from '../api/client';
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
    const token = localStorage.getItem('auth_token');
    if (token && user) {
      fetch(`${API_BASE_URL}/auth/me/preferences`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
        .then(res => (res.ok ? res.json() : null))
        .then(data => {
          if (data && Array.isArray(data.favorite_movie_ids)) {
            setLikedMovies(prev => {
              const merged = Array.from(new Set([...prev, ...data.favorite_movie_ids]));
              return merged;
            });
          }
        })
        .catch(() => {});
    }
  }, [user]);

  const toggleLike = (movieId, movieData = null) => {
    const id = Number(movieId);
    const token = localStorage.getItem('auth_token');

    setLikedMovies(prev => {
      const isAlreadyLiked = prev.includes(id);
      const next = isAlreadyLiked ? prev.filter(mId => mId !== id) : [...prev, id];

      // Emit telemetry to backend
      fetch(`${API_BASE_URL}/events/${isAlreadyLiked ? 'unlike' : 'like'}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-Id': getOrCreateSessionId(),
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        credentials: 'include',
        body: JSON.stringify({ movie_id: id })
      }).catch(() => {});

      return next;
    });
  };

  const setRating = (movieId, rating) => {
    const id = Number(movieId);
    const score = Number(rating);
    const token = localStorage.getItem('auth_token');

    setUserRatings(prev => ({
      ...prev,
      [id]: score
    }));

    fetch(`${API_BASE_URL}/events/rating`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-Id': getOrCreateSessionId(),
        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
      },
      credentials: 'include',
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
      getRating
    }}>
      {children}
    </WatchlistContext.Provider>
  );
}

export const useWatchlist = () => useContext(WatchlistContext);
