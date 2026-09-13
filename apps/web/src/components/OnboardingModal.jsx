import React, { useState } from 'react';
import { Sparkles, Check, ArrowRight } from 'lucide-react';
import { API_BASE_URL } from '../api/client';
import './OnboardingModal.css';

const GENRE_OPTIONS = [
  'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
  'Drama', 'Fantasy', 'Horror', 'Mystery',
  'Romance', 'Science Fiction', 'Thriller'
];

const POPULAR_STARTER_MOVIES = [
  { movieId: 27205, title: 'Inception', year: '2010' },
  { movieId: 157336, title: 'Interstellar', year: '2014' },
  { movieId: 299536, title: 'Avengers: Infinity War', year: '2018' },
  { movieId: 155, title: 'The Dark Knight', year: '2008' },
  { movieId: 550, title: 'Fight Club', year: '1999' },
  { movieId: 680, title: 'Pulp Fiction', year: '1994' },
  { movieId: 13, title: 'Forrest Gump', year: '1994' },
  { movieId: 120, title: 'Lord of the Rings', year: '2001' }
];

export default function OnboardingModal({ isOpen, onClose, onFinish }) {
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [selectedMovies, setSelectedMovies] = useState([]);
  const [submitting, setSubmitting] = useState(false);

  if (!isOpen) return null;

  const toggleGenre = (genre) => {
    if (selectedGenres.includes(genre)) {
      setSelectedGenres(selectedGenres.filter((g) => g !== genre));
    } else {
      setSelectedGenres([...selectedGenres, genre]);
    }
  };

  const toggleMovie = (movieId) => {
    if (selectedMovies.includes(movieId)) {
      setSelectedMovies(selectedMovies.filter((id) => id !== movieId));
    } else {
      setSelectedMovies([...selectedMovies, movieId]);
    }
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    const token = localStorage.getItem('auth_token');

    try {
      const res = await fetch(`${API_BASE_URL}/users/onboarding`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          favorite_genres: selectedGenres,
          favorite_movie_ids: selectedMovies
        })
      });

      if (res.ok) {
        if (onFinish) onFinish();
        onClose();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="onboarding-overlay" onClick={onClose}>
      <div className="onboarding-card" onClick={(e) => e.stopPropagation()}>
        <div className="onboarding-header">
          <div className="onboarding-icon-badge">
            <Sparkles className="w-6 h-6" />
          </div>
          <h2 className="onboarding-title">Personalize Your AI Taste</h2>
          <p className="onboarding-subtitle">
            Select your preferred genres and starter movies to instantly personalize your recommendations feed.
          </p>
        </div>

        {/* 1. Genres */}
        <div className="onboarding-section">
          <label className="onboarding-label">Favorite Genres</label>
          <div className="onboarding-chips">
            {GENRE_OPTIONS.map((g) => {
              const active = selectedGenres.includes(g);
              return (
                <button
                  key={g}
                  type="button"
                  className={`onboarding-chip ${active ? 'active' : ''}`}
                  onClick={() => toggleGenre(g)}
                >
                  {g}
                </button>
              );
            })}
          </div>
        </div>

        {/* 2. Movies */}
        <div className="onboarding-section">
          <label className="onboarding-label">Movies You Love</label>
          <div className="onboarding-movie-grid">
            {POPULAR_STARTER_MOVIES.map((m) => {
              const active = selectedMovies.includes(m.movieId);
              return (
                <button
                  key={m.movieId}
                  type="button"
                  className={`onboarding-movie-btn ${active ? 'active' : ''}`}
                  onClick={() => toggleMovie(m.movieId)}
                >
                  <div className="onboarding-movie-info">
                    <span className="onboarding-movie-title">{m.title}</span>
                    <span className="onboarding-movie-year">{m.year}</span>
                  </div>
                  {active && <Check className="w-4 h-4 text-cyan-400 shrink-0" color="var(--accent-cyan)" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Actions */}
        <div className="onboarding-footer">
          <button
            type="button"
            className="btn-onboarding-skip"
            onClick={onClose}
          >
            Skip for now
          </button>
          <button
            type="button"
            className="btn-onboarding-submit"
            disabled={submitting}
            onClick={handleSubmit}
          >
            <span>{submitting ? 'Saving...' : 'Start Watching'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
