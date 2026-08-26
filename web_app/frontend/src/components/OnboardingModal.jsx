import React, { useState } from 'react';
import { API_BASE_URL } from '../api/client';
import './OnboardingModal.css';

const GENRE_OPTIONS = [
  'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery',
  'Romance', 'Science Fiction', 'Thriller'
];

const POPULAR_STARTER_MOVIES = [
  { movieId: 27205, title: 'Inception', year: '2010' },
  { movieId: 157336, title: 'Interstellar', year: '2014' },
  { movieId: 299536, title: 'Avengers', year: '2018' },
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
        onFinish();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSubmitting(false);
      onClose();
    }
  };

  return (
    <div className="onboard-modal-overlay">
      <div className="onboard-modal-card">
        <h2 className="onboard-title">Welcome to MovieNex! 🎉</h2>
        <p className="onboard-subtitle">
          Select at least 3 favorite genres and a few movies you love so our AI recommendation engine can tailor your profile.
        </p>

        <div>
          <h3 className="onboard-section-heading">1. Favorite Movie Genres:</h3>
          <div className="onboard-genres-grid">
            {GENRE_OPTIONS.map((genre) => (
              <button
                key={genre}
                onClick={() => toggleGenre(genre)}
                className={`onboard-genre-btn ${selectedGenres.includes(genre) ? 'selected' : ''}`}
              >
                {genre}
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="onboard-section-heading">2. Movies You've Enjoyed:</h3>
          <div className="onboard-movies-grid">
            {POPULAR_STARTER_MOVIES.map((movie) => (
              <div
                key={movie.movieId}
                onClick={() => toggleMovie(movie.movieId)}
                className={`onboard-movie-chip ${selectedMovies.includes(movie.movieId) ? 'selected' : ''}`}
              >
                <div className="onboard-m-title">{movie.title}</div>
                <div className="onboard-m-year">{movie.year}</div>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={submitting || (selectedGenres.length === 0 && selectedMovies.length === 0)}
          className="btn-onboard-submit"
        >
          {submitting ? 'Setting up...' : 'Complete & Explore Recommendations'}
        </button>
      </div>
    </div>
  );
}
