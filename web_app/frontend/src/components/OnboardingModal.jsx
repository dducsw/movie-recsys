import React, { useState } from 'react';

const GENRE_OPTIONS = [
  'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery',
  'Romance', 'Science Fiction', 'Thriller', 'War'
];

const POPULAR_STARTER_MOVIES = [
  { movieId: 27205, title: 'Inception', year: '2010', genres: 'Action|Sci-Fi' },
  { movieId: 157336, title: 'Interstellar', year: '2014', genres: 'Sci-Fi|Drama' },
  { movieId: 299536, title: 'Avengers: Infinity War', year: '2018', genres: 'Action|Sci-Fi' },
  { movieId: 155, title: 'The Dark Knight', year: '2008', genres: 'Action|Crime' },
  { movieId: 550, title: 'Fight Club', year: '1999', genres: 'Drama' },
  { movieId: 680, title: 'Pulp Fiction', year: '1994', genres: 'Crime' },
  { movieId: 13, title: 'Forrest Gump', year: '1994', genres: 'Drama|Romance' },
  { movieId: 120, title: 'The Lord of the Rings', year: '2001', genres: 'Adventure|Fantasy' }
];

export default function OnboardingModal({ isOpen, onClose, onComplete }) {
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
      const res = await fetch('http://localhost:8000/api/users/onboarding', {
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
        onComplete({
          favorite_genres: selectedGenres,
          favorite_movie_ids: selectedMovies
        });
        onClose();
      }
    } catch (err) {
      console.error('Failed to submit onboarding:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4">
      <div className="relative w-full max-w-2xl bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl p-6 text-white max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold text-center mb-1 text-red-500">
          Chào Mừng Bạn Đến Với MovieNex!
        </h2>
        <p className="text-sm text-gray-400 text-center mb-6">
          Hãy chọn các thể loại và phim bạn yêu thích để AI chuẩn hóa đề xuất phù hợp nhất với bạn.
        </p>

        {/* Step 1: Favorite Genres */}
        <div className="mb-6">
          <h3 className="text-md font-semibold mb-3 text-gray-200">
            1. Chọn thể loại phim bạn ưa thích:
          </h3>
          <div className="flex flex-wrap gap-2">
            {GENRE_OPTIONS.map((genre) => {
              const active = selectedGenres.includes(genre);
              return (
                <button
                  key={genre}
                  onClick={() => toggleGenre(genre)}
                  className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition border ${
                    active
                      ? 'bg-red-600 border-red-500 text-white shadow-md'
                      : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {genre}
                </button>
              );
            })}
          </div>
        </div>

        {/* Step 2: Starter Movies */}
        <div className="mb-6">
          <h3 className="text-md font-semibold mb-3 text-gray-200">
            2. Chọn một số phim nổi tiếng bạn đã từng xem & thích:
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {POPULAR_STARTER_MOVIES.map((movie) => {
              const active = selectedMovies.includes(movie.movieId);
              return (
                <div
                  key={movie.movieId}
                  onClick={() => toggleMovie(movie.movieId)}
                  className={`cursor-pointer p-3 rounded-xl border text-center transition ${
                    active
                      ? 'bg-red-900/40 border-red-500 ring-2 ring-red-500/50'
                      : 'bg-gray-800/60 border-gray-700/80 hover:border-gray-600'
                  }`}
                >
                  <p className="font-bold text-xs line-clamp-1">{movie.title}</p>
                  <p className="text-[10px] text-gray-400 mt-1">{movie.year} • {movie.genres}</p>
                </div>
              );
            })}
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={submitting}
          className="w-full py-3 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-bold rounded-xl shadow-lg transition"
        >
          {submitting ? 'Đang thiết lập...' : 'Hoàn Tất & Khai Phá Ngay'}
        </button>
      </div>
    </div>
  );
}
