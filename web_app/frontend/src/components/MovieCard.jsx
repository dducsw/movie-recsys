import React, { useState } from 'react';

function MovieCard({ movie, source = 'card', position, onClick, onRate, onWatchlistToggle, isInWatchlist = false }) {
  const [userRating, setUserRating] = useState(movie.user_rating || 0);
  const [inWatchlist, setInWatchlist] = useState(isInWatchlist);
  const [hoverStar, setHoverStar] = useState(0);

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    try {
      const date = new Date(dateStr);
      if (isNaN(date.getTime())) return dateStr;
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (e) {
      return dateStr;
    }
  };

  const handleStarClick = (e, star) => {
    e.stopPropagation();
    setUserRating(star);
    if (onRate) onRate(movie.movieId, star);

    const token = localStorage.getItem('auth_token');
    fetch('http://localhost:8000/api/events/rating', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
      },
      body: JSON.stringify({ movie_id: movie.movieId, rating: star })
    }).catch(err => console.warn('Rating log error:', err));
  };

  const handleWatchlistClick = (e) => {
    e.stopPropagation();
    const nextState = !inWatchlist;
    setInWatchlist(nextState);
    if (onWatchlistToggle) onWatchlistToggle(movie.movieId, nextState);

    const token = localStorage.getItem('auth_token');
    fetch('http://localhost:8000/api/events/watchlist', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
      },
      body: JSON.stringify({ movie_id: movie.movieId })
    }).catch(err => console.warn('Watchlist log error:', err));
  };

  const renderRatingCircle = (score) => {
    const percentage = Math.round((score || 0) * 10);
    const size = 36;
    const strokeWidth = 2.2;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    let strokeColor = '#01d277';
    if (percentage < 70 && percentage >= 40) {
      strokeColor = '#d2d219';
    } else if (percentage < 40) {
      strokeColor = '#db2323';
    }

    return (
      <div className="rating-circle">
        <svg width={size} height={size}>
          <circle
            className="circle-bg"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
          />
          <circle
            className="circle-progress"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            stroke={strokeColor}
          />
        </svg>
        <div className="rating-percent" style={{ fontSize: '10.5px' }}>
          {percentage}
          <span style={{ fontSize: '5px' }}>%</span>
        </div>
      </div>
    );
  };

  return (
    <div
      className="movie-card group relative"
      onClick={onClick}
      data-movie-id={movie.movieId}
      data-source={source}
      data-position={position}
    >
      <div className="poster-container relative overflow-hidden rounded-xl">
        <img
          className="movie-poster w-full object-cover transition-transform duration-300 group-hover:scale-105"
          src={movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300'}
          alt={movie.title}
          loading="lazy"
        />

        {/* Watchlist Bookmark Toggle Button */}
        <button
          onClick={handleWatchlistClick}
          title={inWatchlist ? "Xóa khỏi Danh sách" : "Lưu vào Danh sách xem"}
          className={`absolute top-2 right-2 p-1.5 rounded-full backdrop-blur-md transition ${
            inWatchlist
              ? 'bg-red-600 text-white shadow-lg'
              : 'bg-black/60 text-gray-300 hover:text-white hover:bg-black/80'
          }`}
        >
          <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
            <path d="M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2z" />
          </svg>
        </button>
      </div>

      <div className="movie-info mt-2">
        <div className="rating-badge-container">
          {renderRatingCircle(movie.vote_average)}
        </div>
        <div className="movie-title font-semibold text-sm line-clamp-1">{movie.title}</div>
        <div className="movie-date text-xs text-gray-400">{formatDate(movie.release_date)}</div>

        {/* Star Rating Control (1-5 stars) */}
        <div className="flex items-center gap-0.5 mt-1.5" onClick={(e) => e.stopPropagation()}>
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onMouseEnter={() => setHoverStar(star)}
              onMouseLeave={() => setHoverStar(0)}
              onClick={(e) => handleStarClick(e, star)}
              className="text-xs focus:outline-none transition transform hover:scale-125"
            >
              <span className={(hoverStar || userRating) >= star ? 'text-yellow-400' : 'text-gray-600'}>
                ★
              </span>
            </button>
          ))}
          {userRating > 0 && (
            <span className="text-[10px] text-yellow-400 font-bold ml-1">{userRating}★</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default MovieCard;
