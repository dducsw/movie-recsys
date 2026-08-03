import React from 'react';
import './MovieCard.css';

function MovieCard({ movie, onClick }) {
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

  const renderRatingCircle = (score) => {
    const percentage = Math.round(score * 10);
    const size = 36;
    const strokeWidth = 2.2;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    let strokeColor = '#01d277'; // Green
    if (percentage < 70 && percentage >= 40) {
      strokeColor = '#d2d219'; // Yellow
    } else if (percentage < 40) {
      strokeColor = '#db2323'; // Red
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
    <div className="movie-card" onClick={onClick}>
      <div className="poster-container">
        <img
          className="movie-poster"
          src={
            movie.poster_url && !movie.poster_url.includes('placeholder.com')
              ? movie.poster_url
              : 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300'
          }
          alt={movie.title}
          loading="lazy"
          onError={(e) => {
            e.target.onerror = null;
            e.target.src = 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300';
          }}
        />
        <div className="movie-dots-btn">•••</div>
      </div>
      <div className="movie-info">
        <div className="rating-badge-container">
          {renderRatingCircle(movie.vote_average)}
        </div>
        <div className="movie-title">{movie.title}</div>
        <div className="movie-date">{formatDate(movie.release_date)}</div>
      </div>
    </div>
  );
}

export default MovieCard;