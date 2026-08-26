import React from 'react';
import './RightSidebar.css';

function RightSidebar({ 
  topMovies = [], 
  onSelectMovie, 
  onSeeAllTop, 
  selectedGenre, 
  onSelectGenre 
}) {
  const genresList = [
    'Action', 'Fantasy', 'Comedy', 'Sci-Fi', 
    'Drama', 'Romance', 'Mystery', 'Horror', 
    'Thriller', 'Animation', 'Crime', 'Adventure'
  ];

  const defaultTopMovies = [
    {
      movieId: 278,
      title: "The Shawshank Redemption",
      genres: "Drama",
      vote_average: 9.2,
      poster_url: "https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=200",
      rating_badge: "PG-13"
    },
    {
      movieId: 238,
      title: "The Godfather",
      genres: "Crime • Drama",
      vote_average: 9.2,
      poster_url: "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=200",
      rating_badge: "PG-13"
    },
    {
      movieId: 155,
      title: "The Dark Knight",
      genres: "Action • Crime",
      vote_average: 9.0,
      poster_url: "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?q=80&w=200",
      rating_badge: "PG-13"
    }
  ];

  const displayMovies = topMovies && topMovies.length >= 3 ? topMovies.slice(0, 3) : defaultTopMovies;

  return (
    <aside className="cinemax-right-sidebar">
      {/* Top Movies Section */}
      <div className="right-section">
        <h3 className="section-title">Top Rated Movies</h3>
        <div className="top-movies-list">
          {displayMovies.map((movie) => {
            const genresFormatted = movie.genres ? movie.genres.replace(/\|/g, ' • ').split(' • ').slice(0, 2).join(' • ') : "Drama";
            const voteScore = movie.vote_average ? Number(movie.vote_average).toFixed(1) : "9.0";
            const posterSrc = movie.poster_url && !movie.poster_url.includes('placeholder')
              ? movie.poster_url
              : "https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=200";

            return (
              <div 
                key={movie.movieId || movie.id} 
                className="top-movie-card"
                onClick={() => onSelectMovie && onSelectMovie(movie.movieId || movie.id)}
              >
                <img 
                  src={posterSrc} 
                  alt={movie.title} 
                  className="top-movie-poster"
                  loading="lazy"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = "https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=200";
                  }}
                />
                <div className="top-movie-info">
                  <span className="age-badge">PG-13</span>
                  <div className="top-movie-title" title={movie.title}>{movie.title}</div>
                  <div className="top-movie-genre">{genresFormatted}</div>
                  <div className="top-movie-rating">
                    ★ <span>{voteScore}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* See All Button */}
        <button className="btn-see-all-outline" onClick={onSeeAllTop}>
          See All Movies
        </button>
      </div>

      {/* Favorites Genres Section */}
      <div className="right-section" id="genres-section">
        <h3 className="section-title">Favorite Genres</h3>
        <div className="genre-chips-grid">
          {genresList.map((genre) => (
            <button
              key={genre}
              className={`genre-chip ${selectedGenre === genre ? 'active' : ''}`}
              onClick={() => onSelectGenre && onSelectGenre(genre)}
            >
              {genre}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}

export default RightSidebar;
