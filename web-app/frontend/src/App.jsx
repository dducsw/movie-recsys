import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentSearch, setCurrentSearch] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [activeTab, setActiveTab] = useState('today'); // 'today' or 'week'
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(false);

  // Fetch trending movies on load and when activeTab changes
  useEffect(() => {
    fetchTrending();
  }, [activeTab]);

  const fetchTrending = async () => {
    setLoading(true);
    try {
      // Giả lập lấy dữ liệu (trong CSDL thật, hôm nay/tuần này có thể lấy cùng dữ liệu trending)
      const limit = activeTab === 'today' ? 20 : 30;
      const response = await fetch(`${API_BASE_URL}/movies/trending?limit=${limit}`);
      const data = await response.json();
      setTrendingMovies(data.results || []);
    } catch (error) {
      console.error('Error fetching trending movies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/movies/search?query=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setSearchResults(data.results || []);
      setCurrentSearch(searchQuery);
      setIsSearching(true);
    } catch (error) {
      console.error('Error searching movies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setCurrentSearch('');
    setIsSearching(false);
    setSearchResults([]);
  };

  const handleMovieClick = async (movieId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/movies/${movieId}`);
      if (response.ok) {
        const data = await response.json();
        setSelectedMovie(data);
        setShowModal(true);
      } else {
        console.error('Movie detail not found');
      }
    } catch (error) {
      console.error('Error fetching movie detail:', error);
    }
  };

  // Convert date format "2025-12-17" -> "Dec 17, 2025"
  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (e) {
      return dateStr;
    }
  };

  return (
    <div className="app-container">
      {/* 1. Navbar */}
      <nav className="navbar">
        <div className="navbar-left">
          <div className="navbar-logo" onClick={handleClearSearch}>TMDB</div>
          <ul className="navbar-menu">
            <li>Movies</li>
            <li>TV Shows</li>
            <li>People</li>
            <li>Awards</li>
            <li>More</li>
          </ul>
        </div>
        <div className="navbar-right">
          <span className="nav-icon">+</span>
          <span className="nav-lang">EN</span>
          <span className="nav-icon" style={{ fontSize: '18px' }}>🔔</span>
          <div className="nav-avatar">L</div>
          <span className="nav-icon" style={{ fontSize: '18px', marginLeft: '5px' }}>🔍</span>
        </div>
      </nav>

      {/* 2. Hero Banner */}
      <header className="banner">
        <h1>Welcome.</h1>
        <h2>Millions of movies, TV shows and people to discover. Explore now.</h2>
        <form className="search-container" onSubmit={handleSearchSubmit}>
          <input
            type="text"
            className="search-input"
            placeholder="Search for a movie, tv show, person......"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <button type="submit" className="search-button">
            Search
          </button>
        </form>
      </header>

      {/* 3. Main Content Area */}
      <main className="main-content">
        {isSearching ? (
          /* Search Results View */
          <div>
            <div className="section-header">
              <div className="search-status">
                Search results for: "{currentSearch}" ({searchResults.length} results)
              </div>
              <button 
                onClick={handleClearSearch}
                style={{
                  padding: '4px 12px',
                  borderRadius: '20px',
                  border: '1px solid var(--tmdbDarkBlue)',
                  background: 'none',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '13px'
                }}
              >
                Clear Search
              </button>
            </div>
            {searchResults.length === 0 ? (
              <div className="no-results">No movies found matching your query.</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '30px 20px' }}>
                {searchResults.map((movie) => (
                  <div key={movie.movieId} className="movie-card" onClick={() => handleMovieClick(movie.movieId)}>
                    <div className="poster-container">
                      <img
                        className="movie-poster"
                        src={movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300'}
                        alt={movie.title}
                        loading="lazy"
                      />
                      <div className="movie-dots-btn">•••</div>
                    </div>
                    <div className="movie-info">
                      <div className="movie-title">{movie.title}</div>
                      <div className="movie-date">{formatDate(movie.release_date)}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          /* Normal Trending View */
          <div>
            <div className="section-header">
              <h2>Trending</h2>
              <div className="selector-tabs">
                <div 
                  className={`tab ${activeTab === 'today' ? 'active' : ''}`}
                  onClick={() => setActiveTab('today')}
                >
                  Today
                </div>
                <div 
                  className={`tab ${activeTab === 'week' ? 'active' : ''}`}
                  onClick={() => setActiveTab('week')}
                >
                  This Week
                </div>
              </div>
            </div>

            {loading && trendingMovies.length === 0 ? (
              <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--tmdbTextMuted)' }}>
                Loading movies...
              </div>
            ) : (
              <div className="horizontal-scroll">
                {trendingMovies.map((movie) => (
                  <div key={movie.movieId} className="movie-card" onClick={() => handleMovieClick(movie.movieId)}>
                    <div className="poster-container">
                      <img
                        className="movie-poster"
                        src={movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300'}
                        alt={movie.title}
                        loading="lazy"
                      />
                      <div className="movie-dots-btn">•••</div>
                    </div>
                    <div className="movie-info">
                      <div className="movie-title">{movie.title}</div>
                      <div className="movie-date">{formatDate(movie.release_date)}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* 4. Movie Detail Modal overlay */}
      {showModal && selectedMovie && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close-btn" onClick={() => setShowModal(false)}>
              ✕
            </button>
            <div className="modal-poster">
              <img
                src={selectedMovie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400'}
                alt={selectedMovie.title}
              />
            </div>
            <div className="modal-details">
              <h2 className="modal-title">{selectedMovie.title}</h2>
              <div className="modal-meta">
                <span>{formatDate(selectedMovie.release_date)}</span>
                {selectedMovie.adult && <span className="adult-tag">18+</span>}
              </div>
              <div className="modal-genres">
                {selectedMovie.genres.split('|').map((genre, i) => (
                  <span key={i} className="genre-tag">
                    {genre}
                  </span>
                ))}
              </div>
              <div className="modal-score-container">
                <div className="score-badge">
                  <div className="score-inner">
                    {Math.round(selectedMovie.vote_average * 10)}%
                  </div>
                </div>
                <span className="score-label">User Score</span>
              </div>
              <h3 className="modal-section-title">Overview</h3>
              <p className="modal-overview">{selectedMovie.overview || 'No overview available for this movie.'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
