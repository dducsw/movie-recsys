import React, { useState, useEffect, useRef } from 'react';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  // Navigation & View State
  const [view, setView] = useState('home'); // 'home', 'detail', or 'all'
  const [selectedMovieId, setSelectedMovieId] = useState(null);
  const [selectedMovie, setSelectedMovie] = useState(null);

  // Lists of Movies (Homepage)
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [latestMovies, setLatestMovies] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [likedMoviesDetails, setLikedMoviesDetails] = useState([]);
  const [similarMovies, setSimilarMovies] = useState([]);

  // Swipe Row Refs for scroll buttons
  const recsScrollRef = useRef(null);
  const trendingScrollRef = useRef(null);
  const latestScrollRef = useRef(null);
  const favoritesScrollRef = useRef(null);
  const similarScrollRef = useRef(null);

  // "See All" / Paged View States
  const [allType, setAllType] = useState('trending'); // 'trending' or 'recs'
  const [allMovies, setAllMovies] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loadingAll, setLoadingAll] = useState(false);

  // Search States
  const [searchQuery, setSearchQuery] = useState('');
  const [currentSearch, setCurrentSearch] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  // Tab State (Trending Row)
  const [activeTab, setActiveTab] = useState('today'); // 'today' or 'week'

  // AI Chatbot States
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'bot',
      text: 'Xin chào! Tôi là AI Chatbot gợi ý phim. Bạn có thể hỏi tôi giới thiệu phim theo thể loại (ví dụ: "phim hành động", "phim hoạt hình") hoặc tìm phim tương tự một bộ phim bạn thích (ví dụ: "phim giống Toy Story").'
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  // Loading States
  const [loadingTrending, setLoadingTrending] = useState(false);
  const [loadingLatest, setLoadingLatest] = useState(false);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);

  // Guide Dismiss State
  const [showGuide, setShowGuide] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_hide_guide');
    return saved !== 'true';
  });

  // LocalStorage-based User State
  const [likedMovies, setLikedMovies] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_liked');
    return saved ? JSON.parse(saved) : [];
  });
  const [movieRatings, setMovieRatings] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_ratings');
    return saved ? JSON.parse(saved) : {};
  });

  // Sync Liked Movies to LocalStorage & refetch recs
  useEffect(() => {
    localStorage.setItem('tmdb_recsys_liked', JSON.stringify(likedMovies));
    fetchLikedMovieDetails();
    fetchRecommendations();
    // If viewing recommendations in "See All" view, refresh it
    if (view === 'all' && allType === 'recs') {
      fetchAllMovies('recs', 1);
    }
  }, [likedMovies]);

  // Sync Movie Ratings to LocalStorage
  useEffect(() => {
    localStorage.setItem('tmdb_recsys_ratings', JSON.stringify(movieRatings));
  }, [movieRatings]);

  // Sync Guide Dismiss State to LocalStorage
  const handleDismissGuide = () => {
    setShowGuide(false);
    localStorage.setItem('tmdb_recsys_hide_guide', 'true');
  };

  // Fetch trending movies for homepage on load and when activeTab changes
  useEffect(() => {
    fetchTrending();
  }, [activeTab]);

  // Fetch trending movies for homepage
  const fetchTrending = async () => {
    setLoadingTrending(true);
    try {
      const limit = activeTab === 'today' ? 20 : 35;
      const response = await fetch(`${API_BASE_URL}/movies/trending?limit=${limit}`);
      const data = await response.json();
      setTrendingMovies(data.results || []);
    } catch (error) {
      console.error('Error fetching trending movies:', error);
    } finally {
      setLoadingTrending(false);
    }
  };

  // Fetch latest movies on load
  useEffect(() => {
    fetchLatest();
  }, []);

  // Fetch latest movies for homepage
  const fetchLatest = async () => {
    setLoadingLatest(true);
    try {
      const response = await fetch(`${API_BASE_URL}/movies/latest?limit=20`);
      const data = await response.json();
      setLatestMovies(data.results || []);
    } catch (error) {
      console.error('Error fetching latest movies:', error);
    } finally {
      setLoadingLatest(false);
    }
  };

  // Auto-scroll chat messages
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isTyping]);

  // Handle sending chat messages
  const handleSendChatMessage = async (textToSend) => {
    const text = textToSend || chatInput;
    if (!text.trim()) return;

    if (!textToSend) {
      setChatInput('');
    }

    setChatMessages((prev) => [...prev, { sender: 'user', text }]);
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chatbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await response.json();
      
      setTimeout(() => {
        setChatMessages((prev) => [
          ...prev,
          { sender: 'bot', text: data.text, movies: data.movies || [] }
        ]);
        setIsTyping(false);
      }, 600);
    } catch (error) {
      console.error('Error in chatbot communication:', error);
      setIsTyping(false);
      setChatMessages((prev) => [
        ...prev,
        { sender: 'bot', text: 'Xin lỗi, đã xảy ra lỗi kết nối với máy chủ AI. Vui lòng thử lại sau.' }
      ]);
    }
  };

  // Fetch personalized recommendations for homepage
  const fetchRecommendations = async () => {
    setLoadingRecs(true);
    try {
      const idsParam = likedMovies.join(',');
      const url = idsParam 
        ? `${API_BASE_URL}/recommendations?movie_ids=${idsParam}`
        : `${API_BASE_URL}/recommendations`;
      const response = await fetch(url);
      const data = await response.json();
      setRecommendations(data.results || []);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoadingRecs(false);
    }
  };

  // Fetch full details of liked movies to display them on the homepage
  const fetchLikedMovieDetails = async () => {
    if (likedMovies.length === 0) {
      setLikedMoviesDetails([]);
      return;
    }
    try {
      const promises = likedMovies.map(id => 
        fetch(`${API_BASE_URL}/movies/${id}`).then(res => res.ok ? res.json() : null)
      );
      const results = await promises;
      setLikedMoviesDetails(results.filter(movie => movie !== null));
    } catch (error) {
      console.error('Error fetching liked movie details:', error);
    }
  };

  // Handle Search Submission
  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoadingTrending(true);
    try {
      const response = await fetch(`${API_BASE_URL}/movies/search?query=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setSearchResults(data.results || []);
      setCurrentSearch(searchQuery);
      setIsSearching(true);
      setView('home'); 
      setSelectedMovieId(null);
    } catch (error) {
      console.error('Error searching movies:', error);
    } finally {
      setLoadingTrending(false);
    }
  };

  // Clear Search Results
  const handleClearSearch = () => {
    setSearchQuery('');
    setCurrentSearch('');
    setIsSearching(false);
    setSearchResults([]);
  };

  // Handle click on Movie Card -> Go to Details view
  const handleMovieClick = async (movieId) => {
    setView('detail');
    setSelectedMovieId(movieId);
    setLoadingDetail(true);
    try {
      // Fetch movie detail
      const resDetail = await fetch(`${API_BASE_URL}/movies/${movieId}`);
      if (resDetail.ok) {
        const movieData = await resDetail.json();
        setSelectedMovie(movieData);
      } else {
        console.error('Failed to fetch movie details');
      }

      // Fetch similar movie recommendations
      const resSimilar = await fetch(`${API_BASE_URL}/movies/${movieId}/recommendations?limit=12`);
      if (resSimilar.ok) {
        const similarData = await resSimilar.json();
        setSimilarMovies(similarData.results || []);
      } else {
        setSimilarMovies([]);
      }
    } catch (error) {
      console.error('Error fetching movie details:', error);
    } finally {
      setLoadingDetail(false);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Toggle Like Status of a movie
  const toggleLike = (movieId) => {
    if (likedMovies.includes(movieId)) {
      setLikedMovies(prev => prev.filter(id => id !== movieId));
    } else {
      setLikedMovies(prev => [...prev, movieId]);
    }
  };

  // Handle Star Rating submission
  const handleRateMovie = (movieId, rating) => {
    setMovieRatings(prev => ({
      ...prev,
      [movieId]: prev[movieId] === rating ? 0 : rating 
    }));
  };

  // Scroll Row Helper
  const scrollRow = (ref, direction) => {
    if (ref.current) {
      const scrollAmount = direction === 'left' ? -620 : 620;
      ref.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  // "See All" Handler
  const handleSeeAll = (type) => {
    setView('all');
    setAllType(type);
    setCurrentPage(1);
    fetchAllMovies(type, 1);
  };

  // Fetch paged data for "See All"
  const fetchAllMovies = async (type, page) => {
    setLoadingAll(true);
    try {
      if (type === 'trending') {
        const limit = 20;
        const response = await fetch(`${API_BASE_URL}/movies/trending?page=${page}&limit=${limit}`);
        const data = await response.json();
        setAllMovies(data.results || []);
        setTotalCount(data.total_movies || 10000);
        setTotalPages(Math.ceil((data.total_movies || 10000) / limit));
      } else if (type === 'latest') {
        const limit = 20;
        const response = await fetch(`${API_BASE_URL}/movies/latest?page=${page}&limit=${limit}`);
        const data = await response.json();
        setAllMovies(data.results || []);
        setTotalCount(data.total_movies || 10000);
        setTotalPages(Math.ceil((data.total_movies || 10000) / limit));
      } else if (type === 'recs') {
        // Recommendations fetch all at once (up to 80 items) and paginated in-memory on frontend
        const idsParam = likedMovies.join(',');
        const url = idsParam 
          ? `${API_BASE_URL}/recommendations?movie_ids=${idsParam}`
          : `${API_BASE_URL}/recommendations`;
        
        const response = await fetch(url);
        const data = await response.json();
        const results = data.results || [];
        
        // Paginate in-memory (20 items per page)
        const limit = 20;
        const startIndex = (page - 1) * limit;
        const pagedResults = results.slice(startIndex, startIndex + limit);
        
        setAllMovies(pagedResults);
        setTotalCount(results.length);
        setTotalPages(Math.ceil(results.length / limit) || 1);
      }
    } catch (error) {
      console.error('Error fetching all movies:', error);
    } finally {
      setLoadingAll(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  // Handle page change inside "See All" view
  const handlePageChange = (page) => {
    if (page < 1 || page > totalPages) return;
    setCurrentPage(page);
    fetchAllMovies(allType, page);
  };

  // Helper: Format release date
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

  // Helper: Get Year from date string
  const getYear = (dateStr) => {
    if (!dateStr) return '';
    return dateStr.split('-')[0] || '';
  };

  // Helper: Render Circular SVG rating badge
  const renderRatingCircle = (score, isLarge = false) => {
    const percentage = Math.round(score * 10);
    const size = isLarge ? 58 : 36;
    const strokeWidth = isLarge ? 3.2 : 2.2;
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
      <div className={isLarge ? "detail-score-circle" : "rating-circle"}>
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
        <div className="rating-percent" style={{ fontSize: isLarge ? '15px' : '10.5px' }}>
          {percentage}
          <span style={{ fontSize: isLarge ? '8px' : '5px' }}>%</span>
        </div>
      </div>
    );
  };

  // Render modern pagination buttons for "See All" view
  const renderPagination = () => {
    const pages = [];
    
    // Previous Button
    pages.push(
      <button 
        key="prev" 
        className="pagination-btn"
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage === 1}
      >
        ‹
      </button>
    );

    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(
          <button 
            key={i} 
            className={`pagination-btn ${currentPage === i ? 'active' : ''}`}
            onClick={() => handlePageChange(i)}
          >
            {i}
          </button>
        );
      }
    } else {
      // First page is always visible
      pages.push(
        <button 
          key={1} 
          className={`pagination-btn ${currentPage === 1 ? 'active' : ''}`}
          onClick={() => handlePageChange(1)}
        >
          1
        </button>
      );

      if (currentPage > 3) {
        pages.push(<span key="ell1" className="pagination-ellipsis">...</span>);
      }

      // Middle pages (around current page)
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      for (let i = start; i <= end; i++) {
        pages.push(
          <button 
            key={i} 
            className={`pagination-btn ${currentPage === i ? 'active' : ''}`}
            onClick={() => handlePageChange(i)}
          >
            {i}
          </button>
        );
      }

      if (currentPage < totalPages - 2) {
        pages.push(<span key="ell2" className="pagination-ellipsis">...</span>);
      }

      // Last page is always visible
      pages.push(
        <button 
          key={totalPages} 
          className={`pagination-btn ${currentPage === totalPages ? 'active' : ''}`}
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </button>
      );
    }

    // Next Button
    pages.push(
      <button 
        key="next" 
        className="pagination-btn"
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
      >
        ›
      </button>
    );

    return <div className="pagination-container">{pages}</div>;
  };

  return (
    <div className="app-container">
      {/* 1. Header/Navbar */}
      <nav className="navbar">
        <div className="navbar-left">
          <div className="navbar-logo" onClick={() => { setView('home'); handleClearSearch(); setSelectedMovieId(null); }}>
            TMDB <span>RecSys</span>
          </div>
          <ul className="navbar-menu">
            <li style={view === 'home' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} onClick={() => { setView('home'); handleClearSearch(); setSelectedMovieId(null); }}>Home</li>
            <li style={view === 'all' && allType === 'trending' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} onClick={() => handleSeeAll('trending')}>Movies</li>
            <li style={view === 'chatbot' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} onClick={() => setView('chatbot')}>AI Chatbot</li>
            <li style={view === 'all' && allType === 'recs' ? { color: 'var(--tmdbLightBlue)', opacity: 1 } : {}} onClick={() => handleSeeAll('recs')}>RecSys</li>
          </ul>
        </div>
        <div className="navbar-right">
          <span className="nav-icon" style={{ fontSize: '18px' }} onClick={() => alert("System Status: Recommendation System is active and running!")}>🔔</span>
          <span className="nav-lang">EN</span>
          <div className="nav-avatar">
            <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: '100%', height: '100%', opacity: 0.9 }}>
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
            </svg>
          </div>
          {isSearching && (
            <button className="search-clear-btn" style={{ padding: '2px 10px', fontSize: '12px' }} onClick={handleClearSearch}>
              Close Search
            </button>
          )}
        </div>
      </nav>

      {view === 'home' ? (
        /* ================= HOMEPAGE VIEW ================= */
        <div>
          {/* Hero Banner Section */}
          <header className="banner">
            <h1>Welcome.</h1>
            <h2>Millions of movies, TV shows and recommendation algorithms to discover. Explore now.</h2>
            <form className="search-container" onSubmit={handleSearchSubmit}>
              <input
                type="text"
                className="search-input"
                placeholder="Search for a movie, tv show, person..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <button type="submit" className="search-button">
                Search
              </button>
            </form>
          </header>

          <main className="main-content">
            {isSearching ? (
              /* Search Results Grid */
              <div className="section-wrapper">
                <div className="search-status-wrapper">
                  <div className="search-status">
                    Search results for: "{currentSearch}" ({searchResults.length} movies)
                  </div>
                  <button className="search-clear-btn" onClick={handleClearSearch}>
                    Clear Search
                  </button>
                </div>

                {searchResults.length === 0 ? (
                  <div className="no-results">No movies found matching your query.</div>
                ) : (
                  <div className="search-grid">
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
                          <div className="rating-badge-container">
                            {renderRatingCircle(movie.vote_average)}
                          </div>
                          <div className="movie-title">{movie.title}</div>
                          <div className="movie-date">{formatDate(movie.release_date)}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              /* Standard Homepage Rows */
              <div>
                {/* Glassmorphism Information Alert Guide */}
                {showGuide && (
                  <div className="info-alert-box">
                    <div>
                      <strong>💡 Recommendation System Demo:</strong> This interface simulates the TMDB movie platform.
                      The <strong>"Recommended for You"</strong> row is dynamically computed in real-time using a <em>Content-based Filtering</em> algorithm on the Backend. 
                      Click on any movie card, then click the <strong>❤️ Like</strong> button or select star ratings to teach the system your preferences. 
                      When you return to the Homepage, your recommendations will automatically refresh to reflect your taste!
                    </div>
                    <button className="info-alert-close-btn" onClick={handleDismissGuide} title="Dismiss guide">
                      ✕
                    </button>
                  </div>
                )}

                {/* Row 1: Personalized Recommendations (RecSys) */}
                <div className="section-wrapper">
                  <div className="section-header">
                    <h2>Recommended for You</h2>
                    <span style={{ fontSize: '11px', background: 'rgba(1, 180, 228, 0.1)', color: 'var(--tmdbLightBlue)', border: '1px solid rgba(1, 180, 228, 0.2)', padding: '2px 10px', borderRadius: '12px', fontWeight: 700, marginRight: '10px' }}>
                      RecSys Active
                    </span>
                    {recommendations.length > 0 && (
                      <span className="see-all-link" onClick={() => handleSeeAll('recs')}>See All</span>
                    )}
                  </div>

                  {loadingRecs ? (
                    <div className="spinner-container"><div className="spinner"></div></div>
                  ) : recommendations.length === 0 ? (
                    <div className="no-results" style={{ padding: '30px', background: 'rgba(0,0,0,0.01)', borderRadius: '10px' }}>
                      Like some movies in the Trending section below to start building your personalized recommendation profile!
                    </div>
                  ) : (
                    <div className="scroll-row-wrapper">
                      {/* Swipe Left Arrow */}
                      <button className="scroll-arrow-btn left" onClick={() => scrollRow(recsScrollRef, 'left')}>‹</button>
                      
                      <div className="horizontal-scroll" ref={recsScrollRef}>
                        {recommendations.map((movie) => (
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
                              <div className="rating-badge-container">
                                {renderRatingCircle(movie.vote_average)}
                              </div>
                              <div className="movie-title">{movie.title}</div>
                              <div className="movie-date">{formatDate(movie.release_date)}</div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Swipe Right Arrow */}
                      <button className="scroll-arrow-btn right" onClick={() => scrollRow(recsScrollRef, 'right')}>›</button>
                    </div>
                  )}
                </div>

                {/* Row 2: Trending Row */}
                <div className="section-wrapper">
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
                    <span className="see-all-link" onClick={() => handleSeeAll('trending')}>See All</span>
                  </div>

                  {loadingTrending && trendingMovies.length === 0 ? (
                    <div className="spinner-container"><div className="spinner"></div></div>
                  ) : (
                    <div className="scroll-row-wrapper">
                      {/* Swipe Left Arrow */}
                      <button className="scroll-arrow-btn left" onClick={() => scrollRow(trendingScrollRef, 'left')}>‹</button>
                      
                      <div className="horizontal-scroll" ref={trendingScrollRef}>
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
                              <div className="rating-badge-container">
                                {renderRatingCircle(movie.vote_average)}
                              </div>
                              <div className="movie-title">{movie.title}</div>
                              <div className="movie-date">{formatDate(movie.release_date)}</div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Swipe Right Arrow */}
                      <button className="scroll-arrow-btn right" onClick={() => scrollRow(trendingScrollRef, 'right')}>›</button>
                    </div>
                  )}
                </div>

                {/* Row 3: Latest Row */}
                <div className="section-wrapper">
                  <div className="section-header">
                    <h2>Latest</h2>
                    <span className="see-all-link" onClick={() => handleSeeAll('latest')}>See All</span>
                  </div>

                  {loadingLatest && latestMovies.length === 0 ? (
                    <div className="spinner-container"><div className="spinner"></div></div>
                  ) : (
                    <div className="scroll-row-wrapper">
                      {/* Swipe Left Arrow */}
                      <button className="scroll-arrow-btn left" onClick={() => scrollRow(latestScrollRef, 'left')}>‹</button>
                      
                      <div className="horizontal-scroll" ref={latestScrollRef}>
                        {latestMovies.map((movie) => (
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
                              <div className="rating-badge-container">
                                {renderRatingCircle(movie.vote_average)}
                              </div>
                              <div className="movie-title">{movie.title}</div>
                              <div className="movie-date">{formatDate(movie.release_date)}</div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Swipe Right Arrow */}
                      <button className="scroll-arrow-btn right" onClick={() => scrollRow(latestScrollRef, 'right')}>›</button>
                    </div>
                  )}
                </div>

                {/* Row 3: Liked Movies (Interactive History) */}
                {likedMoviesDetails.length > 0 && (
                  <div className="section-wrapper" style={{ borderTop: '1px solid #f1f5f9', paddingTop: '24px' }}>
                    <div className="section-header">
                      <h2>Your Favorite Movies ({likedMoviesDetails.length})</h2>
                      <button 
                        onClick={() => { if(confirm("Reset all liked history?")) setLikedMovies([]); }}
                        className="search-clear-btn" 
                        style={{ padding: '4px 14px', fontSize: '12px', marginLeft: 'auto' }}
                      >
                        Reset History
                      </button>
                    </div>
                    <div className="scroll-row-wrapper">
                      {/* Swipe Left Arrow */}
                      <button className="scroll-arrow-btn left" onClick={() => scrollRow(favoritesScrollRef, 'left')}>‹</button>

                      <div className="horizontal-scroll" ref={favoritesScrollRef}>
                        {likedMoviesDetails.map((movie) => (
                          <div key={movie.movieId} className="movie-card" onClick={() => handleMovieClick(movie.movieId)}>
                            <div className="poster-container">
                              <img
                                className="movie-poster"
                                src={movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300'}
                                alt={movie.title}
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
                        ))}
                      </div>

                      {/* Swipe Right Arrow */}
                      <button className="scroll-arrow-btn right" onClick={() => scrollRow(favoritesScrollRef, 'right')}>›</button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      ) : view === 'detail' ? (
        /* ================= MOVIE DETAIL PAGE VIEW ================= */
        <div className="detail-page-container">
          <div className="back-btn-container">
            <button className="back-btn" onClick={() => setView('home')}>
              ← Back to Homepage
            </button>
          </div>

          {loadingDetail || !selectedMovie ? (
            <div className="spinner-container" style={{ minHeight: '450px' }}>
              <div className="spinner"></div>
            </div>
          ) : (
            <div>
              {/* Hero Banner with Blur Background Overlay */}
              <div
                className="detail-hero-section"
                style={{
                  backgroundImage: `url(${selectedMovie.poster_url || 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1920'})`
                }}
              >
                <div className="detail-hero-overlay">
                  <div className="detail-hero-content">
                    {/* Movie Poster */}
                    <div className="detail-poster-wrapper">
                      <img
                        className="detail-poster"
                        src={selectedMovie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=400'}
                        alt={selectedMovie.title}
                      />
                    </div>

                    {/* Movie Details */}
                    <div className="detail-info-wrapper">
                      <h1 className="detail-title">
                        {selectedMovie.title} {selectedMovie.release_date && <span>({getYear(selectedMovie.release_date)})</span>}
                      </h1>

                      <div className="detail-meta-row">
                        {selectedMovie.adult && <span className="detail-adult-badge">18+</span>}
                        <span className="detail-meta-item">{formatDate(selectedMovie.release_date)} (US)</span>
                        <span className="detail-meta-item">• 2h 15m</span>
                        <div className="detail-genres-container">
                          {selectedMovie.genres && selectedMovie.genres.split('|').map((genre, i) => (
                            <span key={i} className="detail-genre-tag">
                              {genre}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Score & Actions Row */}
                      <div className="detail-actions-row">
                        {/* Circular Score Badge */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                          {renderRatingCircle(selectedMovie.vote_average, true)}
                          <div className="detail-score-text">User<br />Score</div>
                        </div>

                        {/* Watchlist/Favorite Buttons */}
                        <button
                          className={`circle-action-btn ${likedMovies.includes(selectedMovie.movieId) ? 'liked' : ''}`}
                          onClick={() => toggleLike(selectedMovie.movieId)}
                          title={likedMovies.includes(selectedMovie.movieId) ? "Unlike" : "Mark as Favorite"}
                        >
                          ❤️
                        </button>

                        <button className="circle-action-btn" title="Add to Watchlist" onClick={() => alert("Added to watchlist (Mockup)!")}>
                          🔖
                        </button>

                        {/* Interactive Ratings Bar */}
                        <div className="rate-wrapper">
                          <span className="rate-label">Rate:</span>
                          <div className="stars-container">
                            {[1, 2, 3, 4, 5].map((star) => (
                              <button
                                key={star}
                                className={`star-btn ${movieRatings[selectedMovie.movieId] >= star ? 'active' : ''}`}
                                onClick={() => handleRateMovie(selectedMovie.movieId, star)}
                                title={`Rate ${star} stars`}
                              >
                                ★
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Overview & Crew */}
                      <p className="tagline">Your feedback updates the recommendation algorithms instantly...</p>
                      <h3 className="overview-header">Overview</h3>
                      <p className="overview-text">{selectedMovie.overview || 'No overview available for this movie.'}</p>

                      <div className="crew-container">
                        <div className="crew-member">
                          <span className="crew-name">Christopher Nolan</span>
                          <span className="crew-job">Director, Screenplay</span>
                        </div>
                        <div className="crew-member">
                          <span className="crew-name">Hans Zimmer</span>
                          <span className="crew-job">Composer</span>
                        </div>
                        <div className="crew-member">
                          <span className="crew-name">Emma Thomas</span>
                          <span className="crew-job">Producer</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Similar Recommendations Section (Recsys) */}
              <div className="detail-recs-wrapper">
                <h2>Recommendations</h2>
                <div className="info-alert-box" style={{ background: 'rgba(0,0,0,0.01)', borderLeft: 'none' }}>
                  <span>
                    <strong>🎯 Item-Based Similarities:</strong> The recommended movies below are calculated on-the-fly based on genre overlaps and popularity relative to the current movie. Click any movie to view its detail page!
                  </span>
                </div>

                {similarMovies.length === 0 ? (
                  <div className="no-results">No similar movies found.</div>
                ) : (
                  <div className="scroll-row-wrapper">
                    {/* Swipe Left Arrow */}
                    <button className="scroll-arrow-btn left" onClick={() => scrollRow(similarScrollRef, 'left')}>‹</button>

                    <div className="horizontal-scroll" ref={similarScrollRef}>
                      {similarMovies.map((movie) => (
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
                            <div className="rating-badge-container">
                              {renderRatingCircle(movie.vote_average)}
                            </div>
                            <div className="movie-title">{movie.title}</div>
                            <div className="movie-date">{formatDate(movie.release_date)}</div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Swipe Right Arrow */}
                    <button className="scroll-arrow-btn right" onClick={() => scrollRow(similarScrollRef, 'right')}>›</button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      ) : view === 'chatbot' ? (
        /* ================= AI CHATBOT VIEW ================= */
        <div className="chatbot-container">
          <div className="chatbot-header">
            <h1>AI Movie Recommender</h1>
            <p>Trò chuyện với AI để nhận gợi ý phim theo sở thích cá nhân của bạn</p>
          </div>

          <div className="chatbot-chatbox">
            <div className="chatbot-messages">
              {chatMessages.map((msg, index) => (
                <div key={index} className={`chat-bubble ${msg.sender}`}>
                  <p>{msg.text}</p>
                  
                  {msg.movies && msg.movies.length > 0 && (
                    <div className="chat-movie-list">
                      {msg.movies.map((movie) => (
                        <div 
                          key={movie.movieId} 
                          className="chat-movie-card"
                          onClick={() => handleMovieClick(movie.movieId)}
                          title={`Xem chi tiết ${movie.title}`}
                        >
                          <img 
                            className="chat-movie-poster" 
                            src={movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=150'} 
                            alt={movie.title} 
                          />
                          <div className="chat-movie-info">
                            <div className="chat-movie-title">{movie.title}</div>
                            <div className="chat-movie-date">{getYear(movie.release_date)}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              
              {isTyping && (
                <div className="chatbot-typing">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="chatbot-input-container">
              <input
                type="text"
                className="chatbot-input"
                placeholder="Nhập yêu cầu của bạn (ví dụ: phim hoạt hình lãng mạn, phim giống Toy Story...)"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendChatMessage()}
              />
              <button className="chatbot-send-btn" onClick={() => handleSendChatMessage()}>
                Gửi
              </button>
            </div>
          </div>

          <div className="chatbot-chips">
            <span className="chatbot-chip" onClick={() => handleSendChatMessage("Gợi ý phim hành động viễn tưởng")}>
              🍿 Phim hành động viễn tưởng
            </span>
            <span className="chatbot-chip" onClick={() => handleSendChatMessage("Tôi muốn xem phim hoạt hình gia đình")}>
              👶 Phim hoạt hình gia đình
            </span>
            <span className="chatbot-chip" onClick={() => handleSendChatMessage("Tìm phim giống như Toy Story")}>
              🧸 Phim giống Toy Story
            </span>
            <span className="chatbot-chip" onClick={() => handleSendChatMessage("Gợi ý phim kinh dị kịch tính")}>
              👻 Phim kinh dị kịch tính
            </span>
          </div>
        </div>
      ) : (
        /* ================= PAGINATED GRID VIEW ("SEE ALL") ================= */
        <div className="detail-page-container" style={{ paddingBottom: '80px' }}>
          <div className="back-btn-container">
            <button className="back-btn" onClick={() => setView('home')}>
              ← Back to Homepage
            </button>
          </div>

          <main className="main-content">
            <div className="section-wrapper">
              <div className="search-status-wrapper">
                <div className="search-status">
                  {allType === 'trending' ? 'All Trending Movies' : allType === 'latest' ? 'All Latest Movies' : 'All Personalized Recommendations'}
                  <span style={{ fontSize: '14px', color: 'var(--textSecondary)', fontWeight: 500, marginLeft: '12px' }}>
                    ({totalCount} movies total)
                  </span>
                </div>
              </div>

              {loadingAll ? (
                <div className="spinner-container" style={{ minHeight: '300px' }}><div className="spinner"></div></div>
              ) : allMovies.length === 0 ? (
                <div className="no-results">No movies found.</div>
              ) : (
                <div>
                  <div className="search-grid">
                    {allMovies.map((movie) => (
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
                          <div className="rating-badge-container">
                            {renderRatingCircle(movie.vote_average)}
                          </div>
                          <div className="movie-title">{movie.title}</div>
                          <div className="movie-date">{formatDate(movie.release_date)}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Rendering Pagination Bar */}
                  {totalPages > 1 && renderPagination()}
                </div>
              )}
            </div>
          </main>
        </div>
      )}
    </div>
  );
}

export default App;
