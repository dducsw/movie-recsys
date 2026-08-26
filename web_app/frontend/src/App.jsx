import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar/Sidebar';
import TopNav from './components/TopNav/TopNav';
import HeroBanner from './components/HeroBanner/HeroBanner';
import GenreFilterBar from './components/GenreFilterBar/GenreFilterBar';
import MovieRow from './components/MovieRow/MovieRow';
import MovieCard from './components/MovieCard/MovieCard';
import ChatbotView from './components/ChatbotView/ChatbotView';
import WatchView from './components/WatchView/WatchView';
import AuthModal from './components/AuthModal';
import OnboardingModal from './components/OnboardingModal';
import { API_BASE_URL, getOrCreateSessionId, trackClick } from './api/client';
import './App.css';

function App() {
  // Theme State (Dark mode default)
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('cinemax_dark_mode');
    return saved !== null ? JSON.parse(saved) : true;
  });

  useEffect(() => {
    localStorage.setItem('cinemax_dark_mode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
    }
  }, [darkMode]);

  // Navigation & View State
  const [view, setView] = useState('home'); // 'home', 'all', 'watchlist', 'chatbot', 'watch'
  const [selectedMovieId, setSelectedMovieId] = useState(null);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [activeGenreFilter, setActiveGenreFilter] = useState('Trending');

  // Search and Advanced Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('all'); // 'all', 'released', 'upcoming'
  const [selectedYear, setSelectedYear] = useState('all'); // 'all', '2026', '2025', '2024', '2020-2023', '2010s', '2000s', 'classic'
  const [selectedGenre, setSelectedGenre] = useState('all'); // 'all', 'Action', ...
  const [searchResults, setSearchResults] = useState([]);

  // Hero carousel slide index
  const [heroIndex, setHeroIndex] = useState(0);

  // Active User & Auth Modals state
  const [user, setUser] = useState(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      fetch(`${API_BASE_URL}/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data && data.user) {
            setUser(data.user);
          } else {
            localStorage.removeItem('auth_token');
            setUser(null);
          }
        })
        .catch(() => {});
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_info');
    setUser(null);
    setView('home');
  };

  const handleAuthSuccess = (userData, isNewUser) => {
    setUser(userData);
    if (isNewUser) {
      setIsOnboardingOpen(true);
    }
  };

  // Movie Datasets
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [latestMovies, setLatestMovies] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [likedMoviesDetails, setLikedMoviesDetails] = useState([]);

  // "See All" / Paged View States
  const [allType, setAllType] = useState('trending'); // 'trending', 'latest', 'recs', 'genre', 'search'
  const [allMovies, setAllMovies] = useState([]);

  // Watchlist (Liked Movies) & Star Ratings State
  const [likedMovies, setLikedMovies] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_liked');
    return saved ? JSON.parse(saved) : [];
  });

  const [movieRatings, setMovieRatings] = useState(() => {
    const saved = localStorage.getItem('tmdb_recsys_ratings');
    return saved ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem('tmdb_recsys_liked', JSON.stringify(likedMovies));
    fetchRecommendations();
    fetchLikedMovieDetails();
  }, [likedMovies]);

  useEffect(() => {
    localStorage.setItem('tmdb_recsys_ratings', JSON.stringify(movieRatings));
  }, [movieRatings]);

  // AI Chatbot States & Logic
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'bot',
      text: "Hello! 👋 I'm your MovieNex AI assistant. Ask me anything about movies, actors, plots, or personalized suggestions!\n\nTry asking:\n• \"Recommend top sci-fi thriller movies\"\n• \"Mind-bending movies like Inception\"\n• \"What are the latest trending action releases?\""
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  const [chatSessionId, setChatSessionId] = useState(() => {
    return localStorage.getItem('movienex_chat_session_id') || getOrCreateSessionId();
  });

  const handleSendChatMessage = async (textToSend) => {
    const text = textToSend || chatInput;
    if (!text || !text.trim()) return;

    if (!textToSend) {
      setChatInput('');
    }

    setChatMessages((prev) => [...prev, { sender: 'user', text }]);
    setIsTyping(true);

    try {
      const res = await fetch(`${API_BASE_URL}/chatbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: chatSessionId
        })
      });
      const data = await res.json();

      if (data.session_id && data.session_id !== chatSessionId) {
        setChatSessionId(data.session_id);
        localStorage.setItem('movienex_chat_session_id', data.session_id);
      }

      setTimeout(() => {
        setChatMessages((prev) => [
          ...prev,
          {
            sender: 'bot',
            text: data.text || "Here are some top recommendations for you:",
            movies: data.movies || []
          }
        ]);
        setIsTyping(false);
      }, 500);
    } catch (err) {
      console.error('Chatbot error:', err);
      setIsTyping(false);
      setChatMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: "I had trouble connecting to the AI recommendation engine. Please try again!"
        }
      ]);
    }
  };

  const handleClearHistory = async () => {
    try {
      await fetch(`${API_BASE_URL}/chatbot/history?session_id=${chatSessionId}`, {
        method: 'DELETE'
      });
    } catch (e) {}
    setChatMessages([
      {
        sender: 'bot',
        text: 'Chat history cleared. What kind of movie would you like to explore today?'
      }
    ]);
  };

  const handleNewSession = () => {
    const newSid = crypto.randomUUID ? crypto.randomUUID() : `sid_${Date.now()}`;
    setChatSessionId(newSid);
    localStorage.setItem('movienex_chat_session_id', newSid);
    setChatMessages([
      {
        sender: 'bot',
        text: 'New session started! How can I help you find your next favorite movie?'
      }
    ]);
  };

  // Initial Data Fetching
  useEffect(() => {
    fetchTrending();
    fetchLatest();
    fetchRecommendations();
  }, []);

  const fetchTrending = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/movies/trending?limit=20`);
      const data = await res.json();
      setTrendingMovies(data.results || []);
    } catch (e) {
      console.error('Error fetching trending:', e);
    }
  };

  const fetchLatest = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/movies/latest?limit=20`);
      const data = await res.json();
      setLatestMovies(data.results || []);
    } catch (e) {
      console.error('Error fetching latest:', e);
    }
  };

  const fetchRecommendations = async () => {
    try {
      const likedStr = likedMovies.length > 0 ? likedMovies.join(',') : '';
      const url = likedStr
        ? `${API_BASE_URL}/recommendations?liked_movie_ids=${likedStr}&limit=16`
        : `${API_BASE_URL}/recommendations?limit=16`;
      const res = await fetch(url);
      const data = await res.json();
      setRecommendations(data.results || []);
    } catch (e) {
      console.error('Error fetching recommendations:', e);
    }
  };

  const fetchLikedMovieDetails = async () => {
    if (likedMovies.length === 0) {
      setLikedMoviesDetails([]);
      return;
    }
    try {
      const promises = likedMovies.slice(-20).map((id) =>
        fetch(`${API_BASE_URL}/movies/${id}`).then((r) => (r.ok ? r.json() : null))
      );
      const results = await Promise.all(promises);
      setLikedMoviesDetails(results.filter(Boolean));
    } catch (e) {
      console.error('Error fetching liked details:', e);
    }
  };

  const fetchMovieDetail = async (movieId) => {
    setSelectedMovieId(movieId);
    try {
      const res = await fetch(`${API_BASE_URL}/movies/${movieId}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedMovie(data);
      }
      const simRes = await fetch(`${API_BASE_URL}/movies/${movieId}/recommendations?limit=10`);
      if (simRes.ok) {
        const simData = await simRes.json();
        setSimilarMovies(simData.results || []);
      }
      trackClick(movieId, 'detail_view');
    } catch (e) {
      console.error('Error fetching movie detail:', e);
    }
  };

  // Search & Filter Execution
  const executeSearchWithFilters = async (overrideQuery) => {
    const q = overrideQuery !== undefined ? overrideQuery : searchQuery;
    setView('all');
    setAllType('search');

    const params = new URLSearchParams();
    if (q && q.trim()) params.append('query', q.trim());
    if (selectedGenre && selectedGenre !== 'all') params.append('genre', selectedGenre);
    if (selectedYear && selectedYear !== 'all') params.append('year', selectedYear);
    if (selectedStatus && selectedStatus !== 'all') params.append('status', selectedStatus);
    params.append('limit', '30');

    try {
      const res = await fetch(`${API_BASE_URL}/movies/search?${params.toString()}`);
      const data = await res.json();
      setSearchResults(data.results || []);
      setAllMovies(data.results || []);
    } catch (e) {
      console.error('Error executing search with filters:', e);
    }
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setSelectedStatus('all');
    setSelectedYear('all');
    setSelectedGenre('all');
    setSearchResults([]);
    setActiveGenreFilter('Trending');
  };

  const handleResetFilters = () => {
    setSelectedStatus('all');
    setSelectedYear('all');
    setSelectedGenre('all');
  };

  // Genre Filter Bar Selection
  const handleSelectGenreFilter = (genre) => {
    setActiveGenreFilter(genre);
    if (genre === 'Trending') {
      setView('home');
    } else {
      setView('all');
      setAllType('genre');
      setSelectedGenre(genre);
      fetch(`${API_BASE_URL}/movies/search?genre=${encodeURIComponent(genre)}&limit=30`)
        .then((r) => r.json())
        .then((data) => {
          setAllMovies(data.results || []);
        })
        .catch(() => {});
    }
  };

  // See All Handler
  const handleSeeAll = (type) => {
    setAllType(type);
    setView('all');
    handleClearSearch();
    if (type === 'trending') {
      setAllMovies(trendingMovies);
    } else if (type === 'latest') {
      setAllMovies(latestMovies);
    } else if (type === 'recs') {
      setAllMovies(recommendations);
    }
  };

  // Watchlist (Like) Toggle
  const toggleWatchlist = (movieId) => {
    if (!movieId) return;
    const mid = Number(movieId);
    setLikedMovies((prev) => {
      const exists = prev.includes(mid);
      const updated = exists ? prev.filter((id) => id !== mid) : [...prev, mid];
      return updated;
    });

    // Notify backend
    fetch(`${API_BASE_URL}/events/watchlist`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-Id': getOrCreateSessionId(),
        ...(localStorage.getItem('auth_token') ? { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` } : {})
      },
      body: JSON.stringify({ movie_id: mid })
    }).catch(() => {});
  };

  // Star Rating Handler
  const handleRateMovie = async (movieId, star) => {
    if (!movieId) return;
    const mid = Number(movieId);
    setMovieRatings((prev) => ({ ...prev, [mid]: star }));

    if (star >= 4 && !likedMovies.includes(mid)) {
      setLikedMovies((prev) => [...prev, mid]);
    }

    try {
      await fetch(`${API_BASE_URL}/events/rating`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-Id': getOrCreateSessionId(),
          ...(localStorage.getItem('auth_token') ? { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` } : {})
        },
        body: JSON.stringify({ movie_id: mid, rating: Number(star) })
      });
    } catch (e) {
      console.error('Error saving rating:', e);
    }
  };

  // Hero Spotlight Slider Logic
  const spotlightPool = trendingMovies.length > 0 ? trendingMovies : [
    {
      movieId: 1084244,
      title: "Elio",
      overview: "An underdog with an active imagination finds himself inadvertently beamed up to the Communiverse, an interplanetary organization with representatives from galaxies far and wide.",
      genres: "Family|Adventure|Animation",
      vote_average: 8.8,
      poster_url: "https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1200"
    }
  ];

  const currentHeroMovie = spotlightPool[heroIndex % spotlightPool.length];
  const nextHeroMovie = spotlightPool[(heroIndex + 1) % spotlightPool.length];

  const handleNextHeroSlide = () => {
    setHeroIndex((prev) => (prev + 1) % spotlightPool.length);
  };

  const handlePrevHeroSlide = () => {
    setHeroIndex((prev) => (prev - 1 + spotlightPool.length) % spotlightPool.length);
  };

  // Construct filtered search title description
  const getFilterSummaryText = () => {
    const parts = [];
    if (searchQuery.trim()) parts.push(`"${searchQuery.trim()}"`);
    if (selectedGenre !== 'all') parts.push(`Genre: ${selectedGenre}`);
    if (selectedYear !== 'all') parts.push(`Year: ${selectedYear}`);
    if (selectedStatus !== 'all') parts.push(`Status: ${selectedStatus === 'released' ? 'Released' : 'Upcoming'}`);
    return parts.length > 0 ? parts.join(' • ') : 'All Movies';
  };

  return (
    <div className="streamix-app-layout">
      {/* 1. Left Sidebar Navigation */}
      <Sidebar
        view={view}
        setView={setView}
        allType={view === 'all' ? allType : null}
        handleSeeAll={handleSeeAll}
        setSelectedMovieId={setSelectedMovieId}
        handleClearSearch={handleClearSearch}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        user={user}
        onOpenAuthModal={() => setIsAuthModalOpen(true)}
      />

      {/* 2. Main Full-Width Content Container */}
      <main className="streamix-main-container">
        <TopNav
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          handleSearch={() => executeSearchWithFilters()}
          handleClearSearch={handleClearSearch}
          user={user}
          onLogout={handleLogout}
          onOpenAuthModal={() => setIsAuthModalOpen(true)}
          setView={setView}
          selectedStatus={selectedStatus}
          setSelectedStatus={setSelectedStatus}
          selectedYear={selectedYear}
          setSelectedYear={setSelectedYear}
          selectedGenre={selectedGenre}
          setSelectedGenre={setSelectedGenre}
          onApplyFilters={() => executeSearchWithFilters()}
          onResetFilters={handleResetFilters}
        />

        <div className="streamix-content-body">
          {/* VIEW: Home Dashboard */}
          {view === 'home' && (
            <>
              {/* 1. Hero Spotlight Carousel Banner (16:9 widescreen + peek slide) */}
              <HeroBanner
                movie={currentHeroMovie}
                nextMovie={nextHeroMovie}
                onWatchNow={() => fetchMovieDetail(currentHeroMovie.movieId)}
                onToggleWatchlist={toggleWatchlist}
                isInWatchlist={likedMovies.includes(currentHeroMovie.movieId)}
                onNextSlide={handleNextHeroSlide}
                onPrevSlide={handlePrevHeroSlide}
                onOpenDetail={() => fetchMovieDetail(currentHeroMovie.movieId)}
              />

              {/* 2. Horizontal Genre Filter Bar with Arrow Controls */}
              <GenreFilterBar
                activeFilter={activeGenreFilter}
                onSelectFilter={handleSelectGenreFilter}
              />

              {/* 3. Section: "You Might Like" (AI 3-Stage Recommendations) */}
              <MovieRow
                title="You Might Like"
                badge="AI 3-Stage"
                movies={recommendations}
                onMovieClick={fetchMovieDetail}
                onSeeAll={() => handleSeeAll('recs')}
                likedMovies={likedMovies}
                movieRatings={movieRatings}
                onToggleLike={toggleWatchlist}
                onRateMovie={handleRateMovie}
              />

              {/* 4. Section: "Trending Movies" */}
              <MovieRow
                title="Trending Movies"
                movies={trendingMovies}
                onMovieClick={fetchMovieDetail}
                onSeeAll={() => handleSeeAll('trending')}
                likedMovies={likedMovies}
                movieRatings={movieRatings}
                onToggleLike={toggleWatchlist}
                onRateMovie={handleRateMovie}
              />

              {/* 5. Section: "Recently Added" */}
              <MovieRow
                title="Recently Added"
                movies={latestMovies}
                onMovieClick={fetchMovieDetail}
                onSeeAll={() => handleSeeAll('latest')}
                likedMovies={likedMovies}
                movieRatings={movieRatings}
                onToggleLike={toggleWatchlist}
                onRateMovie={handleRateMovie}
              />
            </>
          )}

          {/* VIEW: All Movies Grid / Search Results / Filtered View */}
          {view === 'all' && (
            <div className="streamix-grid-view">
              <div className="grid-view-header">
                <h1 className="grid-view-title">
                  {allType === 'search' && `Filter Results: ${getFilterSummaryText()}`}
                  {allType === 'genre' && `Genre: ${activeGenreFilter || searchQuery}`}
                  {allType === 'trending' && 'Top Trending Movies'}
                  {allType === 'latest' && 'Recently Added Movies'}
                  {allType === 'recs' && 'Personalized AI Recommendations'}
                </h1>
                <span className="grid-view-count">{allMovies.length} movies found</span>
              </div>

              <div className="streamix-movie-grid">
                {allMovies.map((movie) => {
                  const mid = movie.movieId || movie.id;
                  return (
                    <MovieCard
                      key={mid}
                      movie={movie}
                      onClick={() => fetchMovieDetail(mid)}
                      isLiked={likedMovies.includes(mid)}
                      onToggleLike={toggleWatchlist}
                      userRating={movieRatings[mid] || 0}
                      onRateMovie={handleRateMovie}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* VIEW: Watchlist */}
          {view === 'watchlist' && (
            <div className="streamix-grid-view">
              <div className="grid-view-header">
                <h1 className="grid-view-title">My Favourites ({likedMoviesDetails.length})</h1>
              </div>
              {likedMoviesDetails.length > 0 ? (
                <div className="streamix-movie-grid">
                  {likedMoviesDetails.map((movie) => {
                    const mid = movie.movieId || movie.id;
                    return (
                      <MovieCard
                        key={mid}
                        movie={movie}
                        onClick={() => fetchMovieDetail(mid)}
                        isLiked={likedMovies.includes(mid)}
                        onToggleLike={toggleWatchlist}
                        userRating={movieRatings[mid] || 0}
                        onRateMovie={handleRateMovie}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="empty-state-card">
                  <span className="empty-state-icon">🎬</span>
                  <p className="empty-state-text">Your favourites library is empty. Click the heart icon on any movie to save it here!</p>
                </div>
              )}
            </div>
          )}

          {/* VIEW: Chatbot Assistant (Full Width) */}
          {view === 'chatbot' && (
            <ChatbotView
              chatMessages={chatMessages}
              chatInput={chatInput}
              setChatInput={setChatInput}
              isTyping={isTyping}
              chatEndRef={chatEndRef}
              handleSendChatMessage={handleSendChatMessage}
              handleMovieClick={fetchMovieDetail}
              handleClearHistory={handleClearHistory}
              handleNewSession={handleNewSession}
            />
          )}

          {/* VIEW: Video Player */}
          {view === 'watch' && selectedMovie && (
            <WatchView
              movie={selectedMovie}
              onBack={() => setView('home')}
            />
          )}
        </div>
      </main>

      {/* Movie Detail Modal Overlay */}
      {selectedMovieId && selectedMovie && (
        <div className="movie-detail-overlay" onClick={() => setSelectedMovieId(null)}>
          <div className="movie-detail-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close-btn" onClick={() => setSelectedMovieId(null)}>✕</button>
            <div 
              className="detail-hero-section"
              style={{ backgroundImage: `url(${selectedMovie.poster_url || currentHeroMovie.poster_url})` }}
            >
              <div className="detail-hero-gradient" />
            </div>

            <div className="detail-body-content">
              <img
                src={selectedMovie.poster_url || "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300"}
                alt={selectedMovie.title}
                className="detail-poster-img"
              />
              <div className="detail-main-info">
                <h2 className="detail-title">{selectedMovie.title}</h2>
                <div className="detail-meta-row">
                  <span>{selectedMovie.release_date ? selectedMovie.release_date.split('-')[0] : '2025'}</span>
                  <span>•</span>
                  <span>{selectedMovie.genres ? selectedMovie.genres.replace(/\|/g, ' • ') : 'Drama'}</span>
                  <span>•</span>
                  <span className="detail-star">★ {selectedMovie.vote_average || '8.5'}</span>
                </div>
                <p className="detail-overview">{selectedMovie.overview || "No overview available for this movie."}</p>

                {/* Star Rating & Like Widget inside Detail Modal */}
                <div className="detail-rate-bar">
                  <span className="detail-rate-title">Your Rating:</span>
                  <div className="detail-stars-wrap">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <span
                        key={star}
                        className={`detail-star-btn ${(movieRatings[selectedMovie.movieId] || 0) >= star ? 'active' : ''}`}
                        onClick={() => handleRateMovie(selectedMovie.movieId, star)}
                        title={`Rate ${star} Stars`}
                      >
                        ★
                      </span>
                    ))}
                  </div>
                  {movieRatings[selectedMovie.movieId] && (
                    <span className="detail-rated-text">Rated: {movieRatings[selectedMovie.movieId]} / 5 ⭐</span>
                  )}
                </div>

                <div className="detail-actions-row">
                  <button 
                    className="btn-detail-play"
                    onClick={() => {
                      setSelectedMovieId(null);
                      setView('watch');
                    }}
                  >
                    <svg viewBox="0 0 24 24" fill="currentColor" className="btn-detail-icon">
                      <polygon points="6 4 20 12 6 20 6 4" />
                    </svg>
                    <span>Watch Now</span>
                  </button>
                  <button
                    className={`btn-detail-watchlist ${likedMovies.includes(selectedMovie.movieId) ? 'active' : ''}`}
                    onClick={() => toggleWatchlist(selectedMovie.movieId)}
                  >
                    <svg 
                      viewBox="0 0 24 24" 
                      fill={likedMovies.includes(selectedMovie.movieId) ? "#ef4444" : "none"} 
                      stroke={likedMovies.includes(selectedMovie.movieId) ? "#ef4444" : "currentColor"} 
                      strokeWidth="2.2" 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      className="btn-detail-icon"
                    >
                      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
                    </svg>
                    <span>{likedMovies.includes(selectedMovie.movieId) ? 'In Favourites' : 'Add to Favourites'}</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Similar Movies Section */}
            {similarMovies.length > 0 && (
              <div className="similar-section">
                <MovieRow
                  title="More Like This (Vector Search)"
                  badge="AI Similar"
                  movies={similarMovies}
                  onMovieClick={fetchMovieDetail}
                  likedMovies={likedMovies}
                  movieRatings={movieRatings}
                  onToggleLike={toggleWatchlist}
                  onRateMovie={handleRateMovie}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onAuthSuccess={handleAuthSuccess}
      />

      {/* Onboarding Modal */}
      <OnboardingModal
        isOpen={isOnboardingOpen}
        onClose={() => setIsOnboardingOpen(false)}
        onFinish={() => {
          setIsOnboardingOpen(false);
          fetchRecommendations();
        }}
      />
    </div>
  );
}

export default App;
