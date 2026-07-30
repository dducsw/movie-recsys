import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar/Navbar';
import MovieRow from './components/MovieRow/MovieRow';
import MovieCard from './components/MovieCard/MovieCard';
import ChatbotView from './components/ChatbotView/ChatbotView';
import Footer from './components/Footer/Footer';
import AuthView from './components/AuthView/AuthView';
import WatchView from './components/WatchView/WatchView';


const API_BASE_URL = 'http://localhost:8000/api';

// ── Session tracking ──────────────────────────────────────────────────────────
// Tạo session ID lần đầu, lưu localStorage 30 ngày
const getOrCreateSessionId = () => {
  const KEY = 'movienex_sid';
  let sid = localStorage.getItem(KEY);
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem(KEY, sid);
  }
  return sid;
};

// Gửi click event lên backend (fire-and-forget, không block UI)
const trackClick = (movieId, source, position = null) => {
  fetch(`${API_BASE_URL}/events/click`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Id': getOrCreateSessionId(),
    },
    credentials: 'include',   // gửi cookie movienex_sid
    body: JSON.stringify({ movie_id: movieId, source, position }),
  }).catch(() => {});          // bỏ qua lỗi, không ảnh hưởng UX
};

function App() {
  // Navigation & View State
  const [view, setView] = useState('home'); // 'home', 'detail', 'all', 'auth', 'watch'
  const [selectedMovieId, setSelectedMovieId] = useState(null);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [showTrailer, setShowTrailer] = useState(false);

  // Active User session state
  const [user, setUser] = useState(null);

  useEffect(() => {
    const activeUser = localStorage.getItem('movienex_active_user');
    if (activeUser) {
      setUser(JSON.parse(activeUser).username);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('movienex_active_user');
    setUser(null);
    setView('home');
  };

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
    text: 'Hello! 👋 I am your AI movie recommendation chatbot with memory. I can remember our conversation and provide context-aware recommendations!\n\nTry asking me:\n• "Recommend sci-fi movies"\n• "Movies like Inception"\n• "Tell me more about the second one" (after I recommend movies)'
  }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  const [chatSessionId, setChatSessionId] = useState(() => {
    return localStorage.getItem('movienex_chat_session_id') || null;
  });
  const [chatMessageCount, setChatMessageCount] = useState(0);

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

    // Add user message to local state immediately
    setChatMessages((prev) => [...prev, { sender: 'user', text }]);
    setIsTyping(true);

    try {
      // Use existing session or create new one
      const currentSessionId = chatSessionId || generateSessionId();
      
      const response = await fetch(`${API_BASE_URL}/chatbot/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: text,
          session_id: currentSessionId
        })
      });
      const data = await response.json();
      
      // Save session ID if it's new
      if (!chatSessionId) {
        setChatSessionId(data.session_id);
        localStorage.setItem('movienex_chat_session_id', data.session_id);
      }
      
      // Update message count
      setChatMessageCount(data.message_count || 0);
      
      setTimeout(() => {
        setChatMessages((prev) => [
          ...prev,
          { 
            sender: 'bot', 
            text: data.text, 
            movies: data.movies || [] 
          }
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

  useEffect(() => {
  if (chatEndRef.current) {
    chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }
  }, [chatMessages, isTyping]);

  useEffect(() => {
    if (view === 'chatbot' && chatSessionId) {
      loadChatHistory();
    }
  }, [view, chatSessionId]);

  const loadChatHistory = async () => {
  if (!chatSessionId) return;
  
  try {
    const response = await fetch(`${API_BASE_URL}/chatbot/history/${chatSessionId}`);
    if (response.ok) {
      const data = await response.json();
      if (data.messages && data.messages.length > 0) {
        setChatMessages(
          data.messages.map(msg => ({
            sender: msg.role === 'human' ? 'user' : 'bot',
            text: msg.content
          }))
        );
        setChatMessageCount(data.messages.length);
      } else {
        // Session exists but no messages, show welcome
        setChatMessages([
          {
            sender: 'bot',
            text: 'Hello! 👋 I am your AI movie recommendation chatbot with memory. How can I help you today?'
          }
        ]);
      }
    }
  } catch (error) {
    console.error('Error loading chat history:', error);
    }
  };  

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
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

  // Handle Click-to-Search (e.g., clicking tags, directors, actors, or genres)
  const handleSearchSubmitWithQuery = async (queryText) => {
    if (!queryText.trim()) return;

    setSearchQuery(queryText);
    setLoadingTrending(true);
    try {
      const response = await fetch(`${API_BASE_URL}/movies/search?query=${encodeURIComponent(queryText)}`);
      const data = await response.json();
      setSearchResults(data.results || []);
      setCurrentSearch(queryText);
      setIsSearching(true);
      setView('home'); 
      setSelectedMovieId(null);
    } catch (error) {
      console.error('Error searching movies by metadata:', error);
    } finally {
      setLoadingTrending(false);
    }
  };

  const handleClearChatHistory = async () => {
    if (!chatSessionId) return;
    
    const confirmClear = window.confirm('Clear all conversation history?');
    if (!confirmClear) return;

    try {
      await fetch(`${API_BASE_URL}/chatbot/history/${chatSessionId}`, { 
        method: 'DELETE' 
      });
    } catch (error) {
      console.error('Error clearing history:', error);
    }

    // Reset local state
    setChatMessages([
      {
        sender: 'bot',
        text: 'Conversation cleared! 🧹 How can I help you with movie recommendations?'
      }
    ]);
    setChatMessageCount(0);
  };

  const handleNewChatSession = () => {
    // Generate new session ID
    const newSessionId = generateSessionId();
    setChatSessionId(newSessionId);
    localStorage.setItem('movienex_chat_session_id', newSessionId);
    
    // Reset messages
    setChatMessages([
      {
        sender: 'bot',
        text: 'New conversation started! ✨ I\'m ready to help you find great movies. What would you like to watch?'
      }
    ]);
    setChatMessageCount(0);
    setChatInput('');
  };

  // Clear Search Results
  const handleClearSearch = () => {
    setSearchQuery('');
    setCurrentSearch('');
    setIsSearching(false);
    setSearchResults([]);
  };

  // Handle click on Movie Card -> Go to Details view
  // source: listing nơi user click ('trending'|'latest'|'search'|'recommendations'|'similar'|'chatbot')
  const handleMovieClick = async (movieId, source = 'unknown', position = null) => {
    // Emit click event trước khi navigate (fire-and-forget)
    trackClick(movieId, source, position);

    setView('detail');
    setSelectedMovieId(movieId);
    setLoadingDetail(true);
    try {
      // Fetch movie detail
      const resDetail = await fetch(`${API_BASE_URL}/movies/${movieId}`, {
        credentials: 'include',
      });
      if (resDetail.ok) {
        const movieData = await resDetail.json();
        setSelectedMovie(movieData);
      } else {
        console.error('Failed to fetch movie details');
      }

      // Fetch similar movie recommendations
      const resSimilar = await fetch(`${API_BASE_URL}/movies/${movieId}/recommendations?limit=12`, {
        credentials: 'include',
      });
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
      <Navbar 
        view={view}
        allType={allType}
        setView={setView}
        handleClearSearch={handleClearSearch}
        setSelectedMovieId={setSelectedMovieId}
        handleSeeAll={handleSeeAll}
        isSearching={isSearching}
        user={user}
        onLogout={handleLogout}
      />

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
                    {searchResults.map((movie, idx) => (
                      <MovieCard 
                        key={movie.movieId} 
                        movie={movie} 
                        onClick={() => handleMovieClick(movie.movieId, 'search', idx)} 
                      />
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
                      <strong>💡 Recommendation System Demo:</strong> This interface simulates the MovieNex movie platform.
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
                <MovieRow 
                  title="Recommended for You"
                  movies={recommendations}
                  loading={loadingRecs}
                  scrollRef={recsScrollRef}
                  onScroll={(dir) => scrollRow(recsScrollRef, dir)}
                  onMovieClick={handleMovieClick}
                  onSeeAll={() => handleSeeAll('recs')}
                  source="recommendations"
                  fallbackMessage="Like some movies in the Trending section below to start building your personalized recommendation profile!"
                  headerExtra={
                    <span style={{ fontSize: '11px', background: 'rgba(1, 180, 228, 0.1)', color: 'var(--tmdbLightBlue)', border: '1px solid rgba(1, 180, 228, 0.2)', padding: '2px 10px', borderRadius: '12px', fontWeight: 700, marginRight: '10px' }}>
                      RecSys Active
                    </span>
                  }
                />

                {/* Row 2: Trending Row */}
                <MovieRow 
                  title="Trending"
                  movies={trendingMovies}
                  loading={loadingTrending}
                  scrollRef={trendingScrollRef}
                  onScroll={(dir) => scrollRow(trendingScrollRef, dir)}
                  onMovieClick={handleMovieClick}
                  onSeeAll={() => handleSeeAll('trending')}
                  source="trending"
                  headerExtra={
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
                  }
                />

                {/* Row 3: Latest Row */}
                <MovieRow 
                  title="Latest"
                  movies={latestMovies}
                  loading={loadingLatest}
                  scrollRef={latestScrollRef}
                  onScroll={(dir) => scrollRow(latestScrollRef, dir)}
                  onMovieClick={handleMovieClick}
                  onSeeAll={() => handleSeeAll('latest')}
                  source="latest"
                />

                {/* Row 4: Liked Movies (Interactive History) */}
                {likedMoviesDetails.length > 0 && (
                  <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: '24px' }}>
                    <MovieRow 
                      title={`Your Favorite Movies (${likedMoviesDetails.length})`}
                      movies={likedMoviesDetails}
                      scrollRef={favoritesScrollRef}
                      onScroll={(dir) => scrollRow(favoritesScrollRef, dir)}
                      onMovieClick={handleMovieClick}
                      source="favorites"
                      headerExtra={
                        <button 
                          onClick={() => { if(confirm("Reset all liked history?")) setLikedMovies([]); }}
                          className="search-clear-btn" 
                          style={{ padding: '4px 14px', fontSize: '12px', marginLeft: 'auto' }}
                        >
                          Reset History
                        </button>
                      }
                    />
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
                            <span 
                              key={i} 
                              className="detail-genre-tag clickable-meta-link"
                              onClick={() => handleSearchSubmitWithQuery(genre)}
                            >
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
                          {likedMovies.includes(selectedMovie.movieId) ? (
                            <svg viewBox="0 0 24 24" fill="white" style={{ width: '18px', height: '18px' }}>
                              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                            </svg>
                          ) : (
                            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" style={{ width: '18px', height: '18px' }}>
                              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                            </svg>
                          )}
                        </button>

                        <button className="circle-action-btn" title="Add to Watchlist" onClick={() => alert("Added to watchlist (Mockup)!")}>
                          <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" style={{ width: '18px', height: '18px' }}>
                            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
                          </svg>
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

                      {/* Play Trailer & Watch Movie Buttons */}
                      <div className="play-buttons-row" style={{ display: 'flex', gap: '12px', margin: '15px 0 25px 0' }}>
                        <button
                          className="watch-now-btn"
                          onClick={() => setShowTrailer(true)}
                          style={{
                            background: 'linear-gradient(135deg, #ff007f, #7f00ff)',
                            boxShadow: '0 4px 15px rgba(255, 0, 127, 0.4)'
                          }}
                        >
                          <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: '16px', height: '16px', marginRight: '6px' }}>
                            <path d="M8 5v14l11-7z"/>
                          </svg>
                          Watch Movie
                        </button>

                        <button
                          className="watch-trailer-btn"
                          onClick={() => setShowTrailer(true)}
                          style={{
                            background: 'transparent',
                            border: '1px solid rgba(255, 255, 255, 0.3)',
                            color: 'white',
                            padding: '10px 18px',
                            borderRadius: '20px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            fontWeight: '600',
                            transition: 'all 0.2s',
                            fontSize: '14px'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = 'white';
                            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                            e.currentTarget.style.background = 'transparent';
                          }}
                        >
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ width: '14px', height: '14px', marginRight: '6px' }}>
                            <polygon points="5 3 19 12 5 21 5 3"/>
                          </svg>
                          Play Trailer
                        </button>
                      </div>

                      {/* Overview & Crew */}
                      <h3 className="overview-header">Overview</h3>
                      <p className="overview-text">{selectedMovie.overview || 'No overview available for this movie.'}</p>

                      {selectedMovie.director && (
                        <div className="detail-director-box">
                          <span className="detail-label">Director:</span>
                          <span 
                            className="detail-value clickable-meta-link" 
                            onClick={() => handleSearchSubmitWithQuery(selectedMovie.director)}
                          >
                            {selectedMovie.director}
                          </span>
                        </div>
                      )}

                      {selectedMovie.cast && (
                        <div className="detail-cast-box">
                          <span className="detail-label">Cast:</span>
                          <div className="detail-cast-chips">
                            {selectedMovie.cast.split('|').slice(0, 10).map((actor, idx) => (
                              <span 
                                key={idx} 
                                className="actor-chip clickable-meta-link"
                                onClick={() => handleSearchSubmitWithQuery(actor)}
                              >
                                {actor}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {selectedMovie.keywords && (
                        <div className="detail-tags-box">
                          <span className="detail-label">Tags:</span>
                          <div className="detail-tags-chips">
                            {selectedMovie.keywords.split('|').slice(0, 15).map((tag, idx) => (
                              <span 
                                key={idx} 
                                className="tag-chip clickable-meta-link"
                                onClick={() => handleSearchSubmitWithQuery(tag)}
                              >
                                #{tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Similar Recommendations Section (Recsys) */}
              {/* Similar Recommendations Section (Recsys) */}
              <div className="detail-recs-wrapper">
                <MovieRow 
                  title="Recommendations"
                  movies={similarMovies}
                  scrollRef={similarScrollRef}
                  onScroll={(dir) => scrollRow(similarScrollRef, dir)}
                  onMovieClick={handleMovieClick}
                  source="similar"
                  fallbackMessage="No similar movies found."
                  headerExtra={
                    <div style={{ color: 'var(--textSecondary)', fontSize: '13.5px', margin: '4px 0 16px 0', fontWeight: 500 }}>
                      If you liked <strong>{selectedMovie.title}</strong>, you might also like...
                    </div>
                  }
                />
              </div>
            </div>
          )}
        </div>
      ) : view === 'chatbot' ? (
        /* ================= AI CHATBOT VIEW ================= */
        <ChatbotView 
          chatMessages={chatMessages}
          chatInput={chatInput}
          setChatInput={setChatInput}
          isTyping={isTyping}
          chatEndRef={chatEndRef}
          handleSendChatMessage={handleSendChatMessage}
          handleMovieClick={handleMovieClick}
          // NEW props for session management
          sessionId={chatSessionId}
          handleClearHistory={handleClearChatHistory}
          handleNewSession={handleNewChatSession}
          messageCount={chatMessageCount}
        />
      ) : view === 'auth' ? (
        /* ================= AUTH VIEW ================= */
        <AuthView 
          setView={setView} 
          onLoginSuccess={(username) => setUser(username)} 
        />
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
                    {allMovies.map((movie, idx) => (
                      <MovieCard 
                        key={movie.movieId} 
                        movie={movie} 
                        onClick={() => handleMovieClick(movie.movieId, allType, idx)} 
                      />
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
      {showTrailer && selectedMovie && (
        <WatchView 
          movie={selectedMovie} 
          onClose={() => setShowTrailer(false)} 
        />
      )}
      <Footer />
    </div>
  );
}

export default App;
