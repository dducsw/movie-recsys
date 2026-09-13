import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import HeroBanner from '../components/HeroBanner/HeroBanner';
import GenreFilterBar from '../components/GenreFilterBar/GenreFilterBar';
import MovieRow from '../components/MovieRow/MovieRow';
import { API_BASE_URL } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import { AlertCircle, RefreshCw } from 'lucide-react';
import './HomePage.css';

// Dữ liệu dự phòng bảo đảm màn hình không bao giờ bị đen dù backend đang khởi động
const FALLBACK_MOVIES = [
  {
    movieId: 1339713,
    title: "Obsession",
    release_date: "2026-05-13",
    genres: "Horror|Thriller",
    popularity: 697.4,
    vote_average: 8.3,
    poster_url: "https://image.tmdb.org/t/p/w500/bRwnj8WEKBCvmfeUNOukJPwB43K.jpg",
    overview: "After breaking the mysterious 'One Wish Willow' to win his crush's heart, a hopeless romantic finds himself getting exactly what he asked for but soon discovers that some desires come at a dark, sinister price."
  },
  {
    movieId: 1084244,
    title: "Toy Story 5",
    release_date: "2026-06-17",
    genres: "Adventure|Animation|Comedy|Family",
    popularity: 668.4,
    vote_average: 7.4,
    poster_url: "https://image.tmdb.org/t/p/w500/sfQtVlIHljToOwYjhe21KPGzZWK.jpg",
    overview: "When Bonnie receives a Lilypad tablet as a gift and becomes obsessed, Buzz, Woody, Jessie and the rest of the gang's jobs become exponentially harder when they have to go head to head with the all-new threat to playtime."
  },
  {
    movieId: 1275779,
    title: "Disclosure Day",
    release_date: "2026-06-10",
    genres: "Science Fiction|Thriller",
    popularity: 486.3,
    vote_average: 6.7,
    poster_url: "https://image.tmdb.org/t/p/w500/259wnijEJoJLPuZuscxDTqwnypw.jpg",
    overview: "A cybersecurity expert becomes a whistleblower after uncovering secrets about aliens, putting him on the run from a corporation. Meanwhile, a meteorologist experiencing strange phenomena joins forces with him."
  },
  {
    movieId: 1083381,
    title: "Backrooms",
    release_date: "2026-05-27",
    genres: "Horror|Mystery|Science Fiction",
    popularity: 405.1,
    vote_average: 6.8,
    poster_url: "https://image.tmdb.org/t/p/w500/rhGx6E3qRNMgj3i5su2oukNHwIQ.jpg",
    overview: "A strange doorway appears in the basement of a furniture showroom leading to an endless expanse of liminal offices and terrifying mysteries."
  },
  {
    movieId: 1273221,
    title: "Scary Movie",
    release_date: "2026-06-03",
    genres: "Comedy|Horror",
    popularity: 409.2,
    vote_average: 5.4,
    poster_url: "https://image.tmdb.org/t/p/w500/1KlYdWoOrbL5ux357rW9LC155qw.jpg",
    overview: "Twenty-six years after outrunning a suspiciously familiar masked killer, the Core Four are back in the killer's crosshairs and no horror movie IP is safe."
  }
];

export default function HomePage() {
  const navigate = useNavigate();
  const { likedMovies } = useWatchlist();

  const [trendingMovies, setTrendingMovies] = useState([]);
  const [latestMovies, setLatestMovies] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [backendOffline, setBackendOffline] = useState(false);
  const [activeGenreFilter, setActiveGenreFilter] = useState('Trending');
  const [heroIndex, setHeroIndex] = useState(0);

  const fetchHomeData = () => {
    setIsLoading(true);
    setBackendOffline(false);

    const trendingPromise = fetch(`${API_BASE_URL}/movies/trending?limit=20`, { credentials: 'include' })
      .then(res => res.ok ? res.json() : Promise.reject(new Error('Trending fetch failed')))
      .then(data => {
        if (data.results && data.results.length > 0) {
          setTrendingMovies(data.results);
          return true;
        }
        return false;
      })
      .catch(() => false);

    const latestPromise = fetch(`${API_BASE_URL}/movies/latest?limit=20`, { credentials: 'include' })
      .then(res => res.ok ? res.json() : Promise.reject(new Error('Latest fetch failed')))
      .then(data => {
        if (data.results && data.results.length > 0) {
          setLatestMovies(data.results);
          return true;
        }
        return false;
      })
      .catch(() => false);

    const recsQuery = likedMovies.length > 0 ? `?movie_ids=${likedMovies.join(',')}` : '';
    const recsPromise = fetch(`${API_BASE_URL}/recommendations${recsQuery}`, { credentials: 'include' })
      .then(res => res.ok ? res.json() : Promise.reject(new Error('Recs fetch failed')))
      .then(data => {
        if (data.results && data.results.length > 0) {
          setRecommendations(data.results);
        }
      })
      .catch(() => {});

    Promise.all([trendingPromise, latestPromise, recsPromise])
      .then(([hasTrending, hasLatest]) => {
        setIsLoading(false);
        if (!hasTrending && !hasLatest) {
          // Backend có thể chưa chạy hoặc chưa sẵn sàng
          setBackendOffline(true);
          setTrendingMovies(FALLBACK_MOVIES);
          setLatestMovies(FALLBACK_MOVIES);
        }
      })
      .catch(() => {
        setIsLoading(false);
        setBackendOffline(true);
        setTrendingMovies(FALLBACK_MOVIES);
        setLatestMovies(FALLBACK_MOVIES);
      });
  };

  useEffect(() => {
    fetchHomeData();
  }, [likedMovies]);

  // Movies dùng để hiển thị (ưu tiên live API, fallback nếu rỗng)
  const displayTrending = trendingMovies.length > 0 ? trendingMovies : FALLBACK_MOVIES;
  const displayLatest = latestMovies.length > 0 ? latestMovies : FALLBACK_MOVIES;
  const currentHeroMovie = displayTrending[heroIndex] || displayTrending[0];

  const handleNextSlide = () => {
    if (displayTrending.length > 0) {
      setHeroIndex(prev => (prev + 1) % displayTrending.length);
    }
  };

  const handlePrevSlide = () => {
    if (displayTrending.length > 0) {
      setHeroIndex(prev => (prev - 1 + displayTrending.length) % displayTrending.length);
    }
  };

  const handleGenreSelect = (genre) => {
    setActiveGenreFilter(genre);
    if (genre === 'Trending') {
      navigate('/explore');
    } else {
      navigate(`/explore?genre=${encodeURIComponent(genre)}`);
    }
  };

  return (
    <div className="home-page-container">
      {/* Thông báo nếu backend offline */}
      {backendOffline && (
        <div className="backend-offline-banner">
          <AlertCircle className="w-4 h-4 text-amber-400" />
          <span>Backend đang khởi động hoặc ngoại tuyến. Đang hiển thị danh sách phim hot dự phòng.</span>
          <button onClick={fetchHomeData} className="btn-retry-backend">
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Thử lại</span>
          </button>
        </div>
      )}

      {/* Hero Banner Section */}
      {currentHeroMovie && (
        <HeroBanner
          movie={currentHeroMovie}
          currentIndex={heroIndex}
          totalSlides={Math.min(displayTrending.length, 6)}
          onNextSlide={handleNextSlide}
          onPrevSlide={handlePrevSlide}
        />
      )}

      {/* Genre Filter Quick Bar */}
      <div id="genres-section" className="home-genres-dock">
        <GenreFilterBar
          activeFilter={activeGenreFilter}
          onSelectFilter={handleGenreSelect}
        />
      </div>

      {/* Personalized For You Row */}
      {recommendations.length > 0 && (
        <MovieRow
          title="Recommended For You"
          badge="ML Ranked"
          movies={recommendations}
          source="recs_foryou"
          onSeeAll={() => navigate('/explore?filter=recommendations')}
        />
      )}

      {/* Trending Now Row */}
      {displayTrending.length > 0 && (
        <MovieRow
          title="Trending Now"
          badge="Hot"
          movies={displayTrending}
          source="trending"
          onSeeAll={() => navigate('/explore?filter=trending')}
        />
      )}

      {/* Latest Releases Row */}
      {displayLatest.length > 0 && (
        <MovieRow
          title="New & Upcoming Releases"
          badge="Latest"
          movies={displayLatest}
          source="latest"
          onSeeAll={() => navigate('/explore?filter=latest')}
        />
      )}
    </div>
  );
}
