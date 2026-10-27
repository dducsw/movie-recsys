import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import HeroBanner from '../components/HeroBanner/HeroBanner';
import GenreFilterBar from '../components/GenreFilterBar/GenreFilterBar';
import MovieRow from '../components/MovieRow/MovieRow';
import { apiFetch } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import { RefreshCw } from 'lucide-react';
import './HomePage.css';

const asResults = (data) => (Array.isArray(data) ? data : data?.results) || [];

export default function HomePage() {
  const navigate = useNavigate();
  const { likedMovies } = useWatchlist();

  const [trendingMovies, setTrendingMovies] = useState([]);
  const [latestMovies, setLatestMovies] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [backendOffline, setBackendOffline] = useState(false);
  const [heroIndex, setHeroIndex] = useState(0);

  // Trending/latest chỉ phụ thuộc backend — fetch đúng 1 lần.
  useEffect(() => {
    apiFetch('/movies/trending?limit=20')
      .then((data) => setTrendingMovies(asResults(data)))
      .catch(() => setBackendOffline(true));
    apiFetch('/movies/latest?limit=20')
      .then((data) => setLatestMovies(asResults(data)))
      .catch(() => setBackendOffline(true));
  }, []);

  // Recommendations phụ thuộc lượt like — riêng để khỏi refetch cả trang mỗi lần toggle.
  useEffect(() => {
    const ids = likedMovies.slice(0, 50).join(',');
    apiFetch(`/recommendations${ids ? `?movie_ids=${ids}` : ''}`)
      .then((data) => setRecommendations(asResults(data)))
      .catch(() => {});
  }, [likedMovies]);

  const heroMovie = trendingMovies[heroIndex] || trendingMovies[0];

  return (
    <div className="home-page-container">
      {backendOffline && (
        <div className="backend-offline-banner">
          <span>Backend is not ready — movie lists are unavailable.</span>
          <button onClick={() => window.location.reload()} className="btn-retry-backend">
            <RefreshCw className="retry-icon" />
            <span>Retry</span>
          </button>
        </div>
      )}

      {heroMovie && (
        <HeroBanner
          movie={heroMovie}
          currentIndex={heroIndex}
          totalSlides={Math.min(trendingMovies.length, 6)}
          onSelectSlide={setHeroIndex}
        />
      )}

      <div id="genres-section" className="home-genres-dock">
        <GenreFilterBar
          activeFilter={null}
          onSelectFilter={(genre) =>
            navigate(`/explore?genre=${encodeURIComponent(genre)}`)
          }
        />
      </div>

      {recommendations.length > 0 && (
        <MovieRow
          title="Recommended For You"
          movies={recommendations}
          source="recs_foryou"
          onSeeAll={() => navigate('/explore?filter=recommendations')}
        />
      )}

      {trendingMovies.length > 0 && (
        <MovieRow
          title="Trending Now"
          movies={trendingMovies}
          source="trending"
          onSeeAll={() => navigate('/explore?filter=trending')}
        />
      )}

      {latestMovies.length > 0 && (
        <MovieRow
          title="New & Upcoming Releases"
          movies={latestMovies}
          source="latest"
          onSeeAll={() => navigate('/explore?filter=latest')}
        />
      )}
    </div>
  );
}
