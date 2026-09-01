import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import HeroBanner from '../components/HeroBanner/HeroBanner';
import GenreFilterBar from '../components/GenreFilterBar/GenreFilterBar';
import MovieRow from '../components/MovieRow/MovieRow';
import { API_BASE_URL } from '../api/client';
import { useWatchlist } from '../context/WatchlistContext';
import './HomePage.css';

export default function HomePage() {
  const navigate = useNavigate();
  const { likedMovies } = useWatchlist();

  const [trendingMovies, setTrendingMovies] = useState([]);
  const [latestMovies, setLatestMovies] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [activeGenreFilter, setActiveGenreFilter] = useState('Trending');
  const [heroIndex, setHeroIndex] = useState(0);

  // 1. Fetch Trending Movies
  useEffect(() => {
    fetch(`${API_BASE_URL}/movies/trending?limit=20`, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : { results: [] }))
      .then(data => {
        setTrendingMovies(data.results || []);
      })
      .catch(() => {});
  }, []);

  // 2. Fetch Latest Movies
  useEffect(() => {
    fetch(`${API_BASE_URL}/movies/latest?limit=20`, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : { results: [] }))
      .then(data => {
        setLatestMovies(data.results || []);
      })
      .catch(() => {});
  }, []);

  // 3. Fetch Personalized Recs (3-Stage Engine) when likedMovies changes
  useEffect(() => {
    const query = likedMovies.length > 0 ? `?movie_ids=${likedMovies.join(',')}` : '';
    fetch(`${API_BASE_URL}/recommendations${query}`, { credentials: 'include' })
      .then(res => (res.ok ? res.json() : { results: [] }))
      .then(data => {
        setRecommendations(data.results || []);
      })
      .catch(() => {});
  }, [likedMovies]);

  // Hero carousel slides
  const currentHeroMovie = trendingMovies[heroIndex] || trendingMovies[0];
  const nextHeroMovie = trendingMovies[(heroIndex + 1) % (trendingMovies.length || 1)];

  const handleNextSlide = () => {
    if (trendingMovies.length > 0) {
      setHeroIndex(prev => (prev + 1) % trendingMovies.length);
    }
  };

  const handlePrevSlide = () => {
    if (trendingMovies.length > 0) {
      setHeroIndex(prev => (prev - 1 + trendingMovies.length) % trendingMovies.length);
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
      {/* Full-Width Hero Banner Section */}
      {currentHeroMovie && (
        <HeroBanner
          movie={currentHeroMovie}
          currentIndex={heroIndex}
          totalSlides={trendingMovies.length || 5}
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
          badge="Top Picks"
          movies={recommendations}
          source="recs_foryou"
          onSeeAll={() => navigate('/explore?filter=recommendations')}
        />
      )}

      {/* Trending Now Row */}
      {trendingMovies.length > 0 && (
        <MovieRow
          title="Trending Now"
          badge="Hot"
          movies={trendingMovies}
          source="trending"
          onSeeAll={() => navigate('/explore?filter=trending')}
        />
      )}

      {/* Latest Releases Row */}
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
