import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Play, 
  Plus,
  Check,
  Info, 
  TrendingUp, 
  ChevronLeft, 
  ChevronRight, 
  Star,
  Sparkles,
  Volume2
} from 'lucide-react';
import { useWatchlist } from '../../context/WatchlistContext';
import './HeroBanner.css';

function HeroBanner({ 
  movie, 
  currentIndex = 0,
  totalSlides = 5,
  onNextSlide, 
  onPrevSlide 
}) {
  const navigate = useNavigate();
  const { isLiked, toggleLike } = useWatchlist();

  // Auto slide every 7 seconds
  useEffect(() => {
    if (!onNextSlide) return;
    const interval = setInterval(() => {
      onNextSlide();
    }, 7000);
    return () => clearInterval(interval);
  }, [onNextSlide, currentIndex]);

  if (!movie) return null;

  const movieId = movie.movieId || movie.id;
  const liked = isLiked(movieId);

  const title = movie.title || "Featured Cinematic Masterpiece";
  const genres = movie.genres ? movie.genres.split('|').slice(0, 3) : ["Sci-Fi", "Adventure", "Action"];
  const vote = movie.vote_average ? (Number(movie.vote_average) / 2).toFixed(1) : "4.3";
  const year = movie.release_date ? movie.release_date.split('-')[0] : "2026";
  const overview = movie.overview || "Experience an extraordinary cinematic journey with breathtaking visual fidelity and mind-bending storytelling.";
  
  const backdrop = movie.poster_url && !movie.poster_url.includes('placeholder')
    ? movie.poster_url
    : "https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1600";

  return (
    <div className="cinematic-billboard-wrapper">
      {/* Full-Bleed Edge-to-Edge Backdrop */}
      <div 
        className="billboard-media-bg" 
        style={{ backgroundImage: `url(${backdrop})` }}
      />

      {/* Multi-Directional Gradient Overlay */}
      <div className="billboard-gradient-left" />
      <div className="billboard-gradient-bottom" />

      {/* Billboard Content */}
      <div className="billboard-content-container">
        <div className="billboard-info-block">
          {/* Top Badges */}
          <div className="billboard-badges-row">
            <span className="badge-trending-live">
              <span className="live-dot" />
              <TrendingUp className="w-3.5 h-3.5" />
              <span>#1 IN MOVIES TODAY</span>
            </span>

            <span className="badge-ai-match">
              <Sparkles className="w-3 h-3 text-cyan-400" />
              <span>98% MATCH FOR YOU</span>
            </span>
          </div>

          {/* Title */}
          <h1 className="billboard-title">{title}</h1>

          {/* Sub-row: Rating, Year, Genres */}
          <div className="billboard-meta-row">
            <span className="billboard-rating-pill">
              <Star className="w-4 h-4 fill-amber-400 text-amber-400" />
              <span>{vote}</span>
            </span>
            <span className="meta-dot">•</span>
            <span className="billboard-year-tag">{year}</span>
            <span className="meta-dot">•</span>
            <div className="billboard-genre-list">
              {genres.map((g) => (
                <span key={g} className="billboard-genre-tag">{g}</span>
              ))}
            </div>
          </div>

          {/* Synopsis */}
          <p className="billboard-overview">{overview}</p>

          {/* CTA Action Buttons */}
          <div className="billboard-actions-row">
            <button 
              className="btn-billboard-play" 
              onClick={() => navigate(`/movie/${movieId}`)}
            >
              <Play className="w-5 h-5 fill-current" />
              <span>Watch Now</span>
            </button>

            <button 
              className={`btn-billboard-glass ${liked ? 'active' : ''}`}
              onClick={() => toggleLike(movieId, movie)}
              title={liked ? "In Your Watchlist" : "Add to Watchlist"}
            >
              {liked ? <Check className="w-4 h-4 text-cyan-400" /> : <Plus className="w-4 h-4" />}
              <span>{liked ? 'Watchlist' : 'Add to List'}</span>
            </button>

            <button 
              className="btn-billboard-glass" 
              onClick={() => navigate(`/movie/${movieId}`)}
              title="More Info"
            >
              <Info className="w-4 h-4" />
              <span>Details</span>
            </button>
          </div>
        </div>

        {/* Bottom Right Carousel Dock (Dots + Arrows) */}
        <div className="billboard-carousel-dock">
          <div className="billboard-slide-dots">
            {Array.from({ length: totalSlides }).map((_, idx) => (
              <span 
                key={idx} 
                className={`slide-dot ${idx === currentIndex ? 'active' : ''}`}
              />
            ))}
          </div>

          <div className="billboard-arrow-group">
            {onPrevSlide && (
              <button 
                className="billboard-arrow-btn" 
                onClick={(e) => { e.stopPropagation(); onPrevSlide(); }} 
                title="Previous Slide"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
            )}

            {onNextSlide && (
              <button 
                className="billboard-arrow-btn" 
                onClick={(e) => { e.stopPropagation(); onNextSlide(); }} 
                title="Next Slide"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default HeroBanner;
