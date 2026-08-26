import React from 'react';
import './HeroBanner.css';

function HeroBanner({ 
  movie, 
  nextMovie, 
  onWatchNow, 
  onToggleWatchlist, 
  isInWatchlist, 
  onNextSlide, 
  onPrevSlide,
  onOpenDetail
}) {
  const title = movie?.title || "Elio";
  const genres = movie?.genres ? movie.genres.split('|').slice(0, 2) : ["Family", "Adventure"];
  const vote = movie?.vote_average ? Number(movie.vote_average).toFixed(1) : "8.8";
  const overview = movie?.overview || "An underdog with an active imagination finds himself inadvertently beamed up to the Communiverse, an interplanetary organization with representatives from galaxies far and wide.";
  const backdrop = movie?.poster_url && !movie.poster_url.includes('placeholder')
    ? movie.poster_url
    : "https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1200";

  const nextBackdrop = nextMovie?.poster_url && !nextMovie.poster_url.includes('placeholder')
    ? nextMovie.poster_url
    : "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?q=80&w=600";

  return (
    <div className="streamix-hero-wrapper">
      {/* Main Large Hero Card */}
      <div className="streamix-hero-card">
        <div 
          className="hero-bg-media" 
          style={{ backgroundImage: `url(${backdrop})` }}
        />
        <div className="hero-vignette" />

        {/* Prev Slide Arrow Button */}
        {onPrevSlide && (
          <button 
            className="hero-prev-arrow" 
            onClick={(e) => { e.stopPropagation(); onPrevSlide(); }} 
            title="Previous Movie"
          >
            ‹
          </button>
        )}

        <div className="hero-banner-body">
          {/* Minimalist Transparent Trending Badge */}
          <div className="hero-top-badges">
            <span className="badge-trending">
              <span className="trending-pulsing-dot" />
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="trending-badge-icon">
                <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                <polyline points="17 6 23 6 23 12" />
              </svg>
              <span>Trending Now</span>
            </span>
          </div>

          {/* Genre Chips */}
          <div className="hero-genre-tags">
            {genres.map((g) => (
              <span key={g} className="hero-genre-pill">{g}</span>
            ))}
            <span className="hero-rating-pill">★ {vote}</span>
          </div>

          {/* Title & Description */}
          <h1 className="hero-main-title">{title}</h1>
          <p className="hero-main-desc">{overview}</p>

          {/* Action Buttons */}
          <div className="hero-cta-group">
            <button className="btn-play-now" onClick={() => onWatchNow && onWatchNow(movie)}>
              <svg viewBox="0 0 24 24" fill="currentColor" className="cta-icon">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Watch Now
            </button>

            <button 
              className={`btn-icon-action ${isInWatchlist ? 'active' : ''}`}
              onClick={() => onToggleWatchlist && onToggleWatchlist(movie?.movieId || movie?.id)}
              title={isInWatchlist ? "Remove from Watchlist" : "Add to Watchlist"}
            >
              <svg viewBox="0 0 24 24" fill={isInWatchlist ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2.2" className="cta-icon">
                <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
              </svg>
            </button>

            <button 
              className="btn-icon-action" 
              onClick={() => onOpenDetail && onOpenDetail(movie?.movieId || movie?.id)}
              title="More Details"
            >
              •••
            </button>
          </div>
        </div>
      </div>

      {/* Peek Next Slide Preview */}
      {nextMovie && (
        <div className="streamix-hero-peek" onClick={onNextSlide}>
          <img 
            src={nextBackdrop} 
            alt={nextMovie.title || "Next"} 
            className="peek-poster-img" 
          />
          <div className="peek-overlay" />
          <button className="peek-next-arrow" onClick={(e) => { e.stopPropagation(); onNextSlide(); }} title="Next Slide">
            ›
          </button>
        </div>
      )}
    </div>
  );
}

export default HeroBanner;
