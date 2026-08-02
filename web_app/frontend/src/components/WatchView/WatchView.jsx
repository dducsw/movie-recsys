import React from 'react';
import './WatchView.css';

function WatchView({ movie, onClose }) {
  // Prevent clicks inside the modal from closing it
  const handleModalClick = (e) => {
    e.stopPropagation();
  };

  return (
    <div className="watch-view-overlay" onClick={onClose}>
      <div className="watch-view-modal" onClick={handleModalClick}>
        {/* Play Trailer Header */}
        <div className="watch-header">
          <h3>Play Trailer</h3>
          <button className="watch-close-btn" onClick={onClose} title="Close Trailer">
            ✕
          </button>
        </div>

        {/* Video Screen */}
        <div className="watch-video-screen">
          {movie.trailer_url && movie.trailer_url.includes('/embed/') ? (
            <iframe
              className="watch-video-iframe"
              src={`${movie.trailer_url}?autoplay=1&mute=0`}
              title={`${movie.title} | Official Trailer`}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
              style={{ width: '100%', height: '100%', border: 'none' }}
            ></iframe>
          ) : (
            <div 
              className="watch-video-screen-fallback"
              style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.95)), url(${movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=1200'})`,
                backgroundSize: 'cover',
                backgroundPosition: 'center',
                textAlign: 'center',
                padding: '20px'
              }}
            >
              <p className="watch-overlay-text" style={{ fontSize: '18px', fontWeight: 600, color: 'rgba(255,255,255,0.9)' }}>
                Trailer URL not direct-embeddable
              </p>
              <p style={{ color: 'rgba(255,255,255,0.5)', marginTop: '8px', maxWidth: '450px', fontSize: '14px', lineHeight: 1.4 }}>
                We found a search query on YouTube for this trailer. Please click the button below to watch it on YouTube:
              </p>
              <a 
                href={movie.trailer_url} 
                target="_blank" 
                rel="noopener noreferrer" 
                style={{ marginTop: '20px', display: 'inline-flex', padding: '12px 28px', fontSize: '14px', textDecoration: 'none', background: '#e50914', color: 'white', borderRadius: '4px', fontWeight: 'bold' }}
              >
                Watch Trailer on YouTube ↗
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default WatchView;