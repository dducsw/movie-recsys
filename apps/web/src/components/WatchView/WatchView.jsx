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
          {(() => {
            let embedUrl = null;
            if (movie.trailer_url) {
              if (movie.trailer_url.includes('/embed/')) {
                const parts = movie.trailer_url.split('/embed/');
                const videoId = parts[1]?.split('?')[0];
                if (videoId && !videoId.startsWith('?')) {
                  embedUrl = `https://www.youtube-nocookie.com/embed/${videoId}`;
                } else {
                  embedUrl = movie.trailer_url.replace('www.youtube.com', 'www.youtube-nocookie.com');
                }
              } else if (movie.trailer_url.includes('watch?v=')) {
                const videoId = movie.trailer_url.split('watch?v=')[1]?.split('&')[0];
                if (videoId) embedUrl = `https://www.youtube-nocookie.com/embed/${videoId}`;
              } else if (movie.trailer_url.includes('youtu.be/')) {
                const videoId = movie.trailer_url.split('youtu.be/')[1]?.split('?')[0];
                if (videoId) embedUrl = `https://www.youtube-nocookie.com/embed/${videoId}`;
              }
            }

            // Fallback: If no direct video ID embedUrl found, use YouTube Search Embed
            if (!embedUrl && movie.title) {
              const cleanTitle = movie.title.replace(/\s*\(\d{4}\)/, '').strip ? movie.title.replace(/\s*\(\d{4}\)/, '').trim() : movie.title;
              embedUrl = `https://www.youtube-nocookie.com/embed?listType=search&list=${encodeURIComponent(cleanTitle + ' official trailer')}`;
            }

            const finalSrc = embedUrl 
              ? `${embedUrl}${embedUrl.includes('?') ? '&' : '?'}autoplay=1&mute=0&rel=0&enablejsapi=1`
              : null;

            return finalSrc ? (
              <iframe
                className="watch-video-iframe"
                src={finalSrc}
                title={`${movie.title} | Official Trailer`}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                referrerPolicy="no-referrer-when-downgrade"
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
                  backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.95)), url(${movie.poster_url && !movie.poster_url.includes('placeholder.com') ? movie.poster_url : 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=1200'})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                  textAlign: 'center',
                  padding: '20px'
                }}
              >
                <p className="watch-overlay-text" style={{ fontSize: '18px', fontWeight: 600, color: 'rgba(255,255,255,0.9)' }}>
                  Loading Trailer...
                </p>
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}

export default WatchView;