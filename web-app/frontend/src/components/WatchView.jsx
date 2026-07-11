import React, { useState, useEffect, useRef } from 'react';

function WatchView({ movie, onClose }) {
  // Setup simulated duration: 2h 15m (8100 seconds)
  const defaultDuration = 8100;
  const [duration, setDuration] = useState(defaultDuration);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isBuffering, setIsBuffering] = useState(true);
  const [volume, setVolume] = useState(80); // 0 to 100
  const [isMuted, setIsMuted] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1); // 1, 1.5, 2
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const playerRef = useRef(null);
  const progressInterval = useRef(null);

  // Simulate initial buffering
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsBuffering(false);
      setIsPlaying(true);
    }, 1500); // 1.5s buffering
    return () => clearTimeout(timer);
  }, []);

  // Update time elapsed when playing
  useEffect(() => {
    if (isPlaying && !isBuffering) {
      progressInterval.current = setInterval(() => {
        setCurrentTime((prev) => {
          if (prev >= duration) {
            setIsPlaying(false);
            clearInterval(progressInterval.current);
            return duration;
          }
          return prev + playbackSpeed;
        });
      }, 1000);
    } else {
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
      }
    }

    return () => {
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
      }
    };
  }, [isPlaying, isBuffering, playbackSpeed, duration]);

  // Handle Play / Pause
  const handleTogglePlay = () => {
    if (isBuffering) return;
    setIsPlaying(!isPlaying);
  };

  // Handle Replay
  const handleReplay = () => {
    setCurrentTime(0);
    setIsPlaying(true);
  };

  // Handle Seeking
  const handleProgressChange = (e) => {
    const newTime = parseInt(e.target.value, 10);
    setCurrentTime(newTime);
  };

  // Format seconds to HH:MM:SS
  const formatTime = (totalSeconds) => {
    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = Math.floor(totalSeconds % 60);
    return [
      String(hrs).padStart(2, '0'),
      String(mins).padStart(2, '0'),
      String(secs).padStart(2, '0')
    ].join(':');
  };

  // Toggle Mute
  const handleToggleMute = () => {
    setIsMuted(!isMuted);
  };

  // Toggle Simulated Fullscreen
  const handleToggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  return (
    <div className={`watch-view-container ${isFullscreen ? 'fullscreen-mode' : ''}`} ref={playerRef}>
      {/* Player Header */}
      <div className="watch-header">
        <button className="watch-back-btn" onClick={onClose}>
          ← Back to Details
        </button>
        <div className="watch-movie-title">
          Now Watching: <strong>{movie.title}</strong>
        </div>
        <div className="watch-logo-watermark">
          MovieNex
        </div>
      </div>

      {/* Video Screen */}
      <div 
        className="watch-video-screen" 
        onClick={handleTogglePlay}
        style={{
          backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.85)), url(${movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=1200'})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center'
        }}
      >
        {isBuffering ? (
          <div className="watch-screen-overlay">
            <div className="watch-spinner"></div>
            <p className="watch-overlay-text">Buffering MovieNex secure stream...</p>
          </div>
        ) : !isPlaying ? (
          <div className="watch-screen-overlay pause-overlay">
            <div className="watch-play-icon-overlay">
              <svg viewBox="0 0 24 24" fill="white" style={{ width: '48px', height: '48px' }}>
                <path d="M8 5v14l11-7z"/>
              </svg>
            </div>
            <p className="watch-overlay-text">Paused</p>
          </div>
        ) : (
          <div className="watch-active-indicator">
            <span>● LIVE SIMULATION</span>
          </div>
        )}
      </div>

      {/* Player Controls Bar */}
      <div className="watch-controls-bar">
        {/* Playback Controls */}
        <div className="controls-group left">
          <button 
            className="control-btn play-pause-btn" 
            onClick={handleTogglePlay} 
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              // Pause Icon
              <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
              </svg>
            ) : (
              // Play Icon
              <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                <path d="M8 5v14l11-7z"/>
              </svg>
            )}
          </button>

          <button className="control-btn replay-btn" onClick={handleReplay} title="Replay">
            <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
              <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
            </svg>
          </button>
        </div>

        {/* Timeline Slider */}
        <div className="controls-group timeline-group">
          <span className="time-display elapsed">{formatTime(currentTime)}</span>
          <input
            type="range"
            className="timeline-slider"
            min="0"
            max={duration}
            value={currentTime}
            onChange={handleProgressChange}
          />
          <span className="time-display total">{formatTime(duration)}</span>
        </div>

        {/* Volume & Fullscreen Controls */}
        <div className="controls-group right">
          {/* Speed Selector */}
          <div className="speed-selector-wrapper">
            <button 
              className="control-btn speed-btn" 
              onClick={() => {
                const nextSpeed = playbackSpeed === 1 ? 1.5 : playbackSpeed === 1.5 ? 2 : 1;
                setPlaybackSpeed(nextSpeed);
              }}
              title="Change Speed"
            >
              {playbackSpeed}x
            </button>
          </div>

          {/* Volume control */}
          <div className="volume-wrapper">
            <button className="control-btn volume-btn" onClick={handleToggleMute} title={isMuted ? 'Unmute' : 'Mute'}>
              {isMuted || volume === 0 ? (
                <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                  <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.21.05-.42.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.03a11.02 11.02 0 0 0 3.82-1.72l2.9 2.9 1.28-1.27L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
                </svg>
              ) : volume < 50 ? (
                <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                  <path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z"/>
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                  <path d="M3 9v6h4l5 5V4L9 9H5zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                </svg>
              )}
            </button>
            <input
              type="range"
              className="volume-slider"
              min="0"
              max="100"
              value={isMuted ? 0 : volume}
              onChange={(e) => {
                setVolume(parseInt(e.target.value, 10));
                if (isMuted) setIsMuted(false);
              }}
            />
          </div>

          <button className="control-btn fullscreen-btn" onClick={handleToggleFullscreen} title="Fullscreen">
            {isFullscreen ? (
              // Exit Fullscreen
              <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                <path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/>
              </svg>
            ) : (
              // Enter Fullscreen
              <svg viewBox="0 0 24 24" fill="currentColor" className="icon">
                <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default WatchView;
