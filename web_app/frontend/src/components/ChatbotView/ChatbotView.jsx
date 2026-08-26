import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import MovieCard from '../MovieCard/MovieCard';
import './ChatbotView.css';

function ChatbotView({
  chatMessages = [],
  chatInput = '',
  setChatInput,
  isTyping = false,
  chatEndRef,
  handleSendChatMessage,
  handleMovieClick,
  handleClearHistory,
  handleNewSession
}) {
  const localEndRef = useRef(null);

  useEffect(() => {
    const target = chatEndRef?.current || localEndRef.current;
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isTyping]);

  const quickPrompts = [
    { label: "🚀 Sci-Fi Classics", prompt: "Recommend the best sci-fi movies of all time" },
    { label: "🌀 Like Inception", prompt: "Suggest mind-bending psychological thrillers like Inception" },
    { label: "🍿 Top Animated", prompt: "What are the highest rated animated movies?" },
    { label: "🎭 Deep Drama", prompt: "Recommend emotional and deep drama movies" }
  ];

  return (
    <div className="cinemax-chat-container">
      {/* 1. Header */}
      <header className="chat-top-header">
        <div className="chat-title-group">
          <div className="chat-avatar-icon">🤖</div>
          <div className="chat-title-text">
            <h2 className="chat-app-name">MovieNex AI</h2>
            <span className="chat-tagline">Context-Aware Movie Recommendation Agent</span>
          </div>
        </div>
        <div className="chat-header-actions">
          <button className="chat-ghost-btn" onClick={handleClearHistory} title="Clear history">
            Clear Chat
          </button>
          <button className="chat-action-btn" onClick={handleNewSession} title="New conversation">
            + New Chat
          </button>
        </div>
      </header>

      {/* 2. Scrollable Messages Body */}
      <div className="chat-messages-viewport">
        <div className="chat-messages-list">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`chat-message-row ${msg.sender}`}>
              {msg.sender === 'bot' && <div className="bot-avatar-badge">AI</div>}
              
              <div className={`chat-bubble ${msg.sender}`}>
                {msg.sender === 'bot' ? (
                  <div className="markdown-content">
                    <ReactMarkdown>
                      {msg.text ? msg.text.replace(/!\[.*?\]\(.*?\)/g, '') : ''}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="user-message-text">{msg.text}</p>
                )}

                {/* Recommended Movies Carousel in Chat */}
                {msg.movies && msg.movies.length > 0 && (
                  <div className="chat-movie-deck">
                    <div className="chat-deck-title">
                      <span>🎬 Recommended for you ({msg.movies.length})</span>
                    </div>
                    <div className="chat-movie-scroll-row">
                      {msg.movies.map((movie, movieIdx) => (
                        <div 
                          key={movie.movieId || movie.id || movieIdx} 
                          className="chat-embedded-movie-card"
                          onClick={() => handleMovieClick && handleMovieClick(movie.movieId || movie.id)}
                        >
                          <img
                            src={movie.poster_url || movie.image_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=200'}
                            alt={movie.title}
                            className="chat-card-poster"
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=200';
                            }}
                          />
                          <div className="chat-card-meta">
                            <span className="chat-movie-name" title={movie.title}>{movie.title}</span>
                            <span className="chat-movie-star">★ {movie.vote_average ? Number(movie.vote_average).toFixed(1) : '8.5'}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Typing Indicator */}
          {isTyping && (
            <div className="chat-message-row bot">
              <div className="bot-avatar-badge">AI</div>
              <div className="chat-bubble bot typing-bubble">
                <div className="typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef || localEndRef} />
        </div>
      </div>

      {/* 3. Pinned Bottom Input Zone */}
      <footer className="chat-footer-dock">
        {/* Quick Suggestion Chips */}
        {chatMessages.length <= 1 && (
          <div className="chat-suggestions-bar">
            {quickPrompts.map((item, idx) => (
              <button 
                key={idx}
                className="suggestion-chip"
                onClick={() => handleSendChatMessage(item.prompt)}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}

        {/* Input Bar Form */}
        <div className="chat-input-pill">
          <textarea
            className="chat-textarea"
            placeholder="Ask MovieNex AI for movie recommendations, plot discussions, or actor suggestions..."
            value={chatInput}
            rows={1}
            onChange={(e) => {
              setChatInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendChatMessage();
              }
            }}
            disabled={isTyping}
          />
          <button
            className="chat-send-icon-btn"
            onClick={() => handleSendChatMessage()}
            disabled={isTyping || !chatInput.trim()}
            title="Send Message"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>

        <p className="chat-disclaimer-note">
          MovieNex AI uses hybrid neural retrieval and vector embeddings to recommend movies from your interaction history.
        </p>
      </footer>
    </div>
  );
}

export default ChatbotView;