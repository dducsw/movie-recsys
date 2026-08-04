import React from 'react';
import ReactMarkdown from 'react-markdown';
import MovieCard from '../MovieCard/MovieCard';
import './ChatbotView.css';

function ChatbotView({
  chatMessages,
  chatInput,
  setChatInput,
  isTyping,
  chatEndRef,
  handleSendChatMessage,
  handleMovieClick,
  sessionId,
  handleClearHistory,
  handleNewSession,
  messageCount
}) {
  const getYear = (dateStr) => {
    if (!dateStr) return '';
    if (dateStr.length === 4) return dateStr;
    return dateStr.split('-')[0] || '';
  };

  const getPosterUrl = (movie) => {
    return movie.image_url || movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=150';
  };

  return (
    <div className="modern-chat-shell">
      {/* Minimalist Top Navigation Bar */}
      <header className="modern-header">
        <div className="modern-header-left">
          <h1 className="modern-title">MovieNex AI</h1>
          <span className="modern-model-badge">LLM RecSys</span>
        </div>
        <div className="modern-header-right">
          <button className="modern-ghost-btn" onClick={handleClearHistory} title="Clear conversation history">
            Clear
          </button>
          <button className="modern-primary-btn" onClick={handleNewSession} title="Start a new conversation">
            + New Chat
          </button>
        </div>
      </header>

      {/* Main Chat Content Area */}
      <main className="modern-chat-body">
        <div className="modern-messages-container">
          <div className="modern-messages-inner">
            {chatMessages.map((msg, index) => (
              <div key={index} className={`modern-msg-row ${msg.sender}`}>
                <div className={`modern-bubble ${msg.sender}`}>
                  {msg.sender === 'bot' ? (
                    <div className="modern-markdown">
                      <ReactMarkdown>
                        {msg.text ? msg.text.replace(/!\[.*?\]\(.*?\)/g, '') : ''}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="modern-user-text">{msg.text}</p>
                  )}

                  {/* Movie Recommendations Horizontal Panel */}
                  {msg.movies && msg.movies.length > 0 && (
                    <div className="modern-movie-carousel">
                      <div className="modern-carousel-header">
                        <span className="modern-carousel-label">🎬 Recommended Movies</span>
                        <span className="modern-carousel-count">{msg.movies.length} movies</span>
                      </div>
                      <div className="modern-track-wrapper">
                        <button
                          className="modern-scroll-arrow left"
                          onClick={(e) => {
                            const track = e.currentTarget.nextElementSibling;
                            if (track) track.scrollBy({ left: -360, behavior: 'smooth' });
                          }}
                        >
                          ‹
                        </button>

                        <div className="modern-movie-track">
                          {msg.movies.map((movie, movieIndex) => (
                            <MovieCard
                              key={movie.movieId || movie.id || movieIndex}
                              movie={movie}
                              onClick={() => handleMovieClick(movie.movieId || movie.id, 'chatbot', movieIndex)}
                            />
                          ))}
                        </div>

                        <button
                          className="modern-scroll-arrow right"
                          onClick={(e) => {
                            const track = e.currentTarget.previousElementSibling;
                            if (track) track.scrollBy({ left: 360, behavior: 'smooth' });
                          }}
                        >
                          ›
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Minimal Typing Indicator */}
            {isTyping && (
              <div className="modern-msg-row bot">
                <div className="modern-bubble bot typing-bubble">
                  <div className="modern-typing">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* ChatGPT Style Floating Input Area */}
        <div className="modern-input-zone">
          {chatMessages.length <= 1 && (
            <div className="modern-suggestions">
              <button onClick={() => handleSendChatMessage("Recommend top sci-fi movies")}>🚀 Sci-Fi Classics</button>
              <button onClick={() => handleSendChatMessage("Find movies similar to Inception")}>🌀 Like Inception</button>
              <button onClick={() => handleSendChatMessage("Best mind-bending thrillers?")}>🧠 Mind Thrillers</button>
            </div>
          )}

          <div className="modern-input-box">
            <textarea
              className="modern-textarea"
              placeholder="Message MovieNex AI..."
              value={chatInput}
              rows={1}
              onChange={(e) => {
                setChatInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
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
              className="modern-send-btn"
              onClick={() => handleSendChatMessage()}
              disabled={isTyping || !chatInput.trim()}
              title="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            </button>
          </div>

          <div className="modern-footer-info">
            MovieNex AI can provide tailored recommendations based on your preferences.
          </div>
        </div>
      </main>
    </div>
  );
}

export default ChatbotView;