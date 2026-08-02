import React from 'react';
import ReactMarkdown from 'react-markdown';
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
      {/* Top Navigation Bar */}
      <header className="modern-header">
        <div className="modern-header-left">
          <h1 className="modern-title">🎬 AI Movie Recommender</h1>
        </div>
        <div className="modern-header-right">
          <button className="modern-ghost-btn" onClick={handleClearHistory} title="Clear conversation history">
            🗑️ Clear
          </button>
          <button className="modern-primary-btn" onClick={handleNewSession} title="Start a new conversation">
            ✨ New Chat
          </button>
        </div>
      </header>

      {/* Chat Messages Area */}
      <main className="modern-chat-body">
        <div className="modern-messages-container">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`modern-msg-row ${msg.sender}`}>
              
              {/* Bot Avatar (Left) */}
              {msg.sender === 'bot' && (
                <div className="modern-avatar bot-avatar">🤖</div>
              )}

              <div className={`modern-bubble ${msg.sender}`}>
                {msg.sender === 'bot' ? (
                  <div className="modern-markdown">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="modern-user-text">{msg.text}</p>
                )}
                
                {/* Movie Recommendations */}
                {msg.movies && msg.movies.length > 0 && (
                  <div className="modern-movie-carousel">
                    <div className="modern-carousel-label">🎬 Recommended Movies</div>
                    <div className="modern-movie-track">
                      {msg.movies.map((movie, movieIndex) => (
                        <div 
                          key={movie.movieId || movie.id || movieIndex} 
                          className="modern-movie-card"
                          onClick={() => handleMovieClick(movie.movieId || movie.id, 'chatbot')}
                        >
                          <img 
                            className="modern-movie-poster" 
                            src={getPosterUrl(movie)}
                            alt={movie.title}
                            onError={(e) => {
                              e.currentTarget.src = 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=150';
                            }}
                          />
                          <div className="modern-movie-info">
                            <div className="modern-movie-title">{movie.title}</div>
                            <div className="modern-movie-year">
                              {getYear(movie.year || movie.release_date)}
                            </div>
                            {movie.description && (
                              <div className="modern-movie-desc">
                                {movie.description}
                              </div>
                            )}
                            {movie.genres && (
                              <div className="modern-movie-genres">
                                {Array.isArray(movie.genres) 
                                  ? movie.genres.slice(0, 2).join(' • ')
                                  : movie.genres.split('|').slice(0, 2).join(' • ')
                                }
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* User Avatar (Right) */}
              {msg.sender === 'user' && (
                <div className="modern-avatar user-avatar">👤</div>
              )}
            </div>
          ))}
          
          {/* Typing Indicator */}
          {isTyping && (
            <div className="modern-msg-row bot">
              <div className="modern-avatar bot-avatar">🤖</div>
              <div className="modern-bubble bot">
                <div className="modern-typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="modern-input-zone">
          <div className="modern-suggestions">
            <span onClick={() => handleSendChatMessage("Recommend sci-fi action movies")}>🚀 Sci-Fi Action</span>
            <span onClick={() => handleSendChatMessage("Find movies similar to Toy Story")}>🧸 Like Toy Story</span>
            <span onClick={() => handleSendChatMessage("What are some good comedy movies?")}>😂 Comedy Movies</span>
          </div>
          
          <div className="modern-input-box">
            <input
              type="text"
              className="modern-input"
              placeholder="Ask me anything about movies..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendChatMessage()}
              disabled={isTyping}
            />
            <button 
              className="modern-send-btn" 
              onClick={() => handleSendChatMessage()}
              disabled={isTyping || !chatInput.trim()}
            >
              {isTyping ? '...' : '➤'}
            </button>
          </div>
          
          <div className="modern-footer-info">
            {sessionId && <span className="modern-session-id">Session: {sessionId.slice(0, 8)}...</span>}
            <span className="modern-msg-count">{messageCount} messages</span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default ChatbotView;