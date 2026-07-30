import React from 'react';
import ReactMarkdown from 'react-markdown';

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
  messageCount,
}) {
  const getYear = (dateStr) => {
    if (!dateStr) return '';
    if (dateStr.length === 4) return dateStr;
    return dateStr.split('-')[0] || '';
  };

  const getPosterUrl = (movie) => {
    return (
      movie.image_url ||
      movie.poster_url ||
      'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=150'
    );
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <div className="chatbot-header-content">
          <div>
            <h1>🎬 AI Movie Recommender</h1>
            <p>Chat with AI to get personalized movie recommendations</p>
          </div>

          <div className="chatbot-header-actions">
            <div className="chatbot-session-info">
              <span className="session-badge">
                💬 {messageCount} messages
              </span>

              {sessionId && (
                <span className="session-id" title="Session ID">
                  Session: {sessionId.slice(0, 12)}...
                </span>
              )}
            </div>

            <button
              className="chatbot-header-btn clear-btn"
              onClick={handleClearHistory}
              title="Clear conversation history"
            >
              🗑️ Clear
            </button>

            <button
              className="chatbot-header-btn new-btn"
              onClick={handleNewSession}
              title="Start a new conversation"
            >
              ✨ New Chat
            </button>
          </div>
        </div>
      </div>

      <div className="chatbot-chatbox">
        <div className="chatbot-messages">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`chat-bubble ${msg.sender}`}>
              {msg.sender === 'bot' && (
                <div className="bot-avatar">🤖</div>
              )}

              <div className="bubble-content">
                {/* Markdown Message */}
                <div className="bubble-text markdown-content">
                  <ReactMarkdown>{msg.text || ''}</ReactMarkdown>
                </div>

                {/* Movie Cards */}
                {msg.movies && msg.movies.length > 0 && (
                  <div className="chat-movie-list">
                    <div className="movie-list-label">
                      🎬 Recommended Movies
                    </div>

                    {msg.movies.map((movie, movieIndex) => (
                      <div
                        key={movie.movieId || movie.id || movieIndex}
                        className="chat-movie-card"
                        onClick={() =>
                          handleMovieClick(
                            movie.movieId || movie.id,
                            'chatbot'
                          )
                        }
                        title={`View details of ${movie.title}`}
                      >
                        <div className="chat-movie-poster-wrapper">
                          <img
                            className="chat-movie-poster"
                            src={getPosterUrl(movie)}
                            alt={movie.title}
                            onError={(e) => {
                              e.currentTarget.src =
                                'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=150';
                            }}
                          />
                        </div>

                        <div className="chat-movie-info">
                          <div className="chat-movie-title">
                            {movie.title}
                          </div>

                          <div className="chat-movie-year">
                            {getYear(movie.year || movie.release_date)}
                          </div>

                          {movie.description && (
                            <div className="chat-movie-description">
                              {movie.description}
                            </div>
                          )}

                          {movie.genres && (
                            <div className="chat-movie-genres">
                              {Array.isArray(movie.genres)
                                ? movie.genres
                                    .slice(0, 2)
                                    .join(' • ')
                                : movie.genres
                                    .split('|')
                                    .slice(0, 2)
                                    .join(' • ')}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {msg.sender === 'user' && (
                <div className="user-avatar">👤</div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="chat-bubble bot">
              <div className="bot-avatar">🤖</div>

              <div className="bubble-content">
                <div className="chatbot-typing">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="chatbot-input-container">
          <input
            type="text"
            className="chatbot-input"
            placeholder="Ask me anything about movies... (e.g. 'movies like Inception', 'best sci-fi', 'funny family movies')"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) =>
              e.key === 'Enter' && handleSendChatMessage()
            }
            disabled={isTyping}
          />

          <button
            className="chatbot-send-btn"
            onClick={() => handleSendChatMessage()}
            disabled={isTyping || !chatInput.trim()}
          >
            {isTyping ? '...' : 'Send'}
          </button>
        </div>
      </div>

      <div className="chatbot-chips">
        <span
          className="chatbot-chip"
          onClick={() =>
            handleSendChatMessage('Recommend sci-fi action movies')
          }
        >
          🚀 Sci-Fi Action
        </span>

        <span
          className="chatbot-chip"
          onClick={() =>
            handleSendChatMessage(
              'I want to watch family animation movies'
            )
          }
        >
          👶 Family Animation
        </span>

        <span
          className="chatbot-chip"
          onClick={() =>
            handleSendChatMessage(
              'Find movies similar to Toy Story'
            )
          }
        >
          🧸 Like Toy Story
        </span>

        <span
          className="chatbot-chip"
          onClick={() =>
            handleSendChatMessage(
              'Recommend thriller horror movies'
            )
          }
        >
          👻 Thriller Horror
        </span>

        <span
          className="chatbot-chip"
          onClick={() =>
            handleSendChatMessage(
              'What are some good comedy movies?'
            )
          }
        >
          😂 Comedy Movies
        </span>

        <span
          className="chatbot-chip"
          onClick={() =>
            handleSendChatMessage(
              'Movies like The Dark Knight'
            )
          }
        >
          🦇 Like Dark Knight
        </span>
      </div>
    </div>
  );
}

export default ChatbotView;