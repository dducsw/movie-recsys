import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { useNavigate } from 'react-router-dom';
import { 
  Bot, 
  Send, 
  Sparkles, 
  Trash2, 
  Star, 
  Film,
  User,
  Compass,
  Flame,
  BrainCircuit,
  Zap,
  ArrowUpRight
} from 'lucide-react';
import './ChatbotView.css';

const STARTER_PROMPTS = [
  {
    icon: Compass,
    title: "Mind-Bending Sci-Fi",
    desc: "Movies with deep philosophical concepts and plot twists like Interstellar",
    prompt: "Suggest mind-bending psychological sci-fi films like Inception and Interstellar"
  },
  {
    icon: Flame,
    title: "Dark Neo-Noir Crime",
    desc: "Gritty mystery thrillers directed by David Fincher or Denis Villeneuve",
    prompt: "Recommend dark, gritty crime thrillers directed by David Fincher or Denis Villeneuve"
  },
  {
    icon: BrainCircuit,
    title: "Emotional Masterpieces",
    desc: "Critically acclaimed dramas that leave a lasting emotional impact",
    prompt: "What are the most moving and critically acclaimed drama movies of all time?"
  },
  {
    icon: Zap,
    title: "High-Octane Action",
    desc: "Fast-paced adrenaline cinema with incredible stunt choreography",
    prompt: "Recommend the best high-octane modern action thrillers with great reviews"
  }
];

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
  const navigate = useNavigate();

  useEffect(() => {
    const target = chatEndRef?.current || localEndRef.current;
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isTyping, chatEndRef]);

  const onMovieCardClick = (movieId) => {
    if (handleMovieClick) {
      handleMovieClick(movieId);
    } else {
      navigate(`/movie/${movieId}`);
    }
  };

  const isInitialState = chatMessages.length <= 1;

  return (
    <div className="modern-chat-wrapper">
      {/* 1. Sleek Header Bar */}
      <header className="modern-chat-header">
        <div className="chat-brand-status">
          <div className="ai-status-orb">
            <Bot className="w-5 h-5 text-white" />
            <span className="live-pulse-dot" />
          </div>
          <div className="ai-meta-titles">
            <div className="ai-name-row">
              <h2 className="ai-header-name">MovieNex AI</h2>
              <span className="ai-tech-tag">
                <Sparkles className="w-3 h-3 text-cyan-400" />
                Smart Assistant
              </span>
            </div>
            <span className="ai-sub-status">Your Personal Movie Guide</span>
          </div>
        </div>

        <div className="chat-header-btns">
          <button 
            className="btn-chat-ghost" 
            onClick={handleClearHistory} 
            title="Clear conversation"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        </div>
      </header>

      {/* 2. Messages Container */}
      <div className="modern-chat-scrollarea">
        {/* Welcome Starter Grid when no messages yet */}
        {isInitialState && (
          <div className="chat-welcome-hero animate-fade-in">
            <div className="welcome-ai-icon-box">
              <Sparkles className="w-8 h-8 text-cyan-400" />
            </div>
            <h1 className="welcome-headline">How can I help you discover cinema today?</h1>
            <p className="welcome-subline">
              Ask natural language queries, explore thematic storylines, or select a starter prompt below.
            </p>

            <div className="starter-prompts-grid">
              {STARTER_PROMPTS.map((item, idx) => {
                const IconComp = item.icon;
                return (
                  <button
                    key={idx}
                    type="button"
                    className="starter-bento-card"
                    onClick={() => {
                      if (setChatInput) setChatInput(item.prompt);
                    }}
                  >
                    <div className="bento-card-top">
                      <div className="bento-icon-wrapper">
                        <IconComp className="w-4 h-4 text-cyan-400" />
                      </div>
                      <ArrowUpRight className="w-4 h-4 text-neutral-500 bento-arrow" />
                    </div>
                    <span className="bento-card-title">{item.title}</span>
                    <p className="bento-card-desc">{item.desc}</p>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Message Thread */}
        <div className="chat-thread-container">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`message-row-wrapper ${msg.sender}`}>
              {msg.sender === 'bot' && (
                <div className="bot-avatar-capsule">
                  <Bot className="w-4 h-4" />
                </div>
              )}
              
              <div className={`message-bubble-card ${msg.sender}`}>
                {msg.sender === 'bot' ? (
                  <div className="ai-markdown-render">
                    <ReactMarkdown>
                      {msg.text ? msg.text.replace(/!\[.*?\]\(.*?\)/g, '') : ''}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="user-text-content">{msg.text}</p>
                )}

                {/* Recommended Movies Carousel inside Chat message */}
                {msg.movies && msg.movies.length > 0 && (
                  <div className="chat-recs-shelf">
                    <div className="recs-shelf-label">
                      <Film className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Recommended Movies ({msg.movies.length})</span>
                    </div>
                    <div className="chat-recs-scroll-track">
                      {msg.movies.map((movie, movieIdx) => (
                        <div 
                          key={movie.movieId || movie.id || movieIdx} 
                          className="chat-recs-mini-card"
                          onClick={() => onMovieCardClick(movie.movieId || movie.id)}
                        >
                          <div className="recs-mini-poster-box">
                            <img
                              src={movie.poster_url || movie.image_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=200'}
                              alt={movie.title}
                              className="recs-mini-poster-img"
                              onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=200';
                              }}
                            />
                            <div className="recs-mini-rating">
                              <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                              <span>{movie.vote_average ? (Number(movie.vote_average) / 2).toFixed(1) : '4.3'}</span>
                            </div>
                          </div>
                          <span className="recs-mini-title" title={movie.title}>
                            {movie.title}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {msg.sender === 'user' && (
                <div className="user-avatar-capsule">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {/* Typing Loading Indicator */}
          {isTyping && (
            <div className="message-row-wrapper bot">
              <div className="bot-avatar-capsule">
                <Bot className="w-4 h-4" />
              </div>
              <div className="message-bubble-card bot typing-card">
                <div className="ai-typing-dots">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
                <span className="typing-label">Finding the best matches for you...</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef || localEndRef} />
        </div>
      </div>

      {/* 3. Floating Bottom Input Island */}
      <footer className="modern-chat-input-island">
        <form
          className="chat-input-form-dock"
          onSubmit={(e) => {
            e.preventDefault();
            if (handleSendChatMessage) handleSendChatMessage();
          }}
        >
          <input
            type="text"
            className="modern-chat-input"
            placeholder="Ask anything (e.g. 'Recommend psychological thrillers with shocking endings')..."
            value={chatInput}
            onChange={(e) => setChatInput && setChatInput(e.target.value)}
            disabled={isTyping}
          />
          <button
            type="submit"
            className="modern-send-btn"
            disabled={!chatInput?.trim() || isTyping}
            title="Send query"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </footer>
    </div>
  );
}

export default ChatbotView;
