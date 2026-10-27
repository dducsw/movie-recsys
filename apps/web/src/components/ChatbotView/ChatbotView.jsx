import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { useNavigate } from 'react-router-dom';
import { Bot, Send, Trash2, Star, Film, User } from 'lucide-react';
import { posterUrl } from '../../utils/format';
import './ChatbotView.css';

// ReactMarkdown đã escape HTML; chỉ cần chặn scheme lạ trong link (javascript:,...).
const safeHref = (href) => (/^(https?:|mailto:)/i.test(href || '') ? href : '#');

const STARTER_PROMPTS = [  {
    title: 'Mind-Bending Sci-Fi',
    prompt: 'Suggest mind-bending psychological sci-fi films like Inception and Interstellar'
  },
  {
    title: 'Dark Crime Thrillers',
    prompt: 'Recommend dark, gritty crime thrillers directed by David Fincher or Denis Villeneuve'
  },
  {
    title: 'Emotional Masterpieces',
    prompt: 'What are the most moving and critically acclaimed drama movies of all time?'
  },
  {
    title: 'High-Octane Action',
    prompt: 'Recommend the best high-octane modern action thrillers with great reviews'
  }
];

export default function ChatbotView({
  chatMessages = [],
  chatInput = '',
  setChatInput,
  isTyping = false,
  chatEndRef,
  handleSendChatMessage,
  handleClearHistory
}) {
  const localEndRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    (chatEndRef?.current || localEndRef.current)?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isTyping, chatEndRef]);

  const isInitialState = chatMessages.length <= 1;

  return (
    <div className="chat-wrapper">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-brand">
          <div className="chat-brand-icon">
            <Bot />
          </div>
          <h2 className="chat-header-name">MovieNex AI</h2>
        </div>

        <button className="btn-chat-ghost" onClick={handleClearHistory} title="Clear conversation">
          <Trash2 className="chat-ghost-icon" />
          <span>Clear History</span>
        </button>
      </header>

      {/* Messages */}
      <div className="chat-scrollarea">
        {isInitialState && (
          <div className="chat-welcome animate-fade-in">
            <h1 className="welcome-headline">What do you feel like watching tonight?</h1>

            <div className="starter-prompts">
              {STARTER_PROMPTS.map((item) => (
                <button
                  key={item.title}
                  type="button"
                  className="starter-card"
                  onClick={() => setChatInput && setChatInput(item.prompt)}
                >
                  <span className="starter-card-title">{item.title}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="chat-thread">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`chat-message-row ${msg.sender}`}>
              {msg.sender === 'bot' && (
                <div className="chat-avatar bot">
                  <Bot />
                </div>
              )}

              <div className={`chat-bubble ${msg.sender}`}>
                {msg.sender === 'bot' ? (
                  <div className="chat-markdown">
                    <ReactMarkdown
                      components={{
                        a: ({ href, children }) => (
                          <a href={safeHref(href)} target="_blank" rel="noopener noreferrer">{children}</a>
                        )
                      }}
                    >
                      {(msg.text || '').replace(/!\[.*?\]\(.*?\)/g, '')}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="chat-user-text">{msg.text}</p>
                )}

                {msg.movies?.length > 0 && (
                  <div className="chat-recs-shelf">
                    <div className="recs-shelf-label">
                      <Film className="recs-label-icon" />
                      <span>Recommended Movies ({msg.movies.length})</span>
                    </div>
                    <div className="chat-recs-track">
                      {msg.movies.map((movie) => {
                        const mid = movie.movieId || movie.id;
                        return (
                          <div
                            key={mid}
                            className="chat-rec-card"
                            onClick={() => navigate(`/movie/${mid}`)}
                          >
                            <img
                              src={posterUrl(movie.poster_url || movie.image_url)}
                              alt={movie.title}
                              className="recs-poster-img"
                            />
                            {movie.vote_average ? (
                              <div className="recs-rating">
                                <Star className="recs-star-icon" />
                                <span>{(Number(movie.vote_average) / 2).toFixed(1)}</span>
                              </div>
                            ) : null}
                            <span className="recs-title" title={movie.title}>{movie.title}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {msg.sender === 'user' && (
                <div className="chat-avatar user">
                  <User />
                </div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="chat-message-row bot">
              <div className="chat-avatar bot">
                <Bot />
              </div>
              <div className="chat-bubble bot typing">
                <div className="typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
                <span className="typing-label">Finding matches…</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef || localEndRef} />
        </div>
      </div>

      {/* Input */}
      <footer className="chat-input-footer">
        <form
          className="chat-input-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleSendChatMessage();
          }}
        >
          <input
            type="text"
            className="chat-input"
            placeholder="Ask me anything about movies…"
            value={chatInput}
            onChange={(e) => setChatInput && setChatInput(e.target.value)}
            disabled={isTyping}
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={!chatInput?.trim() || isTyping}
            title="Send"
          >
            <Send className="chat-send-icon" />
          </button>
        </form>
      </footer>
    </div>
  );
}
