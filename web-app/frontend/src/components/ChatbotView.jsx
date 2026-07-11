import React from 'react';

function ChatbotView({ 
  chatMessages, 
  chatInput, 
  setChatInput, 
  isTyping, 
  chatEndRef, 
  handleSendChatMessage, 
  handleMovieClick 
}) {
  const getYear = (dateStr) => {
    if (!dateStr) return '';
    return dateStr.split('-')[0] || '';
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h1>AI Movie Recommender</h1>
        <p>Trò chuyện với AI để nhận gợi ý phim theo sở thích cá nhân của bạn</p>
      </div>

      <div className="chatbot-chatbox">
        <div className="chatbot-messages">
          {chatMessages.map((msg, index) => (
            <div key={index} className={`chat-bubble ${msg.sender}`}>
              <p>{msg.text}</p>
              
              {msg.movies && msg.movies.length > 0 && (
                <div className="chat-movie-list">
                  {msg.movies.map((movie) => (
                    <div 
                      key={movie.movieId} 
                      className="chat-movie-card"
                      onClick={() => handleMovieClick(movie.movieId)}
                      title={`Xem chi tiết ${movie.title}`}
                    >
                      <img 
                        className="chat-movie-poster" 
                        src={movie.poster_url || 'https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=150'} 
                        alt={movie.title} 
                      />
                      <div className="chat-movie-info">
                        <div className="chat-movie-title">{movie.title}</div>
                        <div className="chat-movie-date">{getYear(movie.release_date)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          
          {isTyping && (
            <div className="chatbot-typing">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chatbot-input-container">
          <input
            type="text"
            className="chatbot-input"
            placeholder="Nhập yêu cầu của bạn (ví dụ: phim hoạt hình lãng mạn, phim giống Toy Story...)"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendChatMessage()}
          />
          <button className="chatbot-send-btn" onClick={() => handleSendChatMessage()}>
            Gửi
          </button>
        </div>
      </div>

      <div className="chatbot-chips">
        <span className="chatbot-chip" onClick={() => handleSendChatMessage("Gợi ý phim hành động viễn tưởng")}>
          🍿 Phim hành động viễn tưởng
        </span>
        <span className="chatbot-chip" onClick={() => handleSendChatMessage("Tôi muốn xem phim hoạt hình gia đình")}>
          👶 Phim hoạt hình gia đình
        </span>
        <span className="chatbot-chip" onClick={() => handleSendChatMessage("Tìm phim giống như Toy Story")}>
          🧸 Phim giống Toy Story
        </span>
        <span className="chatbot-chip" onClick={() => handleSendChatMessage("Gợi ý phim kinh dị kịch tính")}>
          👻 Phim kinh dị kịch tính
        </span>
      </div>
    </div>
  );
}

export default ChatbotView;
