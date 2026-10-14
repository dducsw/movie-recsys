import React, { useState, useEffect, useRef } from 'react';
import ChatbotView from '../components/ChatbotView/ChatbotView';
import { apiFetch, getOrCreateSessionId, resetSessionId } from '../api/client';

const INITIAL_GREETING = [{
  sender: 'bot',
  text: "👋 **Xin chào! Tôi là MovieNex AI**, trợ lý tìm kiếm và gợi ý phim thông minh của bạn.\n\nBạn có thể hỏi tôi bất kỳ điều gì về phim ảnh, ví dụ:\n- *'Gợi ý phim khoa học viễn tưởng cốt truyện hại não như Interstellar'*\n- *'Phim của đạo diễn Christopher Nolan điểm cao nhất'*\n- *'Tìm phim năm 2022 có Leonardo DiCaprio đóng'*\n- *'Còn phim nào khác không? (gợi ý thêm phim tiếp theo)'*"
}];

export default function ChatbotPage() {
  // Always reset chat to initial fresh greeting on each page access / mount
  const [chatMessages, setChatMessages] = useState(INITIAL_GREETING);
  const [currentSessionId, setCurrentSessionId] = useState(() => resetSessionId());
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  // Reset chatbot session on component mount (every visit / reload)
  useEffect(() => {
    const newSid = resetSessionId();
    setCurrentSessionId(newSid);
    setChatMessages(INITIAL_GREETING);
  }, []);

  const handleSendChatMessage = async () => {
    if (!chatInput.trim() || isTyping) return;

    const userText = chatInput.trim();
    setChatInput('');

    const newMessages = [...chatMessages, { sender: 'user', text: userText }];
    setChatMessages(newMessages);
    setIsTyping(true);

    try {
      const data = await apiFetch('/chatbot/chat', {
        method: 'POST',
        body: JSON.stringify({
          message: userText,
          session_id: currentSessionId || getOrCreateSessionId()
        })
      });

      setChatMessages([
        ...newMessages,
        {
          sender: 'bot',
          text: data.text || data.answer || data.reply || "Dưới đây là các gợi ý phim phù hợp nhất với yêu cầu của bạn:",
          movies: data.movies || []
        }
      ]);
    } catch (err) {
      setChatMessages([
        ...newMessages,
        {
          sender: 'bot',
          text: "Hệ thống gặp sự cố kết nối tạm thời. Vui lòng kiểm tra dịch vụ backend và thử lại."
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearHistory = async () => {
    const oldSid = currentSessionId;
    const newSid = resetSessionId();
    setCurrentSessionId(newSid);
    setChatMessages(INITIAL_GREETING);

    if (oldSid) {
      try {
        await apiFetch(`/chatbot/history/${encodeURIComponent(oldSid)}`, {
          method: 'DELETE'
        });
      } catch (e) {
        // Ignore background clear errors
      }
    }
  };

  const handleNewSession = () => {
    handleClearHistory();
  };

  return (
    <div className="chatbot-page h-[calc(100vh-6rem)] animate-in fade-in duration-200">
      <ChatbotView
        chatMessages={chatMessages}
        chatInput={chatInput}
        setChatInput={setChatInput}
        isTyping={isTyping}
        chatEndRef={chatEndRef}
        handleSendChatMessage={handleSendChatMessage}
        handleClearHistory={handleClearHistory}
        handleNewSession={handleNewSession}
      />
    </div>
  );
}
