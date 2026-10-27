import { useRef, useState } from 'react';
import ChatbotView from '../components/ChatbotView/ChatbotView';
import { apiFetch, resetSessionId } from '../api/client';

const INITIAL_GREETING = [{
  sender: 'bot',
  text: '**Hi, I\'m MovieNex AI** — your movie recommendation assistant.\n\nTry asking:\n- *Mind-bending sci-fi like Interstellar*\n- *Highest-rated Christopher Nolan movies*\n- *Find 2022 movies starring Leonardo DiCaprio*'
}];

export default function ChatbotPage() {
  const [chatMessages, setChatMessages] = useState(INITIAL_GREETING);
  const [sessionId, setSessionId] = useState(() => resetSessionId());
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

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
        body: JSON.stringify({ message: userText, session_id: sessionId })
      });

      setChatMessages([
        ...newMessages,
        {
          sender: 'bot',
          text: data.text || data.answer || data.reply || 'Here are some recommendations matching your request:',
          movies: data.movies || []
        }
      ]);
    } catch {
      setChatMessages([
        ...newMessages,
        { sender: 'bot', text: 'Something went wrong while connecting. Please try again.' }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearHistory = async () => {
    const oldSid = sessionId;
    const newSid = resetSessionId();
    setSessionId(newSid);
    setChatMessages(INITIAL_GREETING);

    if (oldSid) {
      try {
        await apiFetch(`/chatbot/history/${encodeURIComponent(oldSid)}`, { method: 'DELETE' });
      } catch {
        // Lỗi xóa phía server không chặn session mới phía client.
      }
    }
  };

  return (
    <ChatbotView
      chatMessages={chatMessages}
      chatInput={chatInput}
      setChatInput={setChatInput}
      isTyping={isTyping}
      chatEndRef={chatEndRef}
      handleSendChatMessage={handleSendChatMessage}
      handleClearHistory={handleClearHistory}
    />
  );
}
