import { useState, useRef } from 'react';
import { apiFetch } from '../api/client';

export function useChat() {
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'bot',
      text: 'Hello! 👋 I am your AI movie recommendation chatbot with memory. I can remember our conversation and provide context-aware recommendations!\n\nTry asking me:\n• "Recommend sci-fi movies"\n• "Movies like Inception"\n• "Tell me more about the second one" (after I recommend movies)'
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);
  const isInitialChatLoad = useRef(true);

  const [chatSessionId, setChatSessionId] = useState(() => {
    return localStorage.getItem('movienex_chat_session_id') || null;
  });
  const [chatMessageCount, setChatMessageCount] = useState(0);

  const handleSendChatMessage = async (e, movieId = null, movieTitle = '') => {
    if (e) e.preventDefault();

    let textToSend = chatInput.trim();
    if (movieId && movieTitle) {
      textToSend = `Gợi ý phim tương tự như "${movieTitle}"`;
    }

    if (!textToSend || isTyping) return;

    const userMsg = { sender: 'user', text: textToSend };
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput('');
    setIsTyping(true);

    try {
      const response = await apiFetch('/chatbot/chat', {
        method: 'POST',
        body: JSON.stringify({
          message: textToSend,
          session_id: chatSessionId
        })
      });

      if (response && response.session_id) {
        if (!chatSessionId || chatSessionId !== response.session_id) {
          setChatSessionId(response.session_id);
          localStorage.setItem('movienex_chat_session_id', response.session_id);
        }
      }

      setChatMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          text: response.reply || 'Sorry, I could not generate a response right now.',
          movies: response.recommended_movies || []
        }
      ]);
      setChatMessageCount(prev => prev + 1);
    } catch (err) {
      setChatMessages(prev => [
        ...prev,
        { sender: 'bot', text: 'Error connecting to chatbot server. Please try again.' }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearChatHistory = async () => {
    if (chatSessionId) {
      try {
        await apiFetch(`/chatbot/history/${chatSessionId}`, { method: 'DELETE' });
      } catch (err) {
        // ignore
      }
    }
    localStorage.removeItem('movienex_chat_session_id');
    setChatSessionId(null);
    setChatMessageCount(0);
    setChatMessages([
      {
        sender: 'bot',
        text: 'Chat memory reset! What movies would you like to explore now? 🎬'
      }
    ]);
  };

  return {
    chatMessages,
    setChatMessages,
    chatInput,
    setChatInput,
    isTyping,
    chatEndRef,
    isInitialChatLoad,
    chatSessionId,
    chatMessageCount,
    handleSendChatMessage,
    handleClearChatHistory
  };
}
