import React, { useState, useEffect, useRef } from 'react';
import ChatbotView from '../components/ChatbotView/ChatbotView';
import { API_BASE_URL, getOrCreateSessionId } from '../api/client';

export default function ChatbotPage() {
  const [chatMessages, setChatMessages] = useState(() => {
    const saved = localStorage.getItem('cinemax_chat_history');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {}
    }
    return [{
      sender: 'bot',
      text: "👋 **Hello! I'm MovieNex AI**, your conversational movie recommendation assistant.\n\nAsk me anything like:\n- *'Suggest sci-fi movies with deep plot twists like Interstellar'* \n- *'What are Christopher Nolan's highest rated thriller films?'*\n- *'Recommend emotional anime movies for family night'*"
    }];
  });

  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem('cinemax_chat_history', JSON.stringify(chatMessages));
  }, [chatMessages]);

  const handleSendChatMessage = async () => {
    if (!chatInput.trim() || isTyping) return;

    const userText = chatInput.trim();
    setChatInput('');

    const newMessages = [...chatMessages, { sender: 'user', text: userText }];
    setChatMessages(newMessages);
    setIsTyping(true);

    try {
      const token = localStorage.getItem('auth_token');
      const res = await fetch(`${API_BASE_URL}/chatbot/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-Id': getOrCreateSessionId(),
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        credentials: 'include',
        body: JSON.stringify({
          message: userText,
          session_id: getOrCreateSessionId()
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP Error ${res.status}`);
      }

      const data = await res.json();
      setChatMessages([
        ...newMessages,
        {
          sender: 'bot',
          text: data.text || data.answer || data.reply || "Here are some recommendations based on your request:",
          movies: data.movies || []
        }
      ]);
    } catch (err) {
      setChatMessages([
        ...newMessages,
        {
          sender: 'bot',
          text: "I encountered a brief connection issue. Please verify the backend service is running and try again."
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearHistory = () => {
    const initMsg = [{
      sender: 'bot',
      text: "👋 Chat history cleared. How can I help you find your next favorite movie?"
    }];
    setChatMessages(initMsg);
    localStorage.setItem('cinemax_chat_history', JSON.stringify(initMsg));
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
