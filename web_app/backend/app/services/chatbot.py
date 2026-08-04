from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Dict, List, Any
from app.chatbot import build_chatbot_graph, IntentOutput, GraphState

# ---------------------------------------------------------------------------
# Session Manager – handles multiple users/sessions
# ---------------------------------------------------------------------------

import json
from app.services.recsys import get_redis_client

class ConversationMemory:
    """
    Manage per-session conversation memory.
    Uses Redis persistence with in-memory fallback.
    """
    _graph = None
    _sessions: Dict[str, List[BaseMessage]] = {}
    _max_history: int = 20 # Keep last 20 messages per session

    @classmethod
    def _get_graph(cls):
        if cls._graph is None:
            cls._graph = build_chatbot_graph()
        return cls._graph

    @classmethod
    def get_session_messages(cls, session_id: str) -> List[BaseMessage]:
        """
        Retrieve message history for a session from Redis or in-memory dict.
        """
        r = get_redis_client()
        if r:
            try:
                raw_list = r.lrange(f"chat:{session_id}:history", 0, -1)
                if raw_list:
                    messages = []
                    for raw in raw_list:
                        item = json.loads(raw)
                        if item.get("role") == "human":
                            msg = HumanMessage(content=item["content"])
                        else:
                            msg = AIMessage(content=item["content"])
                        if item.get("movies"):
                            msg.additional_kwargs["movies"] = item["movies"]
                        messages.append(msg)
                    cls._sessions[session_id] = messages
                    return messages
            except Exception:
                pass
        return cls._sessions.get(session_id, [])
    
    @classmethod
    def add_message(cls, session_id: str, message: BaseMessage, movies: List[Dict[str, Any]] = None):
        """
        Add a message to session history with size limit in Redis & memory.
        """
        if movies:
            message.additional_kwargs["movies"] = movies

        if session_id not in cls._sessions:
            cls._sessions[session_id] = []
        cls._sessions[session_id].append(message)
        if len(cls._sessions[session_id]) > cls._max_history:
            cls._sessions[session_id] = cls._sessions[session_id][-cls._max_history:]

        r = get_redis_client()
        if r:
            try:
                role = "human" if isinstance(message, HumanMessage) else "ai"
                payload = {"role": role, "content": message.content}
                if movies:
                    payload["movies"] = movies
                raw = json.dumps(payload, default=str)
                r.rpush(f"chat:{session_id}:history", raw)
                r.ltrim(f"chat:{session_id}:history", -cls._max_history, -1)
                r.expire(f"chat:{session_id}:history", 86400 * 7) # 7 days TTL
            except Exception:
                pass

    @classmethod
    def clear_session(cls, session_id: str):
        """
        Clear a session's history.
        """
        cls._sessions.pop(session_id, None)
        r = get_redis_client()
        if r:
            try:
                r.delete(f"chat:{session_id}:history")
            except Exception:
                pass

    @classmethod
    def get_session_summary(cls, session_id: str) -> Dict[str, Any]:
        """
        Get summary of session for debugging.
        """
        messages = cls.get_session_messages(session_id)
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": [{
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content[:100] + "..." if len(m.content) > 100 else m.content
            } for m in messages[-10:]] # Last 10 messages
        }
    
# ---------------------------------------------------------------------------
# Public service used by the route layer
# ---------------------------------------------------------------------------

class ChatbotService:
    """
    Main service interface for the chatbot.
    """
    @staticmethod
    def get_reply(
        message: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Get chatbot reply with conversation memory.

        Args:
            message: User's input message
            session_id: Unique identifier for the conversation  session

        Returns:
            Dict with "text" (reply) and "movies" (enriched movie list)
        """
        # Add the new user message
        user_message = HumanMessage(content=message)
        ConversationMemory.add_message(session_id, user_message)

        # Get existing conversation history for this session
        history = ConversationMemory.get_session_messages(session_id)

        # Build initial state with full history
        intent_output: IntentOutput = {
            "intent": "",
            "target_title": "",
            "genres": [],
        }

        initial_state: GraphState = {
            "messages": history,
            "intent_output": intent_output,
            "candidate_movies": [],
            "enriched_movies": [],
            "matched_movie": None,
            "final_text": ""
        }

        # Invoke the graph with thread_id for checkpointing
        graph = ConversationMemory._get_graph()
        config = {
            "configurable": {"thread_id": session_id}
        }
        result = graph.invoke(initial_state, config=config)

        # Update session memory with the AI response + recommended movies
        movies_list = result.get("enriched_movies", [])
        ai_message = AIMessage(content=result["final_text"])
        ConversationMemory.add_message(session_id, ai_message, movies=movies_list)

        return {
            "text": result["final_text"],
            "movies": movies_list,
            "session_id": session_id,
            "message_count": len(ConversationMemory.get_session_messages(session_id))
        }
    
    @staticmethod
    def get_history(session_id: str = "default") -> List[Dict[str, Any]]:
        """
        Get conversation history for a session with recommended movies if present.
        """
        messages = ConversationMemory.get_session_messages(session_id)
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content,
                "movies": m.additional_kwargs.get("movies", [])
            }
            for m in messages
        ]

    @staticmethod
    def clear_history(session_id: str = "default") -> Dict[str, str]:
        """Clear conversation history for a session."""
        ConversationMemory.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}

    @staticmethod
    def get_session_info(session_id: str = "default") -> Dict[str, Any]:
        """Get debugging info about a session."""
        return ConversationMemory.get_session_summary(session_id)

