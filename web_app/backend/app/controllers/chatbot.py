from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.chatbot import ChatbotService

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot"])

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    text: str
    movies: List[Dict[str, Any]]
    session_id: str
    message_count: int

class HistoryResponse(BaseModel):
    messages: List[Dict[str, str]]

class ClearResponse(BaseModel):
    status: str
    session_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get AI response with movie recommendation.
    """
    try:
        result = ChatbotService.get_reply(
            message=request.message,
            session_id=request.session_id
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str = "default"):
    """Get conversation history for a session."""
    messages = ChatbotService.get_history(session_id)
    return HistoryResponse(messages=messages)

@router.delete("/history/{session_id}", response_model=ClearResponse)
async def clear_history(session_id: str = "default"):
    """Clear conversation history for a session."""
    result = ChatbotService.clear_history(session_id)
    return ClearResponse(**result)


@router.get("/session/{session_id}")
async def get_session_info(session_id: str = "default"):
    """Get debugging info about a session."""
    return ChatbotService.get_session_info(session_id)
