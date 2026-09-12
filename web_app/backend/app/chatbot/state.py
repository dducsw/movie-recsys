from typing import List, Annotated, TypedDict, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

class IntentOutput(BaseModel):
    intent: str = Field(default="fallback", description="similar, genre, personalized, followup, or fallback")
    target_title: str = Field(default="")
    genres: List[str] = Field(default_factory=list)
    language: str = Field(default="vi", description="'vi' for Vietnamese, 'en' for English")

class GraphState(TypedDict):
    """Graph state. 
    `messages` is automatically reduced by `add_messages`."""
    messages: Annotated[List[BaseMessage], add_messages]

    intent_output: Optional[IntentOutput]
    detected_language: Optional[str]
    matched_movie: Optional[Dict[str, Any]]
    candidate_movies: List[Dict[str, Any]]
    enriched_movies: List[Dict[str, Any]]
    user_id: Optional[int]
    user_liked_ids: Optional[List[int]]

    final_text: str

