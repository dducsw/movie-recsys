from typing import List, Annotated, TypedDict, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

class IntentOutput(BaseModel):
    intent: Literal[
        "similar",
        "genre",
        "followup",
        "fallback"
    ]
    target_title: str = Field(default="")
    genres: List[str] = Field(default_factory=list)

class GraphState(TypedDict):
    """Graph state. 
    `messages` is automatically reduced by `add_messages`."""
    messages: Annotated[List[BaseMessage], add_messages]

    intent_output = Optional[IntentOutput]
    matched_movie: Optional[Dict[str, Any]]
    candidate_movies: List[Dict[str, Any]]
    enriched_movies: List[Dict[str, Any]]

    final_text: str
