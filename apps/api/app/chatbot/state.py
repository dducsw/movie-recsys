from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class IntentOutput(BaseModel):
    intent: str = Field(
        default="fallback",
        description="'similar', 'genre', 'personalized', 'followup', 'sql_query', 'general_qa', or 'fallback'"
    )
    target_title: str = Field(default="")
    genres: list[str] = Field(default_factory=list)
    language: str = Field(default="vi", description="'vi' for Vietnamese, 'en' for English")


class GraphState(TypedDict):
    """
    Graph state for the MovieNex Conversational Assistant.
    `messages` is automatically reduced by `add_messages`.
    """
    messages: Annotated[list[BaseMessage], add_messages]

    intent_output: IntentOutput | None
    detected_language: str | None
    matched_movie: dict[str, Any] | None
    candidate_movies: list[dict[str, Any]]
    enriched_movies: list[dict[str, Any]]
    user_id: int | None
    user_liked_ids: list[int] | None

    # SQL Retrieval context
    sql_query: str | None
    sql_results: list[dict[str, Any]] | None
    sql_message: str | None

    final_text: str
