import os
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

def get_llm(
    model_name="gemini-2.5-flash", 
    temperature=0.3,
    max_tokens=5000
) -> ChatGoogleGenerativeAI:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return "ERROR: GOOGLE_API_KEY environment variable is not configured."
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=google_api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

def get_tavily() -> TavilyClient:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "ERROR: TAVILY_API_KEY environment variable is not configured."

    return TavilyClient(api_key=tavily_api_key)  

from .detect_intent import detect_intent_node
from .generate_answer import generate_answer_node
from .query_movies import query_movies_node
from .tavily_search import tavily_search_node  