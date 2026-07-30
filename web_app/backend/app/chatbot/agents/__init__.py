import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(
    model_name="gemini-2.5-flash", 
    temperature=0.3,
    max_tokens=5000
) -> ChatGoogleGenerativeAI:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("ERROR: GOOGLE_API_KEY environment variable is not configured.")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=google_api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )

from .detect_intent import detect_intent_node
from .generate_answer import generate_answer_node
from .query_movies import query_movies_node
from .enrich_movies import enrich_movies_node