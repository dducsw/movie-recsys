import os
from langchain_openai import ChatOpenAI

def get_llm(
    model_name=None, 
    temperature=0.3,
    max_tokens=4096
) -> ChatOpenAI:
    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if not api_key:
        raise ValueError("ERROR: ZHIPUAI_API_KEY environment variable is not configured.")
    
    base_url = os.getenv("ZHIPUAI_BASE_URL") or os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    model = model_name or os.getenv("ZHIPUAI_MODEL") or os.getenv("ZAI_MODEL", "glm-4.7")
    
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=1,
        timeout=4.0,
    )

from .detect_intent import detect_intent_node
from .generate_answer import generate_answer_node
from .query_movies import query_movies_node
from .enrich_movies import enrich_movies_node