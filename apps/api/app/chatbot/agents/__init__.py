import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024
):
    """
    Multi-provider LLM Factory:
    Priority 1: OpenRouter (supports Gemini 2.0 Flash, Llama 3.3, Claude via OPENROUTER_API_KEY)
    Priority 2: Google Gemini (via GOOGLE_API_KEY or GEMINI_API_KEY)
    Priority 3: ZhipuAI (via ZHIPUAI_API_KEY or ZAI_API_KEY)
    """
    # 1. Google Gemini Direct (Khuyên dùng: siêu nhanh, hỗ trợ đa ngôn ngữ hoàn hảo)
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = model_name or "gemini-2.5-flash"
            logger.info(f"Initializing ChatGoogleGenerativeAI with model '{model}'")
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=15.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Google Gemini LLM: {e}")

    # 2. OpenRouter (Secondary fallback)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            model = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
            logger.info(f"Initializing ChatOpenAI via OpenRouter with model '{model}'")
            return ChatOpenAI(
                model=model,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
                timeout=8.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter LLM: {e}")

    # 3. ZhipuAI
    zhipu_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if zhipu_key:
        base_url = os.getenv("ZHIPUAI_BASE_URL") or os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
        model = model_name or os.getenv("ZHIPUAI_MODEL") or os.getenv("ZAI_MODEL", "glm-4.7-flash")
        logger.info(f"Initializing ChatOpenAI via ZhipuAI with model '{model}'")
        return ChatOpenAI(
            model=model,
            api_key=zhipu_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=1,
            timeout=10.0,
        )

    # Fallback to local default ChatOpenAI instance
    logger.warning("No valid LLM API keys found in environment. Fallback conversational RecSys agent will be utilized.")
    return None


from .detect_intent import detect_intent_node
from .generate_answer import generate_answer_node
from .query_movies import query_movies_node
from .enrich_movies import enrich_movies_node

__all__ = [
    "get_llm",
    "detect_intent_node",
    "generate_answer_node",
    "query_movies_node",
    "enrich_movies_node",
]