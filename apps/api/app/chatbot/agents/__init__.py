import logging
import os
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(".env")
load_dotenv("../../.env")
load_dotenv("../.env")

from langchain_openai import ChatOpenAI
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


def get_openrouter_llm(
    model_name: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024
):
    """
    Dedicated LLM client via OpenRouter for intent detection and structured parameter extraction.
    Uses OPEN_ROUTER_API_KEY or OPENROUTER_API_KEY.
    """
    settings = None
    try:
        settings = get_settings()
    except Exception:
        pass

    openrouter_key = (
        os.getenv("OPEN_ROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or (getattr(settings, "OPENROUTER_API_KEY", None) if settings else None)
    )
    if openrouter_key:
        try:
            model = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
            logger.info(f"Initializing OpenRouter LLM for intent detection with model '{model}'")
            return ChatOpenAI(
                model=model,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
                timeout=12.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter LLM: {e}")

    # Fallback to general LLM if OpenRouter is unavailable
    return get_llm(model_name=model_name, temperature=temperature, max_tokens=max_tokens)


def get_deepseek_llm(
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024
):
    """
    Dedicated DeepSeek LLM Client for intent extraction and JSON output:
    1. Direct DeepSeek API (via DEEPSEEK_API_KEY, https://api.deepseek.com/v1, model 'deepseek-chat')
    2. OpenRouter DeepSeek (via OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY, https://openrouter.ai/api/v1, model 'deepseek/deepseek-chat')
    3. Token Harbor DeepSeek (via TOKEN_HABOR_API_KEY / TOKENHARBOR_API_KEY, https://tokenharbor.ai/v1, model 'deepseek-v4.1-flash:free')
    4. Fallback to get_llm
    """
    settings = None
    try:
        settings = get_settings()
    except Exception:
        pass

    # 1. Direct DeepSeek Official API
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or (getattr(settings, "DEEPSEEK_API_KEY", None) if settings else None)
    if deepseek_key:
        try:
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            logger.info(f"Initializing Direct DeepSeek LLM with model '{model}'")
            return ChatOpenAI(
                model=model,
                api_key=deepseek_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
                timeout=20.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize direct DeepSeek LLM: {e}")

    # 2. OpenRouter DeepSeek
    openrouter_key = (
        os.getenv("OPEN_ROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or (getattr(settings, "OPENROUTER_API_KEY", None) if settings else None)
    )
    if openrouter_key:
        try:
            model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat")
            logger.info(f"Initializing OpenRouter DeepSeek LLM with model '{model}'")
            return ChatOpenAI(
                model=model,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
                timeout=20.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter DeepSeek LLM: {e}")

    # 3. Token Harbor DeepSeek
    tokenharbor_key = (
        os.getenv("TOKEN_HABOR_API_KEY")
        or os.getenv("TOKENHARBOR_API_KEY")
        or (getattr(settings, "TOKENHARBOR_API_KEY", None) if settings else None)
    )
    if tokenharbor_key:
        try:
            base_url = os.getenv("TOKENHARBOR_BASE_URL", "https://tokenharbor.ai/v1")
            model = model_name or os.getenv("TOKENHARBOR_MODEL", "deepseek-v4.1-flash:free")
            logger.info(f"Initializing Token Harbor DeepSeek LLM with model '{model}'")
            return ChatOpenAI(
                model=model,
                api_key=tokenharbor_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
                timeout=20.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Token Harbor DeepSeek LLM: {e}")

    target_model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    fallback_llm = get_llm(model_name=target_model, temperature=temperature, max_tokens=max_tokens)
    if fallback_llm and ("deepseek" in getattr(fallback_llm, "model_name", "").lower() or "deepseek" in str(getattr(fallback_llm, "openai_api_base", "")).lower()):
        return fallback_llm

    return ChatOpenAI(
        model=target_model,
        api_key=os.getenv("DEEPSEEK_API_KEY") or "sk-dummy-deepseek",
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_tokenharbor_llm(
    model_name: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048
):
    """
    Dedicated LLM client via Token Harbor for final answer generation with DeepSeek V4.1 Flash.
    Uses TOKEN_HABOR_API_KEY or TOKENHARBOR_API_KEY with base_url https://tokenharbor.ai/v1.
    """
    tokenharbor_key = (
        os.getenv("TOKEN_HABOR_API_KEY")
        or os.getenv("TOKENHARBOR_API_KEY")
    )
    if tokenharbor_key:
        try:
            base_url = os.getenv("TOKENHARBOR_BASE_URL", "https://tokenharbor.ai/v1")
            model = model_name or os.getenv("TOKENHARBOR_MODEL", "deepseek-v4.1-flash:free")
            logger.info(f"Initializing Token Harbor ChatOpenAI with model '{model}' at '{base_url}'")
            return ChatOpenAI(
                model=model,
                api_key=tokenharbor_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
                timeout=25.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Token Harbor LLM: {e}")

    # Fallback to OpenRouter or general LLM if Token Harbor key not provided
    return get_openrouter_llm(model_name=model_name, temperature=temperature, max_tokens=max_tokens)



def get_llm(
    model_name: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024
):
    """
    Multi-provider LLM Factory:
    Priority 1: Token Harbor (if configured for answer generation)
    Priority 2: OpenRouter (via OPEN_ROUTER_API_KEY / OPENROUTER_API_KEY)
    Priority 3: Google Gemini (via GOOGLE_API_KEY or GEMINI_API_KEY)
    Priority 4: ZhipuAI (via ZHIPUAI_API_KEY or ZAI_API_KEY)
    """
    # 1. OpenRouter
    openrouter_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            model = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
            return ChatOpenAI(
                model=model,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=1,
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenRouter LLM: {e}")

    # 2. Google Gemini Direct
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = model_name or "gemini-2.5-flash"
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=15.0,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Google Gemini LLM: {e}")

    # 3. ZhipuAI
    zhipu_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if zhipu_key:
        base_url = os.getenv("ZHIPUAI_BASE_URL") or os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
        model = model_name or os.getenv("ZHIPUAI_MODEL") or os.getenv("ZAI_MODEL", "glm-4.7-flash")
        return ChatOpenAI(
            model=model,
            api_key=zhipu_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=1,
            timeout=10.0,
        )

    logger.warning("No valid LLM API keys found in environment. Fallback conversational RecSys agent will be utilized.")
    return None


from .detect_intent import detect_intent_node
from .enrich_movies import enrich_movies_node
from .generate_answer import generate_answer_node
from .query_movies import query_movies_node

__all__ = [
    "detect_intent_node",
    "enrich_movies_node",
    "generate_answer_node",
    "get_deepseek_llm",
    "get_llm",
    "get_openrouter_llm",
    "get_tokenharbor_llm",
    "query_movies_node",
]