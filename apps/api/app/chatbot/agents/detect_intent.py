import json
import logging
import re
from typing import Any

from app.chatbot.agents import get_deepseek_llm
from app.chatbot.prompts import INTENT_PROMPT
from app.chatbot.state import (
    GraphState,
    IntentOutput,
    parse_intent_to_output,
)

logger = logging.getLogger(__name__)

# Genre Mapping using distinct keywords with word boundaries
GENRE_MAP = {
    "action": "Action", "hành động": "Action",
    "adventure": "Adventure", "phiêu lưu": "Adventure",
    "animation": "Animation", "hoạt hình": "Animation", "anime": "Animation",
    "comedy": "Comedy", "hài": "Comedy", "hài hước": "Comedy",
    "crime": "Crime", "tội phạm": "Crime", "hình sự": "Crime",
    "documentary": "Documentary", "tài liệu": "Documentary",
    "drama": "Drama", "tâm lý": "Drama", "chính kịch": "Drama",
    "fantasy": "Fantasy", "giả tưởng": "Fantasy",
    "horror": "Horror", "kinh dị": "Horror", "phim ma": "Horror",
    "mystery": "Mystery", "bí ẩn": "Mystery", "trinh thám": "Mystery",
    "romance": "Romance", "lãng mạn": "Romance", "tình cảm": "Romance",
    "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "khoa học viễn tưởng": "Sci-Fi", "viễn tưởng": "Sci-Fi",
    "thriller": "Thriller", "giật gân": "Thriller", "hồi hộp": "Thriller",
    "war": "War", "chiến tranh": "War",
    "western": "Western", "cao bồi": "Western"
}


def detect_user_language(text: str) -> str:
    """Detect whether user input is Vietnamese ('vi') or English ('en')."""
    text_lower = text.lower().strip()
    vi_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    if any(c in vi_chars for c in text_lower):
        return "vi"

    vi_tokens = {
        "phim", "tìm", "gợi", "cho", "tôi", "muốn", "xem", "nào", "hay", "không",
        "hài", "kinh", "dị", "chiến", "tranh", "tình", "cảm", "hành", "động",
        "hoạt", "hình", "chào", "bạn", "giúp", "ơi", "nhé", "ạ", "ơn", "mà",
        "đạo", "diễn", "viên", "bao", "nhiêu", "giải", "thích", "kết", "thúc",
        "thêm", "nữa", "tiếp", "khác"
    }
    tokens = set(re.findall(r"\b\w+\b", text_lower))
    if len(tokens & vi_tokens) > 0:
        return "vi"

    return "en"


def extract_json_from_llm_response(text: str) -> dict[str, Any] | None:
    """Robustly extract and parse JSON object from LLM text output."""
    if not text:
        return None
    cleaned = text.strip()
    # Check for markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    # Find outermost braces
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        cleaned = brace_match.group(0).strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        logger.debug(f"JSON parse error from LLM response: {e}")
        return None


def heuristic_detect_intent(user_text: str) -> IntentOutput:
    """Fast, deterministic fallback intent and entity classifier across dataset fields."""
    text_lower = user_text.lower().strip()
    lang = detect_user_language(user_text)

    # 1. Check for Greetings / General QA / Plot / Trivia / Chitchat
    qa_patterns = [
        r"\b(?:chào|hello|hi|hey|alo)\b",
        r"\b(?:bạn là ai|bạn có thể|who are you|what can you do)\b",
        r"\b(?:giải thích|cốt truyện|kết thúc|ý nghĩa|plot of|ending of)\b"
    ]
    if any(re.search(pat, text_lower) for pat in qa_patterns):
        return IntentOutput(intent="general_qa", target_title="", genres=[], language=lang)

    # 2. Check for 'More / Next Recommendations' Intent
    more_patterns = [
        r"(?:cho thêm|gợi ý thêm|còn phim nào|còn bộ nào|xem thêm|tiếp theo|nữa không|thêm phim|phim khác)",
        r"(?:more movies|show me more|give me more|what else|other movies|next recommendations)"
    ]
    if any(re.search(pat, text_lower) for pat in more_patterns):
        return IntentOutput(
            intent="more_recommendations",
            is_more_request=True,
            language=lang
        )

    # 3. Check for Movie Detail Intent (e.g. 'chi tiết batman begins', 'thông tin phim Interstellar')
    detail_patterns = [
        r"(?:chi tiết|thông tin|nội dung|cốt truyện|giới thiệu|review|tìm hiểu|về bộ phim|về phim)\s+(?:về\s+)?['\"]?([^'\"?.,!]+)['\"]?",
        r"(?:details of|about movie|tell me about|info on|synopsis of)\s+['\"]?([^'\"?.,!]+)['\"]?"
    ]
    for pat in detail_patterns:
        match = re.search(pat, text_lower)
        if match:
            target = match.group(1).strip()
            target = re.sub(r"\s+(?:đi|nhé|ạ|cho tôi|nào)$", "", target).strip()
            if target and target not in ("gì", "nào"):
                return IntentOutput(intent="movie_detail", target_title=target, genres=[], language=lang)

    # 4. Check for Similar Movie Intent
    similar_patterns = [
        r"(?:similar to|movies like|films like|resemble|tương tự|giống phim|giống như)\s+['\"]?([^'\"?.,!]+)['\"]?",
        r"(?:phim nào giống|phim tương tự)\s+['\"]?([^'\"?.,!]+)['\"]?"
    ]
    for pat in similar_patterns:
        match = re.search(pat, text_lower)
        if match:
            target = match.group(1).strip()
            return IntentOutput(intent="similar", target_title=target, genres=[], language=lang)

    # 5. Check for Director / Actor / Year / Rating search fields
    director = ""
    dir_match = re.search(r"(?:đạo diễn|directed by|director)\s+(?:là\s+)?(.+?)(?=\s+(?:năm|điểm|diễn|hay|và)|$)", text_lower)
    if dir_match:
        director = dir_match.group(1).strip()

    actors = []
    actor_match = re.search(r"(?:diễn viên|starring|đóng bởi|diễn xuất bởi|có sự tham gia của)\s+(?:là\s+)?(.+?)(?=\s+(?:đóng|năm|điểm|đạo diễn|hay|và)|$)", text_lower)
    if not actor_match and "có" in text_lower and "có thể" not in text_lower:
        actor_match = re.search(r"\bcó\s+([A-Za-zÀ-ỹ\s]+?)(?=\s+(?:đóng|năm|điểm|đạo diễn|hay|và)|$)", text_lower)

    if actor_match:
        act_val = actor_match.group(1).strip()
        if act_val and act_val not in ("gì", "thể", "nào", "gì không"):
            actors = [act_val]

    year = ""
    year_match = re.search(r"\b(?:năm|year)?\s*(19\d\d|20\d\d|2010s|2020s)\b", text_lower)
    if year_match:
        year = year_match.group(1)

    min_rating = None
    rating_match = re.search(r"(?:điểm|rating|imdb)\s*(?:trên|>=|>)?\s*(\d(?:\.\d)?)", text_lower)
    if rating_match:
        try:
            min_rating = float(rating_match.group(1))
        except ValueError:
            pass

    # 5. Check for Specific SQL Database Counts / Watchlist
    sql_indicators = [
        r"\b(?:bao nhiêu phim|tổng số phim|how many movies|total movies)\b",
        r"\b(?:danh sách theo dõi|watchlist|phim tôi đã lưu)\b",
    ]
    if any(re.search(pat, text_lower) for pat in sql_indicators):
        return IntentOutput(intent="sql_query", target_title="", director=director, actors=actors, year=year, min_rating=min_rating, language=lang)

    # 6. Check for Genre Intent with Word Boundary Matching
    found_genres: list[str] = []
    for keyword, canonical_genre in GENRE_MAP.items():
        if (
            re.search(r'(?:\b|_)' + re.escape(keyword) + r'(?:\b|_)', text_lower)
            and canonical_genre not in found_genres
        ):
            found_genres.append(canonical_genre)

    # Check for single specific criteria intents
    active_filters = sum([bool(year), bool(director), bool(actors), bool(found_genres), bool(min_rating is not None)])

    if active_filters == 1:
        if year:
            return IntentOutput(intent="year", year=year, language=lang)
        if director:
            return IntentOutput(intent="director", director=director, language=lang)
        if actors:
            return IntentOutput(intent="actor", actors=actors, language=lang)
        if found_genres:
            return IntentOutput(intent="genre", genres=found_genres, language=lang)
        if min_rating is not None:
            return IntentOutput(intent="rating", min_rating=min_rating, language=lang)

    # Multiple filters combined -> search_filter
    if active_filters > 1:
        return IntentOutput(
            intent="search_filter",
            target_title="",
            genres=found_genres,
            director=director,
            actors=actors,
            year=year,
            min_rating=min_rating,
            language=lang
        )

    # 7. Check for Personalized / General Recommendation Intent
    personal_keywords = [
        "recommend", "suggest", "for me", "what to watch", "gợi ý", "cho tôi",
        "phim hay", "nên xem", "top movie", "best movie", "bộ phim nào"
    ]
    if any(k in text_lower for k in personal_keywords):
        return IntentOutput(intent="personalized", target_title="", genres=[], language=lang)

    # 8. Check for Follow-up
    if any(k in text_lower for k in ["first", "second", "third", "thứ nhất", "thứ hai", "bộ đó", "phim đó"]):
        return IntentOutput(intent="followup", target_title="", genres=[], language=lang)

    return IntentOutput(intent="fallback", target_title="", genres=[], language=lang)


def detect_intent_node(state: GraphState) -> dict:
    """
    Classify user intent and extract query parameters into JSON using DeepSeek LLM.
    Validates output with specialized BaseModel per query case:
    - 'year': { "intent": "year", "year": "..." }
    - 'director': { "intent": "director", "director": "..." }
    - 'actor': { "intent": "actor", "actors": [...] }
    - 'genre': { "intent": "genre", "genres": [...] }
    - 'rating': { "intent": "rating", "min_rating": ... }
    - 'similar': { "intent": "similar", "target_title": "..." }
    - 'search_filter': multi-criteria combined
    - 'sql_query': catalog stats / counts
    - 'more_recommendations': next batch request
    - 'personalized': user recommendations
    - 'general_qa': plot, ending, trivia, chitchat
    """
    messages = state.get("messages", [])
    last_text = messages[-1].content if messages else ""
    lang = detect_user_language(last_text)

    # 1. Fast deterministic check for 'more recommendations'
    heuristic_res = heuristic_detect_intent(last_text)
    if heuristic_res.is_more_request:
        heuristic_res.language = lang
        logger.info(f"Detected 'more_recommendations' request via heuristic for session {state.get('session_id')}")
        return {"intent_output": heuristic_res, "detected_language": lang}

    # 2. Consult DeepSeek LLM for structured JSON intent and entity classification
    llm = get_deepseek_llm(temperature=0.0, max_tokens=300)
    if llm is not None:
        try:
            prompt = INTENT_PROMPT
            chain = prompt | llm
            response = chain.invoke({"history": messages})
            raw_content = response.content if hasattr(response, "content") else str(response)
            parsed_json = extract_json_from_llm_response(raw_content)

            if parsed_json and isinstance(parsed_json, dict):
                # Validate and convert using specialized BaseModels
                intent_output = parse_intent_to_output(parsed_json, lang=lang)
                logger.info(
                    f"DeepSeek LLM detected intent='{intent_output.intent}', "
                    f"year='{intent_output.year}', director='{intent_output.director}', "
                    f"actors={intent_output.actors}, genres={intent_output.genres}, "
                    f"min_rating={intent_output.min_rating}, target='{intent_output.target_title}'"
                )
                return {"intent_output": intent_output, "detected_language": lang}
        except Exception as e:
            logger.warning(f"DeepSeek LLM intent detection encountered: {e}")

    # 3. Fallback to heuristic detector
    heuristic_res.language = lang
    return {"intent_output": heuristic_res, "detected_language": lang}