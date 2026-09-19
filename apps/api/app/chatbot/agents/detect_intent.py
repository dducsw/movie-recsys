import logging
import re

from app.chatbot.agents import get_llm
from app.chatbot.prompts import INTENT_PROMPT
from app.chatbot.state import GraphState, IntentOutput

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
        "đạo", "diễn", "viên", "bao", "nhiêu", "giải", "thích", "kết", "thúc"
    }
    tokens = set(re.findall(r"\b\w+\b", text_lower))
    if len(tokens & vi_tokens) > 0:
        return "vi"

    return "en"


def heuristic_detect_intent(user_text: str) -> IntentOutput:
    """Fast, deterministic fallback intent classifier for RecSys and SQL queries."""
    text_lower = user_text.lower().strip()
    lang = detect_user_language(user_text)

    # 1. Check for Similar Movie Intent
    similar_patterns = [
        r"(?:similar to|movies like|films like|resemble|tương tự|giống phim|giống như)\s+['\"]?([^'\"?.,!]+)['\"]?",
        r"(?:phim nào giống|phim tương tự)\s+['\"]?([^'\"?.,!]+)['\"]?"
    ]
    for pat in similar_patterns:
        match = re.search(pat, text_lower)
        if match:
            target = match.group(1).strip()
            return IntentOutput(intent="similar", target_title=target, genres=[], language=lang)

    # 2. Check for Specific SQL Queries (Director, Actor, Count, Watchlist, Year stats)
    sql_indicators = [
        r"\b(?:đạo diễn|directed by|director)\b",
        r"\b(?:diễn viên|starring|đóng bởi|diễn xuất)\b",
        r"\b(?:bao nhiêu phim|tổng số phim|how many movies|total movies)\b",
        r"\b(?:danh sách theo dõi|watchlist|phim tôi đã lưu)\b",
        r"\b(?:phim năm 20\d\d|phim năm 19\d\d|top phim 20\d\d)\b"
    ]
    if any(re.search(pat, text_lower) for pat in sql_indicators):
        return IntentOutput(intent="sql_query", target_title="", genres=[], language=lang)

    # 3. Check for Genre Intent with Word Boundary Matching
    found_genres: list[str] = []
    for keyword, canonical_genre in GENRE_MAP.items():
        if (
            re.search(r'(?:\b|_)' + re.escape(keyword) + r'(?:\b|_)', text_lower)
            and canonical_genre not in found_genres
        ):
            found_genres.append(canonical_genre)

    if found_genres:
        return IntentOutput(intent="genre", target_title="", genres=found_genres, language=lang)

    # 4. Check for Greetings / General QA / Plot / Trivia / Chitchat
    qa_patterns = [
        r"\b(?:chào|hello|hi|hey|alo)\b",
        r"\b(?:bạn là ai|bạn có thể|who are you|what can you do)\b",
        r"\b(?:giải thích|cốt truyện|kết thúc|ý nghĩa|plot of|ending of)\b"
    ]
    if any(re.search(pat, text_lower) for pat in qa_patterns):
        return IntentOutput(intent="general_qa", target_title="", genres=[], language=lang)

    # 5. Check for Personalized / General Recommendation Intent
    personal_keywords = [
        "recommend", "suggest", "for me", "what to watch", "gợi ý", "cho tôi",
        "phim hay", "nên xem", "top movie", "best movie", "bộ phim nào"
    ]
    if any(k in text_lower for k in personal_keywords):
        return IntentOutput(intent="personalized", target_title="", genres=[], language=lang)

    # 6. Check for Follow-up
    if any(k in text_lower for k in ["first", "second", "third", "thứ nhất", "thứ hai", "bộ đó", "phim đó"]):
        return IntentOutput(intent="followup", target_title="", genres=[], language=lang)

    return IntentOutput(intent="fallback", target_title="", genres=[], language=lang)


def detect_intent_node(state: GraphState) -> dict:
    """Classify user intent using fast deterministic detector, with LLM for open-ended queries."""
    messages = state.get("messages", [])
    last_text = messages[-1].content if messages else ""
    lang = detect_user_language(last_text)

    # 1. Fast deterministic check
    heuristic_res = heuristic_detect_intent(last_text)
    if heuristic_res.intent not in ("fallback", "general_qa"):
        heuristic_res.language = lang
        logger.info(f"Heuristic intent detected: {heuristic_res.intent}, target: {heuristic_res.target_title}, genres: {heuristic_res.genres}, lang: {lang}")
        return {"intent_output": heuristic_res, "detected_language": lang}

    # 2. Consult LLM for nuanced or open-ended classification
    llm = get_llm()
    if llm is not None:
        try:
            prompt = INTENT_PROMPT
            structured_llm = llm.with_structured_output(IntentOutput)
            chain = prompt | structured_llm
            intent_output = chain.invoke({"history": messages})
            if intent_output and intent_output.intent:
                intent_output.language = lang
                logger.info(f"LLM detected intent: {intent_output.intent}, target: {intent_output.target_title}, genres: {intent_output.genres}, lang: {lang}")
                return {"intent_output": intent_output, "detected_language": lang}
        except Exception as e:
            logger.warning(f"LLM intent detection encounter: {e}")

    heuristic_res.language = lang
    return {"intent_output": heuristic_res, "detected_language": lang}