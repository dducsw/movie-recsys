import json
import logging
from typing import List, Dict, Any
from langchain_core.messages import AIMessage
from app.chatbot.state import GraphState
from app.chatbot.prompts import ANSWER_PROMPT
from app.chatbot.agents import get_llm

logger = logging.getLogger(__name__)


def build_fallback_response(
    user_query: str,
    intent: str,
    matched_movie: Dict[str, Any],
    enriched: List[Dict[str, Any]],
    lang: str = "vi",
    genres: List[str] = None
) -> str:
    """Generate a clean, beautiful Markdown response matching the user's input language."""
    is_vi = (lang == "vi")
    genres_list = genres or []
    genre_display = ", ".join(genres_list) if genres_list else ("hài" if is_vi else "comedy")

    if not enriched:
        if is_vi:
            return (
                "Chào bạn! Tôi là **MovieNex AI** — trợ lý gợi ý phim thông minh của bạn. 🎬\n\n"
                "Tôi có thể giúp bạn tìm kiếm phim theo sở thích cá nhân, gợi ý phim tương tự, hoặc tìm theo thể loại bạn yêu thích.\n\n"
                "👉 *Bạn có thể thử hỏi:* \n"
                "- *'Gợi ý cho tôi phim hài hước nhẹ nhàng'* \n"
                "- *'Có phim hành động khoa học viễn tưởng nào giống Inception không?'*"
            )
        else:
            return (
                "Hello! I am **MovieNex AI** — your intelligent movie discovery assistant. 🎬\n\n"
                "I can help you discover personalized movies, find similar films, or filter by your favorite genres.\n\n"
                "👉 *You can try asking:* \n"
                "- *'Recommend me some lighthearted comedy movies'* \n"
                "- *'What are some sci-fi action movies like Inception?'*"
            )

    # Header
    if matched_movie and intent == "similar":
        orig_title = matched_movie.get("title", "phim bạn yêu cầu" if is_vi else "your requested movie")
        header = (
            f"Dựa trên bộ phim **{orig_title}**, hệ thống MovieNex AI đã tuyển chọn những bộ phim tương đồng nhất dành cho bạn:\n\n"
            if is_vi else
            f"Based on **{orig_title}**, here are the most relevant movie recommendations curated for you:\n\n"
        )
    elif intent == "genre":
        header = (
            f"Dưới đây là các tác phẩm đặc sắc nhất thuộc thể loại **{genre_display}** dành cho bạn:\n\n"
            if is_vi else
            f"Here are top-rated **{genre_display}** movies curated for you:\n\n"
        )
    else:
        header = (
            "Hệ thống MovieNex AI đã tuyển chọn những bộ phim phù hợp nhất với yêu cầu của bạn:\n\n"
            if is_vi else
            "Here are the best movie picks matching your request:\n\n"
        )

    items_text = []
    for idx, m in enumerate(enriched[:5], 1):
        title = m.get("title", "Không tên" if is_vi else "Untitled")
        year = str(m.get("release_date") or "2024").split("-")[0]
        g_str = m.get("genres", "").replace("|", ", ")
        vote = round(float(m.get("vote_average") or 0.0), 1)
        overview = m.get("description") or m.get("overview") or (
            "Một tác phẩm điện ảnh xuất sắc với cốt truyện hấp dẫn và diễn xuất ấn tượng."
            if is_vi else
            "A compelling cinematic masterpiece with outstanding storytelling and performances."
        )
        if len(overview) > 160:
            overview = overview[:157] + "..."

        if is_vi:
            block = (
                f"### {idx}. 🎬 **{title}** *({year})*\n"
                f"- **Thể loại:** {g_str} | **Đánh giá:** ⭐ {vote}/10\n"
                f"- **Nội dung:** {overview}\n"
            )
        else:
            block = (
                f"### {idx}. 🎬 **{title}** *({year})*\n"
                f"- **Genres:** {g_str} | **Rating:** ⭐ {vote}/10\n"
                f"- **Plot:** {overview}\n"
            )
        items_text.append(block)

    footer = (
        "\n---\n💡 *Bạn có thể nhấn trực tiếp vào thẻ phim bên dưới để xem chi tiết, trailer hoặc thêm vào danh sách yêu thích!*"
        if is_vi else
        "\n---\n💡 *Click on any movie card below to view full details, watch trailers, or add to your watchlist!*"
    )
    return header + "\n".join(items_text) + footer


def generate_answer_node(state: GraphState) -> dict:
    """Generate contextual response strictly in the user's detected language."""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""
    enriched = state.get("enriched_movies", [])
    intent_output = state.get("intent_output")
    intent = intent_output.intent if intent_output else "fallback"
    matched = state.get("matched_movie")
    genres = intent_output.genres if intent_output else []

    detected_lang = state.get("detected_language") or (intent_output.language if intent_output else "vi")
    response_lang_name = "Vietnamese" if detected_lang == "vi" else "English"

    final_text = ""
    filtered_movies = enriched

    llm = get_llm()
    if llm is not None:
        try:
            movie_context = ""
            if matched and intent == "similar":
                movie_context += f"Referenced original movie: {matched.get('title', 'Unknown')}\n\n"

            if enriched:
                movie_list_str = json.dumps(
                    [
                        {
                            "title": m.get("title"),
                            "release_date": m.get("release_date"),
                            "genres": m.get("genres"),
                            "description": m.get("description", ""),
                            "image_url": m.get("image_url")
                        }
                        for m in enriched[:8]
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                movie_context += f"List of recommended candidate movies:\n{movie_list_str}"
            else:
                movie_context = "No specific movies found for this query."

            prompt = ANSWER_PROMPT
            chain = prompt | llm
            response = chain.invoke({
                "history": messages,
                "movie_context": movie_context,
                "response_language": response_lang_name
            })
            final_text = response.content.strip()

            # Filter or order candidates mentioned in LLM answer
            if enriched and final_text:
                mentioned = []
                for m in enriched:
                    t = m.get("title", "")
                    if t and t.lower() in final_text.lower():
                        mentioned.append(m)
                if mentioned:
                    filtered_movies = mentioned
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}. Generating localized fallback response.")
            final_text = ""

    if not final_text:
        final_text = build_fallback_response(
            user_query=last_msg,
            intent=intent,
            matched_movie=matched,
            enriched=enriched,
            lang=detected_lang,
            genres=genres
        )

    return {
        "final_text": final_text,
        "enriched_movies": filtered_movies,
        "messages": [AIMessage(content=final_text)],
    }