import json
import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.chatbot.agents import get_llm
from app.chatbot.prompts import ANSWER_PROMPT
from app.chatbot.state import GraphState

logger = logging.getLogger(__name__)


def build_fallback_response(
    user_query: str,
    intent: str,
    matched_movie: dict[str, Any] | None,
    enriched: list[dict[str, Any]],
    sql_results: list[dict[str, Any]] | None = None,
    lang: str = "vi",
    genres: list[str] | None = None
) -> str:
    """Generate a clean, beautiful Markdown response matching the user's input language when LLM is offline."""
    is_vi = (lang == "vi")
    genres_list = genres or []
    genre_display = ", ".join(genres_list) if genres_list else ("hài" if is_vi else "comedy")

    # 1. SQL Result Fallback
    if sql_results is not None:
        if not sql_results:
            return (
                "Rất tiếc, hệ thống không tìm thấy bản ghi nào trong cơ sở dữ liệu phù hợp với yêu cầu của bạn. 🔍"
                if is_vi else
                "Sorry, no matching records were found in the database for your query. 🔍"
            )

        # Case A: Aggregate Count
        first_row = sql_results[0]
        if "total_movies" in first_row or "count" in first_row:
            val = first_row.get("total_movies") or first_row.get("count")
            return (
                f"📊 Theo thống kê cơ sở dữ liệu MovieNex, hiện có tổng cộng **{val:,}** bản ghi trong hệ thống."
                if is_vi else
                f"📊 According to MovieNex database records, there are currently **{val:,}** items in the system."
            )

        # Case B: Entity records / Movies
        header = (
            "Dưới đây là kết quả tra cứu trực tiếp từ cơ sở dữ liệu MovieNex:\n\n"
            if is_vi else
            "Here are the direct records retrieved from the MovieNex catalog:\n\n"
        )
        lines = []
        for i, row in enumerate(sql_results[:8], 1):
            title = row.get("title") or row.get("name") or "Untitled"
            year = (row.get("release_date") or "")[:4]
            year_str = f" *({year})*" if year else ""
            director = f" | Đạo diễn: {row['director']}" if row.get("director") else ""
            rating = f" | ⭐ {row['vote_average']}/10" if row.get("vote_average") else ""
            lines.append(f"{i}. **{title}**{year_str}{rating}{director}")

        return header + "\n".join(lines)

    # 2. Recommendation Fallbacks
    if not enriched:
        if is_vi:
            return (
                "Chào bạn! Tôi là **MovieNex AI** — trợ lý điện ảnh & gợi ý phim thông minh của bạn. 🎬\n\n"
                "Tôi có thể giúp bạn:\n"
                "- **Tra cứu cơ sở dữ liệu:** Diễn viên, đạo diễn, năm phát hành, điểm IMDb, thống kê.\n"
                "- **Gợi ý phim:** Phim tương tự, cá nhân hóa theo sở thích, lọc theo thể loại.\n"
                "- **Thảo luận điện ảnh:** Giải thích cốt truyện, phân tích cái kết, thảo luận trivia.\n\n"
                "👉 *Bạn có thể thử hỏi:* \n"
                "- *'Phim của đạo diễn Christopher Nolan'*\n"
                "- *'Gợi ý cho tôi phim hành động khoa học viễn tưởng hay'*\n"
                "- *'Giải thích ý nghĩa cái kết phim Shutter Island'*"
            )
        else:
            return (
                "Hello! I am **MovieNex AI** — your intelligent movie concierge and recommendation assistant. 🎬\n\n"
                "I can help you with:\n"
                "- **Catalog Lookup:** Direct queries on directors, actors, ratings, and stats.\n"
                "- **Personalized Recs:** 3-stage discovery, similar titles, and genre feeds.\n"
                "- **Cinema Discussions:** Plot analysis, endings explanation, and trivia.\n\n"
                "👉 *You can try asking:* \n"
                "- *'Movies directed by Christopher Nolan'*\n"
                "- *'Recommend some mind-bending sci-fi movies'*\n"
                "- *'Explain the ending of Shutter Island'*"
            )

    # Format recommended movies
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
            "Here are the most suitable movies curated for you by MovieNex AI:\n\n"
        )

    sections = []
    for i, m in enumerate(enriched[:5], 1):
        title = m.get("title", "Unknown Title")
        rel_date = m.get("release_date", "")
        year = rel_date[:4] if rel_date and len(rel_date) >= 4 else ""
        year_str = f" *({year})*" if year else ""
        desc = m.get("description") or m.get("overview") or ""
        if len(desc) > 180:
            desc = desc[:177] + "..."
        sections.append(f"{i}. ### 🎬 **{title}**{year_str}\n   {desc}\n")

    footer = (
        "\n👉 *Bạn có muốn tôi chia sẻ thêm thông tin chi tiết hoặc gợi ý thêm tác phẩm nào khác không?*"
        if is_vi else
        "\n👉 *Would you like more details on any of these titles or additional recommendations?*"
    )
    return header + "\n".join(sections) + footer


def generate_answer_node(state: GraphState) -> dict:
    """Synthesizes context (SQL records, candidate movies, user history) into an articulate response."""
    messages = state.get("messages", [])
    last_msg = messages[-1].content if messages else ""
    intent_output = state.get("intent_output")
    intent = intent_output.intent if intent_output else "fallback"
    genres = intent_output.genres if intent_output else []

    detected_lang = state.get("detected_language") or "vi"
    response_lang_name = "Vietnamese" if detected_lang == "vi" else "English"

    matched = state.get("matched_movie")
    enriched = state.get("enriched_movies", [])
    sql_query = state.get("sql_query")
    sql_results = state.get("sql_results")
    filtered_movies = enriched

    final_text = ""
    llm = get_llm()

    if llm is not None:
        try:
            context_blocks = []

            # 1. SQL database findings
            if sql_results is not None:
                context_blocks.append(
                    f"[DATABASE QUERY EXECUTED]\nQuery: {sql_query}\n"
                    f"Results: {json.dumps(sql_results[:10], ensure_ascii=False, indent=2)}"
                )

            # 2. Referenced movie
            if matched:
                context_blocks.append(f"[REFERENCED MOVIE]\nTitle: {matched.get('title', 'Unknown')}")

            # 3. Recommended / Candidate movies
            if enriched:
                movie_list_str = json.dumps(
                    [
                        {
                            "title": m.get("title"),
                            "release_date": m.get("release_date"),
                            "genres": m.get("genres"),
                            "description": m.get("description", ""),
                            "director": m.get("director", ""),
                            "rating": m.get("vote_average", 0.0)
                        }
                        for m in enriched[:8]
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                context_blocks.append(f"[CANDIDATE MOVIES]\n{movie_list_str}")

            context_data = "\n\n".join(context_blocks) if context_blocks else "No specific database entities matched. Answer using general cinema knowledge."

            prompt = ANSWER_PROMPT
            chain = prompt | llm
            response = chain.invoke({
                "history": messages,
                "context_data": context_data,
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
            sql_results=sql_results,
            lang=detected_lang,
            genres=genres
        )

    return {
        "final_text": final_text,
        "enriched_movies": filtered_movies,
        "messages": [AIMessage(content=final_text)],
    }