from typing import Annotated, Any, Literal, TypedDict, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIZED BASE MODELS FOR DATABASE QUERIES & INTENT CASES
# ─────────────────────────────────────────────────────────────────────────────

class YearQueryModel(BaseModel):
    """Truy vấn phim theo năm hoặc khoảng thời gian phát hành."""
    intent: Literal["year"] = "year"
    year: str = Field(description="Năm hoặc thập niên phát hành phim (ví dụ: '2024', '1999', '2010s', '2015-2020')")
    language: str = Field(default="vi", description="'vi' cho Tiếng Việt, 'en' cho English")


class DirectorQueryModel(BaseModel):
    """Truy vấn phim theo tên đạo diễn."""
    intent: Literal["director"] = "director"
    director: str = Field(description="Tên đạo diễn (ví dụ: 'Christopher Nolan', 'Quentin Tarantino')")
    language: str = Field(default="vi")


class ActorQueryModel(BaseModel):
    """Truy vấn phim theo tên diễn viên / dàn cast."""
    intent: Literal["actor"] = "actor"
    actors: list[str] = Field(default_factory=list, description="Danh sách tên diễn viên (ví dụ: ['Leonardo DiCaprio'])")
    language: str = Field(default="vi")


class GenreQueryModel(BaseModel):
    """Truy vấn phim theo thể loại (Action, Sci-Fi, Horror, Animation...)."""
    intent: Literal["genre"] = "genre"
    genres: list[str] = Field(default_factory=list, description="Danh sách thể loại phim chuẩn tiếng Anh (ví dụ: ['Action', 'Sci-Fi'])")
    language: str = Field(default="vi")


class RatingQueryModel(BaseModel):
    """Truy vấn phim theo điểm số tối thiểu."""
    intent: Literal["rating"] = "rating"
    min_rating: float = Field(description="Điểm đánh giá tối thiểu theo thang 10 (ví dụ: 8.0, 7.5)")
    language: str = Field(default="vi")


class SimilarQueryModel(BaseModel):
    """Tìm phim tương tự hoặc tra cứu theo tên một bộ phim mẫu."""
    intent: Literal["similar"] = "similar"
    target_title: str = Field(description="Tên bộ phim mục tiêu (ví dụ: 'Inception', 'Interstellar')")
    language: str = Field(default="vi")


class MovieDetailQueryModel(BaseModel):
    """Tra cứu chi tiết, cốt truyện, diễn viên, đạo diễn, đánh giá của một bộ phim cụ thể."""
    intent: Literal["movie_detail"] = "movie_detail"
    target_title: str = Field(description="Tên bộ phim cần xem chi tiết (ví dụ: 'Batman Begins', 'Interstellar')")
    language: str = Field(default="vi")


class SearchFilterQueryModel(BaseModel):
    """Tìm kiếm kết hợp đa điều kiện (năm, đạo diễn, diễn viên, thể loại, điểm số...)."""
    intent: Literal["search_filter"] = "search_filter"
    target_title: str = Field(default="", description="Tên phim nếu có")
    director: str = Field(default="", description="Tên đạo diễn nếu có")
    actors: list[str] = Field(default_factory=list, description="Danh sách diễn viên")
    genres: list[str] = Field(default_factory=list, description="Danh sách thể loại")
    year: str = Field(default="", description="Năm hoặc thập niên")
    min_rating: float | None = Field(default=None, description="Điểm đánh giá tối thiểu")
    keywords: list[str] = Field(default_factory=list, description="Từ khóa chủ đề")
    language: str = Field(default="vi")


class SQLStatsQueryModel(BaseModel):
    """Truy vấn thống kê hoặc sự kiện đếm số lượng từ cơ sở dữ liệu."""
    intent: Literal["sql_query"] = "sql_query"
    query_topic: str = Field(default="stats", description="Chủ đề truy vấn SQL (count, aggregate, catalog query)")
    language: str = Field(default="vi")


class MoreRecommendationsQueryModel(BaseModel):
    """Yêu cầu gợi ý thêm các bộ phim khác tiếp nối danh sách trước."""
    intent: Literal["more_recommendations"] = "more_recommendations"
    is_more_request: bool = Field(default=True, description="Luôn là True khi người dùng muốn xem thêm phim")
    language: str = Field(default="vi")


class PersonalizedQueryModel(BaseModel):
    """Yêu cầu gợi ý phim cá nhân hóa hoặc gợi ý phim hay hôm nay."""
    intent: Literal["personalized"] = "personalized"
    language: str = Field(default="vi")


class GeneralQAQueryModel(BaseModel):
    """Thảo luận điện ảnh, phân tích cốt truyện, giải thích cái kết hoặc đàm thoại chung."""
    intent: Literal["general_qa"] = "general_qa"
    topic: str = Field(default="", description="Chủ đề thảo luận hoặc phân tích")
    target_title: str = Field(default="", description="Tên phim cần giải thích (nếu có)")
    language: str = Field(default="vi")


class FallbackQueryModel(BaseModel):
    """Không xác định được intent rõ ràng."""
    intent: Literal["fallback"] = "fallback"
    language: str = Field(default="vi")


MovieQueryModelUnion = Union[
    YearQueryModel,
    DirectorQueryModel,
    ActorQueryModel,
    GenreQueryModel,
    RatingQueryModel,
    SimilarQueryModel,
    MovieDetailQueryModel,
    SearchFilterQueryModel,
    SQLStatsQueryModel,
    MoreRecommendationsQueryModel,
    PersonalizedQueryModel,
    GeneralQAQueryModel,
    FallbackQueryModel,
]



# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED INTENT OUTPUT FOR DOWNSTREAM NODES
# ─────────────────────────────────────────────────────────────────────────────

class IntentOutput(BaseModel):
    intent: str = Field(
        default="fallback",
        description="'year', 'director', 'actor', 'genre', 'rating', 'similar', 'search_filter', 'sql_query', 'more_recommendations', 'personalized', 'general_qa', or 'fallback'"
    )
    target_title: str = Field(default="", description="Target movie title if comparing or looking for similar")
    genres: list[str] = Field(default_factory=list, description="Target genres (e.g. Action, Sci-Fi, Drama)")
    director: str = Field(default="", description="Director name if specified")
    actors: list[str] = Field(default_factory=list, description="Actor/cast names if specified")
    year: str = Field(default="", description="Release year or decade (e.g. 2024, 2010s, 2020-2023, classic)")
    keywords: list[str] = Field(default_factory=list, description="Key plot topics, themes, or tags")
    min_rating: float | None = Field(default=None, description="Minimum vote average rating (e.g. 7.5, 8.0)")
    is_more_request: bool = Field(default=False, description="True if user is asking for more/next recommendations")
    language: str = Field(default="vi", description="'vi' for Vietnamese, 'en' for English")


def parse_intent_to_output(payload: dict[str, Any], lang: str = "vi") -> IntentOutput:
    """Validate raw payload against specialized model, then return unified IntentOutput."""
    intent_val = (payload.get("intent") or "fallback").strip().lower()
    payload_lang = payload.get("language") or lang

    try:
        if intent_val == "year":
            m_year = YearQueryModel(**payload)
            return IntentOutput(intent="year", year=str(m_year.year), language=m_year.language)
        elif intent_val == "director":
            m_dir = DirectorQueryModel(**payload)
            return IntentOutput(intent="director", director=m_dir.director, language=m_dir.language)
        elif intent_val == "actor":
            m_act = ActorQueryModel(**payload)
            return IntentOutput(intent="actor", actors=m_act.actors, language=m_act.language)
        elif intent_val == "genre":
            m_gen = GenreQueryModel(**payload)
            return IntentOutput(intent="genre", genres=m_gen.genres, language=m_gen.language)
        elif intent_val == "rating":
            m_rat = RatingQueryModel(**payload)
            return IntentOutput(intent="rating", min_rating=float(m_rat.min_rating), language=m_rat.language)
        elif intent_val in ("similar", "title"):
            m_sim = SimilarQueryModel(**payload)
            return IntentOutput(intent="similar", target_title=m_sim.target_title, language=m_sim.language)
        elif intent_val in ("movie_detail", "detail", "movie_info", "info"):
            title = payload.get("target_title") or payload.get("title") or ""
            return IntentOutput(intent="movie_detail", target_title=title, language=payload_lang)
        elif intent_val == "search_filter":
            m_flt = SearchFilterQueryModel(**payload)
            return IntentOutput(
                intent="search_filter",
                target_title=m_flt.target_title,
                director=m_flt.director,
                actors=m_flt.actors,
                genres=m_flt.genres,
                year=str(m_flt.year) if m_flt.year else "",
                min_rating=m_flt.min_rating,
                keywords=m_flt.keywords,
                language=m_flt.language
            )
        elif intent_val == "sql_query":
            m_sql = SQLStatsQueryModel(**payload)
            return IntentOutput(intent="sql_query", language=m_sql.language)
        elif intent_val == "more_recommendations":
            m_more = MoreRecommendationsQueryModel(**payload)
            return IntentOutput(intent="more_recommendations", is_more_request=True, language=m_more.language)
        elif intent_val == "personalized":
            m_pers = PersonalizedQueryModel(**payload)
            return IntentOutput(intent="personalized", language=m_pers.language)
        elif intent_val == "general_qa":
            m_qa = GeneralQAQueryModel(**payload)
            return IntentOutput(intent="general_qa", target_title=m_qa.target_title, language=m_qa.language)
    except Exception:
        pass

    # Generic or fallback
    return IntentOutput(
        intent=intent_val if intent_val in (
            "year", "director", "actor", "genre", "rating", "similar", "movie_detail",
            "search_filter", "sql_query", "more_recommendations", "personalized", "general_qa"
        ) else "fallback",
        target_title=payload.get("target_title") or "",
        genres=payload.get("genres") or [],
        director=payload.get("director") or "",
        actors=payload.get("actors") or [],
        year=str(payload.get("year")) if payload.get("year") else "",
        keywords=payload.get("keywords") or [],
        min_rating=float(payload.get("min_rating")) if payload.get("min_rating") is not None else None,
        is_more_request=bool(payload.get("is_more_request", False)),
        language=payload_lang
    )



class GraphState(TypedDict):
    """
    Graph state for the MovieNex Conversational Assistant.
    `messages` is automatically reduced by `add_messages`.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str | None

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
