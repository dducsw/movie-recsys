import random
from typing import Dict, Any
from app.models.movie import MovieModel
from app.services.recsys import RecsysService

class ChatbotService:
    @staticmethod
    def get_reply(message: str) -> Dict[str, Any]:
        message_lower = message.lower()
        
        # 1. Similar movies keyword detection
        similar_keywords = ["giống", "like", "tương tự", "similar"]
        is_similar_query = any(kw in message_lower for kw in similar_keywords)
        
        target_title = ""
        if is_similar_query:
            cleaned = message_lower
            for kw in similar_keywords:
                cleaned = cleaned.replace(kw, "")
            for filler in ["phim", "gợi ý", "recommend", "show me", "tìm", "find", "như", "những"]:
                cleaned = cleaned.replace(filler, "")
            target_title = cleaned.strip()

        if target_title and len(target_title) > 2:
            search_results = MovieModel.search(target_title, limit=1)
            if search_results:
                matched_movie = search_results[0]
                similar_movies = RecsysService.get_similar_movies(matched_movie["movieId"], limit=5)
                # Filter the current movie from the list if present
                similar_movies = [m for m in similar_movies if m["movieId"] != matched_movie["movieId"]][:5]
                return {
                    "text": f"Dựa trên bộ phim **{matched_movie['title']}** mà bạn quan tâm, tôi gợi ý cho bạn một số phim tương tự sau:",
                    "movies": similar_movies
                }

        # 2. Genre detection
        genre_map = {
            "hành động": "Action", "action": "Action",
            "hài": "Comedy", "comedy": "Comedy",
            "viễn tưởng": "Sci-Fi", "sci-fi": "Sci-Fi", "khoa học viễn tưởng": "Sci-Fi",
            "kinh dị": "Horror", "horror": "Horror",
            "tình cảm": "Romance", "lãng mạn": "Romance", "romance": "Romance",
            "hoạt hình": "Animation", "animation": "Animation",
            "phiêu lưu": "Adventure", "adventure": "Adventure",
            "tâm lý": "Drama", "drama": "Drama", "kịch tính": "Drama",
            "giật gân": "Thriller", "thriller": "Thriller",
            "bí ẩn": "Mystery", "mystery": "Mystery",
            "ảo tưởng": "Fantasy", "fantasy": "Fantasy",
            "gia đình": "Family", "family": "Family"
        }
        
        detected_genres = []
        for kw, genre in genre_map.items():
            if kw in message_lower:
                if genre not in detected_genres:
                    detected_genres.append(genre)
                    
        if detected_genres:
            all_movies = MovieModel.get_all_genres_and_popularity()
            
            matching_movies = []
            for movie in all_movies:
                movie_genres = [g.strip() for g in movie.get("genres", "").split("|") if g.strip()]
                if any(dg in movie_genres for dg in detected_genres):
                    matching_movies.append(movie)
            
            # Sort by popularity
            matching_movies.sort(key=lambda x: x.get("popularity", 0.0), reverse=True)
            recommended = matching_movies[:5]
            
            genres_str = ", ".join(detected_genres)
            if recommended:
                return {
                    "text": f"Tôi tìm thấy một số phim thuộc thể loại **{genres_str}** phổ biến nhất dành cho bạn:",
                    "movies": recommended
                }
            else:
                return {
                    "text": f"Xin lỗi, tôi chưa tìm thấy phim nào thuộc thể loại **{genres_str}** trong cơ sở dữ liệu.",
                    "movies": []
                }

        # 3. Fallback / Greeting
        greeting_replies = [
            "Xin chào! Tôi là AI Chatbot gợi ý phim. Bạn có thể yêu cầu tôi gợi ý phim theo thể loại (hành động, hài hước, kinh dị...) hoặc tìm phim tương tự một bộ phim bạn thích (ví dụ: 'phim giống Toy Story').",
            "Chào bạn! Hôm nay bạn muốn tìm phim gì? Hãy nói cho tôi biết thể loại hoặc tên phim bạn yêu thích nhé!",
            "Xin chào! Tôi có thể giúp gì cho bạn hôm nay? Hãy nhập thể loại phim bạn muốn xem (ví dụ: 'phim hoạt hình' hoặc 'phim khoa học viễn tưởng')."
        ]
        return {
            "text": random.choice(greeting_replies),
            "movies": []
        }
