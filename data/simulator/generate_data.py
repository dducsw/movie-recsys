import os
import argparse
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Fix seed for reproducibility
np.random.seed(42)
random.seed(42)

def load_real_movies(csv_path):
    """
    Load real movie dataset from crawl. Raises error if file doesn't exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[Error] Movie dataset not found at: {csv_path}\n"
            f"Please run the crawler first:\n"
            f"python data/crawler/crawl_movies.py --pages 500"
        )
    
    df = pd.read_csv(csv_path)
    df["genres"] = df["genres"].fillna("(no genres listed)")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(1.0)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(5.0)
    df["release_date"] = df["release_date"].fillna("2010-01-01")
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("")
    return df

def generate_advanced_simulator(movies_csv, num_users, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load real movies
    df_movies = load_real_movies(movies_csv)
    num_movies = len(df_movies)
    print(f"[Success] Loaded {num_movies} real movies from: {movies_csv}")
    
    # Trích xuất danh sách tất cả thể loại độc bản
    all_genres = set()
    for item in df_movies["genres"].str.split("|"):
        all_genres.update(item)
    all_genres = list(all_genres - {"(no genres listed)", ""})
    if not all_genres:
        all_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
        
    # Trích xuất danh sách Đạo diễn phổ biến (tối thiểu xuất hiện 2 lần để đảm bảo có thể gợi ý)
    director_counts = df_movies["director"].value_counts()
    all_directors = director_counts[(director_counts.index != "Unknown") & (director_counts >= 2)].index.tolist()
    if not all_directors:
        all_directors = director_counts[director_counts.index != "Unknown"].head(50).index.tolist()
    if not all_directors:
        all_directors = ["Christopher Nolan", "Steven Spielberg", "Quentin Tarantino", "James Cameron"]

    # Trích xuất danh sách Diễn viên phổ biến (lấy top 300 diễn viên xuất hiện nhiều nhất)
    all_actors = []
    for cast_str in df_movies["cast"].str.split("|").dropna():
        all_actors.extend(cast_str)
    actor_counts = pd.Series([a for a in all_actors if a.strip()]).value_counts()
    all_cast = actor_counts.head(300).index.tolist()
    if not all_cast:
        all_cast = ["Leonardo DiCaprio", "Tom Hanks", "Brad Pitt", "Scarlett Johansson", "Robert Downey Jr."]

    print(f"Detected genres: {len(all_genres)} categories")
    print(f"Detected popular directors: {len(all_directors)} names")
    print(f"Detected popular cast members: {len(all_cast)} names")

    # 2. Simulate Users (Khởi tạo profile gốc tĩnh)
    users_data = []
    for u_id in range(1, num_users + 1):
        fav_genres = random.sample(all_genres, k=random.randint(1, 3))
        fav_director = random.choice(all_directors) if all_directors else "Unknown"
        fav_cast = random.sample(all_cast, k=random.randint(1, 2)) if all_cast else []
        activity = int(np.random.gamma(shape=2.5, scale=40)) + 15 
        user_bias = np.random.normal(loc=0.0, scale=0.5)
        
        users_data.append({
            "user_id": u_id,
            "favorite_genres": "|".join(fav_genres),
            "favorite_directors": fav_director,
            "favorite_cast": "|".join(fav_cast),
            "activity_level": activity,
            "user_bias": user_bias
        })
    df_users = pd.DataFrame(users_data)
    df_users.to_csv(os.path.join(output_dir, "sim_users.csv"), index=False)
    print(f"Generated {num_users} users profiles saved to: sim_users.csv")

    # 3. Simulate interactions (Clicks, Ratings, Sessions, & Impressions)
    click_events = []        # Ghi nhận các sự kiện click tương thích ngược (chỉ gồm click, detail_view, watch_start, watch_complete)
    ratings_data = []        # Ghi nhận đánh giá sao ratings
    ctr_impressions = []     # Ghi nhận toàn bộ vết hiển thị (Impression Logs) để phục vụ CTR/SOTA models
    
    start_time = datetime(2026, 1, 1)
    df_movies["release_datetime"] = pd.to_datetime(df_movies["release_date"], errors="coerce")
    default_release = pd.to_datetime("2010-01-01")
    
    for idx, user in df_users.iterrows():
        u_id = user["user_id"]
        
        # Sao chép sở thích để phục vụ mô phỏng dịch chuyển sở thích (Concept Drift)
        user_fav_genres = user["favorite_genres"].split("|")
        user_fav_director = user["favorite_directors"]
        user_fav_cast = user["favorite_cast"].split("|")
        
        activity = user["activity_level"]
        u_bias = user["user_bias"]
        
        user_time = start_time + timedelta(hours=random.randint(0, 720))
        
        # Định danh phiên hoạt động (Sessionization)
        session_counter = 1
        session_id = f"sess_{u_id}_{session_counter}"
        last_event_time = user_time
        current_week = 0

        # Lặp qua các lần mở ứng dụng (mỗi user mở app nhiều lần tùy mức độ active)
        num_sessions = max(3, activity // 15)
        for s_idx in range(num_sessions):
            # Tạo ngẫu nhiên thời gian giãn cách giữa các phiên (ví dụ từ vài giờ đến vài ngày)
            session_delay_hours = random.randint(12, 120)
            user_time += timedelta(hours=session_delay_hours)
            
            # Kiểm tra và áp dụng dịch chuyển sở thích (Concept Drift) theo từng tuần
            week = (user_time - start_time).days // 7
            if week > current_week:
                current_week = week
                # 25% cơ hội thay đổi 1 thể loại ưa thích mỗi tuần
                if random.random() < 0.25 and len(user_fav_genres) > 0:
                    idx_to_swap = random.randint(0, len(user_fav_genres) - 1)
                    new_genre = random.choice(list(set(all_genres) - set(user_fav_genres)))
                    user_fav_genres[idx_to_swap] = new_genre
                # 10% cơ hội đổi đạo diễn ưa thích mỗi tuần
                if random.random() < 0.10:
                    user_fav_director = random.choice(all_directors)
            
            # Cấp phát session_id mới cho phiên hoạt động này
            session_counter += 1
            session_id = f"sess_{u_id}_{session_counter}"
            
            # --- Sinh danh sách hiển thị (Impression Logs - Hệ thống hiển thị 20 phim trên trang chủ) ---
            # Tính điểm hấp dẫn để xếp hạng hiển thị
            movie_attraction_scores = []
            for _, movie in df_movies.iterrows():
                base_pop = movie["popularity"]
                
                # Suy giảm theo thời gian (Time Decay)
                m_release = movie["release_datetime"]
                if pd.isna(m_release):
                    m_release = default_release
                delta_days = (user_time - m_release).days
                decay_factor = np.exp(-0.00015 * max(0, delta_days))
                decay_popularity = base_pop * decay_factor
                
                # Khớp gu thể loại
                movie_genres = movie["genres"].split("|")
                genre_fit = any(g in user_fav_genres for g in movie_genres)
                genre_multiplier = 3.5 if genre_fit else 1.0
                
                # Khớp gu đạo diễn
                director_fit = (movie["director"] == user_fav_director) and (movie["director"] != "Unknown")
                director_multiplier = 1.8 if director_fit else 1.0
                
                # Khớp gu diễn viên
                movie_cast = str(movie["cast"]).split("|")
                cast_fit = any(c in user_fav_cast for c in movie_cast)
                cast_multiplier = 1.4 if cast_fit else 1.0
                
                attraction_score = decay_popularity * genre_multiplier * director_multiplier * cast_multiplier
                movie_attraction_scores.append(attraction_score)
            
            # Lấy Top 20 phim có điểm hấp dẫn cao nhất kèm theo một chút nhiễu (Gumbel noise)
            # mô phỏng hệ thống gợi ý hiển thị Top-20 phim trên giao diện
            movie_attraction_scores = np.array(movie_attraction_scores)
            noise = np.random.gumbel(0, 0.1, size=len(movie_attraction_scores))
            ranked_indices = np.argsort(-(movie_attraction_scores + noise))[:20]
            
            # --- Duyệt danh sách hiển thị (Position Bias & click simulation) ---
            for pos, idx_m in enumerate(ranked_indices, start=1):
                movie_info = df_movies.iloc[idx_m]
                m_id = movie_info["movieId"]
                movie_genres = movie_info["genres"].split("|")
                
                # Độ khớp thuộc tính của phim cụ thể
                genre_fit = any(g in user_fav_genres for g in movie_genres)
                director_fit = (movie_info["director"] == user_fav_director) and (movie_info["director"] != "Unknown")
                movie_cast = str(movie_info["cast"]).split("|")
                cast_fit = any(c in user_fav_cast for c in movie_cast)
                
                # Tính xác suất click có áp dụng Phạt Vị Trí (Position Bias Penalty)
                # DCG weight: 1.0 / log2(pos + 1)
                position_penalty = 1.0 / np.log2(pos + 1)
                
                # Xác suất click cơ sở tỷ lệ với điểm thu hút tương đối
                max_att = max(1.0, movie_attraction_scores.max())
                base_click_prob = (movie_attraction_scores[idx_m] / max_att) * 0.45
                
                # Áp dụng Position Bias
                p_click = base_click_prob * position_penalty
                p_click = min(max(p_click, 0.02), 0.90)  # Bọc trong khoảng an toàn
                
                is_clicked = random.random() < p_click
                
                # Ghi nhận vết hiển thị (Impression Logs - Chứa cả Negative Samples)
                ctr_impressions.append({
                    "userId": u_id,
                    "movieId": m_id,
                    "timestamp": int(user_time.timestamp()),
                    "session_id": session_id,
                    "position": pos,
                    "clicked": 1 if is_clicked else 0
                })
                
                # Nếu người dùng click vào phim
                if is_clicked:
                    user_time += timedelta(minutes=random.randint(1, 10))
                    
                    # 1. Ghi nhận sự kiện Click
                    click_events.append({
                        "userId": u_id,
                        "movieId": m_id,
                        "timestamp": user_time,
                        "event_type": "click",
                        "session_id": session_id
                    })
                    
                    # 2. Xem chi tiết (Detail View)
                    p_detail = 0.55 + (0.15 if genre_fit else 0.0) + (0.05 if director_fit else 0.0) + (0.05 if cast_fit else 0.0)
                    p_detail = min(max(p_detail, 0.1), 0.95)
                    
                    if random.random() < p_detail:
                        user_time += timedelta(seconds=random.randint(10, 80))
                        click_events.append({
                            "userId": u_id,
                            "movieId": m_id,
                            "timestamp": user_time,
                            "event_type": "detail_view",
                            "session_id": session_id
                        })
                        
                        # 3. Quyết định Xem phim thật (watch_start)
                        # Phụ thuộc điểm IMDb trung bình và độ khớp thuộc tính
                        p_watch = 0.35 + (0.15 if movie_info["vote_average"] >= 7.0 else 0.0) + (0.10 if genre_fit else 0.0) + (0.05 if director_fit else 0.0)
                        p_watch = min(max(p_watch, 0.1), 0.95)
                        
                        if random.random() < p_watch:
                            user_time += timedelta(seconds=5)
                            click_events.append({
                                "userId": u_id,
                                "movieId": m_id,
                                "timestamp": user_time,
                                "event_type": "watch_start",
                                "session_id": session_id
                            })
                            
                            # 4. Hoàn thành xem phim (watch_complete)
                            p_complete = 0.60 + (0.15 if movie_info["vote_average"] >= 7.5 else 0.0) + (0.10 if genre_fit else 0.0) - (0.20 if movie_info["vote_average"] <= 5.0 else 0.0)
                            p_complete = min(max(p_complete, 0.1), 0.95)
                            
                            is_completed = random.random() < p_complete
                            watch_duration = random.randint(3600, 7200) if is_completed else random.randint(60, 1800)
                            user_time += timedelta(seconds=watch_duration)
                            
                            if is_completed:
                                click_events.append({
                                    "userId": u_id,
                                    "movieId": m_id,
                                    "timestamp": user_time,
                                    "event_type": "watch_complete",
                                    "session_id": session_id
                                })
                            
                            # 5. Sinh đánh giá Ratings (J-Shaped Beta Distribution)
                            if random.random() < 0.35:
                                item_bias = (movie_info["vote_average"] - 5.0) / 2.0
                                genre_bonus = np.random.uniform(0.5, 1.2) if genre_fit else 0.0
                                director_bonus = np.random.uniform(0.5, 1.2) if director_fit else 0.0
                                cast_bonus = np.random.uniform(0.3, 0.8) if cast_fit else 0.0
                                completion_bonus = np.random.uniform(0.3, 0.8) if is_completed else np.random.uniform(-1.5, -0.5)
                                
                                affinity = 0.5 + u_bias + item_bias + genre_bonus + director_bonus + cast_bonus + completion_bonus
                                p_rating = 1.0 / (1.0 + np.exp(-1.2 * affinity))
                                
                                S = 6.0
                                alpha = max(0.1, p_rating * S)
                                beta = max(0.1, (1.0 - p_rating) * S)
                                
                                rating_score = 1.0 + 4.0 * np.random.beta(alpha, beta)
                                rating = min(max(round(rating_score * 2) / 2, 0.5), 5.0)
                                
                                ratings_data.append({
                                    "userId": u_id,
                                    "movieId": m_id,
                                    "rating": rating,
                                    "timestamp": int(user_time.timestamp())
                                })
                                
    # 4. Save output files
    df_click_events = pd.DataFrame(click_events)
    df_ratings = pd.DataFrame(ratings_data)
    df_ctr_impressions = pd.DataFrame(ctr_impressions)
    
    # Lưu tệp tin tương thích ngược (chỉ gồm clicks, views, watches)
    df_click_events.to_csv(os.path.join(output_dir, "sim_click_events.csv"), index=False)
    df_ratings.to_csv(os.path.join(output_dir, "sim_ratings.csv"), index=False)
    
    # Lưu tệp tin vết hiển thị nâng cao (Impression Logs) phục vụ các thuật toán CTR/SOTA
    df_ctr_impressions.to_csv(os.path.join(output_dir, "sim_impressions.csv"), index=False)
    
    print(f"\n[Success] Generated simulation datasets saved in '{output_dir}':")
    print(f"- {len(df_click_events):,} click funnel events -> sim_click_events.csv (100% backward compatible)")
    print(f"- {len(df_ratings):,} rating instances -> sim_ratings.csv")
    print(f"- {len(df_ctr_impressions):,} impression logs (with negative samples) -> sim_impressions.csv (New SOTA dataset!)")

if __name__ == "__main__":
    # Resolve default movies crawled path dynamically (under data/crawler/)
    default_movies_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "crawler", "movies_crawled.csv")
    )
    
    # Resolve default output directory dynamically (data/simulator/)
    default_output_dir = os.path.abspath(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(description="Advanced Movie RecSys Interaction Simulator")
    parser.add_argument("--movies_csv", type=str, default=default_movies_csv,
                        help="Path to real movies CSV file")
    parser.add_argument("--num_users", type=int, default=200,
                        help="Number of simulated users to create (default: 200)")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help="Output directory for generated CSV files")
                        
    args = parser.parse_args()
    
    generate_advanced_simulator(
        movies_csv=args.movies_csv,
        num_users=args.num_users,
        output_dir=args.output_dir
    )
