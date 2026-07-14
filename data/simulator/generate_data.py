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

    # 2. Simulate Users
    users_data = []
    for u_id in range(1, num_users + 1):
        # Favorite genres (1 to 3 categories)
        fav_genres = random.sample(all_genres, k=random.randint(1, 3))
        # Favorite director (1)
        fav_director = random.choice(all_directors) if all_directors else "Unknown"
        # Favorite cast members (1 to 2)
        fav_cast = random.sample(all_cast, k=random.randint(1, 2)) if all_cast else []
        
        # Activity level (Gamma distribution)
        activity = int(np.random.gamma(shape=2.5, scale=40)) + 15 
        # User Bias (b_u) - standard normal distribution
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

    # 3. Simulate interactions (Clicks & Ratings)
    click_events = []
    ratings_data = []
    
    start_time = datetime(2026, 1, 1)
    df_movies["release_datetime"] = pd.to_datetime(df_movies["release_date"], errors="coerce")
    default_release = pd.to_datetime("2010-01-01")
    
    for idx, user in df_users.iterrows():
        u_id = user["user_id"]
        fav_genres = user["favorite_genres"].split("|")
        fav_director = user["favorite_directors"]
        fav_cast = user["favorite_cast"].split("|")
        
        activity = user["activity_level"]
        u_bias = user["user_bias"]
        
        # Phân phối thời gian hoạt động của user trong 1 tháng
        user_time = start_time + timedelta(hours=random.randint(0, 720))
        
        # Tính toán điểm hấp dẫn của phim đối với người dùng
        movie_attraction_scores = []
        for _, movie in df_movies.iterrows():
            # a. Popularity baseline
            base_pop = movie["popularity"]
            
            # b. Time Decay: e^(-lambda * dt)
            m_release = movie["release_datetime"]
            if pd.isna(m_release):
                m_release = default_release
                
            delta_days = (user_time - m_release).days
            decay_factor = np.exp(-0.00015 * max(0, delta_days))
            decay_popularity = base_pop * decay_factor
            
            # c. Genre Preference Alignment (x3.5 nếu trùng thể loại yêu thích)
            movie_genres = movie["genres"].split("|")
            genre_fit = any(g in fav_genres for g in movie_genres)
            genre_multiplier = 3.5 if genre_fit else 1.0
            
            # d. Đạo diễn ưa thích Alignment (x1.8 điểm thu hút)
            director_fit = (movie["director"] == fav_director) and (movie["director"] != "Unknown")
            director_multiplier = 1.8 if director_fit else 1.0
            
            # e. Diễn viên ưa thích Alignment (x1.4 điểm thu hút)
            movie_cast = str(movie["cast"]).split("|")
            cast_fit = any(c in fav_cast for c in movie_cast)
            cast_multiplier = 1.4 if cast_fit else 1.0
            
            # f. Final attraction score
            attraction_score = decay_popularity * genre_multiplier * director_multiplier * cast_multiplier
            movie_attraction_scores.append(attraction_score)
            
        # Chuẩn hóa điểm số thành phân phối xác suất
        movie_attraction_scores = np.array(movie_attraction_scores)
        movie_probs = movie_attraction_scores / sum(movie_attraction_scores)
        
        # Chọn các phim tương tác
        chosen_movies = np.random.choice(
            df_movies["movieId"], 
            size=min(activity, num_movies), 
            replace=False, 
            p=movie_probs
        )
        
        for m_id in chosen_movies:
            movie_info = df_movies[df_movies["movieId"] == m_id].iloc[0]
            movie_genres = movie_info["genres"].split("|")
            
            # --- Phễu tương tác ngầm định (Implicit Click Events Funnel) ---
            user_time += timedelta(minutes=random.randint(1, 45))
            click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "click"})
            
            # 65% xem chi tiết
            if random.random() < 0.65:
                user_time += timedelta(seconds=random.randint(10, 80))
                click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "detail_view"})
                
                # 45% bắt đầu xem
                if random.random() < 0.45:
                    user_time += timedelta(seconds=5)
                    click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "watch_start"})
                    
                    # 75% hoàn thành xem phim
                    is_completed = random.random() < 0.75
                    watch_duration = random.randint(3600, 7200) if is_completed else random.randint(60, 1800)
                    user_time += timedelta(seconds=watch_duration)
                    
                    if is_completed:
                        click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "watch_complete"})
                    
                    # --- Sinh đánh giá tường minh (Explicit Rating Generation) ---
                    # 35% cơ hội chấm điểm sau khi xem
                    if random.random() < 0.35:
                        # Mô hình tuyến tính chấm điểm:
                        # Rating = Global_Mean (3.5) + User_Bias + Item_Bias + Genre_Fit_Bonus + Director_Bonus + Cast_Bonus + Completion_Bonus + Noise
                        item_bias = (movie_info["vote_average"] - 5.0) / 2.0
                        
                        genre_bonus = np.random.uniform(0.5, 1.2) if any(g in fav_genres for g in movie_genres) else 0.0
                        director_bonus = np.random.uniform(0.5, 1.2) if (movie_info["director"] == fav_director) and (movie_info["director"] != "Unknown") else 0.0
                        
                        movie_cast = str(movie_info["cast"]).split("|")
                        cast_bonus = np.random.uniform(0.3, 0.8) if any(c in fav_cast for c in movie_cast) else 0.0
                        
                        completion_bonus = np.random.uniform(0.3, 0.8) if is_completed else np.random.uniform(-1.5, -0.5)
                        noise = np.random.normal(loc=0.0, scale=0.35)
                        
                        rating_score = 3.5 + u_bias + item_bias + genre_bonus + director_bonus + cast_bonus + completion_bonus + noise
                        
                        # Giới hạn điểm [0.5, 5.0] và làm tròn đến nửa sao gần nhất
                        rating = min(max(rating_score, 0.5), 5.0)
                        rating = round(rating * 2) / 2
                        
                        ratings_data.append({
                            "userId": u_id,
                            "movieId": m_id,
                            "rating": rating,
                            "timestamp": int(user_time.timestamp())
                        })
                        
    df_click_events = pd.DataFrame(click_events)
    df_ratings = pd.DataFrame(ratings_data)
    
    df_click_events.to_csv(os.path.join(output_dir, "sim_click_events.csv"), index=False)
    df_ratings.to_csv(os.path.join(output_dir, "sim_ratings.csv"), index=False)
    
    print(f"\n[Success] Generated simulation datasets saved in '{output_dir}':")
    print(f"- {len(df_click_events):,} click funnel events -> sim_click_events.csv")
    print(f"- {len(df_ratings):,} rating instances -> sim_ratings.csv")

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
