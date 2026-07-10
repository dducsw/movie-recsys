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
            f"python data/crawler/crawl_movies.py --pages 500 --output {csv_path}"
        )
    
    df = pd.read_csv(csv_path)
    df["genres"] = df["genres"].fillna("(no genres listed)")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(1.0)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(5.0)
    df["release_date"] = df["release_date"].fillna("2010-01-01")
    return df

def generate_advanced_simulator(movies_csv, num_users, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load real movies
    df_movies = load_real_movies(movies_csv)
    num_movies = len(df_movies)
    print(f"[Success] Loaded {num_movies} real movies from: {movies_csv}")
    
    # Extract unique genres from data
    all_genres = set()
    for item in df_movies["genres"].str.split("|"):
        all_genres.update(item)
    all_genres = list(all_genres - {"(no genres listed)", ""})
    if not all_genres:
        all_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
        
    print(f"Detected genres: {all_genres}")

    # 2. Simulate Users
    users_data = []
    for u_id in range(1, num_users + 1):
        # Favorite genres (1 to 3 categories)
        fav_genres = random.sample(all_genres, k=random.randint(1, 3))
        # Activity level (Gamma distribution)
        activity = int(np.random.gamma(shape=2.5, scale=40)) + 15 
        # User Bias (b_u) - standard normal distribution
        user_bias = np.random.normal(loc=0.0, scale=0.5)
        
        users_data.append({
            "user_id": u_id,
            "favorite_genres": "|".join(fav_genres),
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
        activity = user["activity_level"]
        u_bias = user["user_bias"]
        
        # Distribute active time across 1 month
        user_time = start_time + timedelta(hours=random.randint(0, 720))
        
        # Calculate dynamic movie attraction score for this user
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
            
            # c. Genre Preference Alignment
            movie_genres = movie["genres"].split("|")
            genre_fit = any(g in fav_genres for g in movie_genres)
            genre_multiplier = 3.5 if genre_fit else 1.0
            
            # d. Final attraction score
            attraction_score = decay_popularity * genre_multiplier
            movie_attraction_scores.append(attraction_score)
            
        # Normalize attraction scores to probability distribution
        movie_attraction_scores = np.array(movie_attraction_scores)
        movie_probs = movie_attraction_scores / sum(movie_attraction_scores)
        
        # Select movies the user interacts with
        chosen_movies = np.random.choice(
            df_movies["movieId"], 
            size=min(activity, num_movies), 
            replace=False, 
            p=movie_probs
        )
        
        for m_id in chosen_movies:
            movie_info = df_movies[df_movies["movieId"] == m_id].iloc[0]
            movie_genres = movie_info["genres"].split("|")
            
            # --- Implicit Click Events Funnel ---
            user_time += timedelta(minutes=random.randint(1, 45))
            click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "click"})
            
            # 65% chance read details
            if random.random() < 0.65:
                user_time += timedelta(seconds=random.randint(10, 80))
                click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "detail_view"})
                
                # 45% chance start watch
                if random.random() < 0.45:
                    user_time += timedelta(seconds=5)
                    click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "watch_start"})
                    
                    # 75% chance complete watch
                    is_completed = random.random() < 0.75
                    watch_duration = random.randint(3600, 7200) if is_completed else random.randint(60, 1800)
                    user_time += timedelta(seconds=watch_duration)
                    
                    if is_completed:
                        click_events.append({"userId": u_id, "movieId": m_id, "timestamp": user_time, "event_type": "watch_complete"})
                    
                    # --- Explicit Rating Generation ---
                    # 35% chance to leave a rating after watching
                    if random.random() < 0.35:
                        # Linear Model: Rating = Global_Mean (3.5) + User_Bias + Item_Bias + Genre_Fit_Bonus + Completion_Bonus + Noise
                        item_bias = (movie_info["vote_average"] - 5.0) / 2.0
                        
                        genre_bonus = np.random.uniform(0.5, 1.2) if any(g in fav_genres for g in movie_genres) else 0.0
                        completion_bonus = np.random.uniform(0.3, 0.8) if is_completed else np.random.uniform(-1.5, -0.5)
                        
                        noise = np.random.normal(loc=0.0, scale=0.35)
                        
                        rating_score = 3.5 + u_bias + item_bias + genre_bonus + completion_bonus + noise
                        
                        # Clip rating to standard range [0.5, 5.0] and round to nearest 0.5 star
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
    parser = argparse.ArgumentParser(description="Advanced Movie RecSys Interaction Simulator")
    parser.add_argument("--movies_csv", type=str, default="data/movies_crawled.csv",
                        help="Path to real movies CSV file (default: data/movies_crawled.csv)")
    parser.add_argument("--num_users", type=int, default=200,
                        help="Number of simulated users to create (default: 200)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for generated CSV files (default: data)")
                        
    args = parser.parse_args()
    
    generate_advanced_simulator(
        movies_csv=args.movies_csv,
        num_users=args.num_users,
        output_dir=args.output_dir
    )
