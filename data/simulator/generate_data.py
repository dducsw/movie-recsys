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
    if "runtime" in df.columns:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    return df

def infer_preferences_from_movies(fav_movies_list, movie_id_genres, movie_id_director, movie_id_cast):
    """
    Infer user preferences (favorite genres, directors, cast) from a list of movie IDs using O(1) lookups.
    """
    if not fav_movies_list:
        return [], "Unknown", []
        
    genres_list = []
    directors = []
    cast_list = []
    
    for m in fav_movies_list:
        try:
            m_id = int(m)
        except ValueError:
            continue
            
        # Genres
        g_str = movie_id_genres.get(m_id)
        if g_str and g_str != "(no genres listed)":
            genres_list.extend(str(g_str).split("|"))
            
        # Director
        d = movie_id_director.get(m_id)
        if d and d != "Unknown":
            directors.append(d)
            
        # Cast
        c_str = movie_id_cast.get(m_id)
        if c_str:
            cast_list.extend(str(c_str).split("|"))
            
    user_fav_genres = list(set(genres_list)) if genres_list else []
    user_fav_director = max(set(directors), key=directors.count) if directors else "Unknown"
    user_fav_cast = list(set(c for c in cast_list if c)) if cast_list else []
    
    return user_fav_genres, user_fav_director, user_fav_cast

def simulate_users(num_users, df_movies, all_genres, all_directors, all_cast, cold_start_users_ratio=0.10):
    """
    Simulate user profiles, including cold-start and active users, with seed favorite movies.
    """
    users_data = []
    num_cold_start = int(num_users * cold_start_users_ratio)
    
    # Pre-map movie attributes to movieIds for lightning fast user favorite movie sampling
    genre_to_movie_ids = {}
    director_to_movie_ids = {}
    cast_to_movie_ids = {}
    
    for _, row in df_movies.iterrows():
        m_id = int(row["movieId"])
        # Genres
        for g in str(row["genres"]).split("|"):
            if g and g != "(no genres listed)":
                genre_to_movie_ids.setdefault(g, []).append(m_id)
        # Director
        d = row["director"]
        if d and d != "Unknown":
            director_to_movie_ids.setdefault(d, []).append(m_id)
        # Cast
        for c in str(row["cast"]).split("|"):
            if c:
                cast_to_movie_ids.setdefault(c, []).append(m_id)

    for u_id in range(1, num_users + 1):
        is_cold_start = u_id <= num_cold_start
        
        if is_cold_start:
            fav_genres = []
            fav_director = "Unknown"
            fav_cast = []
            fav_movies = []
            activity = int(np.random.gamma(shape=1.5, scale=20)) + 5  # New users are less active
            user_bias = np.random.normal(loc=-0.2, scale=0.5)         # Slightly more critical
        else:
            fav_genres = random.sample(all_genres, k=random.randint(1, 3))
            fav_director = random.choice(all_directors) if all_directors else "Unknown"
            fav_cast = random.sample(all_cast, k=random.randint(1, 2)) if all_cast else []
            activity = int(np.random.gamma(shape=2.5, scale=40)) + 15 
            user_bias = np.random.normal(loc=0.0, scale=0.5)
            
            # Select 2-5 favorite movies matching their genre, director, or cast preferences via dictionary lookup (O(1))
            candidate_ids = set()
            for g in fav_genres:
                if g in genre_to_movie_ids:
                    candidate_ids.update(genre_to_movie_ids[g])
            if fav_director in director_to_movie_ids:
                candidate_ids.update(director_to_movie_ids[fav_director])
            for c in fav_cast:
                if c in cast_to_movie_ids:
                    candidate_ids.update(cast_to_movie_ids[c])
                    
            matching_movie_ids = list(candidate_ids)
            
            if len(matching_movie_ids) >= 2:
                fav_movies_sample = np.random.choice(matching_movie_ids, size=min(len(matching_movie_ids), random.randint(2, 5)), replace=False)
            else:
                # Fallback to general popular movies
                popular_movie_ids = df_movies.head(100)["movieId"].values
                fav_movies_sample = np.random.choice(popular_movie_ids, size=random.randint(2, 5), replace=False)
                
            fav_movies = [str(mid) for mid in fav_movies_sample]
        
        users_data.append({
            "user_id": u_id,
            "favorite_genres": "|".join(fav_genres),
            "favorite_directors": fav_director,
            "favorite_cast": "|".join(fav_cast),
            "favorite_movies": "|".join(fav_movies),
            "activity_level": activity,
            "user_bias": user_bias,
            "is_cold_start": 1 if is_cold_start else 0
        })
    return pd.DataFrame(users_data)


def simulate_funnel(u_id, m_id, user_bias, user_fav_genres, user_fav_director, user_fav_cast,
                    m_genres, m_director, m_cast, m_vote_average, m_runtime, session_id, user_time):
    """
    Simulate the click funnel for a single movie: click -> detail_view -> watch_start -> watch_complete -> rating.
    Returns:
        funnel_events: list of dicts representing event logs.
        rating_data: dict of rating (if generated), else None.
        updated_time: datetime representing user's time after interaction.
    """
    funnel_events = []
    rating_data = None
    
    # 1. Click Event
    user_time += timedelta(minutes=random.randint(1, 10))
    funnel_events.append({
        "userId": u_id,
        "movieId": m_id,
        "timestamp": user_time,
        "event_type": "click",
        "session_id": session_id
    })
    
    # Get genre, director, cast info for matching
    movie_genres = m_genres.split("|")
    genre_fit = any(g in user_fav_genres for g in movie_genres) if user_fav_genres else False
    director_fit = (m_director == user_fav_director) and (m_director != "Unknown")
    movie_cast = str(m_cast).split("|")
    cast_fit = any(c in user_fav_cast for c in movie_cast) if user_fav_cast else False
    
    # Extract runtime if present
    if m_runtime is not None and pd.notna(m_runtime):
        runtime_sec = int(m_runtime) * 60
    else:
        runtime_sec = random.randint(80, 150) * 60
    
    # 2. Xem chi tiết (Detail View)
    p_detail = 0.55 + (0.15 if genre_fit else 0.0) + (0.05 if director_fit else 0.0) + (0.05 if cast_fit else 0.0)
    p_detail = min(max(p_detail, 0.1), 0.95)
    
    if random.random() < p_detail:
        user_time += timedelta(seconds=random.randint(10, 80))
        funnel_events.append({
            "userId": u_id,
            "movieId": m_id,
            "timestamp": user_time,
            "event_type": "detail_view",
            "session_id": session_id
        })
        
        # 3. Quyết định Xem phim thật (watch_start)
        p_watch = 0.35 + (0.15 if m_vote_average >= 7.0 else 0.0) + (0.10 if genre_fit else 0.0) + (0.05 if director_fit else 0.0)
        p_watch = min(max(p_watch, 0.1), 0.95)
        
        if random.random() < p_watch:
            user_time += timedelta(seconds=5)
            funnel_events.append({
                "userId": u_id,
                "movieId": m_id,
                "timestamp": user_time,
                "event_type": "watch_start",
                "session_id": session_id
            })
            
            # 4. Hoàn thành xem phim (watch_complete)
            p_complete = 0.60 + (0.15 if m_vote_average >= 7.5 else 0.0) + (0.10 if genre_fit else 0.0) - (0.20 if m_vote_average <= 5.0 else 0.0)
            
            # Adjust completion probability based on runtime (penalize longer runtimes: e.g., -5% per 30 minutes over 2 hours)
            if runtime_sec > 120 * 60:
                p_complete -= 0.05 * ((runtime_sec - 120 * 60) / 1800)
                
            p_complete = min(max(p_complete, 0.1), 0.95)
            is_completed = random.random() < p_complete
            
            # Set watch duration based on runtime
            if is_completed:
                watch_duration = int(random.uniform(0.9, 1.0) * runtime_sec)
            else:
                watch_duration = int(random.uniform(0.01, 0.8) * runtime_sec)
                
            user_time += timedelta(seconds=watch_duration)
            
            if is_completed:
                funnel_events.append({
                    "userId": u_id,
                    "movieId": m_id,
                    "timestamp": user_time,
                    "event_type": "watch_complete",
                    "session_id": session_id
                })
            
            # 5. Sinh đánh giá Ratings (J-Shaped Beta Distribution)
            if random.random() < 0.35:
                item_bias = (m_vote_average - 5.0) / 2.0
                genre_bonus = np.random.uniform(0.5, 1.2) if genre_fit else 0.0
                director_bonus = np.random.uniform(0.5, 1.2) if director_fit else 0.0
                cast_bonus = np.random.uniform(0.3, 0.8) if cast_fit else 0.0
                completion_bonus = np.random.uniform(0.3, 0.8) if is_completed else np.random.uniform(-1.5, -0.5)
                
                affinity = 0.5 + user_bias + item_bias + genre_bonus + director_bonus + cast_bonus + completion_bonus
                p_rating = 1.0 / (1.0 + np.exp(-1.2 * affinity))
                
                S = 6.0
                alpha = max(0.1, p_rating * S)
                beta = max(0.1, (1.0 - p_rating) * S)
                
                rating_score = 1.0 + 4.0 * np.random.beta(alpha, beta)
                rating = min(max(round(rating_score * 2) / 2, 0.5), 5.0)
                
                rating_data = {
                    "userId": u_id,
                    "movieId": m_id,
                    "rating": rating,
                    "timestamp": int(user_time.timestamp())
                }
                
    return funnel_events, rating_data, user_time

def simulate_session(u_id, user_bias, user_fav_genres, user_fav_director, user_fav_cast,
                     session_id, user_time, df_movies, movie_genres_matrix, movie_cast_matrix,
                     movie_directors_array, movie_release_datetime, movie_popularity,
                     movie_available_after, genre_to_idx, cast_to_idx, exploration_ratio,
                     is_exploration_session, start_time,
                     movie_ids, movie_genres_list, movie_directors_list, movie_casts_list,
                     movie_vote_averages, movie_runtimes):
    """
    Simulate recommendation listing generation and user interactions within one session.
    """
    session_events = []
    session_ratings = []
    total_impressions = 0
    total_clicks = 0
    
    # Establish Session Intent Genre
    session_intent_genre = None
    if user_fav_genres:
        session_intent_genre = random.choice(user_fav_genres)
        
    num_movies = len(df_movies)
    
    # Vectorized Attraction Score Calculation
    # 1. Time Decay
    delta_days = (np.datetime64(user_time) - movie_release_datetime) / np.timedelta64(1, 'D')
    delta_days = np.clip(delta_days, 0, None)
    decay_factor = np.exp(-0.00015 * delta_days)
    decay_popularity = movie_popularity * decay_factor
    
    # 2. Exploit Mode vs. Exploration Mode
    if is_exploration_session or not user_fav_genres:
        attraction_scores = decay_popularity
    else:
        # Optimize by summing active columns rather than full matrix multiplication
        active_genre_indices = [genre_to_idx[g] for g in user_fav_genres if g in genre_to_idx]
        genre_fit = np.sum(movie_genres_matrix[:, active_genre_indices], axis=1) > 0 if active_genre_indices else np.zeros(num_movies, dtype=bool)
        genre_multiplier = np.where(genre_fit, 3.5, 1.0)
        
        # Intent Match
        if session_intent_genre and session_intent_genre in genre_to_idx:
            intent_idx = genre_to_idx[session_intent_genre]
            intent_fit = movie_genres_matrix[:, intent_idx] > 0
            genre_multiplier = np.where(intent_fit, genre_multiplier * 2.0, genre_multiplier)
            
        # Director Match
        director_fit = (movie_directors_array == user_fav_director) & (movie_directors_array != "Unknown")
        director_multiplier = np.where(director_fit, 1.8, 1.0)
        
        # Optimize cast matching by summing active columns
        active_cast_indices = [cast_to_idx[c] for c in user_fav_cast if c in cast_to_idx]
        cast_fit = np.sum(movie_cast_matrix[:, active_cast_indices], axis=1) > 0 if active_cast_indices else np.zeros(num_movies, dtype=bool)
        cast_multiplier = np.where(cast_fit, 1.4, 1.0)
        
        attraction_scores = decay_popularity * genre_multiplier * director_multiplier * cast_multiplier
        
    # Apply Item Cold-Start Masking
    is_available = movie_available_after <= np.datetime64(user_time)
    attraction_scores = np.where(is_available, attraction_scores, -1e9)
    
    # Ranked Top 20 Candidates with Gumbel Noise (argpartition is O(N) vs argsort O(N log N))
    noise = np.random.gumbel(0, 0.1, size=num_movies)
    top20_partition = np.argpartition(-(attraction_scores + noise), 20)[:20]
    ranked_indices = top20_partition[np.argsort(-(attraction_scores + noise)[top20_partition])]
    
    # Process Impression candidates & Clicks
    for pos, idx_m in enumerate(ranked_indices, start=1):
        m_id = movie_ids[idx_m]
        
        total_impressions += 1
        
        # Click probability calculations (Position Bias & base click probability)
        position_penalty = 1.0 / np.log2(pos + 1)
        
        max_att = max(1.0, attraction_scores.max())
        base_click_prob = (attraction_scores[idx_m] / max_att) * 0.45
        
        p_click = base_click_prob * position_penalty
        p_click = min(max(p_click, 0.02), 0.90)
        
        is_clicked = random.random() < p_click
        
        if is_clicked:
            total_clicks += 1
            funnel_evs, rating, updated_time = simulate_funnel(
                u_id=u_id,
                m_id=m_id,
                user_bias=user_bias,
                user_fav_genres=user_fav_genres,
                user_fav_director=user_fav_director,
                user_fav_cast=user_fav_cast,
                m_genres=movie_genres_list[idx_m],
                m_director=movie_directors_list[idx_m],
                m_cast=movie_casts_list[idx_m],
                m_vote_average=movie_vote_averages[idx_m],
                m_runtime=movie_runtimes[idx_m] if movie_runtimes is not None else None,
                session_id=session_id,
                user_time=user_time
            )
            session_events.extend(funnel_evs)
            if rating:
                session_ratings.append(rating)
            user_time = updated_time
            
    return session_events, session_ratings, total_impressions, total_clicks, user_time

def generate_advanced_simulator(movies_csv, num_users, output_dir, cold_start_users_ratio=0.10, exploration_ratio=0.05, cold_start_movies_ratio=0.20):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load movies
    df_movies = load_real_movies(movies_csv)
    num_movies = len(df_movies)
    print(f"[Success] Loaded {num_movies} real movies from: {movies_csv}")
    
    # Extract unique genres
    all_genres = set()
    for item in df_movies["genres"].str.split("|"):
        all_genres.update(item)
    all_genres = list(all_genres - {"(no genres listed)", ""})
    if not all_genres:
        all_genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
        
    # Extract popular directors
    director_counts = df_movies["director"].value_counts()
    all_directors = director_counts[(director_counts.index != "Unknown") & (director_counts >= 2)].index.tolist()
    if not all_directors:
        all_directors = director_counts[director_counts.index != "Unknown"].head(50).index.tolist()
    if not all_directors:
        all_directors = ["Christopher Nolan", "Steven Spielberg", "Quentin Tarantino", "James Cameron"]

    # Extract popular cast members
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

    # 2. Setup precomputation arrays/matrices for vectorization
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres = len(all_genres)
    movie_genres_matrix = np.zeros((num_movies, num_genres), dtype=float)
    for idx, genres_str in enumerate(df_movies["genres"]):
        for g in str(genres_str).split("|"):
            if g in genre_to_idx:
                movie_genres_matrix[idx, genre_to_idx[g]] = 1.0

    cast_to_idx = {c: i for i, c in enumerate(all_cast)}
    num_cast = len(all_cast)
    movie_cast_matrix = np.zeros((num_movies, num_cast), dtype=float)
    for idx, cast_str in enumerate(df_movies["cast"]):
        for c in str(cast_str).split("|"):
            if c in cast_to_idx:
                movie_cast_matrix[idx, cast_to_idx[c]] = 1.0

    movie_directors_array = df_movies["director"].values
    movie_popularity = df_movies["popularity"].values

    start_time = datetime(2026, 1, 1)
    df_movies["release_datetime"] = pd.to_datetime(df_movies["release_date"], errors="coerce")
    default_release = pd.to_datetime("2010-01-01")

    # 3. Setup Item Cold-Start
    num_cold_start_movies = int(num_movies * cold_start_movies_ratio)
    if num_cold_start_movies > 0:
        cold_start_movie_indices = np.random.choice(num_movies, size=num_cold_start_movies, replace=False)
    else:
        cold_start_movie_indices = np.array([], dtype=int)

    movie_available_after = np.full(num_movies, start_time, dtype='datetime64[ns]')
    for idx in cold_start_movie_indices:
        random_days = random.randint(1, 45)
        random_hours = random.randint(0, 23)
        movie_available_after[idx] = np.datetime64(start_time + timedelta(days=random_days, hours=random_hours))

    # Precompute release date datetime array for time decay, mapping NaT to default_release
    movie_release_datetime = df_movies["release_datetime"].values.copy()
    movie_release_datetime[pd.isna(movie_release_datetime)] = np.datetime64(default_release)
    if len(cold_start_movie_indices) > 0:
        movie_release_datetime[cold_start_movie_indices] = movie_available_after[cold_start_movie_indices]

    # Pre-extract movie columns for O(1) list/array indexing in loops (bypasses slow pandas .iloc)
    movie_ids = df_movies["movieId"].values
    movie_genres_list = df_movies["genres"].tolist()
    movie_directors_list = df_movies["director"].tolist()
    movie_casts_list = df_movies["cast"].tolist()
    movie_vote_averages = df_movies["vote_average"].values
    movie_runtimes = df_movies["runtime"].values if "runtime" in df_movies.columns else None

    # Pre-map movie properties for instant O(1) lookups during user preference inference
    movie_id_genres = df_movies.set_index("movieId")["genres"].to_dict()
    movie_id_director = df_movies.set_index("movieId")["director"].to_dict()
    movie_id_cast = df_movies.set_index("movieId")["cast"].to_dict()

    # 4. Simulate Users
    df_users = simulate_users(num_users, df_movies, all_genres, all_directors, all_cast, cold_start_users_ratio)
    
    # Save only OLTP-friendly simplified columns to users CSV (do not write favorite_genres/directors/cast)
    cols_to_save = ["user_id", "favorite_movies", "activity_level", "user_bias", "is_cold_start"]
    df_users[cols_to_save].to_csv(os.path.join(output_dir, "sim_users.csv"), index=False)
    
    num_cold_start = int(num_users * cold_start_users_ratio)
    print(f"Generated {num_users} users profiles (Cold-start: {num_cold_start}) saved to: sim_users.csv")

    # 5. Simulate Interactions
    click_events = []
    ratings_data = []
    
    total_sessions_count = 0
    total_impressions_count = 0
    total_clicks_count = 0

    for idx, user in df_users.iterrows():
        u_id = int(user["user_id"])
        
        # Dynamically infer preferences from favorite movies list using precomputed O(1) lookups
        fav_movies_list = [m for m in str(user.get("favorite_movies", "")).split("|") if m]
        user_fav_genres, user_fav_director, user_fav_cast = infer_preferences_from_movies(
            fav_movies_list, movie_id_genres, movie_id_director, movie_id_cast
        )
        
        activity = user["activity_level"]
        u_bias = user["user_bias"]
        
        user_time = start_time + timedelta(hours=random.randint(0, 720))
        current_week = 0
        
        num_sessions = max(3, activity // 15)
        for s_idx in range(num_sessions):
            total_sessions_count += 1
            
            session_delay_hours = random.randint(12, 120)
            user_time += timedelta(hours=session_delay_hours)
            
            if random.random() < 0.70:
                gold_hour = random.randint(19, 23)
                user_time = user_time.replace(hour=gold_hour, minute=random.randint(0, 59), second=random.randint(0, 59))
                
            week = (user_time - start_time).days // 7
            if week > current_week:
                current_week = week
                if random.random() < 0.25 and len(user_fav_genres) > 0:
                    idx_to_swap = random.randint(0, len(user_fav_genres) - 1)
                    new_genre = random.choice(list(set(all_genres) - set(user_fav_genres)))
                    user_fav_genres[idx_to_swap] = new_genre
                if random.random() < 0.10:
                    user_fav_director = random.choice(all_directors) if all_directors else "Unknown"
                    
            session_id = f"sess_{u_id}_{s_idx + 1}"
            is_exploration_session = random.random() < exploration_ratio
            
            sess_evs, sess_ratings, sess_imps, sess_clicks, updated_time = simulate_session(
                u_id=u_id,
                user_bias=u_bias,
                user_fav_genres=user_fav_genres,
                user_fav_director=user_fav_director,
                user_fav_cast=user_fav_cast,
                session_id=session_id,
                user_time=user_time,
                df_movies=df_movies,
                movie_genres_matrix=movie_genres_matrix,
                movie_cast_matrix=movie_cast_matrix,
                movie_directors_array=movie_directors_array,
                movie_release_datetime=movie_release_datetime,
                movie_popularity=movie_popularity,
                movie_available_after=movie_available_after,
                genre_to_idx=genre_to_idx,
                cast_to_idx=cast_to_idx,
                exploration_ratio=exploration_ratio,
                is_exploration_session=is_exploration_session,
                start_time=start_time,
                movie_ids=movie_ids,
                movie_genres_list=movie_genres_list,
                movie_directors_list=movie_directors_list,
                movie_casts_list=movie_casts_list,
                movie_vote_averages=movie_vote_averages,
                movie_runtimes=movie_runtimes
            )
            
            click_events.extend(sess_evs)
            ratings_data.extend(sess_ratings)
            total_impressions_count += sess_imps
            total_clicks_count += sess_clicks
            user_time = updated_time

    # 6. Save outputs
    df_click_events = pd.DataFrame(click_events)
    df_ratings = pd.DataFrame(ratings_data)
    
    df_click_events.to_csv(os.path.join(output_dir, "sim_click_events.csv"), index=False)
    df_ratings.to_csv(os.path.join(output_dir, "sim_ratings.csv"), index=False)
    
    print(f"\n[Success] Generated simulation datasets saved in '{output_dir}':")
    print(f"- {len(df_click_events):,} click funnel events -> sim_click_events.csv (100% backward compatible)")
    print(f"- {len(df_ratings):,} rating instances -> sim_ratings.csv")

    # 7. Print QC Summary
    if len(df_click_events) > 0:
        event_counts = df_click_events["event_type"].value_counts()
        total_clicks = event_counts.get("click", 0)
        total_details = event_counts.get("detail_view", 0)
        total_starts = event_counts.get("watch_start", 0)
        total_completes = event_counts.get("watch_complete", 0)
        
        ctr = total_clicks / total_impressions_count if total_impressions_count > 0 else 0
        click_to_detail = total_details / total_clicks if total_clicks > 0 else 0
        detail_to_start = total_starts / total_details if total_details > 0 else 0
        start_to_complete = total_completes / total_starts if total_starts > 0 else 0
    else:
        total_clicks = total_details = total_starts = total_completes = 0
        ctr = click_to_detail = detail_to_start = start_to_complete = 0
        
    num_ratings = len(df_ratings)
    avg_rating = df_ratings["rating"].mean() if num_ratings > 0 else 0.0
    rating_sparsity = 1.0 - (num_ratings / (num_users * num_movies)) if num_movies > 0 and num_users > 0 else 1.0
    
    print("\n" + "="*80)
    print("                              QC DATA SUMMARY")
    print("="*80)
    print(f"Total Users:           {num_users:<8} (Cold-start: {num_cold_start})")
    print(f"Total Movies:          {num_movies:<8} (Cold-start: {num_cold_start_movies})")
    print(f"Total Sessions:        {total_sessions_count:<8}")
    print(f"Total Impressions:     {total_impressions_count:<8}")
    print(f"Total Clicks:          {total_clicks:<8}")
    print(f"Overall CTR:           {ctr:.4f} ({ctr*100:.2f}%)")
    print("\nFunnel Breakdown:")
    print(f"- Clicks:              {total_clicks:<8}")
    print(f"- Detail Views:        {total_details:<8} (Conversion from Click: {click_to_detail*100:.2f}%)")
    print(f"- Watch Starts:        {total_starts:<8} (Conversion from Detail: {detail_to_start*100:.2f}%)")
    print(f"- Watch Completes:     {total_completes:<8} (Completion Rate of Starts: {start_to_complete*100:.2f}%)")
    print("\nRatings Metrics:")
    print(f"- Total Ratings:       {num_ratings:<8}")
    print(f"- Average Rating:      {avg_rating:.2f}")
    print(f"- Rating Sparsity:     {rating_sparsity*100:.2f}% (Density: {(1 - rating_sparsity)*100:.4f}%)")
    print("="*80 + "\n")

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
    parser.add_argument("--cold_start_users_ratio", type=float, default=0.10,
                        help="Ratio of cold start users with empty profiles (default: 0.10)")
    parser.add_argument("--exploration_ratio", type=float, default=0.05,
                        help="Ratio of exploration (serendipity) clicks based on global popularity (default: 0.05)")
    parser.add_argument("--cold_start_movies_ratio", type=float, default=0.20,
                        help="Ratio of cold start movies appearing late in simulation (default: 0.20)")
                        
    args = parser.parse_args()
    
    generate_advanced_simulator(
        movies_csv=args.movies_csv,
        num_users=args.num_users,
        output_dir=args.output_dir,
        cold_start_users_ratio=args.cold_start_users_ratio,
        exploration_ratio=args.exploration_ratio,
        cold_start_movies_ratio=args.cold_start_movies_ratio
    )
