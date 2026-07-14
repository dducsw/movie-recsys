from app.models.movie import MovieModel
from typing import List, Dict, Any
from collections import Counter

class RecsysService:
    @staticmethod
    def get_similar_movies(movie_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        """
        Calculates similar movies based on genre overlap and popularity.
        """
        # Fetch the target movie details
        target_movie = MovieModel.get_by_id(movie_id)
        if not target_movie:
            return []
        
        target_genres = set(target_movie["genres"].split("|")) if target_movie["genres"] else set()
        
        # If no genres, fall back to trending
        if not target_genres:
            return MovieModel.get_trending(page=1, limit=limit)

        # Get all movies to calculate similarities
        all_movies = MovieModel.get_all_genres_and_popularity()
        scored_movies = []

        for movie in all_movies:
            if movie["movieId"] == movie_id:
                continue
            
            movie_genres = set(movie["genres"].split("|")) if movie["genres"] else set()
            shared_genres = target_genres.intersection(movie_genres)
            
            if not shared_genres:
                continue
            
            # Score formula: number of shared genres + logarithmic scale of popularity
            # We add a slight boost for movies with higher vote_average
            vote_avg = movie.get("vote_average", 0.0) or 0.0
            popularity = movie.get("popularity", 0.0) or 0.0
            
            overlap_score = len(shared_genres)
            # Normalize popularity boost
            pop_boost = min(popularity / 50.0, 5.0) 
            score = (overlap_score * 10) + pop_boost + (vote_avg * 0.5)
            
            scored_movies.append((movie, score))

        # Sort by score descending, then popularity descending
        scored_movies.sort(key=lambda x: (x[1], x[0].get("popularity", 0.0) or 0.0), reverse=True)
        
        # Prepare list of movies to return
        results = []
        for movie, _ in scored_movies[:limit]:
            results.append({
                "movieId": movie["movieId"],
                "title": movie["title"],
                "release_date": movie["release_date"],
                "genres": movie["genres"],
                "popularity": movie["popularity"],
                "vote_average": movie.get("vote_average", 0.0),
                "poster_url": movie["poster_url"]
            })
            
        # Fallback to trending if not enough similar movies
        if len(results) < limit:
            trending = MovieModel.get_trending(page=1, limit=limit - len(results))
            # avoid duplicates
            existing_ids = {m["movieId"] for m in results}
            for m in trending:
                if m["movieId"] not in existing_ids and m["movieId"] != movie_id:
                    results.append(m)
                    if len(results) >= limit:
                        break
                        
        return results

    @staticmethod
    def get_personalized_recommendations(liked_movie_ids: List[int], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Generates personalized recommendations based on the user's liked movies.
        Analyzes genres of liked movies to build a profile, then scores all other movies.
        """
        # If no liked movies (cold start), return trending movies
        if not liked_movie_ids:
            return MovieModel.get_trending(page=1, limit=limit)

        # Get details of the liked movies
        liked_movies = MovieModel.get_by_ids(liked_movie_ids)
        if not liked_movies:
            return MovieModel.get_trending(page=1, limit=limit)

        # Build genre profile (count frequencies)
        genre_counts = Counter()
        for movie in liked_movies:
            genres = movie["genres"].split("|") if movie["genres"] else []
            for genre in genres:
                if genre:
                    genre_counts[genre] += 1

        # Normalize weights
        total_likes = len(liked_movies)
        genre_weights = {genre: count / total_likes for genre, count in genre_counts.items()}

        # Fetch all movies to score
        all_movies = MovieModel.get_all_genres_and_popularity()
        scored_movies = []
        liked_set = set(liked_movie_ids)

        for movie in all_movies:
            if movie["movieId"] in liked_set:
                continue

            movie_genres = movie["genres"].split("|") if movie["genres"] else []
            # Calculate match score based on user profile weights
            match_score = sum(genre_weights.get(genre, 0.0) for genre in movie_genres)
            
            if match_score == 0:
                continue

            # Incorporate popularity and average rating
            popularity = movie.get("popularity", 0.0) or 0.0
            vote_avg = movie.get("vote_average", 0.0) or 0.0
            
            pop_boost = min(popularity / 100.0, 3.0)
            score = (match_score * 15) + pop_boost + (vote_avg * 0.3)
            
            scored_movies.append((movie, score))

        # Sort by score descending, then popularity descending
        scored_movies.sort(key=lambda x: (x[1], x[0].get("popularity", 0.0) or 0.0), reverse=True)

        results = []
        for movie, _ in scored_movies[:limit]:
            results.append({
                "movieId": movie["movieId"],
                "title": movie["title"],
                "release_date": movie["release_date"],
                "genres": movie["genres"],
                "popularity": movie["popularity"],
                "vote_average": movie.get("vote_average", 0.0),
                "poster_url": movie["poster_url"]
            })

        # Fill up with trending if we don't have enough recommendations
        if len(results) < limit:
            trending = MovieModel.get_trending(page=1, limit=limit - len(results))
            existing_ids = {m["movieId"] for m in results}
            for m in trending:
                if m["movieId"] not in existing_ids and m["movieId"] not in liked_set:
                    results.append(m)
                    if len(results) >= limit:
                        break

        return results
