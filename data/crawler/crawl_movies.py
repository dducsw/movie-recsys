import os
import argparse
import time
import requests
import pandas as pd

def load_dotenv(env_path=".env"):
    """
    Manually load .env file to populate environment variables
    without requiring third-party dotenv library.
    """
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    os.environ[key] = value

def fetch_genres(api_key, session=None):
    """
    Fetch genre list from TMDB to map genre IDs to text names.
    """
    url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": api_key, "language": "en-US"}
    try:
        get_func = session.get if session else requests.get
        response = get_func(url, params=params)
        if response.status_code == 200:
            genres_data = response.json().get("genres", [])
            return {item["id"]: item["name"] for item in genres_data}
        else:
            print(f"[Warning] Failed to fetch genres (HTTP {response.status_code}). Will keep raw genre IDs.")
            return {}
    except Exception as e:
        print(f"[Warning] Error fetching genres: {e}. Will keep raw genre IDs.")
        return {}

def fetch_movie_details(api_key, movie_id, session=None):
    """
    Fetch credits (director, top cast) and keywords from TMDB.
    """
    credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
    keywords_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"
    params = {"api_key": api_key}
    
    director = "Unknown"
    cast_list = []
    keywords_list = []
    
    try:
        get_func = session.get if session else requests.get
        # Fetch credits
        res_credits = get_func(credits_url, params=params)
        if res_credits.status_code == 200:
            data = res_credits.json()
            # Lấy đạo diễn (Director)
            crew = data.get("crew", [])
            for member in crew:
                if member.get("job") == "Director":
                    director = member.get("name")
                    break
            # Lấy Top 10 Diễn viên (Cast) để tăng độ phong phú của dữ liệu gợi ý
            cast = data.get("cast", [])
            cast_list = [member.get("name") for member in cast[:10]]
            
        # Fetch keywords
        res_keywords = get_func(keywords_url, params=params)
        if res_keywords.status_code == 200:
            data = res_keywords.json()
            keywords = data.get("keywords", [])
            keywords_list = [kw.get("name") for kw in keywords]
            
    except Exception as e:
        # Ghi nhận cảnh báo nhưng không làm gián đoạn chương trình
        pass
        
    return {
        "director": director,
        "cast": "|".join(cast_list),
        "keywords": "|".join(keywords_list)
    }

def crawl_movies(api_key, start_year, end_year, max_pages, output_file):
    if not api_key or api_key == "YOUR_TMDB_API_KEY_HERE":
        print("[Error] Invalid TMDB API Key! Please update the .env file or pass the key using the --api_key parameter.")
        return
        
    print(f"Starting TMDB crawl for Hollywood movies ({start_year} - {end_year})...")
    print(f"- Target: Fetching up to {max_pages} pages (max {max_pages * 20} movies)")
    
    # Khởi tạo requests.Session để tái sử dụng TCP connections (HTTP Keep-Alive), tăng tốc độ đáng kể
    session = requests.Session()
    
    # 1. Fetch genre metadata mapping
    genre_mapping = fetch_genres(api_key, session)
    
    movies_list = []
    base_url = "https://api.themoviedb.org/3/discover/movie"
    
    # 2. Fetch movies page-by-page
    for page in range(1, max_pages + 1):
        print(f"Fetching page {page}/{max_pages}...")
        params = {
            "api_key": api_key,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "primary_release_date.gte": f"{start_year}-01-01",
            "primary_release_date.lte": f"{end_year}-12-31",
            "with_original_language": "en", # English language movies
            "page": page
        }
        
        try:
            response = session.get(base_url, params=params)
            if response.status_code != 200:
                print(f"[Error] Failed to fetch page {page}: HTTP {response.status_code}")
                break
                
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                print("No more movies available to fetch.")
                break
                
            for item in results:
                movie_id = item.get("id")
                
                # Crawl chi tiết: Director, Cast, Keywords (sử dụng session chung)
                details = fetch_movie_details(api_key, movie_id, session)
                # Tăng thời gian ngủ lên 0.3s để an toàn tuyệt đối tránh rate limit
                time.sleep(0.3)
                
                # Map genre IDs sang chuỗi ngăn cách bởi '|'
                genre_ids = item.get("genre_ids", [])
                genres_names = [genre_mapping.get(g_id, str(g_id)) for g_id in genre_ids]
                genres_str = "|".join(genres_names)
                
                poster_path = item.get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
                
                movies_list.append({
                    "movieId": movie_id,
                    "title": item.get("title"),
                    "release_date": item.get("release_date"),
                    "genres": genres_str,
                    "popularity": item.get("popularity"),
                    "adult": item.get("adult", False),
                    "overview": item.get("overview"),
                    "vote_average": item.get("vote_average"),
                    "vote_count": item.get("vote_count"),
                    "poster_url": poster_url,
                    "director": details["director"],
                    "cast": details["cast"],
                    "keywords": details["keywords"]
                })
                
            # Rate limiting delay cho trang discover tăng lên 0.6s
            time.sleep(0.6)
            
        except Exception as e:
            print(f"[Error] An error occurred on page {page}: {e}")
            break
            
    if not movies_list:
        print("[Error] No movie data crawled. Please check your configuration.")
        return
        
    # 3. Save output to CSV file
    df = pd.DataFrame(movies_list)
    
    # Create directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n[Success] Saved {len(df)} movies to: {output_file}")

if __name__ == "__main__":
    # Resolve .env relative to this script (two directories up)
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    
    parser = argparse.ArgumentParser(description="TMDB Hollywood Movies Crawler (2000-2026)")
    parser.add_argument("--api_key", type=str, default=os.getenv("TMDB_API_KEY"),
                        help="TMDB API Key (default: loaded from .env file)")
    parser.add_argument("--start_year", type=int, default=2000,
                        help="Start year to crawl (default: 2000)")
    parser.add_argument("--end_year", type=int, default=2026,
                        help="End year to crawl (default: 2026)")
    parser.add_argument("--pages", type=int, default=10,
                        help="Number of pages to fetch, 20 movies per page (default: 10)")
    parser.add_argument("--output", type=str, default="data/movies_crawled.csv",
                        help="Output path for CSV file (default: data/movies_crawled.csv)")
                        
    args = parser.parse_args()
    
    crawl_movies(
        api_key=args.api_key,
        start_year=args.start_year,
        end_year=args.end_year,
        max_pages=args.pages,
        output_file=args.output
    )
