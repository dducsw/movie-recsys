import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(title="Movie Recommendation System API", version="1.0.0")

# Enable CORS so the React Frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development ease
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection configuration nạp từ môi trường hoặc mặc định
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5435")
DB_NAME = os.getenv("DB_NAME", "movie_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")

def get_db_connection():
    """
    Tạo và trả về một kết nối cơ sở dữ liệu PostgreSQL.
    Sử dụng RealDictCursor để trả về kết quả truy vấn dưới dạng dictionary (key-value)
    giúp dễ dàng chuyển đổi sang JSON.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"[Error] Failed to connect to PostgreSQL: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Database connection failure. Please make sure the PostgreSQL container is running."
        )

@app.get("/api/movies/trending")
def get_trending_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Lấy danh sách các bộ phim phổ biến nhất (Trending) từ CSDL.
    Sắp xếp theo độ phổ biến (popularity) từ cao xuống thấp.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    offset = (page - 1) * limit
    
    try:
        # Lấy danh sách phim phân trang
        query = """
            SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
            FROM movies 
            ORDER BY popularity DESC 
            LIMIT %s OFFSET %s
        """
        cur.execute(query, (limit, offset))
        movies = cur.fetchall()
        
        # Đếm tổng số phim để phân trang
        cur.execute('SELECT COUNT(*) FROM movies')
        total_count = cur.fetchone()['count']
        
        return {
            "page": page,
            "limit": limit,
            "total_movies": total_count,
            "results": movies
        }
    except Exception as e:
        print(f"[Error] Failed to fetch trending movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch data from movies table.")
    finally:
        cur.close()
        conn.close()

@app.get("/api/movies/search")
def search_movies(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=50)
):
    """
    Tìm kiếm phim theo từ khóa tiêu đề (case-insensitive) sử dụng ILIKE.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Sử dụng ILIKE để tìm kiếm không phân biệt hoa thường và sắp xếp theo độ phổ biến
        search_query = f"%{query}%"
        sql_query = """
            SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
            FROM movies 
            WHERE title ILIKE %s 
            ORDER BY popularity DESC 
            LIMIT %s
        """
        cur.execute(sql_query, (search_query, limit))
        results = cur.fetchall()
        
        return {
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        print(f"[Error] Search failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute search query.")
    finally:
        cur.close()
        conn.close()

@app.get("/api/movies/{movie_id}")
def get_movie_detail(movie_id: int):
    """
    Lấy thông tin chi tiết của một bộ phim cụ thể qua movieId.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        query = """
            SELECT movieid AS "movieId", title, release_date, genres, popularity, adult, overview, vote_average, vote_count, poster_url 
            FROM movies 
            WHERE movieid = %s
        """
        cur.execute(query, (movie_id,))
        movie = cur.fetchone()
        
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found.")
            
        return movie
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Failed to fetch movie detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to query movie detail.")
    finally:
        cur.close()
        conn.close()
