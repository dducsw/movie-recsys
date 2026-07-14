# Movie Recommender Web Application (React + FastAPI)

Tài liệu này hướng dẫn chi tiết cách thiết lập, thiết kế kiến trúc và xây dựng một ứng dụng web (Web App) hoàn chỉnh để trình chiếu và tương tác với các phim được gợi ý. Ứng dụng sử dụng **React (Vite) + Tailwind CSS** cho Frontend và **FastAPI (Python)** cho Backend.

---

## 1. Kiến trúc Tổng quan (System Architecture)

Ứng dụng hoạt động theo mô hình Client-Server không đồng bộ:

```
┌────────────────────────────────────────┐
│          React Frontend (Client)       │
│  - Giao diện người dùng (Netflix UI)   │
│  - Gửi sự kiện: Click, Rating          │
└──────────────────┬─────────────────────┘
                   │
                   │ HTTP Requests (REST API)
                   ▼
┌────────────────────────────────────────┐
│          FastAPI Backend (Server)      │
│  - API Endpoints (Routing & Validation)│
│  - Đọc/Ghi cơ sở dữ liệu (ORM)          │
└──────────────────┬─────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌────────────────────┐
│   Database      │ │ Recommender Engine │
│  - SQLite/Post  │ │ - Load model SVD   │
│  - Metadata, Log│ │ - Predict ratings  │
└─────────────────┘ └────────────────────┘
```

---

## 2. Cấu trúc Thư mục Dự án (Project Structure)

Chúng ta tổ chức thư mục tách biệt giữa Frontend và Backend để dễ quản lý và triển khai độc lập:

```
webapp/
├── backend/
│   ├── main.py            # Khởi chạy ứng dụng FastAPI & định nghĩa routes
│   ├── database.py        # Kết nối CSDL (SQLAlchemy)
│   ├── models.py          # Khai báo cấu trúc bảng (ORM Models)
│   ├── schemas.py         # Kiểm tra định dạng dữ liệu đầu vào (Pydantic)
│   ├── recommender.py     # Gọi mô hình SVD/Collaborative Filtering để dự đoán
│   ├── requirements.txt   # Các thư viện Python cần thiết
│   └── movies.db          # File CSDL SQLite tạm thời
└── frontend/
    ├── package.json       # Quản lý dependencies của Nodejs
    ├── tailwind.config.js # Cấu hình Tailwind CSS
    ├── index.html
    └── src/
        ├── main.jsx       # Điểm khởi chạy React
        ├── App.jsx        # Component chính quản lý giao diện và state
        └── components/    # Các UI components tái sử dụng
            ├── MovieCard.jsx
            └── MovieRow.jsx
```

---

## 3. Hướng dẫn xây dựng Backend (FastAPI)

### Bước 1: Khai báo các thư viện cần thiết
Tạo file `backend/requirements.txt`:
```text
fastapi>=0.100.0
uvicorn>=0.22.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-surprise>=1.1.3

# Chatbot
langchain-core==1.4.6
langchain-google-genai==4.2.1
langgraph>=1.2.2
tavily-python>=0.7.26
```

### Bước 2: Thiết kế Database Models (`backend/models.py`)
Sử dụng SQLAlchemy để định nghĩa các bảng lưu thông tin phim và tương tác người dùng:

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()

class Movie(Base):
    __tablename__ = "movies"
    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    genres = Column(String)
    poster_url = Column(String)
    overview = Column(String)
    vote_average = Column(Float)

class UserInteraction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"))
    event_type = Column(String)  # click, detail_view, watch_start, watch_complete
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class UserRating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.movie_id"))
    rating = Column(Float)  # 0.5 - 5.0
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
```

### Bước 3: Tích hợp Bộ gợi ý (`backend/recommender.py`)
Mô phỏng hàm gọi mô hình Collaborative Filtering (ví dụ dùng thư viện `Surprise` đã được huấn luyện offline):

```python
import os
import pickle
import pandas as pd

class Recommender:
    def __init__(self):
        # Trong thực tế, load model SVD đã được lưu dưới dạng file pickle
        # self.model = pickle.load(open("models/svd_model.pkl", "rb"))
        self.model = None

    def get_personal_recommendations(self, user_id: int, db_session, top_n=10):
        # 1. Lấy tất cả danh sách phim từ database
        from models import Movie
        all_movies = db_session.query(Movie).all()
        
        predictions = []
        for movie in all_movies:
            # Dự đoán rating mà user sẽ chấm cho phim này
            # Nếu có model: pred = self.model.predict(user_id, movie.movie_id).est
            # Tạm thời sinh điểm ngẫu nhiên giả lập nếu chưa có model
            pred = round(pd.Series([2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).sample().values[0], 1)
            predictions.append((movie, pred))
            
        # Sắp xếp các phim theo điểm số dự đoán giảm dần
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in predictions[:top_n]]

rec_engine = Recommender()
```

### Bước 4: Xây dựng các API Endpoints (`backend/main.py`)
Khởi tạo FastAPI, giải quyết cấu hình CORS để cho phép React (chạy trên port 5173) gọi dữ liệu, và định nghĩa các API routes:

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, recommender

# Khởi tạo DB tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Movie Recommender API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Port mặc định của Vite React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency để kết nối DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/movies/popular")
def get_popular_movies(db: Session = Depends(get_db)):
    # Trả về top 20 phim có điểm đánh giá chung cao nhất
    return db.query(models.Movie).order_by(models.Movie.vote_average.desc()).limit(20).all()

@app.get("/api/recommendations/{user_id}")
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    return recommender.rec_engine.get_personal_recommendations(user_id, db)

@app.post("/api/interactions")
def log_interaction(user_id: int, movie_id: int, event_type: str, db: Session = Depends(get_db)):
    interaction = models.UserInteraction(user_id=user_id, movie_id=movie_id, event_type=event_type)
    db.add(interaction)
    db.commit()
    return {"status": "success", "message": f"Logged {event_type} event."}

@app.post("/api/ratings")
def submit_rating(user_id: int, movie_id: int, rating: float, db: Session = Depends(get_db)):
    if rating < 0.5 or rating > 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")
    user_rating = models.UserRating(user_id=user_id, movie_id=movie_id, rating=rating)
    db.add(user_rating)
    db.commit()
    return {"status": "success", "message": "Rating submitted."}
```

---

## 4. Hướng dẫn xây dựng Frontend (React + Vite)

### Bước 1: Khởi tạo dự án
Trong thư mục `webapp/`, chạy lệnh khởi tạo:
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Cấu hình `frontend/tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```
Thêm `@tailwind base; @tailwind components; @tailwind utilities;` vào đầu file `frontend/src/index.css`.

### Bước 2: Viết mã nguồn giao diện chính (`frontend/src/App.jsx`)
Giao diện hiển thị danh sách phim dạng dòng (rows) giống Netflix, cho phép bấm chấm điểm và gửi tương tác:

```jsx
import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000/api';
const CURRENT_USER_ID = 1; // Giả định user đang đăng nhập là ID 1

function App() {
  const [recommendedMovies, setRecommendedMovies] = useState([]);
  const [popularMovies, setPopularMovies] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [rating, setRating] = useState(0);

  useEffect(() => {
    // 1. Tải danh sách phim phổ biến
    fetch(`${API_BASE}/movies/popular`)
      .then(res => res.json())
      .then(data => setPopularMovies(data));

    // 2. Tải danh sách phim gợi ý cá nhân hóa
    fetch(`${API_BASE}/recommendations/${CURRENT_USER_ID}`)
      .then(res => res.json())
      .then(data => setRecommendedMovies(data));
  }, []);

  const handleMovieClick = (movie) => {
    setSelectedMovie(movie);
    // Gửi sự kiện click về backend
    fetch(`${API_BASE}/interactions?user_id=${CURRENT_USER_ID}&movie_id=${movie.movie_id}&event_type=click`, {
      method: 'POST'
    });
  };

  const handleRateSubmit = (movieId) => {
    fetch(`${API_BASE}/ratings?user_id=${CURRENT_USER_ID}&movie_id=${movieId}&rating=${rating}`, {
      method: 'POST'
    }).then(() => {
      alert("Cảm ơn bạn đã đánh giá!");
      setSelectedMovie(null);
      setRating(0);
      // Có thể gọi lại API tải danh sách gợi ý để cập nhật theo thời gian thực
    });
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-extrabold text-red-600 tracking-wider">NETFLIX REC-SYS</h1>
      </header>

      {/* Row 1: Phim gợi ý cho bạn */}
      <section className="mb-12">
        <h2 className="text-xl font-bold mb-4 text-gray-300">Gợi Ý Dành Riêng Cho Bạn</h2>
        <div className="flex space-x-4 overflow-x-auto pb-4 scrollbar-hide">
          {recommendedMovies.map(movie => (
            <div 
              key={movie.movie_id} 
              className="flex-none w-48 bg-neutral-900 rounded-lg overflow-hidden cursor-pointer hover:scale-105 transition-transform duration-300"
              onClick={() => handleMovieClick(movie)}
            >
              <img src={movie.poster_url || 'https://via.placeholder.com/150x225'} alt={movie.title} className="w-full h-72 object-cover" />
              <div className="p-3">
                <h3 className="font-semibold text-sm truncate">{movie.title}</h3>
                <span className="text-xs text-yellow-500">⭐ {movie.vote_average.toFixed(1)}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Row 2: Phim phổ biến */}
      <section className="mb-12">
        <h2 className="text-xl font-bold mb-4 text-gray-300">Phổ Biến Trên Hệ Thống</h2>
        <div className="flex space-x-4 overflow-x-auto pb-4">
          {popularMovies.map(movie => (
            <div 
              key={movie.movie_id} 
              className="flex-none w-48 bg-neutral-900 rounded-lg overflow-hidden cursor-pointer hover:scale-105 transition-transform duration-300"
              onClick={() => handleMovieClick(movie)}
            >
              <img src={movie.poster_url || 'https://via.placeholder.com/150x225'} alt={movie.title} className="w-full h-72 object-cover" />
              <div className="p-3">
                <h3 className="font-semibold text-sm truncate">{movie.title}</h3>
                <span className="text-xs text-yellow-500">⭐ {movie.vote_average.toFixed(1)}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Modal chi tiết phim */}
      {selectedMovie && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center p-4 z-50">
          <div className="bg-neutral-900 rounded-xl max-w-2xl w-full p-6 relative flex flex-col md:flex-row gap-6">
            <button className="absolute top-4 right-4 text-gray-400 hover:text-white" onClick={() => setSelectedMovie(null)}>✕</button>
            <img src={selectedMovie.poster_url} alt={selectedMovie.title} className="w-48 h-72 object-cover rounded-lg" />
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-2">{selectedMovie.title}</h2>
              <p className="text-sm text-gray-400 mb-4">{selectedMovie.overview}</p>
              
              <div className="mb-4">
                <span className="block text-sm text-gray-400 mb-2">Đánh giá của bạn:</span>
                <div className="flex gap-2">
                  {[1, 2, 3, 4, 5].map(star => (
                    <button 
                      key={star} 
                      className={`text-2xl ${rating >= star ? 'text-yellow-500' : 'text-gray-600'}`}
                      onClick={() => setRating(star)}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
              <button 
                className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-6 rounded-lg transition-colors"
                onClick={() => handleRateSubmit(selectedMovie.movie_id)}
              >
                Gửi Đánh Giá
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
```

---

## 5. Hướng dẫn Khởi chạy Dự án

### Khởi động Backend:
1.  Di chuyển vào thư mục backend và cài đặt thư viện:
    ```bash
    cd webapp/backend
    pip install -r requirements.txt
    ```
2.  Khởi động server phát triển bằng Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```
    *Server backend sẽ chạy tại: `http://localhost:8000` (Bạn có thể xem tài liệu API tự động tại `http://localhost:8000/docs`).*

### Khởi động Frontend:
1.  Di chuyển vào thư mục frontend:
    ```bash
    cd webapp/frontend
    npm run dev
    ```
    *Giao diện Web App sẽ khả dụng tại: `http://localhost:5173`.*
