# Movie Recommendation System Data Simulator

Tài liệu này hướng dẫn chi tiết phương pháp thu thập dữ liệu phim Hollywood (giai đoạn 2000 - 2025) và cách xây dựng thuật toán giả lập dữ liệu tương tác người dùng (ratings, click events) mô phỏng theo hành vi thực tế trên các ứng dụng như Netflix và Prime Video.

---

## 1. Phương pháp Thu thập Dữ liệu Phim thực tế (Hollywood 2000 - 2025)

Để xây dựng hệ thống gợi ý chất lượng, trước tiên ta cần một danh sách phim thực tế đầy đủ thông tin (tiêu đề, poster, tóm tắt nội dung, thể loại, điểm đánh giá chung).

### Tại sao nên dùng API thay vì Web Scraping?
Việc tự viết scraper (sử dụng BeautifulSoup/Selenium) để cào dữ liệu từ IMDb hay Wikipedia có nhược điểm lớn:
*   Dễ bị khóa IP (Rate limit).
*   Cấu trúc HTML thay đổi thường xuyên khiến code bị lỗi.
*   Dữ liệu không đồng bộ, tốn nhiều thời gian làm sạch.

### Giải pháp khuyên dùng: Sử dụng API của TMDB (The Movie Database)
TMDB cung cấp API hoàn toàn miễn phí cho nhà phát triển cá nhân với dữ liệu được cập nhật liên tục và cấu trúc JSON chuẩn hóa.

#### Hướng dẫn lấy dữ liệu qua TMDB API:
1.  Đăng ký tài khoản tại [TMDb](https://www.themoviedb.org/) và tạo một API Key (trong mục Settings -> API).
2.  Sử dụng endpoint `discover/movie` với các tham số lọc phim chiếu rạp tiếng Anh sản xuất từ năm 2000 đến 2025.

#### Mã nguồn Python thu thập dữ liệu phim:

```python
import requests
import pandas as pd
import time

API_KEY = "YOUR_TMDB_API_KEY"  # Thay bằng API Key của bạn
BASE_URL = "https://api.themoviedb.org/3"

def fetch_hollywood_movies(start_year=2000, end_year=2025, max_pages=10):
    movies_list = []
    
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "language": "en-US",
            "sort_by": "popularity.desc", # Ưu tiên phim phổ biến nhất trước
            "primary_release_date.gte": f"{start_year}-01-01",
            "primary_release_date.lte": f"{end_year}-12-31",
            "with_original_language": "en",
            "page": page
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Lỗi tải trang {page}: {response.text}")
            break
            
        data = response.json()
        results = data.get("results", [])
        
        for item in results:
            poster_path = item.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            
            movies_list.append({
                "movie_id": item.get("id"),
                "title": item.get("title"),
                "release_date": item.get("release_date"),
                "overview": item.get("overview"),
                "vote_average": item.get("vote_average"), # Rating chung từ cộng đồng
                "vote_count": item.get("vote_count"),
                "poster_url": poster_url,
                "genres_ids": item.get("genre_ids")
            })
            
        time.sleep(0.2) # Tránh bị rate limit
        
    return pd.DataFrame(movies_list)
```

---

## 2. Phương pháp & Cơ sở Toán học của việc Giả lập Hành vi

Hành vi của người dùng thực tế trên các nền tảng xem phim trực tuyến tuân theo các quy luật phân phối thống kê rõ rệt. Để sinh dữ liệu giả lập (simulated data) chất lượng cao, mô hình giả lập cần mô phỏng các quy luật này:

### A. Quy luật Đuôi dài (Long-tail Distribution) cho Phim
Trong thực tế, một lượng rất nhỏ phim bom tấn (như *Avengers, Avatar, Forrest Gump*) chiếm đại đa số lượt tương tác (views, clicks, ratings), trong khi hàng ngàn bộ phim khác ít khi được chú ý tới. 
Chúng ta mô hình hóa mức độ phổ biến này bằng **Phân phối lũy thừa (Exponential Distribution)**:

$$f(x; \lambda) = \lambda e^{-\lambda x} \quad (x \ge 0)$$

*   Điểm phổ biến (Popularity Score) của phim được sinh ngẫu nhiên từ phân phối lũy thừa. Phim có điểm phổ biến cao sẽ có xác suất được người dùng click xem cao hơn nhiều lần.

### B. Tần suất hoạt động của Người dùng (User Activity)
Một số ít người dùng hoạt động cực kỳ tích cực (Heavy Users - cày phim liên tục), còn lại phần lớn là người dùng bình thường (Light Users - thỉnh thoảng mới xem). 
Tần suất hoạt động này được mô hình hóa bằng **Phân phối Gamma (Gamma Distribution)**:

$$f(x; k, \theta) = \frac{x^{k-1} e^{-\frac{x}{\theta}}}{\theta^k \Gamma(k)} \quad (x > 0)$$

*   Phân phối Gamma tạo ra một đồ thị lệch phải có đỉnh ở mức hoạt động thấp và đuôi dài kéo về phía hoạt động cao, phản ánh đúng thực tế hành vi người dùng trực tuyến.

### C. Logic Phễu Hành Vi (Behavior Funnel)
Mô phỏng chuỗi sự kiện tương tác của người dùng trên giao diện ứng dụng (như Netflix):

```
       [ Click Poster ] (Xác xuất cao nhất)
              │
              ▼ (60% Cơ hội)
      [ View Detail (Đọc tóm tắt phim) ]
              │
              ▼ (40% Cơ hội)
      [ Watch Start (Bắt đầu xem phim) ]
         /         \
   (30% Cơ hội)    (70% Cơ hội)
       /             \
[ Tắt giữa chừng ]   [ Watch Complete (Xem hết phim) ]
                      │
                      ▼ (30% Cơ hội)
             [ Chấm điểm Rating ]
```

### D. Điểm đánh giá Cá nhân hóa (Personalized Ratings)
Người dùng sẽ đánh giá điểm cao cho phim nếu:
*   Phim thuộc thể loại yêu thích (Favorite Genres) của họ.
*   Họ đã xem hết phim (`watch_complete`).
*   Ngược lại, nếu họ tắt phim giữa chừng, điểm đánh giá sẽ rất thấp hoặc họ sẽ không chấm điểm.

Công thức mô phỏng điểm số:
$$\text{Rating} = 3.0 + \text{Genre Bonus} + \text{Completion Bonus} - \text{Dropout Penalty} + \epsilon$$
*(Trong đó $\epsilon$ là nhiễu ngẫu nhiên phân phối chuẩn $\mathcal{N}(0, 0.5)$).* Điểm số cuối cùng được giới hạn trong khoảng $[0.5, 5.0]$ và làm tròn về mức $0.5$ sao gần nhất.

---

## 3. Mã nguồn Python Giả lập Hoàn chỉnh

Đoạn code Python tự chứa (self-contained) dưới đây giả lập toàn bộ quá trình trên và xuất ra 4 file dữ liệu CSV trực tiếp vào máy của bạn để sử dụng ngay lập tức:

```python
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta

# Thiết lập seed cố định để kết quả đồng bộ
np.random.seed(42)
random.seed(42)

def generate_simulated_data(num_users=200, num_movies=500, output_dir="data_simulated"):
    os.makedirs(output_dir, exist_ok=True)
    
    genres_list = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Horror", "Thriller", "Adventure", "Fantasy"]
    
    # 1. Tạo danh sách Phim giả lập
    movies_data = []
    for m_id in range(1, num_movies + 1):
        popularity = np.random.exponential(scale=10.0) # Power-law
        genres = random.sample(genres_list, k=random.randint(1, 3))
        movies_data.append({
            "movie_id": m_id,
            "title": f"Movie_{m_id}",
            "genres": "|".join(genres),
            "popularity_score": popularity
        })
    df_movies = pd.DataFrame(movies_data)
    df_movies.to_csv(os.path.join(output_dir, "sim_movies.csv"), index=False)
    
    # 2. Tạo danh sách Người dùng giả lập
    users_data = []
    for u_id in range(1, num_users + 1):
        fav_genres = random.sample(genres_list, k=random.randint(1, 2))
        activity = int(np.random.gamma(shape=2, scale=30)) + 10 # Gamma distribution
        users_data.append({
            "user_id": u_id,
            "favorite_genres": fav_genres,
            "activity_level": activity
        })
    df_users = pd.DataFrame(users_data)
    df_users.to_csv(os.path.join(output_dir, "sim_users.csv"), index=False)
    
    # 3. Giả lập hành vi tương tác (Click events & Ratings)
    click_events = []
    ratings_data = []
    start_time = datetime(2026, 1, 1)
    
    for _, user in df_users.iterrows():
        u_id = user["user_id"]
        fav_genres = user["favorite_genres"]
        activity = user["activity_level"]
        
        # Tính toán xác suất tương tác dựa trên độ hot của phim và sở thích thể loại của user
        movie_scores = []
        for _, movie in df_movies.iterrows():
            score = movie["popularity_score"]
            movie_genres = movie["genres"].split("|")
            has_fav_genre = any(g in fav_genres for g in movie_genres)
            if has_fav_genre:
                score *= 3.0 # Tăng gấp 3 lần cơ hội tương tác nếu thuộc thể loại ưa thích
            movie_scores.append(score)
            
        movie_probs = np.array(movie_scores) / sum(movie_scores)
        
        # Lựa chọn ngẫu nhiên các phim user sẽ tương tác
        chosen_movies = np.random.choice(
            df_movies["movie_id"], 
            size=min(activity, num_movies), 
            replace=False, 
            p=movie_probs
        )
        
        user_time = start_time + timedelta(hours=random.randint(0, 720)) # Phân tán thời gian trong 1 tháng
        
        for m_id in chosen_movies:
            movie_info = df_movies[df_movies["movie_id"] == m_id].iloc[0]
            movie_genres = movie_info["genres"].split("|")
            
            # --- Sinh Click Events ---
            user_time += timedelta(minutes=random.randint(1, 30))
            click_events.append({"user_id": u_id, "movie_id": m_id, "timestamp": user_time, "event_type": "click"})
            
            # 60% đọc chi tiết nội dung phim
            if random.random() < 0.60:
                user_time += timedelta(seconds=random.randint(10, 60))
                click_events.append({"user_id": u_id, "movie_id": m_id, "timestamp": user_time, "event_type": "detail_view"})
                
                # 40% bấm nút Xem phim
                if random.random() < 0.40:
                    user_time += timedelta(seconds=5)
                    click_events.append({"user_id": u_id, "movie_id": m_id, "timestamp": user_time, "event_type": "watch_start"})
                    
                    # 70% xem hết bộ phim
                    is_completed = random.random() < 0.70
                    watch_duration = random.randint(3600, 7200) if is_completed else random.randint(60, 1800)
                    user_time += timedelta(seconds=watch_duration)
                    
                    if is_completed:
                        click_events.append({"user_id": u_id, "movie_id": m_id, "timestamp": user_time, "event_type": "watch_complete"})
                    
                    # --- Sinh Ratings (Chỉ đánh giá khi đã bắt đầu xem) ---
                    if random.random() < 0.30: # 30% tỷ lệ rating
                        rating_score = 3.0
                        if any(g in fav_genres for g in movie_genres):
                            rating_score += np.random.uniform(0.5, 1.5) # Cộng điểm thể loại yêu thích
                        if is_completed:
                            rating_score += np.random.uniform(0.2, 0.8) # Cộng điểm vì xem hết phim
                        else:
                            rating_score -= np.random.uniform(0.5, 1.5) # Trừ điểm do tắt giữa chừng
                            
                        # Thêm nhiễu và giới hạn điểm số [0.5, 5.0]
                        rating_score += np.random.normal(0, 0.3)
                        rating = min(max(rating_score, 0.5), 5.0)
                        rating = round(rating * 2) / 2 # Làm tròn về bước 0.5 sao
                        
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
    
    print(f"Sinh dữ liệu giả lập thành công tại thư mục '{output_dir}':")
    print(f"- {len(df_movies)} phim -> sim_movies.csv")
    print(f"- {len(df_users)} người dùng -> sim_users.csv")
    print(f"- {len(df_click_events)} sự kiện click -> sim_click_events.csv")
    print(f"- {len(df_ratings)} lượt chấm điểm -> sim_ratings.csv")

if __name__ == "__main__":
    generate_simulated_data()
```
Tập tin [eda_analysis.ipynb](../eda_analysis.ipynb) đã chỉ ra các quy luật tương tự trên dữ liệu thật của MovieLens, và code giả lập phía trên đã được cấu hình các hệ số Gamma và lũy thừa tương ứng để đảm bảo phân bố dữ liệu giả lập có độ tương đồng tối đa với dữ liệu thật.
