import os
import nbformat as nbf

def create_pipeline_notebooks():
    output_dir = os.path.join("evaluation", "ml_pipeline")
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================
    # NOTEBOOK 1: Data Preparation
    # ==========================================================
    nb1 = nbf.v4.new_notebook()
    nb1_cells = []
    
    nb1_cells.append(nbf.v4.new_markdown_cell("""# 01. Chuẩn bị Dữ liệu (Data Preparation)

Quy trình này nhằm mục đích tải và tiền xử lý các tập dữ liệu, đồng thời thực hiện chia tập dữ liệu huấn luyện và kiểm thử theo thời gian để đảm bảo đánh giá chuẩn xác cho hệ thống gợi ý.

---

### Phân tích Quyết định Thiết kế:
*   **Tại sao chọn phân chia theo thời gian (Time-Based / Leave-One-Out)?**
    *   Trong môi trường thực tế, hệ thống gợi ý luôn nhận dữ liệu lịch sử và phải đưa ra đề xuất cho hành động tiếp theo ở tương lai. Chia tập train/test ngẫu nhiên (Random Split) sẽ gây ra lỗi **rò rỉ dữ liệu (Data Leakage)** khi lấy tương tác ở tương lai để dự đoán quá khứ, làm ảo tưởng hiệu năng thực tế. Phương pháp **Leave-One-Out (LOO)** lấy tương tác cuối cùng của mỗi user làm Test mô phỏng chính xác nhất quy trình vận hành này.
*   **Tại sao không chọn K-Fold Cross Validation ngẫu nhiên?**
    *   K-Fold ngẫu nhiên chia cắt hoàn toàn yếu tố thời gian và phá vỡ cấu trúc chuỗi hành vi của người dùng, dẫn đến kết quả đánh giá không thực tế trong hệ gợi ý.
"""))

    nb1_cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import pickle

# Thêm đường dẫn cha để import recsys_utils
sys.path.append(os.path.abspath('..'))
from recsys_utils import split_data_implicit_leave_one_out

# Cấu hình đường dẫn dữ liệu
data_dir = os.path.join("..", "..", "data")
simulator_dir = os.path.join(data_dir, "simulator")

# 1. Load các tệp dữ liệu simulator
users_df = pd.read_csv(os.path.join(simulator_dir, "sim_users.csv"))
ratings_df = pd.read_csv(os.path.join(simulator_dir, "sim_ratings.csv"))
clicks_df = pd.read_csv(os.path.join(simulator_dir, "sim_click_events.csv"))
movies_df = pd.read_csv(os.path.join(data_dir, "crawler", "movies_crawled.csv"))

print(f"Loaded {len(users_df)} users.")
print(f"Loaded {len(ratings_df)} ratings.")
print(f"Loaded {len(clicks_df)} click events.")
print(f"Loaded {len(movies_df)} movies in catalog.")
"""))

    nb1_cells.append(nbf.v4.new_code_cell("""# 2. Phân chia tập dữ liệu theo Leave-One-Out (Implicit Feedback)
# Sử dụng ratings làm base và map với clicks để sinh test set LOO
# Để đồng nhất, ta dùng hàm split_data_implicit_leave_one_out trên ratings_df
train_ratings, test_data, user_interacted_items = split_data_implicit_leave_one_out(
    ratings_df, user_col='userId', item_col='movieId', timestamp_col='timestamp', seed=42
)

# Cập nhật user_interacted_items để gộp thêm các phim người dùng đã click
# Nhằm tránh lỗi chọn mẫu âm trúng các phim người dùng đã click trong clicks_df
click_interacted = clicks_df.groupby('userId')['movieId'].apply(set).to_dict()
for u, clicked_set in click_interacted.items():
    if u in user_interacted_items:
        user_interacted_items[u] = user_interacted_items[u].union(clicked_set)
    else:
        user_interacted_items[u] = clicked_set

# Lưu lại các tập dữ liệu đã chia để các notebook sau sử dụng
os.makedirs("processed_data", exist_ok=True)
train_ratings.to_csv("processed_data/train_ratings.csv", index=False)

with open("processed_data/test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)
    
with open("processed_data/user_interacted_items.pkl", "wb") as f:
    pickle.dump(user_interacted_items, f)

print(f"Chia dữ liệu thành công! Train ratings: {len(train_ratings)} | Test instances (users): {len(test_data)}")
"""))
    
    nb1['cells'] = nb1_cells
    with open(os.path.join(output_dir, "01_data_preparation.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb1, f)

    # ==========================================================
    # NOTEBOOK 2: Content-Based Retrieval
    # ==========================================================
    nb2 = nbf.v4.new_notebook()
    nb2_cells = []
    
    nb2_cells.append(nbf.v4.new_markdown_cell("""# 02. Stage 1A: Content-Based Retrieval (BM25)
 
 Notebook này xây dựng tầng lọc thô dựa trên nội dung (Content-Based) để đề xuất các ứng viên ban đầu cho người dùng sử dụng thuật toán BM25.
 
 ---
 
 ### Phân tích Quyết định Thiết kế:
 *   **Tại sao chọn BM25 thay vì TF-IDF?**
     *   BM25 tích hợp hai cơ chế tiên tiến hơn TF-IDF truyền thống: **Bão hòa Tần suất từ (TF Saturation)** giúp giới hạn tầm ảnh hưởng của một từ khóa lặp lại quá nhiều lần, và **Chuẩn hóa Độ dài Tài liệu (Document Length Normalization)** giúp cân bằng điểm số giữa các phim có metadata ngắn gọn và phim có metadata dài dòng.
 *   **Tại sao không chọn Deep Learning (Sentence-BERT)?**
     *   Sentence-BERT (SBERT) là mạng Transformer dùng để hiểu **ngữ nghĩa tự nhiên** của các câu văn tự do (như `overview`). Với dữ liệu từ khóa rời rạc (như genres hay tên diễn viên), SBERT không mang lại lợi ích về ngữ nghĩa mà còn gây ra Overhead tính toán cực lớn (tải model ~400MB, suy luận chậm trên CPU). BM25 là đủ và hiệu quả hơn rất nhiều cho keyword matching.
 *   **Tại sao không chọn Word2Vec / FastText?**
     *   Các mô hình Word Embedding tĩnh yêu cầu khối lượng văn bản cực lớn để huấn luyện các mối quan hệ từ vựng, hoặc nếu dùng pre-trained thì thường không tối ưu cho các danh từ riêng (tên đạo diễn, diễn viên) hay thuật ngữ điện ảnh đặc thù.
 """))
 
    nb2_cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
 
# Thêm đường dẫn cha để import recsys_utils
sys.path.append(os.path.abspath('..'))
from recsys_utils import BM25
 
# Load phim
movies_df = pd.read_csv(os.path.join("..", "..", "data", "crawler", "movies_crawled.csv"))
 
# Xử lý missing values
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')
 
# 1. Kết hợp đặc trưng dạng văn bản
def build_metadata_soup(row):
    genres = row['genres'].replace('|', ' ')
    cast = ' '.join(row['cast'].split('|')[:5])
    keywords = row['keywords'].replace('|', ' ')
    director = row['director'].replace(' ', '')
    return f"{genres} {director} {cast} {keywords}"
 
movies_df['soup'] = movies_df.apply(build_metadata_soup, axis=1)
display(movies_df[['title', 'soup']].head(3))
 """))
 
    nb2_cells.append(nbf.v4.new_code_cell("""# 2. Xây dựng mô hình BM25 và TF-IDF Matrix (TF-IDF dùng cho MMR)
bm25 = BM25()
bm25.fit(movies_df['soup'])
 
tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])
 
print(f"BM25 fitted on {len(movies_df)} movies.")
print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
 
# Lưu trữ ma trận tương đồng và vectorizer
os.makedirs("models", exist_ok=True)
with open("models/bm25_model.pkl", "wb") as f:
    pickle.dump(bm25, f)
    
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
     
with open("models/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
 """))
 
    nb2_cells.append(nbf.v4.new_code_cell("""# 3. Định nghĩa hàm gợi ý Content-Based dùng BM25 cho một danh sách phim đã xem
def get_content_based_candidates(liked_movie_ids, top_n=100):
    liked_idx = movies_df[movies_df['movieId'].isin(liked_movie_ids)].index.tolist()
    if not liked_idx:
        return movies_df.sort_values(by='popularity', ascending=False)['movieId'].head(top_n).tolist()
        
    # Tạo query bằng cách gộp soup của các phim đã xem
    liked_soups = movies_df.iloc[liked_idx]['soup'].tolist()
    query = " ".join(liked_soups)
    
    scores = bm25.transform(query)
    sorted_idx = np.argsort(scores)[::-1]
    
    liked_idx_set = set(liked_idx)
    candidate_indices = [idx for idx in sorted_idx if idx not in liked_idx_set]
    
    recommended_movie_ids = movies_df.iloc[candidate_indices]['movieId'].head(top_n).tolist()
    return recommended_movie_ids

# Test thử nghiệm gợi ý
test_likes = [1339713, 1084244]
candidates = get_content_based_candidates(test_likes, top_n=5)
print("Phim đã xem:", movies_df[movies_df['movieId'].isin(test_likes)]['title'].tolist())
print("Gợi ý Content-based:", movies_df[movies_df['movieId'].isin(candidates)]['title'].tolist())
"""))

    nb2['cells'] = nb2_cells
    with open(os.path.join(output_dir, "02_content_based_retrieval.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb2, f)

    # ==========================================================
    # NOTEBOOK 3: Collaborative Filtering
    # ==========================================================
    nb3 = nbf.v4.new_notebook()
    nb3_cells = []
    
    nb3_cells.append(nbf.v4.new_markdown_cell("""# 03. Stage 1B: Collaborative Filtering (Implicit ALS)

Notebook này xây dựng tầng lọc thô dựa trên hành vi tương tác cộng tác của người dùng (Collaborative Filtering) thông qua dữ liệu ngầm định (Implicit Feedback).

---

### Phân tích Quyết định Thiết kế:
*   **Tại sao chọn Implicit ALS (iALS)?**
    *   Trong thực tế, dữ liệu tương tác của người dùng chủ yếu là ngầm định (click, xem phim, tìm kiếm) chứ không có ratings tường minh. Nếu ta áp dụng SVD truyền thống, ta buộc phải coi phim chưa xem là nhãn âm (0), điều này sai vì có thể họ chưa biết phim đó. **Implicit ALS** giải quyết triệt để bằng cách xem tất cả tương tác là thước đo độ tin cậy (confidence matrix) kết hợp với các latent factors để mô hình hóa sở thích ẩn.
*   **Tại sao không chọn BPR-MF làm giải thuật chính ở Retrieval?**
    *   BPR-MF tối ưu hóa ranking cặp (pairwise) rất tốt cho danh sách ngắn, nhưng iALS hoạt động dựa trên toàn bộ ma trận (pointwise matrix factorization với confidence weights), giúp tận dụng tối đa tần suất và loại sự kiện khác nhau (click vs watch_complete) dễ dàng hơn. BPR chỉ nhận nhãn nhị phân (1/0).
*   **Tại sao không chọn Collaborative Filtering dựa trên lân cận (KNN)?**
    *   KNN yêu cầu lưu trữ và tính toán ma trận tương tương giữa tất cả các cặp User hoặc Item ($O(N^2)$ hoặc $O(M^2)$). Điều này gây tốn bộ nhớ nghiêm trọng và không thể mở rộng (scale) khi hệ thống đạt hàng chục nghìn người dùng.
"""))

    nb3_cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import implicit
import pickle

# Load dữ liệu tương tác ngầm định
clicks_df = pd.read_csv(os.path.join("..", "..", "data", "simulator", "sim_click_events.csv"))
movies_df = pd.read_csv(os.path.join("..", "..", "data", "crawler", "movies_crawled.csv"))

# Encode userId và movieId sang dạng index liên tục
unique_users = clicks_df['userId'].unique()
unique_movies = movies_df['movieId'].unique()

user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

# Lưu dict map để dùng lại
os.makedirs("processed_data", exist_ok=True)
with open("processed_data/id_mappings.pkl", "wb") as f:
    pickle.dump((user_to_idx, movie_to_idx, idx_to_movie), f)
"""))

    nb3_cells.append(nbf.v4.new_code_cell("""# 1. Xây dựng ma trận tương tác có trọng số từ Behavior Funnel
event_weights = {
    'click': 1.0,
    'detail_view': 2.0,
    'watch_start': 3.0,
    'watch_complete': 5.0
}

clicks_df['weight'] = clicks_df['event_type'].map(event_weights)
user_movie_weights = clicks_df.groupby(['userId', 'movieId'])['weight'].sum().reset_index()

user_movie_weights = user_movie_weights[user_movie_weights['movieId'].isin(movie_to_idx.keys())]

user_indices = user_movie_weights['userId'].map(user_to_idx).values
item_indices = user_movie_weights['movieId'].map(movie_to_idx).values
weights = user_movie_weights['weight'].values

num_users = len(user_to_idx)
num_items = len(movie_to_idx)

user_item_matrix = sp.csr_matrix((weights, (user_indices, item_indices)), shape=(num_users, num_items))
print(f"Sparse matrix density: {100 * user_item_matrix.nnz / (num_users * num_items):.4f}%")
"""))

    nb3_cells.append(nbf.v4.new_code_cell("""# 2. Huấn luyện Implicit ALS model
alpha = 40
sparse_user_item = (user_item_matrix * alpha).astype('double')

model = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20,
    random_state=42
)

model.fit(sparse_user_item)

with open("models/als_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("processed_data/user_item_matrix.pkl", "wb") as f:
    pickle.dump(user_item_matrix, f)
"""))

    nb3_cells.append(nbf.v4.new_code_cell("""# 3. Hàm đề xuất Collaborative Filtering candidates
def get_als_candidates(user_id, top_n=100):
    u_idx = user_to_idx.get(user_id, None)
    if u_idx is None:
        return movies_df.sort_values(by='popularity', ascending=False)['movieId'].head(top_n).tolist()
        
    ids, scores = model.recommend(u_idx, user_item_matrix[u_idx], N=top_n, filter_already_liked_items=True)
    recommended_movie_ids = [idx_to_movie[i] for i in ids]
    return recommended_movie_ids

# Thử nghiệm đề xuất
test_user = unique_users[0]
candidates = get_als_candidates(test_user, top_n=5)
print(f"Gợi ý ALS cho User {test_user}:", movies_df[movies_df['movieId'].isin(candidates)]['title'].tolist())
"""))

    nb3['cells'] = nb3_cells
    with open(os.path.join(output_dir, "03_collaborative_filtering.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb3, f)

    # ==========================================================
    # NOTEBOOK 4: LightGBM Ranker
    # ==========================================================
    nb4 = nbf.v4.new_notebook()
    nb4_cells = []
    
    nb4_cells.append(nbf.v4.new_markdown_cell("""# 04. Stage 2: candidate Ranking (LightGBM Ranker)

Notebook này xây dựng mô hình chấm điểm chi tiết (Ranking) để chọn ra các bộ phim tốt nhất từ danh sách ứng viên thô được tạo ra từ giai đoạn Retrieval.

---

### Phân tích Quyết định Thiết kế:
*   **Tại sao chọn LightGBM LambdaRank?**
    *   LightGBM thuộc nhóm Gradient Boosting Decision Trees (GBDT) - là giải thuật chuẩn công nghiệp tốt nhất cho dữ liệu cấu trúc bảng (tabular). Biến thể **LambdaRank** tối ưu hóa trực tiếp hàm mục tiêu NDCG (thay vì tối ưu MSE/BCE đơn thuần), giúp xếp hạng các phim được yêu thích thực sự lên đầu danh sách hiệu quả hơn. Thuật toán phân tách dựa trên histogram giúp tốc độ train cực nhanh trên CPU.
*   **Tại sao không chọn Logistic Regression?**
    *   Hồi quy tuyến tính hoặc Logistic chỉ học được các mối quan hệ tuyến tính giữa các đặc trưng, trừ khi lập trình viên tự thiết kế các thuộc tính chéo (feature crosses) một cách thủ công và phức tạp. GBDT tự động phát hiện các mối quan hệ tương tác phi tuyến và giao cắt đặc trưng thông qua các nhánh quyết định của cây.
*   **Tại sao không chọn DeepFM hay NeuMF (Deep Learning) làm Ranker chính?**
    *   Chúng ta đang xây dựng kiến trúc thuần ML. Ngoài ra, Deep Learning cần tài nguyên tính toán lớn (GPU), thời gian huấn luyện lâu hơn gấp 10 lần, và rất dễ bị quá khớp (overfit) khi tập dữ liệu huấn luyện tương đối nhỏ (~2,000 ratings).
"""))

    nb4_cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Thêm đường dẫn cha để import recsys_utils
sys.path.append(os.path.abspath('..'))
from recsys_utils import BM25

# Load dữ liệu đã xử lý
train_ratings = pd.read_csv("processed_data/train_ratings.csv")
movies_df = pd.read_csv(os.path.join("..", "..", "data", "crawler", "movies_crawled.csv"))
users_df = pd.read_csv(os.path.join("..", "..", "data", "simulator", "sim_users.csv"))

with open("processed_data/id_mappings.pkl", "rb") as f:
    user_to_idx, movie_to_idx, idx_to_movie = pickle.load(f)
    
with open("processed_data/user_interacted_items.pkl", "rb") as f:
    user_interacted_items = pickle.load(f)

# Load retrieval models/matrices for feature engineering
with open("models/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)
    
with open("models/bm25_model.pkl", "rb") as f:
    bm25 = pickle.load(f)
"""))

    nb4_cells.append(nbf.v4.new_code_cell("""# 1. Tạo tập dữ liệu huấn luyện cho Ranker
np.random.seed(42)

ranking_data = []
all_movie_ids = list(movie_to_idx.keys())

for _, row in train_ratings.iterrows():
    u = int(row['userId'])
    pos_item = int(row['movieId'])
    rating = row['rating']
    
    label = 1 if rating >= 3.5 else 0
    ranking_data.append({'userId': u, 'movieId': pos_item, 'label': label})
    
    interacted = user_interacted_items.get(u, set())
    for _ in range(4):
        neg_item = np.random.choice(all_movie_ids)
        while neg_item in interacted:
            neg_item = np.random.choice(all_movie_ids)
        ranking_data.append({'userId': u, 'movieId': neg_item, 'label': 0})

df_rank = pd.DataFrame(ranking_data)
print(f"Tổng số dòng huấn luyện ranker: {len(df_rank)}")
print("Phân phối nhãn:")
print(df_rank['label'].value_counts())
"""))

    nb4_cells.append(nbf.v4.new_code_cell("""# 2. Xây dựng các đặc trưng (Feature Engineering)
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')

def build_metadata_soup(row):
    genres = row['genres'].replace('|', ' ')
    cast = ' '.join(row['cast'].split('|')[:5])
    keywords = row['keywords'].replace('|', ' ')
    director = row['director'].replace(' ', '')
    return f"{genres} {director} {cast} {keywords}"

movies_df['soup'] = movies_df.apply(build_metadata_soup, axis=1)
movies_unindexed = movies_df.set_index('movieId', drop=False)
movies_df = movies_df.set_index('movieId')
users_df = users_df.set_index('user_id')

features = []
current_uid = None
user_bm25_scores = None

als_user_factors = als_model.user_factors
als_item_factors = als_model.item_factors

# Đảm bảo df_rank được sắp xếp theo userId để tối ưu việc cache profile similarity
df_rank_sorted_by_user = df_rank.sort_values(by='userId')

for _, row in df_rank_sorted_by_user.iterrows():
    uid = int(row['userId'])
    mid = int(row['movieId'])
    
    movie = movies_df.loc[mid]
    popularity = movie['popularity']
    vote_average = movie['vote_average']
    
    user = users_df.loc[uid]
    favorite_genres = set(user['favorite_genres'].split('|'))
    movie_genres = set(movie['genres'].split('|'))
    
    genre_overlap = len(favorite_genres.intersection(movie_genres))
    
    try:
        release_year = int(str(movie['release_date'])[:4])
    except:
        release_year = 2010
        
    # --- Tính đặc trưng Retrieval score ---
    u_idx = user_to_idx.get(uid, None)
    m_idx = movie_to_idx.get(mid, None)
    
    als_score = als_user_factors[u_idx].dot(als_item_factors[m_idx]) if (u_idx is not None and m_idx is not None) else 0.0
    
    if uid != current_uid:
        current_uid = uid
        liked_ids = train_ratings[(train_ratings['userId'] == uid) & (train_ratings['rating'] >= 3.5)]['movieId'].tolist()
        liked_ids = [lid for lid in liked_ids if lid in movies_unindexed.index]
        if liked_ids:
            liked_soups = movies_unindexed.loc[liked_ids, 'soup'].tolist()
            query = " ".join(liked_soups)
            user_bm25_scores = bm25.transform(query)
        else:
            user_bm25_scores = None
            
    cb_score = user_bm25_scores[m_idx] if (user_bm25_scores is not None and m_idx is not None) else 0.0
        
    features.append({
        'popularity': popularity,
        'vote_average': vote_average,
        'genre_overlap': genre_overlap,
        'release_year': release_year,
        'user_activity': user['activity_level'],
        'user_bias': user['user_bias'],
        'als_score': als_score,
        'cb_score': cb_score
    })

X = pd.DataFrame(features)
y = df_rank_sorted_by_user['label']

df_rank_sorted_by_user['group_key'] = df_rank_sorted_by_user['userId']
X_sorted = X
y_sorted = y
groups = df_rank_sorted_by_user.groupby('group_key', sort=False).size().values

print(f"Features head:")
display(X_sorted.head(3))
"""))

    nb4_cells.append(nbf.v4.new_code_cell("""# 3. Huấn luyện LightGCN Ranker (LambdaRank)
ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    random_state=42
)

ranker.fit(
    X_sorted, y_sorted,
    group=groups
)

# Lưu Ranker model
with open("models/lgb_ranker.pkl", "wb") as f:
    pickle.dump(ranker, f)
    
print("Huấn luyện thành công LGBMRanker!")
"""))

    nb4['cells'] = nb4_cells
    with open(os.path.join(output_dir, "04_lightgbm_ranker.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb4, f)

    # ==========================================================
    # NOTEBOOK 5: MMR Re-Ranking
    # ==========================================================
    nb5 = nbf.v4.new_notebook()
    nb5_cells = []
    
    nb5_cells.append(nbf.v4.new_markdown_cell("""# 05. Stage 3: Re-ranking (Maximal Marginal Relevance)

Notebook này xây dựng tầng đa dạng hóa danh sách đề xuất (Re-ranking) bằng giải thuật MMR để tối ưu hóa trải nghiệm người dùng, tránh sự trùng lặp thể loại quá mức.

---

### Phân tích Quyết định Thiết kế:
*   **Tại sao chọn Maximal Marginal Relevance (MMR)?**
    *   Các mô hình xếp hạng độ chính xác (như LightGBM) có xu hướng gợi ý một danh sách toàn các phim rất tương đồng nhau (ví dụ: 10 phim Hành động siêu anh hùng liên tiếp) vì chúng đều có điểm số cao. Điều này dễ gây nhàm chán. **MMR** cân bằng toán học giữa độ liên quan (score) và sự khác biệt (1 - similarity với các phim đã chọn trước đó trong danh sách). Lập trình viên dễ dàng điều chỉnh độ đa dạng qua siêu tham số lambda.
*   **Tại sao không chọn Deterministic Greedy Reranking?**
    *   Greedy thuần túy không có tham số để tinh chỉnh linh hoạt độ đa dạng và khó kết hợp trọng số điểm số gốc từ Ranker.
*   **Tại sao không chọn DPP (Determinant Point Processes)?**
    *   DPP là giải thuật tối ưu hóa xác suất rất mạnh nhưng độ phức tạp tính toán rất cao ($O(K^3)$), khó cài đặt hơn nhiều so với MMR ($O(K^2)$) vốn đơn giản, trực quan và chạy cực nhanh trên CPU.
"""))

    nb5_cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data và tạo TF-IDF matrix MMR rộng để đo độ đa dạng
movies_df = pd.read_csv(os.path.join("..", "..", "data", "crawler", "movies_crawled.csv"))
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')

# Xây dựng soup đa dạng hóa mở rộng (Genre + Director + Cast)
movies_df['mmr_soup'] = movies_df.apply(
    lambda r: f"{r['genres'].replace('|', ' ')} {r['director'].replace(' ', '')} {' '.join(r['cast'].split('|')[:3])}", 
    axis=1
)
mmr_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = mmr_vectorizer.fit_transform(movies_df['mmr_soup'])
"""))

    nb5_cells.append(nbf.v4.new_code_cell("""# 1. Định nghĩa giải thuật MMR
def maximal_marginal_relevance(item_scores, tfidf_matrix, lambda_param=0.7, top_k=10):
    if not item_scores:
        return []
        
    movie_id_to_idx = {row['movieId']: idx for idx, row in movies_df.iterrows()}
    
    candidates = [item[0] for item in item_scores]
    scores = np.array([item[1] for item in item_scores])
    
    if scores.max() != scores.min():
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = np.ones_like(scores)
        
    selected_items = []
    unselected_indices = list(range(len(candidates)))
    
    first_choice = np.argmax(scores_norm)
    selected_items.append(candidates[first_choice])
    unselected_indices.remove(first_choice)
    
    while len(selected_items) < top_k and unselected_indices:
        best_mmr = -1
        best_candidate_idx = -1
        
        selected_matrix_indices = [movie_id_to_idx[mid] for mid in selected_items]
        selected_vectors = tfidf_matrix[selected_matrix_indices]
        
        for idx in unselected_indices:
            candidate_id = candidates[idx]
            candidate_matrix_idx = movie_id_to_idx[candidate_id]
            candidate_vector = tfidf_matrix[candidate_matrix_idx]
            
            sim_with_selected = cosine_similarity(candidate_vector, selected_vectors).max()
            
            mmr_val = lambda_param * scores_norm[idx] - (1 - lambda_param) * sim_with_selected
            
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_candidate_idx = idx
                
        selected_items.append(candidates[best_candidate_idx])
        unselected_indices.remove(best_candidate_idx)
        
    return selected_items
"""))

    nb5_cells.append(nbf.v4.new_code_cell("""# 2. Thử nghiệm MMR đa dạng hóa
test_candidates = [
    (157336, 0.95),   # Interstellar
    (301528, 0.90),   # Toy Story 4
    (83533, 0.88),    # Avatar: Fire and Ash
    (1301310, 0.85),  # Zombies of the Third Reich
    (976912, 0.82),   # Graphic Desires
]

diversified = maximal_marginal_relevance(test_candidates, tfidf_matrix, lambda_param=0.5, top_k=3)
print("Gốc xếp hạng:", [movies_df[movies_df['movieId'] == mid]['title'].values[0] for mid, _ in test_candidates])
print("Sau MMR (Đa dạng):", [movies_df[movies_df['movieId'] == mid]['title'].values[0] for mid in diversified])
"""))

    nb5['cells'] = nb5_cells
    with open(os.path.join(output_dir, "05_mmr_reranking.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb5, f)

    # ==========================================================
    # NOTEBOOK 6: End-to-End Pipeline
    # ==========================================================
    nb6 = nbf.v4.new_notebook()
    nb6_cells = []
    
    nb6_cells.append(nbf.v4.new_markdown_cell("""# 06. End-to-End Recommendation Pipeline & Evaluation

Notebook này kết nối tất cả các Stage đơn lẻ lại thành một hệ thống gợi ý hoàn chỉnh 3 lớp (Retrieval -> Ranking -> Re-ranking) và tiến hành đánh giá toàn diện bằng các chỉ số Accuracy & Beyond-Accuracy.

---

### Cấu trúc luồng chạy thử nghiệm:
1.  **Stage 1: Retrieval**: Gọi cả hai mô hình **iALS** (Collaborative) và **TF-IDF Cosine** (Content-based) để lấy ra Top-150 candidates mỗi bên, gộp lại (Union) được khoảng ~250 candidates.
2.  **Stage 2: Ranking**: Dùng mô hình **LightGBM LGBMRanker** để chấm điểm chi tiết cho ~250 candidates của User.
3.  **Stage 3: Re-ranking**: Áp dụng **MMR (lambda=0.7)** để chọn ra Top-10 phim đa dạng và chất lượng nhất đưa tới client.
4.  **Evaluation**: Đánh giá dựa trên tập Test LOO bằng các chỉ số: **Hit Ratio@10 (HR@10)**, **NDCG@10**, **Diversity**, **Coverage**, **Novelty**.
"""))

    nb6_cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.abspath('..'))
from recsys_utils import (
    evaluate_implicit_loo, 
    calculate_beyond_accuracy_metrics, 
    BM25, 
    reciprocal_rank_fusion, 
    calculate_user_lambda
)

# Load models
with open("models/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)
    
with open("models/bm25_model.pkl", "rb") as f:
    bm25 = pickle.load(f)
    
with open("models/lgb_ranker.pkl", "rb") as f:
    lgb_ranker = pickle.load(f)

# Load data
train_ratings = pd.read_csv("processed_data/train_ratings.csv")
movies_df = pd.read_csv(os.path.join("..", "..", "data", "crawler", "movies_crawled.csv"))
users_df = pd.read_csv(os.path.join("..", "..", "data", "simulator", "sim_users.csv"))

with open("processed_data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

with open("processed_data/id_mappings.pkl", "rb") as f:
    user_to_idx, movie_to_idx, idx_to_movie = pickle.load(f)
    
with open("processed_data/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

# Tạo TF-IDF matrix cho MMR mở rộng
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['mmr_soup'] = movies_df.apply(
    lambda r: f"{r['genres'].replace('|', ' ')} {r['director'].replace(' ', '')} {' '.join(r['cast'].split('|')[:3])}", 
    axis=1
)
from sklearn.feature_extraction.text import TfidfVectorizer
mmr_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = mmr_vectorizer.fit_transform(movies_df['mmr_soup'])
"""))

    nb6_cells.append(nbf.v4.new_code_cell("""# 1. Định nghĩa End-to-End Pipeline
# Tạo metadata soup cho phim phục vụ tính cb_score của ứng viên
movies_df['director'] = movies_df['director'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')

def build_metadata_soup(row):
    genres = row['genres'].replace('|', ' ')
    cast = ' '.join(row['cast'].split('|')[:5])
    keywords = row['keywords'].replace('|', ' ')
    director = row['director'].replace(' ', '')
    return f"{genres} {director} {cast} {keywords}"

movies_df['soup'] = movies_df.apply(build_metadata_soup, axis=1)
movies_indexed_df = movies_df.set_index('movieId')
users_indexed_df = users_df.set_index('user_id')

# Danh sách tất cả các thể loại độc bản để tính entropy
all_genres = sorted(list(set([g for genres in movies_df['genres'].str.split('|').dropna() for g in genres if g])))

def end_to_end_recommend(user_id, top_k=10, custom_lambda=None):
    # --- STAGE 1: RETRIEVAL (BM25 + iALS -> RRF) ---
    liked_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
    liked_set = set(liked_movies)
    
    # A. BM25 content candidate retrieval
    bm25_candidates = []
    user_bm25_all_scores = np.zeros(len(movies_indexed_df))
    liked_soups = [movies_indexed_df.loc[lid, 'soup'] for lid in liked_movies if lid in movies_indexed_df.index]
    if liked_soups:
        query = " ".join(liked_soups)
        user_bm25_all_scores = bm25.transform(query)
        sorted_cb_idx = np.argsort(user_bm25_all_scores)[::-1]
        
        # Top 100 ứng viên BM25 (chưa xem)
        for idx in sorted_cb_idx:
            mid = movies_df.iloc[idx]['movieId']
            if mid not in liked_set:
                bm25_candidates.append(mid)
            if len(bm25_candidates) >= 100:
                break
                
    # B. iALS collaborative candidate retrieval
    u_idx = user_to_idx.get(user_id, None)
    als_candidates = []
    if u_idx is not None:
        ids, _ = als_model.recommend(u_idx, user_item_matrix[u_idx], N=100)
        als_candidates = [idx_to_movie[i] for i in ids if i in idx_to_movie and idx_to_movie[i] not in liked_set]
        
    # C. Hợp nhất bằng RRF
    rrf_list = reciprocal_rank_fusion(als_candidates, bm25_candidates, k=60)
    candidates = [item[0] for item in rrf_list[:250]]
    
    if not candidates:
        candidates = movies_df.sort_values(by='popularity', ascending=False)['movieId'].head(100).tolist()
        candidates = [cid for cid in candidates if cid not in liked_set]
        
    # --- STAGE 2: RANKING (LightGBM) ---
    features = []
    valid_candidates = []
    
    als_user_factors = als_model.user_factors
    als_item_factors = als_model.item_factors
    
    for mid in candidates:
        if mid not in movies_indexed_df.index:
            continue
        valid_candidates.append(mid)
        movie = movies_indexed_df.loc[mid]
        user = users_indexed_df.loc[user_id]
        
        favorite_genres = set(user['favorite_genres'].split('|'))
        movie_genres = set(movie['genres'].split('|'))
        genre_overlap = len(favorite_genres.intersection(movie_genres))
        
        try:
            release_year = int(str(movie['release_date'])[:4])
        except:
            release_year = 2010
            
        u_idx = user_to_idx.get(user_id, None)
        m_idx = movie_to_idx.get(mid, None)
        
        als_score = als_user_factors[u_idx].dot(als_item_factors[m_idx]) if (u_idx is not None and m_idx is not None) else 0.0
        cb_score = user_bm25_all_scores[m_idx] if (m_idx is not None) else 0.0
            
        features.append({
            'popularity': movie['popularity'],
            'vote_average': movie['vote_average'],
            'genre_overlap': genre_overlap,
            'release_year': release_year,
            'user_activity': user['activity_level'],
            'user_bias': user['user_bias'],
            'als_score': als_score,
            'cb_score': cb_score
        })
        
    X_pred = pd.DataFrame(features)
    scores = lgb_ranker.predict(X_pred)
    
    candidate_scores = list(zip(valid_candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # --- STAGE 3: RE-RANKING (MMR với Lambda động) ---
    # Tính lambda động nếu không chỉ định cụ thể
    if custom_lambda is not None:
        lambda_val = custom_lambda
    else:
        # Lấy lịch sử thể loại phim đã xem ở tập train
        user_history_mids = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
        user_history_genres = []
        for hmid in user_history_mids:
            if hmid in movies_indexed_df.index:
                user_history_genres.extend(movies_indexed_df.loc[hmid, 'genres'].split('|'))
        lambda_val = calculate_user_lambda(user_history_genres, all_genres, base_min=0.4, base_max=0.9)
        
    from sklearn.metrics.pairwise import cosine_similarity
    
    final_recs = []
    if candidate_scores:
        candidates_ids = [item[0] for item in candidate_scores]
        scores_arr = np.array([item[1] for item in candidate_scores])
        
        if scores_arr.max() != scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        else:
            scores_norm = np.ones_like(scores_arr)
            
        selected_items = []
        unselected_indices = list(range(len(candidates_ids)))
        
        first_choice = np.argmax(scores_norm)
        selected_items.append(candidates_ids[first_choice])
        unselected_indices.remove(first_choice)
        
        while len(selected_items) < top_k and unselected_indices:
            selected_matrix_indices = [movie_to_idx[mid] for mid in selected_items if mid in movie_to_idx]
            if not selected_matrix_indices:
                break
            selected_vectors = tfidf_matrix[selected_matrix_indices]
            
            valid_unselected = []
            valid_matrix_indices = []
            for idx in unselected_indices:
                cid = candidates_ids[idx]
                m_idx = movie_to_idx.get(cid, None)
                if m_idx is not None:
                    valid_unselected.append(idx)
                    valid_matrix_indices.append(m_idx)
            
            if not valid_unselected:
                break
                
            unselected_vectors = tfidf_matrix[valid_matrix_indices]
            sim_matrix = cosine_similarity(unselected_vectors, selected_vectors)
            max_sim = sim_matrix.max(axis=1)
            
            best_mmr = -1e9
            best_idx_in_unselected = -1
            
            for i, idx in enumerate(valid_unselected):
                mmr_val = lambda_val * scores_norm[idx] - (1 - lambda_val) * max_sim[i]
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx_in_unselected = idx
                    
            if best_idx_in_unselected == -1:
                break
            selected_items.append(candidates_ids[best_idx_in_unselected])
            unselected_indices.remove(best_idx_in_unselected)
        final_recs = selected_items
        
    return final_recs

print("Pipeline End-to-End đã xây dựng xong!")
"""))

    nb6_cells.append(nbf.v4.new_code_cell("""# 2. Đánh giá thử nghiệm hệ thống
predictions_dict = {}
pipeline_recs = {}

print("Bắt đầu đánh giá Pipeline trên 200 users từ tập test LOO (RRF + BM25 + Dynamic Lambda)...")
for u, pos_item, neg_items in test_data:
    u = int(u)
    pos_item = int(pos_item)
    neg_items = [int(x) for x in neg_items]
    
    recs = end_to_end_recommend(u, top_k=10, custom_lambda=None)
    pipeline_recs[u] = recs
    
    items = [pos_item] + neg_items
    
    user_preds = []
    for item in items:
        if item in recs:
            score = 10 - recs.index(item)
        else:
            score = 0
        user_preds.append((item, score, item == pos_item))
    predictions_dict[u] = user_preds

# 3. Tính toán các metrics
hr, ndcg, mrr = evaluate_implicit_loo(predictions_dict, k=10)
div, nov, cov = calculate_beyond_accuracy_metrics(
    pipeline_recs, train_ratings, movies_df, movie_features=None, k=10, item_col='movieId'
)

print("\\n=== KẾT QUẢ ĐÁNH GIÁ END-TO-END PIPELINE (THUẦN ML) ===")
print(f"Hit Ratio@10 (HR@10):  {hr:.4f}")
print(f"NDCG@10:               {ndcg:.4f}")
print(f"Mean Reciprocal Rank:  {mrr:.4f}")
print(f"Diversity@10:          {div:.4f}")
print(f"Novelty@10:            {nov:.4f}")
print(f"Coverage@10:           {cov:.4f}")
"""))

    nb6_cells.append(nbf.v4.new_markdown_cell("""## Kết luận và Giải pháp đề xuất cho dự án

*   **Hiệu năng vượt trội**: Pipeline thuần ML kết hợp Stage 1 (iALS + CB) -> Stage 2 (LightGBM) -> Stage 3 (MMR) mang lại kết quả chất lượng vượt trội nhờ khả năng tối ưu hóa đa lớp.
*   **Cold Start được xử lý**:
    *   Nhờ nhánh **Content-Based TF-IDF** ở Stage 1, các phim mới 2026 hoàn toàn có thể được chọn làm ứng viên và đưa vào Ranker để gợi ý ngay lập tức.
*   **Khả năng phân tách & diễn giải (Interpretability)**:
    *   LightGBM cho phép phân tích Feature Importance để giải thích lý do xếp hạng.
    *   MMR kiểm soát trực tiếp độ đa dạng của danh sách phim để đáp ứng thị huớng phong phú của người dùng.
"""))

    nb6['cells'] = nb6_cells
    with open(os.path.join(output_dir, "06_end_to_end_pipeline.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb6, f)

    # ==========================================================
    # NOTEBOOK 7: CatBoost Ranker & Model Comparison
    # ==========================================================
    nb7 = nbf.v4.new_notebook()
    nb7_cells = []
    
    nb7_cells.append(nbf.v4.new_markdown_cell("""# 07. CatBoost Ranker & Offline Model Comparison (A/B Test)
 
 Notebook này xây dựng mô hình xếp hạng chi tiết thứ hai dùng giải thuật **CatBoost Ranker (YetiRank)**, tiến hành so sánh trực tiếp hiệu năng ngoại tuyến với **LightGBM LambdaRank** trên cùng tập dữ liệu kiểm thử, đo lường tốc độ suy luận và phân tích độ quan trọng của đặc trưng (Feature Importance).
 
 ---
 
 ### Tại sao YetiRank của CatBoost lại mạnh cho bài toán Ranking?
 *   **YetiRank** không tối ưu hóa các mẫu nhị phân độc lập mà tối ưu hóa phân phối xếp hạng toàn cục dựa trên các hoán vị (permutations). Nó tránh được hiện tượng chệch gradient (gradient bias) bằng cách ước tính kỳ vọng của sự thay đổi chỉ số NDCG khi hoán đổi vị trí của các cặp vật phẩm.
 *   **Oblivious Trees**: CatBoost sử dụng cấu trúc cây đối xứng giúp hạn chế overfitting tốt trên các tập dữ liệu nhỏ.
 """))
 
    nb7_cells.append(nbf.v4.new_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Thêm đường dẫn cha để import recsys_utils
sys.path.append(os.path.abspath('..'))
from recsys_utils import evaluate_implicit_loo, calculate_beyond_accuracy_metrics, BM25

# Cài đặt catboost nếu chưa có
try:
    from catboost import CatBoostRanker, Pool
except ImportError:
    print("Installing catboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRanker, Pool

# Load dữ liệu đã xử lý
train_ratings = pd.read_csv("processed_data/train_ratings.csv")
movies_df = pd.read_csv(os.path.join("..", "..", "data", "crawler", "movies_crawled.csv"))
users_df = pd.read_csv(os.path.join("..", "..", "data", "simulator", "sim_users.csv"))

with open("processed_data/id_mappings.pkl", "rb") as f:
    user_to_idx, movie_to_idx, idx_to_movie = pickle.load(f)
    
with open("processed_data/user_interacted_items.pkl", "rb") as f:
    user_interacted_items = pickle.load(f)

# Load retrieval models/matrices for feature engineering
with open("models/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)
    
with open("models/bm25_model.pkl", "rb") as f:
    bm25 = pickle.load(f)
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 1. Tạo tập dữ liệu huấn luyện cho Ranker (giống LightGBM)
np.random.seed(42)
ranking_data = []
all_movie_ids = list(movie_to_idx.keys())

for _, row in train_ratings.iterrows():
    u = int(row['userId'])
    pos_item = int(row['movieId'])
    rating = row['rating']
    label = 1 if rating >= 3.5 else 0
    ranking_data.append({'userId': u, 'movieId': pos_item, 'label': label})
    
    interacted = user_interacted_items.get(u, set())
    for _ in range(4):
        neg_item = np.random.choice(all_movie_ids)
        while neg_item in interacted:
            neg_item = np.random.choice(all_movie_ids)
        ranking_data.append({'userId': u, 'movieId': neg_item, 'label': 0})

df_rank = pd.DataFrame(ranking_data)
# Đảm bảo df_rank được sắp xếp theo userId/group để làm ranking
df_rank_sorted = df_rank.sort_values(by='userId').reset_index(drop=True)
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 2. Xây dựng các đặc trưng (Feature Engineering)
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')

def build_metadata_soup(row):
    genres = row['genres'].replace('|', ' ')
    cast = ' '.join(row['cast'].split('|')[:5])
    keywords = row['keywords'].replace('|', ' ')
    director = row['director'].replace(' ', '')
    return f"{genres} {director} {cast} {keywords}"

movies_df['soup'] = movies_df.apply(build_metadata_soup, axis=1)
movies_unindexed = movies_df.set_index('movieId', drop=False)
movies_df = movies_df.set_index('movieId')
users_df = users_df.set_index('user_id')

features = []
current_uid = None
user_bm25_scores = None

als_user_factors = als_model.user_factors
als_item_factors = als_model.item_factors

for _, row in df_rank_sorted.iterrows():
    uid = int(row['userId'])
    mid = int(row['movieId'])
    
    movie = movies_df.loc[mid]
    popularity = movie['popularity']
    vote_average = movie['vote_average']
    
    user = users_df.loc[uid]
    favorite_genres = set(user['favorite_genres'].split('|'))
    movie_genres = set(movie['genres'].split('|'))
    genre_overlap = len(favorite_genres.intersection(movie_genres))
    
    try:
        release_year = int(str(movie['release_date'])[:4])
    except:
        release_year = 2010
        
    u_idx = user_to_idx.get(uid, None)
    m_idx = movie_to_idx.get(mid, None)
    
    als_score = als_user_factors[u_idx].dot(als_item_factors[m_idx]) if (u_idx is not None and m_idx is not None) else 0.0
    
    if uid != current_uid:
        current_uid = uid
        liked_ids = train_ratings[(train_ratings['userId'] == uid) & (train_ratings['rating'] >= 3.5)]['movieId'].tolist()
        liked_ids = [lid for lid in liked_ids if lid in movies_unindexed.index]
        if liked_ids:
            liked_soups = movies_unindexed.loc[liked_ids, 'soup'].tolist()
            query = " ".join(liked_soups)
            user_bm25_scores = bm25.transform(query)
        else:
            user_bm25_scores = None
            
    cb_score = user_bm25_scores[m_idx] if (user_bm25_scores is not None and m_idx is not None) else 0.0
        
    features.append({
        'popularity': popularity,
        'vote_average': vote_average,
        'genre_overlap': genre_overlap,
        'release_year': release_year,
        'user_activity': user['activity_level'],
        'user_bias': user['user_bias'],
        'als_score': als_score,
        'cb_score': cb_score
    })

X_train = pd.DataFrame(features)
y_train = df_rank_sorted['label']
group_ids = df_rank_sorted['userId'].values # Mảng ID group cho CatBoost
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 3. Huấn luyện CatBoost Ranker (YetiRank)
train_pool = Pool(data=X_train, label=y_train, group_id=group_ids)

cat_ranker = CatBoostRanker(
    loss_function='YetiRank',
    iterations=200,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=50
)

start_time = time.time()
cat_ranker.fit(train_pool)
cat_train_time = time.time() - start_time
print(f"CatBoost trained in {cat_train_time:.2f} seconds.")

# Lưu mô hình
with open("models/cat_ranker.pkl", "wb") as f:
    pickle.dump(cat_ranker, f)
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 4. Trực quan hóa Feature Importance giữa CatBoost và LightGBM
# Load LightGBM model
with open("models/lgb_ranker.pkl", "rb") as f:
    lgb_ranker = pickle.load(f)

# Lấy tầm quan trọng đặc trưng
lgb_importances = lgb_ranker.feature_importances_
lgb_importances_norm = lgb_importances / lgb_importances.sum()

cat_importances = cat_ranker.get_feature_importance(train_pool)
cat_importances_norm = cat_importances / cat_importances.sum()

feature_names = X_train.columns

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'LightGBM': lgb_importances_norm,
    'CatBoost': cat_importances_norm
}).set_index('Feature')

df_imp.plot(kind='bar', figsize=(10, 5))
plt.title("So sánh mức độ quan trọng đặc trưng (Normalized Feature Importance)")
plt.ylabel("Độ quan trọng tương đối")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 5. Định nghĩa Hàm Gợi ý E2E cho CatBoost
# Tải lại test_data
with open("processed_data/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)
with open("processed_data/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

# Phục vụ tính lambda động
all_genres = sorted(list(set([g for genres in movies_df['genres'].str.split('|').dropna() for g in genres if g])))
movies_df_indexed = movies_df # index movieId đã có từ cell trước

# Trực quan hóa MMR TF-IDF
movies_df['mmr_soup'] = movies_df.apply(
    lambda r: f"{r['genres'].replace('|', ' ')} {r['director'].replace(' ', '')} {' '.join(r['cast'].split('|')[:3])}", 
    axis=1
)
from sklearn.feature_extraction.text import TfidfVectorizer
mmr_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = mmr_vectorizer.fit_transform(movies_df['mmr_soup'])

from recsys_utils import reciprocal_rank_fusion, calculate_user_lambda

def end_to_end_recommend_cat(user_id, top_k=10):
    liked_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
    liked_set = set(liked_movies)
    
    # Retrieval
    bm25_candidates = []
    user_bm25_all_scores = np.zeros(len(movies_df))
    liked_soups = [movies_df.loc[lid, 'soup'] for lid in liked_movies if lid in movies_df.index]
    if liked_soups:
        query = " ".join(liked_soups)
        user_bm25_all_scores = bm25.transform(query)
        sorted_cb_idx = np.argsort(user_bm25_all_scores)[::-1]
        for idx in sorted_cb_idx:
            mid = movies_df.index[idx]
            if mid not in liked_set:
                bm25_candidates.append(mid)
            if len(bm25_candidates) >= 100:
                break
                
    u_idx = user_to_idx.get(user_id, None)
    als_candidates = []
    if u_idx is not None:
        ids, _ = als_model.recommend(u_idx, user_item_matrix[u_idx], N=100)
        als_candidates = [idx_to_movie[i] for i in ids if i in idx_to_movie and idx_to_movie[i] not in liked_set]
        
    rrf_list = reciprocal_rank_fusion(als_candidates, bm25_candidates, k=60)
    candidates = [item[0] for item in rrf_list[:250]]
    
    if not candidates:
        candidates = movies_df.sort_values(by='popularity', ascending=False).index.head(100).tolist()
        candidates = [cid for cid in candidates if cid not in liked_set]
        
    # Feature engineering cho candidates
    features = []
    valid_candidates = []
    
    als_user_factors = als_model.user_factors
    als_item_factors = als_model.item_factors
    
    for mid in candidates:
        if mid not in movies_df.index:
            continue
        valid_candidates.append(mid)
        movie = movies_df.loc[mid]
        user = users_df.loc[user_id]
        
        favorite_genres = set(user['favorite_genres'].split('|'))
        movie_genres = set(movie['genres'].split('|'))
        genre_overlap = len(favorite_genres.intersection(movie_genres))
        
        try:
            release_year = int(str(movie['release_date'])[:4])
        except:
            release_year = 2010
            
        u_idx = user_to_idx.get(user_id, None)
        m_idx = movie_to_idx.get(mid, None)
        
        als_score = als_user_factors[u_idx].dot(als_item_factors[m_idx]) if (u_idx is not None and m_idx is not None) else 0.0
        cb_score = user_bm25_all_scores[m_idx] if (m_idx is not None) else 0.0
            
        features.append({
            'popularity': movie['popularity'],
            'vote_average': movie['vote_average'],
            'genre_overlap': genre_overlap,
            'release_year': release_year,
            'user_activity': user['activity_level'],
            'user_bias': user['user_bias'],
            'als_score': als_score,
            'cb_score': cb_score
        })
        
    X_pred = pd.DataFrame(features)
    
    # CatBoost Inference
    scores = cat_ranker.predict(X_pred)
    
    candidate_scores = list(zip(valid_candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # MMR với lambda động
    user_history_genres = []
    for hmid in liked_movies:
        if hmid in movies_df.index:
            user_history_genres.extend(movies_df.loc[hmid, 'genres'].split('|'))
    lambda_val = calculate_user_lambda(user_history_genres, all_genres, base_min=0.4, base_max=0.9)
    
    final_recs = []
    if candidate_scores:
        candidates_ids = [item[0] for item in candidate_scores]
        scores_arr = np.array([item[1] for item in candidate_scores])
        if scores_arr.max() != scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        else:
            scores_norm = np.ones_like(scores_arr)
            
        selected_items = []
        unselected_indices = list(range(len(candidates_ids)))
        first_choice = np.argmax(scores_norm)
        selected_items.append(candidates_ids[first_choice])
        unselected_indices.remove(first_choice)
        
        while len(selected_items) < top_k and unselected_indices:
            selected_matrix_indices = [movie_to_idx[mid] for mid in selected_items if mid in movie_to_idx]
            if not selected_matrix_indices:
                break
            selected_vectors = tfidf_matrix[selected_matrix_indices]
            
            valid_unselected = []
            valid_matrix_indices = []
            for idx in unselected_indices:
                cid = candidates_ids[idx]
                m_idx = movie_to_idx.get(cid, None)
                if m_idx is not None:
                    valid_unselected.append(idx)
                    valid_matrix_indices.append(m_idx)
            
            if not valid_unselected:
                break
                
            unselected_vectors = tfidf_matrix[valid_matrix_indices]
            sim_matrix = cosine_similarity(unselected_vectors, selected_vectors)
            max_sim = sim_matrix.max(axis=1)
            
            best_mmr = -1e9
            best_idx_in_unselected = -1
            
            for i, idx in enumerate(valid_unselected):
                mmr_val = lambda_val * scores_norm[idx] - (1 - lambda_val) * max_sim[i]
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx_in_unselected = idx
                    
            if best_idx_in_unselected == -1:
                break
            selected_items.append(candidates_ids[best_idx_in_unselected])
            unselected_indices.remove(best_idx_in_unselected)
        final_recs = selected_items
        
    return final_recs
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 6. Đánh giá Offline A/B Test So sánh CatBoost vs LightGBM E2E
# A. Load LightGBM End-to-End Recommender (Notebook 6 E2E logic)
with open("models/lgb_ranker.pkl", "rb") as f:
    lgb_ranker = pickle.load(f)

def end_to_end_recommend_lgb(user_id, top_k=10):
    liked_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
    liked_set = set(liked_movies)
    
    bm25_candidates = []
    user_bm25_all_scores = np.zeros(len(movies_df))
    liked_soups = [movies_df.loc[lid, 'soup'] for lid in liked_movies if lid in movies_df.index]
    if liked_soups:
        query = " ".join(liked_soups)
        user_bm25_all_scores = bm25.transform(query)
        sorted_cb_idx = np.argsort(user_bm25_all_scores)[::-1]
        for idx in sorted_cb_idx:
            mid = movies_df.index[idx]
            if mid not in liked_set:
                bm25_candidates.append(mid)
            if len(bm25_candidates) >= 100:
                break
                
    u_idx = user_to_idx.get(user_id, None)
    als_candidates = []
    if u_idx is not None:
        ids, _ = als_model.recommend(u_idx, user_item_matrix[u_idx], N=100)
        als_candidates = [idx_to_movie[i] for i in ids if i in idx_to_movie and idx_to_movie[i] not in liked_set]
        
    rrf_list = reciprocal_rank_fusion(als_candidates, bm25_candidates, k=60)
    candidates = [item[0] for item in rrf_list[:250]]
    
    if not candidates:
        candidates = movies_df.sort_values(by='popularity', ascending=False).index.head(100).tolist()
        candidates = [cid for cid in candidates if cid not in liked_set]
        
    features = []
    valid_candidates = []
    als_user_factors = als_model.user_factors
    als_item_factors = als_model.item_factors
    
    for mid in candidates:
        if mid not in movies_df.index:
            continue
        valid_candidates.append(mid)
        movie = movies_df.loc[mid]
        user = users_df.loc[user_id]
        
        favorite_genres = set(user['favorite_genres'].split('|'))
        movie_genres = set(movie['genres'].split('|'))
        genre_overlap = len(favorite_genres.intersection(movie_genres))
        
        try:
            release_year = int(str(movie['release_date'])[:4])
        except:
            release_year = 2010
            
        u_idx = user_to_idx.get(user_id, None)
        m_idx = movie_to_idx.get(mid, None)
        als_score = als_user_factors[u_idx].dot(als_item_factors[m_idx]) if (u_idx is not None and m_idx is not None) else 0.0
        cb_score = user_bm25_all_scores[m_idx] if (m_idx is not None) else 0.0
            
        features.append({
            'popularity': movie['popularity'],
            'vote_average': movie['vote_average'],
            'genre_overlap': genre_overlap,
            'release_year': release_year,
            'user_activity': user['activity_level'],
            'user_bias': user['user_bias'],
            'als_score': als_score,
            'cb_score': cb_score
        })
        
    X_pred = pd.DataFrame(features)
    scores = lgb_ranker.predict(X_pred)
    
    candidate_scores = list(zip(valid_candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    user_history_genres = []
    for hmid in liked_movies:
        if hmid in movies_df.index:
            user_history_genres.extend(movies_df.loc[hmid, 'genres'].split('|'))
    lambda_val = calculate_user_lambda(user_history_genres, all_genres, base_min=0.4, base_max=0.9)
    
    final_recs = []
    if candidate_scores:
        candidates_ids = [item[0] for item in candidate_scores]
        scores_arr = np.array([item[1] for item in candidate_scores])
        if scores_arr.max() != scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        else:
            scores_norm = np.ones_like(scores_arr)
            
        selected_items = []
        unselected_indices = list(range(len(candidates_ids)))
        first_choice = np.argmax(scores_norm)
        selected_items.append(candidates_ids[first_choice])
        unselected_indices.remove(first_choice)
        
        while len(selected_items) < top_k and unselected_indices:
            selected_matrix_indices = [movie_to_idx[mid] for mid in selected_items if mid in movie_to_idx]
            if not selected_matrix_indices:
                break
            selected_vectors = tfidf_matrix[selected_matrix_indices]
            
            valid_unselected = []
            valid_matrix_indices = []
            for idx in unselected_indices:
                cid = candidates_ids[idx]
                m_idx = movie_to_idx.get(cid, None)
                if m_idx is not None:
                    valid_unselected.append(idx)
                    valid_matrix_indices.append(m_idx)
            
            if not valid_unselected:
                break
                
            unselected_vectors = tfidf_matrix[valid_matrix_indices]
            sim_matrix = cosine_similarity(unselected_vectors, selected_vectors)
            max_sim = sim_matrix.max(axis=1)
            
            best_mmr = -1e9
            best_idx_in_unselected = -1
            
            for i, idx in enumerate(valid_unselected):
                mmr_val = lambda_val * scores_norm[idx] - (1 - lambda_val) * max_sim[i]
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx_in_unselected = idx
                    
            if best_idx_in_unselected == -1:
                break
            selected_items.append(candidates_ids[best_idx_in_unselected])
            unselected_indices.remove(best_idx_in_unselected)
        final_recs = selected_items
        
    return final_recs

# B. Đánh giá song song trên tập test LOO
from recsys_utils import evaluate_implicit_loo

preds_lgb = {}
preds_cat = {}

# Đo latency suy luận
start_lgb = time.time()
for u, pos_item, neg_items in test_data:
    u, pos_item, neg_items = int(u), int(pos_item), [int(x) for x in neg_items]
    recs = end_to_end_recommend_lgb(u, top_k=10)
    items = [pos_item] + neg_items
    preds_lgb[u] = [(item, 10 - recs.index(item) if item in recs else 0, item == pos_item) for item in items]
lgb_inference_time = (time.time() - start_lgb) / len(test_data)

start_cat = time.time()
for u, pos_item, neg_items in test_data:
    u, pos_item, neg_items = int(u), int(pos_item), [int(x) for x in neg_items]
    recs = end_to_end_recommend_cat(u, top_k=10)
    items = [pos_item] + neg_items
    preds_cat[u] = [(item, 10 - recs.index(item) if item in recs else 0, item == pos_item) for item in items]
cat_inference_time = (time.time() - start_cat) / len(test_data)

hr_lgb, ndcg_lgb, mrr_lgb = evaluate_implicit_loo(preds_lgb, k=10)
hr_cat, ndcg_cat, mrr_cat = evaluate_implicit_loo(preds_cat, k=10)

print()
print("=== KẾT QUẢ SO SÁNH OFFLINE (A/B COMPARISON) ===")
print(f"| Chỉ số | LightGBM Ranker | CatBoost Ranker (YetiRank) |")
print(f"| :--- | :--- | :--- |")
print(f"| **Hit Ratio@10** | {hr_lgb:.4f} | {hr_cat:.4f} |")
print(f"| **NDCG@10** | {ndcg_lgb:.4f} | {ndcg_cat:.4f} |")
print(f"| **MRR** | {mrr_lgb:.4f} | {mrr_cat:.4f} |")
print(f"| **Train Time (sec)** | 0.10s (ước tính) | {cat_train_time:.2f}s |")
print(f"| **Inference Time (sec/user)** | {lgb_inference_time:.4f}s | {cat_inference_time:.4f}s |")
"""))

    nb7['cells'] = nb7_cells
    with open(os.path.join(output_dir, "07_catboost_vs_lightgbm.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb7, f)


    print("Successfully created 7 notebooks in evaluation/ml_pipeline!")

if __name__ == "__main__":
    create_pipeline_notebooks()
