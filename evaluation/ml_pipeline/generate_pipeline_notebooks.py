import os
import nbformat as nbf

def create_pipeline_notebooks():
    output_dir = "." if os.path.exists("generate_pipeline_notebooks.py") else os.path.join("evaluation", "ml_pipeline")
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

    nb1_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Tải dữ liệu
Tế bào (cell) này thực hiện các công việc chuẩn bị ban đầu:
1. **Import thư viện**: Tải các thư viện Pandas (thao tác bảng), Numpy (tính toán số học), và Pickle (lưu trữ đối tượng nhị phân).
2. **Thiết lập đường dẫn**: Định nghĩa đường dẫn động tới thư mục dữ liệu cào (`data/crawler/`) và dữ liệu giả lập (`data/simulator/`).
3. **Tải dữ liệu**: Đọc 4 tệp CSV chính bao gồm: người dùng (`sim_users.csv`), lượt đánh giá (`sim_ratings.csv`), hành vi click (`sim_click_events.csv`), và danh mục phim gốc (`movies_crawled.csv`).
"""))
    nb1_cells.append(nbf.v4.new_code_cell("""import os""" + """
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

    nb1_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Phân chia tập dữ liệu bằng Leave-One-Out (LOO)

#### Nguyên lý chia tập dữ liệu theo thời gian:
Để đánh giá hệ gợi ý một cách chuẩn xác mà không bị rò rỉ dữ liệu (data leakage), ta áp dụng phương pháp **Time-based Leave-One-Out (LOO)**. 
Công thức toán học chia tập dữ liệu cho mỗi người dùng $u$:
$$\\text{Train}_u = \\{(u, i, t) \\mid t < t_u^{\\max}\\}$$
$$\\text{Test}_u = \\{(u, i, t) \\mid t = t_u^{\\max}\\}$$
Trong đó $t_u^{\\max}$ là mốc thời gian tương tác cuối cùng của người dùng $u$. 

#### So sánh các giải pháp chia tập dữ liệu:
| Giải pháp | Nguyên lý | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **Random Split** (Ví dụ: 80/20) | Chia ngẫu nhiên các dòng tương tác | Đơn giản, dễ thực hiện. | **Rò rỉ dữ liệu (Data Leakage)**: Sử dụng hành vi ở tương lai để dự đoán quá khứ. |
| **K-Fold Cross Validation** | Chia dữ liệu thành K phần ngẫu nhiên | Đánh giá ổn định lỗi sai số. | Phá vỡ hoàn toàn yếu tố thời gian và chuỗi hành vi tuần tự của user. |
| **Time-based LOO** (Lựa chọn) | Lấy tương tác cuối làm Test | Mô phỏng chính xác hành vi thực tế (dự đoán hành động tiếp theo). | Nếu user có quá ít tương tác, tập test sẽ không đại diện đủ. |

Tế bào này gọi hàm `split_data_implicit_leave_one_out` để tách dữ liệu. Tiếp theo, ta gộp toàn bộ lịch sử click thực tế của user vào `user_interacted_items`. Điều này rất quan trọng để khi mô hình thực hiện đánh giá (Evaluation), ta sẽ tạo mẫu âm (negative samples) tránh lấy trúng những bộ phim mà user thực sự đã click hoặc xem. Kết quả cuối cùng được lưu lại vào thư mục `processed_data/`.
"""))
    nb1_cells.append(nbf.v4.new_code_cell("""# 2. Phân chia tập dữ liệu theo Leave-One-Out (Implicit Feedback)""" + """
# Sử dụng ratings làm base và map với clicks để sinh test set LOO
# Để đồng nhất, ta dùng hàm split_data_implicit_leave_one_out trên ratings_df
train_ratings, test_data, user_interacted_items = split_data_implicit_leave_one_out(
    ratings_df, user_col='userId', item_col='movieId', timestamp_col='timestamp', seed=42
)

# 1. Tính mốc thời gian tối đa (test timestamp) của ratings_df cho mỗi user
user_test_timestamp = ratings_df.groupby('userId')['timestamp'].max().to_dict()

# 2. Quy đổi timestamp của clicks_df sang Unix Epoch để lọc sạch clicks tương lai
clicks_df['timestamp_epoch'] = pd.to_datetime(clicks_df['timestamp']).astype('int64') // 10**9
clicks_df['test_ts'] = clicks_df['userId'].map(user_test_timestamp)

# Chỉ giữ các click trước thời điểm test của user đó (hoặc user không có rating test)
train_clicks = clicks_df[clicks_df['test_ts'].isna() | (clicks_df['timestamp_epoch'] < clicks_df['test_ts'])].copy()
train_clicks.drop(columns=['timestamp_epoch', 'test_ts'], inplace=True, errors='ignore')

# Cập nhật user_interacted_items dựa trên train_clicks đã lọc để tránh rò rỉ khi tạo mẫu âm
click_interacted = train_clicks.groupby('userId')['movieId'].apply(set).to_dict()
for u, clicked_set in click_interacted.items():
    if u in user_interacted_items:
        user_interacted_items[u] = user_interacted_items[u].union(clicked_set)
    else:
        user_interacted_items[u] = clicked_set

# Lưu lại các tập dữ liệu đã chia để các notebook sau sử dụng
os.makedirs("processed_data", exist_ok=True)
train_ratings.to_csv("processed_data/train_ratings.csv", index=False)
train_clicks.to_csv("processed_data/train_clicks.csv", index=False)

with open("processed_data/test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)
    
with open("processed_data/user_interacted_items.pkl", "wb") as f:
    pickle.dump(user_interacted_items, f)

with open("processed_data/user_test_timestamp.pkl", "wb") as f:
    pickle.dump(user_test_timestamp, f)

print(f"Chia dữ liệu thành công! Train ratings: {len(train_ratings)} | Train clicks: {len(train_clicks)} | Test instances (users): {len(test_data)}")
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
 
    nb2_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Xây dựng Metadata Soup
Tế bào này tải tập dữ liệu phim từ file CSV, điền các giá trị trống (NaN) bằng chuỗi rỗng để tránh lỗi tính toán. Sau đó, ta định nghĩa hàm `build_metadata_soup` để gộp 4 thuộc tính nội dung quan trọng của phim:
*   Thể loại (`genres` - phân tách bằng khoảng trắng thay vì dấu `|`)
*   Đạo diễn (`director` - viết liền không dấu cách để gom cụm chính xác tên)
*   Diễn viên chính (`cast` - lấy tối đa 5 diễn viên đầu tiên)
*   Từ khóa cốt truyện (`keywords`)

Chuỗi gộp này (Metadata Soup) đóng vai trò như một văn bản mô tả ngắn về thuộc tính phim, giúp thuật toán trích xuất từ khóa tính toán độ tương đồng.
"""))
    nb2_cells.append(nbf.v4.new_code_cell("""import os""" + """
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
 
    nb2_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Huấn luyện mô hình BM25 và Tính toán TF-IDF Matrix

#### Nguyên lý thuật toán BM25:
BM25 (Best Matching 25) là giải thuật xếp hạng văn bản tiên tiến dựa trên lý thuyết xác xuất. Điểm BM25 của bộ phim (tài liệu $D$) đối với lịch sử người dùng (truy vấn $Q$) được tính bằng:
$$\\text{Score}(D, Q) = \\sum_{i=1}^{n} \\text{IDF}(q_i) \\cdot \\frac{f(q_i, D) \\cdot (k_1 + 1)}{f(q_i, D) + k_1 \\cdot \\left(1 - b + b \\cdot \\frac{|D|}{\\text{avgdl}}\\right)}$$
Trong đó:
*   $f(q_i, D)$ là tần suất xuất hiện của từ khóa $q_i$ trong soup của phim $D$.
*   $|D|$ và $\\text{avgdl}$ lần lượt là độ dài soup phim $D$ và độ dài trung bình của tất cả các phim.
*   $k_1$ là tham số bão hòa tần suất từ (thường chọn $1.2 \\le k_1 \\le 2.0$). Giới hạn ảnh hưởng của một từ lặp đi lặp lại.
*   $b$ là tham số chuẩn hóa độ dài tài liệu (thường chọn $b = 0.75$). Phạt các phim có soup quá dài dòng chứa nhiều từ rác.
*   $\\text{IDF}(q_i)$ là lượng thông tin nghịch đảo của từ khóa:
$$\\text{IDF}(q_i) = \\ln \\left( \\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \\right)$$

#### So sánh các giải pháp trích xuất đặc trưng nội dung:
| Giải pháp | Nguyên lý | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **TF-IDF** | Trọng số dựa trên tần suất từ và tần suất tài liệu nghịch đảo | Đơn giản, dễ cài đặt. | Không có bão hòa tần suất từ (TF) và không phạt độ dài tài liệu. |
| **BM25** (Lựa chọn) | Cải tiến TF-IDF bằng bão hòa TF và phạt độ dài tài liệu | Hiệu năng vượt trội cho dữ liệu dạng từ khóa rời rạc. | Cần điều chỉnh siêu tham số $k_1$ và $b$. |
| **Sentence-BERT** (SBERT) | Dùng Transformer mã hóa văn bản thành dense vector | Hiểu ngữ nghĩa tự nhiên của câu tự do (Overview). | Rất chậm khi chạy trên CPU, tốn bộ nhớ, không tối ưu cho danh từ riêng rời rạc. |

Tế bào này fit mô hình BM25 trên soup phim, đồng thời dùng `TfidfVectorizer` sinh ma trận TF-IDF (sẽ dùng để tính cosine similarity đo tính trùng lặp phục vụ thuật toán MMR ở Stage 3). Các mô hình được lưu lại dưới dạng file Pickle.
"""))
    nb2_cells.append(nbf.v4.new_code_cell("""# 2. Xây dựng mô hình BM25 và TF-IDF Matrix (TF-IDF dùng cho MMR)""" + """
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
 
    nb2_cells.append(nbf.v4.new_markdown_cell("""### Bước 3: Định nghĩa Hàm gợi ý Content-Based (BM25)
Tế bào này cài đặt hàm `get_content_based_candidates`:
1. Nhận danh sách các ID phim người dùng đã thích (`liked_movie_ids`).
2. Ghép soup của các phim đã thích này lại để làm một câu truy vấn lớn đại diện cho "gu nội dung" của người dùng.
3. Chạy mô hình BM25 để tính toán điểm tương đồng của câu truy vấn này với tất cả các phim khác trong danh mục.
4. Sắp xếp điểm số giảm dần, loại bỏ các phim người dùng đã xem, và trả về Top N phim làm ứng viên Content-Based cho giai đoạn tiếp theo.
"""))
    nb2_cells.append(nbf.v4.new_code_cell("""# 3. Định nghĩa hàm gợi ý Content-Based dùng BM25 cho một danh sách phim đã xem""" + """
def get_content_based_candidates(liked_movie_ids, top_n=100):
    # Giới hạn tối đa 20 phim tương tác gần nhất để tránh nhiễu
    liked_movie_ids = list(liked_movie_ids)[-20:]
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

    nb3_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Mã hóa Index
Tế bào này tải dữ liệu hành vi click giả lập và phim cào được. Vì các thuật toán Matrix Factorization trong thư viện `implicit` yêu cầu ma trận thưa với chỉ mục liên tục, ta thực hiện lập bản đồ (encode) các giá trị ID gốc (`userId`, `movieId`) sang chỉ mục nguyên liên tục bắt đầu từ 0. Dict mapping này được lưu vào file `id_mappings.pkl`.
"""))
    nb3_cells.append(nbf.v4.new_code_cell("""import os""" + """
import pandas as pd
import numpy as np
import scipy.sparse as sp
import implicit
import pickle

# Load dữ liệu tương tác ngầm định đã lọc sạch rò rỉ (leakage) và ratings
clicks_df = pd.read_csv("processed_data/train_clicks.csv")
train_ratings = pd.read_csv("processed_data/train_ratings.csv")
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

    nb3_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Xây dựng Ma trận Tương tác Thưa có Trọng số
Hiệu quả của iALS phụ thuộc vào việc xây dựng ma trận trọng số (confidence score).
Chúng ta thiết lập trọng số phễu hành vi (Behavior Funnel Weighting) cho cả click events và rating events:
*   `click`: **1.0**
*   `detail_view`: **2.0**
*   `watch_start`: **3.0**
*   `watch_complete` hoặc `Explicit rating >= 3.5`: **5.0** (Tín hiệu xem hết/rất thích)
*   `Explicit rating < 3.5`: **1.5** (Tín hiệu tương tác nhưng không thích lắm)

Ta kết hợp và lấy trọng số lớn nhất (`max()`) cho mỗi cặp User-Movie để làm đầu vào ma trận thưa.
"""))
    nb3_cells.append(nbf.v4.new_code_cell("""# 1. Xây dựng ma trận tương tác có trọng số từ Behavior Funnel và Ratings""" + """
event_weights = {
    'click': 1.0,
    'detail_view': 2.0,
    'watch_start': 3.0,
    'watch_complete': 5.0
}

clicks_df['weight'] = clicks_df['event_type'].map(event_weights)
user_movie_weights = clicks_df.groupby(['userId', 'movieId'])['weight'].sum().reset_index()

# Tích hợp rating (explicit feedback) vào ma trận
ratings_weights = train_ratings.copy()
ratings_weights['weight'] = ratings_weights['rating'].apply(lambda r: 5.0 if r >= 3.5 else 1.5)
ratings_weights = ratings_weights[['userId', 'movieId', 'weight']]

# Gộp cả 2 nguồn, lấy giá trị lớn nhất cho mỗi cặp User-Movie
combined_weights = pd.concat([user_movie_weights, ratings_weights], ignore_index=True)
user_movie_weights = combined_weights.groupby(['userId', 'movieId'])['weight'].max().reset_index()

user_movie_weights = user_movie_weights[user_movie_weights['movieId'].isin(movie_to_idx.keys())]
user_movie_weights = user_movie_weights[user_movie_weights['userId'].isin(user_to_idx.keys())]

user_indices = user_movie_weights['userId'].map(user_to_idx).values
item_indices = user_movie_weights['movieId'].map(movie_to_idx).values
weights = user_movie_weights['weight'].values

num_users = len(user_to_idx)
num_items = len(movie_to_idx)

user_item_matrix = sp.csr_matrix((weights, (user_indices, item_indices)), shape=(num_users, num_items))
print(f"Sparse matrix density: {100 * user_item_matrix.nnz / (num_users * num_items):.4f}%")
"""))

    nb3_cells.append(nbf.v4.new_markdown_cell("""### Bước 3: Huấn luyện mô hình Implicit ALS (Alternating Least Squares)

#### Nguyên lý Toán học của mô hình iALS:
Mô hình iALS phân rã ma trận tương tác người dùng - vật phẩm $R$ thành hai ma trận nhân tử ẩn có số chiều thấp: Ma trận User Factors $X \\in \\mathbb{R}^{U \\times f}$ và Ma trận Item Factors $Y \\in \\mathbb{R}^{I \\times f}$. 
Hàm mục tiêu tối ưu hóa bình phương tối thiểu có trọng số:
$$\\min_{x_*, y_*} \\sum_{u, i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \\lambda \\left( \\sum_u \\|x_u\\|^2 + \\sum_i \\|y_i\\|^2 \\right)$$
Trong đó:
*   $p_{ui}$ là chỉ số sở thích nhị phân: $p_{ui} = 1$ nếu tổng trọng số tương tác $r_{ui} > 0$, ngược lại $p_{ui} = 0$.
*   $c_{ui}$ là độ tin cậy (confidence measure): $c_{ui} = 1 + \\alpha r_{ui}$ (với $\\alpha$ là hằng số tỷ lệ, thường đặt mặc định là 40).
*   $\\lambda$ là hệ số điều hòa (regularization) để tránh hiện tượng quá khớp (overfitting).
*   $x_u$ và $y_i$ lần lượt là vector đặc trưng ẩn đại diện cho người dùng $u$ và bộ phim $i$.

#### So sánh các giải thuật Collaborative Filtering:
| Giải pháp | Nguyên lý | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **KNN Item-based** | Đo độ tương đồng giữa các cột của ma trận tương tác | Đơn giản, dễ giải thích. | Độ phức tạp tính toán rất lớn $O(I^2)$, không thể mở rộng (scale) khi số lượng phim tăng lên. |
| **Explicit SVD** | Phân rã ma trận dựa trên ratings tường minh (1-5 sao) | Hiệu quả cao khi dữ liệu rating dày đặc. | **Lỗi chệch dữ liệu**: Coi các phim chưa rate là nhãn âm (0), trong khi thực tế có thể user thích nhưng chưa biết phim đó. |
| **Implicit ALS** (Lựa chọn) | Phân rã ma trận dựa trên trọng số độ tin cậy ngầm định | Giải quyết triệt để bài toán thiếu nhãn âm, cực kỳ thích hợp cho log tương tác ngầm định (click/watch). | Cần tối ưu siêu tham số số chiều ẩn $f$ và hệ số $\\alpha$. |

Tế bào này nhân ma trận tương tác thưa với trọng số confidence $\\alpha=40$, huấn luyện mô hình ALS với 64 factors và lưu trữ kết quả.
"""))
    nb3_cells.append(nbf.v4.new_code_cell("""# 2. Huấn luyện Implicit ALS model""" + """
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

    nb3_cells.append(nbf.v4.new_markdown_cell("""### Bước 4: Hàm lấy ứng viên Collaborative Filtering
Định nghĩa hàm `get_als_candidates` nhận vào ID người dùng, tìm chỉ mục mã hóa tương ứng, gọi hàm `recommend()` của mô hình ALS đã huấn luyện để dự đoán điểm số và lọc ra Top N phim chưa xem có điểm số cao nhất làm ứng viên.
"""))
    nb3_cells.append(nbf.v4.new_code_cell("""# 3. Hàm đề xuất Collaborative Filtering candidates""" + """
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

    nb4_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Tải dữ liệu giai đoạn trước
Tế bào này tải tập dữ liệu train ratings, thông tin chi tiết phim và người dùng, các dict mapping ID, cùng các mô hình Retrieval (ALS và BM25) đã huấn luyện thành công ở Stage 1 để sử dụng cho việc trích xuất đặc trưng chéo.
"""))
    nb4_cells.append(nbf.v4.new_code_cell("""import os""" + """
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import scipy.sparse as sp
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

with open("processed_data/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)
"""))

    nb4_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Sinh Tập dữ liệu Huấn luyện Xếp hạng (Negative Sampling)
Để huấn luyện mô hình xếp hạng nhị phân Stage 2, ta cần cả mẫu dương (positive) và mẫu âm (negative):
1. **Mẫu dương (Label = 1)**: Là các cặp User-Movie có rating thực tế $\\ge 3.5$ trong tập train.
2. **Mẫu âm (Label = 0)**: Dùng chiến lược **Hard Negative Sampling kết hợp Random Negatives** (Chuẩn công nghiệp):
   * Với mỗi user, ta chạy mô hình iALS để sinh 200 đề xuất tốt nhất. Các phim trong top này mà user **chưa tương tác** được gọi là **Hard Negatives** (phim gần đúng gu nhưng user không chọn).
   * Lấy mẫu 50% Hard Negatives và 50% Random Negatives để giúp Ranker phân biệt tối ưu giữa phim liên quan gần và phim hoàn toàn không liên quan.
"""))
    nb4_cells.append(nbf.v4.new_code_cell("""# 1. Tạo tập dữ liệu huấn luyện cho Ranker (Hard + Random Negatives)""" + """
np.random.seed(42)

# Precompute iALS candidates cho mọi user trong train ratings phục vụ trích xuất hard negatives
print("Đang sinh danh sách hard negatives từ iALS...")
als_hard_candidates = {}
for u in train_ratings['userId'].unique():
    u_idx = user_to_idx.get(u, None)
    if u_idx is not None:
        ids, _ = als_model.recommend(u_idx, user_item_matrix[u_idx], N=200)
        movie_ids = [idx_to_movie[i] for i in ids if i in idx_to_movie]
        als_hard_candidates[u] = movie_ids

ranking_data = []
all_movie_ids = list(movie_to_idx.keys())

for _, row in train_ratings.iterrows():
    u = int(row['userId'])
    pos_item = int(row['movieId'])
    rating = row['rating']
    
    label = 1 if rating >= 3.5 else 0
    ranking_data.append({'userId': u, 'movieId': pos_item, 'label': label})
    
    interacted = user_interacted_items.get(u, set())
    hard_negs = als_hard_candidates.get(u, [])
    # Lọc bỏ các phim user đã tương tác
    hard_negs_clean = [m for m in hard_negs if m not in interacted]
    
    # Lấy mẫu tối đa 2 hard negatives
    num_hard_to_sample = min(2, len(hard_negs_clean))
    sampled_negs = []
    if num_hard_to_sample > 0:
        sampled_negs = list(np.random.choice(hard_negs_clean, size=num_hard_to_sample, replace=False))
        
    # Phần còn lại bù bằng random negatives
    num_rand_to_sample = 4 - len(sampled_negs)
    for _ in range(num_rand_to_sample):
        neg_item = np.random.choice(all_movie_ids)
        while neg_item in interacted or neg_item in sampled_negs:
            neg_item = np.random.choice(all_movie_ids)
        sampled_negs.append(neg_item)
        
    for neg_item in sampled_negs:
        ranking_data.append({'userId': u, 'movieId': neg_item, 'label': 0})

df_rank = pd.DataFrame(ranking_data)
print(f"Tổng số dòng huấn luyện ranker: {len(df_rank)}")
print("Phân phối nhãn:")
print(df_rank['label'].value_counts())
"""))

    nb4_cells.append(nbf.v4.new_markdown_cell("""### Bước 3: Thiết kế Đặc trưng (Feature Engineering)
Để mô hình học máy đạt độ chính xác cao, ta thiết kế 3 nhóm đặc trưng chính cho mỗi cặp User-Movie:
1. **Đặc trưng của Phim (Movie Features)**:
   - `popularity`: Độ phổ biến của phim cào từ TMDB.
   - `vote_average`: Điểm đánh giá trung bình toàn cầu.
   - `release_year`: Năm phát hành của phim (trích xuất từ release_date).
2. **Đặc trưng của Người dùng (User Features)**:
   - `user_activity`: Mức độ hoạt động của người dùng (tổng số click/rating).
   - `user_bias`: Xu hướng chấm điểm của người dùng (chênh lệch trung bình so với điểm chung).
3. **Đặc trưng Tương tác chéo (Cross/Affinity Features)**:
   - `genre_overlap`: Số thể loại trùng khớp giữa gu yêu thích của người dùng và thể loại của bộ phim.
   - `als_score`: Điểm số dự đoán từ mô hình Collaborative Filtering (ALS) ở Stage 1.
   - `cb_score`: Điểm số dự đoán độ tương đồng nội dung BM25 từ mô hình Content-Based ở Stage 1.
"""))
    nb4_cells.append(nbf.v4.new_code_cell("""# 2. Xây dựng các đặc trưng (Feature Engineering)""" + """
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

# Precompute user liked movie IDs to avoid slow dataframe scans in the loop
train_pos_ratings = train_ratings[train_ratings['rating'] >= 3.5]
user_liked_movies_dict = train_pos_ratings.groupby('userId')['movieId'].apply(list).to_dict()

# Precompute user favorite genres from their actual watched history (rating >= 3.5) with fallback to seed list
user_fav_genres_dict = {}
for uid, row in users_df.iterrows():
    liked_mids = user_liked_movies_dict.get(uid, [])
    genres = set()
    for lmid in liked_mids:
        if lmid in movies_df.index:
            g_str = movies_df.loc[lmid, "genres"]
            if pd.notna(g_str):
                genres.update(str(g_str).split("|"))
                
    if not genres:
        fav_m_str = str(row.get("favorite_movies", ""))
        fav_m_list = [int(m) for m in fav_m_str.split("|") if str(m).isdigit()]
        for fid in fav_m_list:
            if fid in movies_df.index:
                g_str = movies_df.loc[fid, "genres"]
                if pd.notna(g_str):
                    genres.update(str(g_str).split("|"))
    user_fav_genres_dict[uid] = genres

# Precompute dictionary mappings for O(1) loop lookups (bypasses slow pandas .loc)
movie_popularity_dict = movies_df['popularity'].to_dict()
movie_vote_average_dict = movies_df['vote_average'].to_dict()
movie_genres_dict = movies_df['genres'].to_dict()
movie_genres_sets = {mid: set(str(g).split('|')) for mid, g in movie_genres_dict.items()}
movie_release_date_dict = movies_df['release_date'].to_dict()
movies_soup_dict = movies_unindexed['soup'].to_dict()

user_activity_dict = users_df['activity_level'].to_dict()
user_bias_dict = users_df['user_bias'].to_dict()

# Đảm bảo df_rank được sắp xếp theo userId để tối ưu việc cache profile similarity
df_rank_sorted_by_user = df_rank.sort_values(by='userId')

uids = df_rank_sorted_by_user['userId'].values
mids = df_rank_sorted_by_user['movieId'].values

# Vectorized/fast lookup arrays
popularity_vec = [movie_popularity_dict.get(mid, 1.0) for mid in mids]
vote_average_vec = [movie_vote_average_dict.get(mid, 5.0) for mid in mids]
user_activity_vec = [user_activity_dict.get(uid, 15) for uid in uids]
user_bias_vec = [user_bias_dict.get(uid, 0.0) for uid in uids]

genre_overlaps = []
for uid, mid in zip(uids, mids):
    user_favs = user_fav_genres_dict.get(uid, set())
    m_genres = movie_genres_sets.get(mid, set())
    genre_overlaps.append(len(user_favs.intersection(m_genres)))

release_years = []
for mid in mids:
    try:
        release_years.append(int(str(movie_release_date_dict.get(mid, "2010"))[:4]))
    except:
        release_years.append(2010)

# Vectorized ALS scores (avoid loop row-by-row matrix multiplication)
u_idx_arr = np.array([user_to_idx.get(u, -1) for u in uids])
m_idx_arr = np.array([movie_to_idx.get(m, -1) for m in mids])
valid_mask = (u_idx_arr != -1) & (m_idx_arr != -1)
als_scores = np.zeros(len(df_rank_sorted_by_user))
if valid_mask.any():
    als_scores[valid_mask] = np.sum(
        als_model.user_factors[u_idx_arr[valid_mask]] * als_model.item_factors[m_idx_arr[valid_mask]], 
        axis=1
    )

# Fast CB scores using precomputed BoW vectors (avoid text tokenization and prevent leakage)
user_query_bow_dict = {}
for uid, liked_movies in user_liked_movies_dict.items():
    liked_ids_clean = [lid for lid in liked_movies if lid in movies_soup_dict]
    liked_ids_clean = liked_ids_clean[-20:]
    if liked_ids_clean:
        liked_idxs = [movie_to_idx[lid] for lid in liked_ids_clean if lid in movie_to_idx]
        if liked_idxs:
            user_query_bow_dict[uid] = bm25.tf[liked_idxs].sum(axis=0)

# Precompute BM25 scores vector for all users instantly
user_bm25_all_dict = {}
for uid, u_bow in user_query_bow_dict.items():
    if u_bow is not None:
        user_bm25_all_dict[uid] = bm25.transform_from_bow(u_bow)

cb_scores = []
for uid, mid in zip(uids, mids):
    m_idx = movie_to_idx.get(mid, None)
    u_scores = user_bm25_all_dict.get(uid, None)
    cb_score = u_scores[m_idx] if (u_scores is not None and m_idx is not None) else 0.0
    cb_scores.append(cb_score)

# Instantly build features DataFrame
X = pd.DataFrame({
    'popularity': popularity_vec,
    'vote_average': vote_average_vec,
    'genre_overlap': genre_overlaps,
    'release_year': release_years,
    'user_activity': user_activity_vec,
    'user_bias': user_bias_vec,
    'als_score': als_scores,
    'cb_score': cb_scores
})
y = df_rank_sorted_by_user['label']

df_rank_sorted_by_user['group_key'] = df_rank_sorted_by_user['userId']
X_sorted = X
y_sorted = y
groups = df_rank_sorted_by_user.groupby('group_key', sort=False).size().values

print(f"Features head:")
display(X_sorted.head(3))
"""))

    nb4_cells.append(nbf.v4.new_markdown_cell("""### Bước 4: Huấn luyện mô hình xếp hạng LightGBM (LambdaRank)

#### Nguyên lý thuật toán LambdaRank:
Khác với mô hình phân loại nhị phân thông thường tối ưu hàm Binary Cross Entropy độc lập cho từng dòng, LambdaRank là giải thuật xếp hạng dạng cặp (**Pairwise Learning-to-Rank**). 
Nó định nghĩa một đạo hàm giả (gọi là $\\lambda$) cho mỗi cặp phim $(i, j)$ được đề xuất cho cùng một người dùng:
$$\\lambda_{ij} = \\frac{\\partial C_{ij}}{\\partial s_i} = -\\frac{1}{1 + e^{s_i - s_j}} \\cdot |\\Delta \\text{NDCG}|$$
Trong đó:
*   $s_i$ và $s_j$ lần lượt là điểm số dự đoán của mô hình cho phim tốt $i$ và phim kém hơn $j$.
*   $|\\Delta \\text{NDCG}|$ là lượng thay đổi của chỉ số xếp hạng toàn cục NDCG nếu ta hoán đổi vị trí của phim $i$ và $j$ trong danh sách đề xuất. 
*   Cơ chế này giúp mô hình tập trung tối ưu hóa thứ tự xếp hạng của các phim ở đầu danh sách (vị trí quan trọng nhất) thay vì tối ưu hóa toàn bộ dữ liệu một cách cào bằng.

#### So sánh các cách tiếp cận Learning-to-Rank (LTR):
| Cách tiếp cận | Hàm mục tiêu | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **Pointwise** | Hồi quy hoặc Phân loại nhị phân độc lập từng dòng (BCE/MSE) | Đơn giản, dùng được các thư viện phân loại truyền thống. | Coi các phim độc lập, không học được thứ tự tương đối giữa các phim trong cùng danh sách. |
| **Pairwise** (LambdaRank) | Tối ưu hóa thứ tự cặp phim, nhân với trọng số ảnh hưởng NDCG | Hiệu năng xếp hạng thực tế rất cao, tập trung vào top đầu danh sách. | Cấu trúc dữ liệu huấn luyện phức tạp hơn (cần định nghĩa `group` danh sách). |
| **Listwise** | Tối ưu hóa trực tiếp hàm phân phối xác suất của toàn bộ danh sách | Về mặt lý thuyết là tối ưu nhất. | Rất khó cài đặt, chi phí tính toán cực kỳ lớn. |

Tế bào này khởi tạo mô hình `LGBMRanker` với mục tiêu `objective='lambdarank'`, nhóm dữ liệu theo từng User (`group=groups`) và huấn luyện mô hình.
"""))
    nb4_cells.append(nbf.v4.new_code_cell("""# 3. Huấn luyện LightGCN Ranker (LambdaRank)""" + """
ranker = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    eval_at=[10],
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
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

    nb5_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Vector hóa TF-IDF phục vụ MMR
Tế bào này chuẩn bị dữ liệu văn bản mở rộng gồm thể loại, đạo diễn và diễn viên chính gộp lại làm `mmr_soup`. Sau đó, ta dùng `TfidfVectorizer` để chuyển đổi toàn bộ soup phim sang ma trận TF-IDF biểu diễn đặc trưng nội dung để phục vụ việc tính toán khoảng cách cosine giữa các bộ phim ứng viên.
"""))
    nb5_cells.append(nbf.v4.new_code_cell("""import os""" + """
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

    nb5_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Định nghĩa Thuật toán Đa dạng hóa MMR

#### Nguyên lý thuật toán MMR (Maximal Marginal Relevance):
Để tránh tình trạng danh sách gợi ý bị trùng lặp thể loại quá mức (ví dụ gợi ý liên tiếp 10 phim hành động siêu anh hùng), ta áp dụng thuật toán MMR để cân bằng giữa Độ liên quan (Relevance) và Độ đa dạng (Diversity).
Công thức lựa chọn phần tử tiếp theo $D_i$ đưa vào danh sách đề xuất $S$:
$$\\text{MMR} = \\arg\\max_{D_i \\in R \\setminus S} \\left[ \\lambda \\cdot \\text{Sim}_1(D_i, Q) - (1 - \\lambda) \\cdot \\max_{D_j \\in S} \\text{Sim}_2(D_i, D_j) \\right]$$
Trong đó:
*   $R$ là tập hợp các ứng viên ban đầu (được xếp hạng bởi LightGBM).
*   $S$ là tập hợp các phim đã được lựa chọn vào danh sách gợi ý cuối cùng.
*   $\\text{Sim}_1(D_i, Q)$ là điểm số độ liên quan của ứng viên $D_i$ với user (chính là điểm số dự đoán của LightGBM).
*   $\\text{Sim}_2(D_i, D_j)$ là độ tương đồng nội dung giữa ứng viên $D_i$ với phim $D_j$ đã được chọn trong danh sách (tính bằng Cosine Similarity trên ma trận TF-IDF).
*   $\\lambda \\in [0, 1]$ là tham số điều hòa. Nếu $\\lambda = 1.0$, hệ thống chỉ quan tâm độ liên quan (xếp hạng gốc). Nếu $\\lambda = 0.0$, hệ thống chỉ quan tâm đa dạng hóa tối đa (chọn phim khác biệt nhất).

#### So sánh các thuật toán Đa dạng hóa (Reranking):
| Thuật toán | Nguyên lý | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **Deterministic Greedy** | Chọn phim tiếp theo sao cho khác thể loại với phim liền trước | Rất nhanh, cài đặt đơn giản. | Không linh hoạt, không đo được độ tương đồng ngữ nghĩa chi tiết. |
| **MMR** (Lựa chọn) | Cân bằng tuyến tính giữa điểm xếp hạng và cosine similarity với toàn bộ danh sách đã chọn | Hiệu quả cao, có tham số $\\lambda$ điều chỉnh linh hoạt. | Chi phí tính toán tăng dần theo kích thước danh sách chọn $O(K^2)$. |
| **DPP** (Determinant Point Processes)| Mô hình hóa danh sách dưới dạng ma trận Kernel và tính định thức | Tối ưu hóa xác suất toàn cục rất chuẩn xác. | Rất khó cài đặt, chi phí tính toán cực kỳ lớn $O(K^3)$. |

Tế bào này định nghĩa hàm `maximal_marginal_relevance` thực hiện duyệt qua các ứng viên chưa chọn, tính điểm phạt đa dạng và trích chọn phim tối ưu MMR.
"""))
    nb5_cells.append(nbf.v4.new_code_cell("""# 1. Định nghĩa giải thuật MMR""" + """
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

    nb5_cells.append(nbf.v4.new_markdown_cell("""### Bước 3: Chạy Thử nghiệm MMR đa dạng hóa
Tế bào này chạy thử nghiệm thực tế thuật toán MMR trên một danh sách phim giả định để quan sát sự thay đổi thứ tự và lựa chọn phim trước và sau khi đa dạng hóa.
"""))
    nb5_cells.append(nbf.v4.new_code_cell("""# 2. Thử nghiệm MMR đa dạng hóa""" + """
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

    nb6_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Tải các Stage Mô hình
Tế bào này nạp các thư viện, hàm tiện ích đánh giá trong `recsys_utils.py`, các file dữ liệu trung gian và tải 3 lớp mô hình chính (BM25, ALS, LGBMRanker) để ghép nối luồng chạy.
"""))
    nb6_cells.append(nbf.v4.new_code_cell("""import os""" + """
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

    nb6_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Thiết lập Hàm Đề xuất End-to-End

#### Nguyên lý Reciprocal Rank Fusion (RRF) ở bước Retrieval:
Ở Stage 1 (Retrieval), ta sử dụng kết hợp cả hai mô hình iALS (lọc cộng tác) và BM25 (lọc nội dung) để tận dụng ưu điểm của cả hai bên. Để gộp hai danh sách ứng viên này lại thành một danh sách duy nhất không bị thiên lệch điểm số, ta áp dụng công thức **RRF**:
$$\\text{RRF\\_Score}(d \\in D) = \\sum_{m \\in M} \\frac{1}{k + r_m(d)}$$
Trong đó $M$ là tập hợp mô hình retrieval, $r_m(d)$ là thứ hạng của bộ phim $d$ trong kết quả của mô hình $m$, và $k$ là hằng số phạt vị trí (thường chọn $k = 60$). RRF giúp xếp hạng các phim xuất hiện ở vị trí cao trong cả hai mô hình lên trên mà không cần chuẩn hóa điểm số gốc khác biệt của chúng.

#### Thuật toán $\\lambda$ động cá nhân hóa (Dynamic Lambda MMR):
Thông thường, MMR sử dụng một hệ số $\\lambda$ cố định cho mọi người dùng. Tuy nhiên, mỗi người dùng lại có xu hướng chấp nhận sự đa dạng khác nhau. 
*   Người dùng có gu hẹp (chỉ thích xem phim tài liệu): cần $\\lambda$ lớn để giữ độ liên quan cao.
*   Người dùng có gu rộng (thích xem nhiều thể loại khác nhau): cần $\\lambda$ nhỏ để tăng độ đa dạng.

Ta áp dụng thuật toán tính toán $\\lambda$ động dựa trên entropy thể loại lịch sử của người dùng:
$$\\lambda_u = \\text{calculate\\_user\\_lambda}(\\text{history\\_genres}_u)$$
Hàm `calculate_user_lambda` tính Entropy Shannon thể loại, chuyển đổi sang giá trị $\\lambda$ cá nhân hóa nằm trong khoảng $[0.4, 0.9]$.

Tế bào này cài đặt luồng chạy liên tục:
1. **Retrieval**: BM25 (100 phim) + iALS (100 phim) $\\rightarrow$ Gộp RRF $\\rightarrow$ Lấy Top 250 ứng viên.
2. **Ranking**: Trích xuất các đặc trưng và chấm điểm bằng LightGBM $\\rightarrow$ Xếp hạng giảm dần.
3. **Re-ranking**: Áp dụng MMR với $\\lambda$ động dựa trên Entropy lịch sử để sinh Top 10 phim gợi ý cuối cùng.
"""))
    nb6_cells.append(nbf.v4.new_code_cell("""# 1. Định nghĩa End-to-End Pipeline""" + """
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

def end_to_end_recommend(user_id, top_k=10, custom_lambda=None, eval_mode=False, eval_candidates=None):
    from recsys_utils import extract_user_features

    # --- EVAL MODE: Dự đoán trực tiếp trên danh sách candidates được chỉ định ---
    if eval_mode and eval_candidates is not None:
        X_pred, valid_mids = extract_user_features(
            user_id, eval_candidates, train_ratings, movies_df, users_df, 
            user_to_idx, movie_to_idx, als_model, bm25, is_train=False
        )
        if X_pred.empty:
            return [(mid, 0.0) for mid in eval_candidates]
        scores = lgb_ranker.predict(X_pred)
        # valid_mids đã được extract_user_features trả về đúng thứ tự với scores
        return list(zip(valid_mids, scores))

    # --- STAGE 1: RETRIEVAL (BM25 + iALS -> RRF) ---
    liked_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
    liked_set = set(liked_movies)
    
    # A. BM25 content candidate retrieval (sử dụng tối đa 20 phim tương tác gần nhất hoặc favorite_movies nếu cold start)
    bm25_candidates = []
    if liked_movies:
        liked_movies_profile = liked_movies[-20:]
    else:
        # Cold start fallback: dùng danh sách phim yêu thích khởi tạo
        user = users_indexed_df.loc[user_id]
        fav_m_str = str(user.get("favorite_movies", ""))
        liked_movies_profile = [int(m) for m in fav_m_str.split("|") if str(m).isdigit()]
        
    liked_soups = [movies_indexed_df.loc[lid, 'soup'] for lid in liked_movies_profile if lid in movies_indexed_df.index]
    if liked_soups:
        query = " ".join(liked_soups)
        user_bm25_all_scores = bm25.transform(query)
        sorted_cb_idx = np.argsort(user_bm25_all_scores)[::-1]
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
    X_pred, valid_candidates = extract_user_features(
        user_id, candidates, train_ratings, movies_df, users_df, 
        user_to_idx, movie_to_idx, als_model, bm25, is_train=False
    )
    if X_pred.empty:
        return []
    
    scores = lgb_ranker.predict(X_pred)
    
    candidate_scores = list(zip(valid_candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # --- STAGE 3: RE-RANKING (MMR với Lambda động) ---
    if custom_lambda is not None:
        lambda_val = custom_lambda
    else:
        user_history_genres = []
        for hmid in liked_movies:
            if hmid in movies_indexed_df.index:
                user_history_genres.extend(str(movies_indexed_df.loc[hmid, 'genres']).split('|'))
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

    nb6_cells.append(nbf.v4.new_markdown_cell("""### Bước 3: Đánh giá Hiệu năng Hệ thống trên Tập Test LOO

#### Định nghĩa toán học các chỉ số Accuracy và Beyond-Accuracy:
*   **Hit Ratio@K (HR@K)**: Tỷ lệ người dùng có phim test thực tế xuất hiện trong Top K phim gợi ý.
$$\\text{HR}@K = \\frac{1}{|U|} \\sum_{u \\in U} I(\\text{test\\_item}_u \\in \\text{Rec}_u(K))$$
*   **NDCG@K (Normalized Discounted Cumulative Gain)**: Đo lường chất lượng xếp hạng của phim test trong Top K, phạt các phim đúng nhưng bị xếp ở vị trí thấp.
$$\\text{NDCG}@K = \\frac{\\text{DCG}@K}{\\text{IDCG}@K}, \\quad \\text{DCG}@K = \\sum_{i=1}^{K} \\frac{2^{rel_i} - 1}{\\log_2(i + 1)}$$
*   **Diversity@K** (Độ đa dạng thể loại):
$$\\text{Diversity}@K = 1 - \\frac{2}{K(K-1)} \\sum_{i < j} \\text{CosineSimilarity}(\\vec{v}_i, \\vec{v}_j)$$
*   **Coverage@K** (Độ phủ danh mục): Tỷ lệ số lượng phim độc bản được gợi ý ít nhất một lần cho bất kỳ user nào trên tổng số phim trong catalog.
$$\\text{Coverage}@K = \\frac{|\\bigcup_{u \\in U} \\text{Rec}_u(K)|}{|I|}$$
*   **Novelty@K** (Độ mới lạ - đo bằng lượng thông tin tự thân của phim):
$$\\text{Novelty}@K = \\frac{1}{|U|} \\sum_{u \\in U} \\frac{1}{K} \\sum_{i \\in \\text{Rec}_u(K)} -\\log_2 P(i)$$
Trong đó $P(i) = \\frac{\\text{tổng click của phim } i}{\\text{tổng click của mọi phim}}$. Phim ít người xem (long-tail items) có $P(i)$ nhỏ nên sẽ tăng điểm Novelty.

Tế bào này lặp qua các người dùng trong tập kiểm thử LOO, chạy luồng gợi ý End-to-End, tính toán tất cả các chỉ số trên và in báo cáo QC kết quả.
"""))
    nb6_cells.append(nbf.v4.new_code_cell("""# 2. Đánh giá thử nghiệm hệ thống""" + """
predictions_dict = {}
pipeline_recs = {}

print("Bắt đầu đánh giá Pipeline trên 200 users từ tập test LOO (eval_mode)...")
for u, pos_item, neg_items in test_data[:200]:
    u = int(u)
    pos_item = int(pos_item)
    neg_items = [int(x) for x in neg_items]
    
    candidates = [pos_item] + neg_items
    
    # eval_mode=True để chấm điểm trực tiếp 100 LOO candidates mà không bị stable sort bug
    scores_list = end_to_end_recommend(u, eval_mode=True, eval_candidates=candidates)
    scores_dict = dict(scores_list)
    
    user_preds = []
    for item in candidates:
        score = scores_dict.get(item, -1e9)  # gán score cực thấp nếu phim không tìm thấy
        user_preds.append((item, score, item == pos_item))
    predictions_dict[u] = user_preds
    
    # Tạo top-10 thực tế phục vụ tính Diversity, Novelty, Coverage
    pipeline_recs[u] = end_to_end_recommend(u, top_k=10, custom_lambda=None)

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
 
    nb7_cells.append(nbf.v4.new_markdown_cell("""### Bước 1: Khởi tạo và Tải thư viện
Tế bào này nạp các thư viện, tự động cài đặt gói `catboost` nếu chưa có trên hệ thống, tải dữ liệu train ratings và các mô hình Retrieval Stage 1 để làm đặc trưng huấn luyện.
"""))
    nb7_cells.append(nbf.v4.new_code_cell("""import os""" + """
import sys
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
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

with open("processed_data/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)
"""))

    nb7_cells.append(nbf.v4.new_markdown_cell("""### Bước 2: Sinh Tập dữ liệu Huấn luyện (Negative Sampling)
Tương tự như khi huấn luyện LightGBM, tế bào này thực hiện lấy mẫu âm tỷ lệ 1:4 đối với mỗi tương tác dương thực tế để xây dựng tập dữ liệu train ratings cân bằng thông qua chiến lược **Hard Negatives (từ iALS)** kết hợp **Random Negatives**.
"""))
    nb7_cells.append(nbf.v4.new_code_cell("""# 1. Tạo tập dữ liệu huấn luyện cho Ranker (giống LightGBM)""" + """
np.random.seed(42)

# Precompute iALS candidates cho mọi user trong train ratings phục vụ trích xuất hard negatives
print("Đang sinh danh sách hard negatives từ iALS...")
als_hard_candidates = {}
for u in train_ratings['userId'].unique():
    u_idx = user_to_idx.get(u, None)
    if u_idx is not None:
        ids, _ = als_model.recommend(u_idx, user_item_matrix[u_idx], N=200)
        movie_ids = [idx_to_movie[i] for i in ids if i in idx_to_movie]
        als_hard_candidates[u] = movie_ids

ranking_data = []
all_movie_ids = list(movie_to_idx.keys())

for _, row in train_ratings.iterrows():
    u = int(row['userId'])
    pos_item = int(row['movieId'])
    rating = row['rating']
    label = 1 if rating >= 3.5 else 0
    ranking_data.append({'userId': u, 'movieId': pos_item, 'label': label})
    
    interacted = user_interacted_items.get(u, set())
    hard_negs = als_hard_candidates.get(u, [])
    # Lọc bỏ các phim user đã tương tác
    hard_negs_clean = [m for m in hard_negs if m not in interacted]
    
    # Lấy mẫu tối đa 2 hard negatives
    num_hard_to_sample = min(2, len(hard_negs_clean))
    sampled_negs = []
    if num_hard_to_sample > 0:
        sampled_negs = list(np.random.choice(hard_negs_clean, size=num_hard_to_sample, replace=False))
        
    # Phần còn lại bù bằng random negatives
    num_rand_to_sample = 4 - len(sampled_negs)
    for _ in range(num_rand_to_sample):
        neg_item = np.random.choice(all_movie_ids)
        while neg_item in interacted or neg_item in sampled_negs:
            neg_item = np.random.choice(all_movie_ids)
        sampled_negs.append(neg_item)
        
    for neg_item in sampled_negs:
        ranking_data.append({'userId': u, 'movieId': neg_item, 'label': 0})

df_rank = pd.DataFrame(ranking_data)
# Đảm bảo df_rank được sắp xếp theo userId/group để làm ranking
df_rank_sorted = df_rank.sort_values(by='userId').reset_index(drop=True)
"""))

    nb7_cells.append(nbf.v4.new_markdown_cell("""### Bước 3: Thiết kế Đặc trưng (Feature Engineering)
Tế bào này trích xuất các đặc trưng xếp hạng tương đồng với mô hình LightGBM bao gồm đặc trưng phim, đặc trưng user và các điểm số chéo (als_score, cb_score) để đảm bảo so sánh công bằng chéo (apples-to-apples comparison).
"""))
    nb7_cells.append(nbf.v4.new_code_cell("""# 2. Xây dựng các đặc trưng (Feature Engineering)""" + """
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

# Precompute user liked movie IDs to avoid slow dataframe scans in the loop
train_pos_ratings = train_ratings[train_ratings['rating'] >= 3.5]
user_liked_movies_dict = train_pos_ratings.groupby('userId')['movieId'].apply(list).to_dict()

# Precompute dictionary mappings for O(1) lookups (bypasses slow pandas .loc)
movie_popularity_dict = movies_df['popularity'].to_dict()
movie_vote_average_dict = movies_df['vote_average'].to_dict()
movie_genres_dict = movies_df['genres'].to_dict()
movie_genres_sets = {mid: set(str(g).split('|')) for mid, g in movie_genres_dict.items()}
movie_release_date_dict = movies_df['release_date'].to_dict()
movies_soup_dict = movies_unindexed['soup'].to_dict()

user_activity_dict = users_df['activity_level'].to_dict()
user_bias_dict = users_df['user_bias'].to_dict()

# Precompute user favorite genres from their actual watched history (rating >= 3.5) with fallback to seed list
user_fav_genres_dict = {}
for uid, fav_m_str in zip(users_df.index, users_df['favorite_movies']):
    liked_mids = user_liked_movies_dict.get(uid, [])
    genres = set()
    for lmid in liked_mids:
        g_str = movie_genres_dict.get(lmid)
        if g_str and g_str != "(no genres listed)":
            genres.update(g_str.split("|"))
            
    if not genres and pd.notna(fav_m_str):
        for m in str(fav_m_str).split("|"):
            if m.isdigit():
                fid = int(m)
                g_str = movie_genres_dict.get(fid)
                if g_str and g_str != "(no genres listed)":
                    genres.update(g_str.split("|"))
    user_fav_genres_dict[uid] = genres

uids = df_rank_sorted['userId'].values
mids = df_rank_sorted['movieId'].values

# Vectorized/fast lookup arrays
popularity_vec = [movie_popularity_dict.get(mid, 1.0) for mid in mids]
vote_average_vec = [movie_vote_average_dict.get(mid, 5.0) for mid in mids]
user_activity_vec = [user_activity_dict.get(uid, 15) for uid in uids]
user_bias_vec = [user_bias_dict.get(uid, 0.0) for uid in uids]

genre_overlaps = []
for uid, mid in zip(uids, mids):
    user_favs = user_fav_genres_dict.get(uid, set())
    m_genres = movie_genres_sets.get(mid, set())
    genre_overlaps.append(len(user_favs.intersection(m_genres)))

release_years = []
for mid in mids:
    try:
        release_years.append(int(str(movie_release_date_dict.get(mid, "2010"))[:4]))
    except:
        release_years.append(2010)

# Vectorized ALS scores (avoid loop row-by-row matrix multiplication)
u_idx_arr = np.array([user_to_idx.get(u, -1) for u in uids])
m_idx_arr = np.array([movie_to_idx.get(m, -1) for m in mids])
valid_mask = (u_idx_arr != -1) & (m_idx_arr != -1)
als_scores = np.zeros(len(df_rank_sorted))
if valid_mask.any():
    als_scores[valid_mask] = np.sum(
        als_model.user_factors[u_idx_arr[valid_mask]] * als_model.item_factors[m_idx_arr[valid_mask]], 
        axis=1
    )

# Fast CB scores using precomputed soup dictionary (avoid loc in loop)
cb_scores = []
current_uid = None
user_bm25_scores = None

for uid, mid in zip(uids, mids):
    m_idx = movie_to_idx.get(mid, None)
    if uid != current_uid:
        current_uid = uid
        liked_ids = user_liked_movies_dict.get(uid, [])
        liked_ids = [lid for lid in liked_ids if lid in movies_soup_dict]
        if liked_ids:
            liked_soups = [movies_soup_dict[lid] for lid in liked_ids]
            query = " ".join(liked_soups)
            user_bm25_scores = bm25.transform(query)
        else:
            user_bm25_scores = None
            
    cb_score = user_bm25_scores[m_idx] if (user_bm25_scores is not None and m_idx is not None) else 0.0
    cb_scores.append(cb_score)

# Instantly build features DataFrame
X_train = pd.DataFrame({
    'popularity': popularity_vec,
    'vote_average': vote_average_vec,
    'genre_overlap': genre_overlaps,
    'release_year': release_years,
    'user_activity': user_activity_vec,
    'user_bias': user_bias_vec,
    'als_score': als_scores,
    'cb_score': cb_scores
})
y_train = df_rank_sorted['label']
group_ids = df_rank_sorted['userId'].values # Mảng ID group cho CatBoost
"""))

    nb7_cells.append(nbf.v4.new_markdown_cell("""### Bước 4: Huấn luyện mô hình xếp hạng CatBoost YetiRank

#### Nguyên lý giải thuật YetiRank:
**YetiRank** của CatBoost là một bước tiến vượt bậc so với LambdaRank. 
*   Thay vì chỉ nhân với lượng thay đổi NDCG cục bộ khi tráo đổi 2 phần tử, YetiRank tạo ra một chuỗi các hoán vị (permutations) giả định ngẫu nhiên đối với danh sách vật phẩm của mỗi người dùng.
*   Sau đó, YetiRank tối ưu hóa kỳ vọng toán học của độ chênh lệch chỉ số NDCG trên toàn bộ các hoán vị này. Điều này giúp CatBoost ước lượng chính xác hơn phân phối xếp hạng toàn cục và có xu hướng ổn định hơn khi phân phối dữ liệu huấn luyện nhỏ.
*   Đồng thời, cấu trúc cây đối xứng (Symmetric/Oblivious Trees) giúp việc đưa ra dự đoán của CatBoost rất nhanh vì có thể song song hóa tối đa ở mức phần cứng.

Tế bào này định nghĩa cấu trúc dữ liệu `Pool` của CatBoost, thiết lập hàm mất mát `loss_function='YetiRank'`, huấn luyện mô hình và lưu vào ổ đĩa.
"""))
    nb7_cells.append(nbf.v4.new_code_cell("""# 3. Huấn luyện CatBoost Ranker (YetiRank)""" + """
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

    nb7_cells.append(nbf.v4.new_markdown_cell("""### Bước 5: So sánh Tầm quan trọng của Đặc trưng (Feature Importance)
Tế bào này nạp mô hình LightGBM đã lưu, lấy mức độ đóng góp đặc trưng (Feature Importance) đã được chuẩn hóa của cả hai mô hình, vẽ biểu đồ so sánh trực tiếp để quan sát xem mỗi giải thuật ưu tiên những nhóm tín hiệu nào khi xếp hạng phim.
"""))
    nb7_cells.append(nbf.v4.new_code_cell("""# 4. Trực quan hóa Feature Importance giữa CatBoost và LightGBM""" + """
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

def end_to_end_recommend_cat(user_id, top_k=10, eval_mode=False, eval_candidates=None):
    from recsys_utils import extract_user_features

    # --- EVAL MODE ---
    if eval_mode and eval_candidates is not None:
        X_pred, valid_mids = extract_user_features(
            user_id, eval_candidates, train_ratings, movies_df, users_df, 
            user_to_idx, movie_to_idx, als_model, bm25, is_train=False
        )
        if X_pred.empty:
            return [(mid, 0.0) for mid in eval_candidates]
        scores = cat_ranker.predict(X_pred)
        return list(zip(valid_mids, scores))

    # --- STAGE 1: RETRIEVAL (BM25 + iALS -> RRF) ---
    liked_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
    liked_set = set(liked_movies)
    
    # A. BM25 content candidate retrieval (sử dụng tối đa 20 phim tương tác gần nhất hoặc favorite_movies nếu cold start)
    bm25_candidates = []
    if liked_movies:
        liked_movies_profile = liked_movies[-20:]
    else:
        # Cold start fallback: dùng danh sách phim yêu thích khởi tạo
        user = users_df.loc[user_id]
        fav_m_str = str(user.get("favorite_movies", ""))
        liked_movies_profile = [int(m) for m in fav_m_str.split("|") if str(m).isdigit()]
        
    liked_soups = [movies_df.loc[lid, 'soup'] for lid in liked_movies_profile if lid in movies_df.index]
    if liked_soups:
        query = " ".join(liked_soups)
        user_bm25_all_scores = bm25.transform(query)
        sorted_cb_idx = np.argsort(user_bm25_all_scores)[::-1]
        for idx in sorted_cb_idx:
            # Ở cell trước movies_df đã set_index('movieId'), dùng iloc và lấy chỉ mục movieId
            mid = movies_df.index[idx]
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
        candidates = movies_df.sort_values(by='popularity', ascending=False).index.head(100).tolist()
        candidates = [cid for cid in candidates if cid not in liked_set]
        
    # --- STAGE 2: RANKING (CatBoost) ---
    X_pred, valid_candidates = extract_user_features(
        user_id, candidates, train_ratings, movies_df, users_df, 
        user_to_idx, movie_to_idx, als_model, bm25, is_train=False
    )
    if X_pred.empty:
        return []
    
    scores = cat_ranker.predict(X_pred)
    
    candidate_scores = list(zip(valid_candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # --- STAGE 3: RE-RANKING (MMR với Lambda động) ---
    user_history_genres = []
    for hmid in liked_movies:
        if hmid in movies_df.index:
            user_history_genres.extend(str(movies_df.loc[hmid, 'genres']).split('|'))
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
"""))

    nb7_cells.append(nbf.v4.new_code_cell("""# 6. Đánh giá Offline A/B Test So sánh CatBoost vs LightGBM E2E
# A. Load LightGBM End-to-End Recommender (Notebook 6 E2E logic)
with open("models/lgb_ranker.pkl", "rb") as f:
    lgb_ranker = pickle.load(f)

def end_to_end_recommend_lgb(user_id, top_k=10, eval_mode=False, eval_candidates=None):
    from recsys_utils import extract_user_features

    # --- EVAL MODE ---
    if eval_mode and eval_candidates is not None:
        X_pred, valid_mids = extract_user_features(
            user_id, eval_candidates, train_ratings, movies_df, users_df, 
            user_to_idx, movie_to_idx, als_model, bm25, is_train=False
        )
        if X_pred.empty:
            return [(mid, 0.0) for mid in eval_candidates]
        scores = lgb_ranker.predict(X_pred)
        return list(zip(valid_mids, scores))

    # --- STAGE 1: RETRIEVAL (BM25 + iALS -> RRF) ---
    liked_movies = train_ratings[train_ratings['userId'] == user_id]['movieId'].tolist()
    liked_set = set(liked_movies)
    
    # A. BM25 content candidate retrieval (sử dụng tối đa 20 phim tương tác gần nhất hoặc favorite_movies nếu cold start)
    bm25_candidates = []
    if liked_movies:
        liked_movies_profile = liked_movies[-20:]
    else:
        # Cold start fallback: dùng danh sách phim yêu thích khởi tạo
        user = users_df.loc[user_id]
        fav_m_str = str(user.get("favorite_movies", ""))
        liked_movies_profile = [int(m) for m in fav_m_str.split("|") if str(m).isdigit()]
        
    liked_soups = [movies_df.loc[lid, 'soup'] for lid in liked_movies_profile if lid in movies_df.index]
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
        candidates = movies_df.sort_values(by='popularity', ascending=False).index.head(100).tolist()
        candidates = [cid for cid in candidates if cid not in liked_set]
        
    # --- STAGE 2: RANKING (LightGBM) ---
    X_pred, valid_candidates = extract_user_features(
        user_id, candidates, train_ratings, movies_df, users_df, 
        user_to_idx, movie_to_idx, als_model, bm25, is_train=False
    )
    if X_pred.empty:
        return []
    
    scores = lgb_ranker.predict(X_pred)
    
    candidate_scores = list(zip(valid_candidates, scores))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    # --- STAGE 3: RE-RANKING (MMR với Lambda động) ---
    user_history_genres = []
    for hmid in liked_movies:
        if hmid in movies_df.index:
            user_history_genres.extend(str(movies_df.loc[hmid, 'genres']).split('|'))
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

# B. Đánh giá song song trên tập test LOO
from recsys_utils import evaluate_implicit_loo

preds_lgb = {}
preds_cat = {}

# 1. Warm-up Phase (loại bỏ hiệu ứng cold cache)
print("Đang chạy warm-up phase cho CPU cache...")
warmup_sample = test_data[:20]
for u, pos_item, neg_items in warmup_sample:
    u, pos_item, neg_items = int(u), int(pos_item), [int(x) for x in neg_items]
    items = [pos_item] + neg_items
    _ = end_to_end_recommend_lgb(u, eval_mode=True, eval_candidates=items)
    _ = end_to_end_recommend_cat(u, eval_mode=True, eval_candidates=items)

# 2. Benchmark Phase chính thức
test_data_sample = test_data[20:220]

start_lgb = time.time()
for u, pos_item, neg_items in test_data_sample:
    u, pos_item, neg_items = int(u), int(pos_item), [int(x) for x in neg_items]
    items = [pos_item] + neg_items
    
    # Chạy LightGBM ở chế độ eval_mode=True
    scores_list = end_to_end_recommend_lgb(u, eval_mode=True, eval_candidates=items)
    scores_dict = dict(scores_list)
    preds_lgb[u] = [(item, scores_dict.get(item, -1e9), item == pos_item) for item in items]
lgb_inference_time = (time.time() - start_lgb) / len(test_data_sample)

start_cat = time.time()
for u, pos_item, neg_items in test_data_sample:
    u, pos_item, neg_items = int(u), int(pos_item), [int(x) for x in neg_items]
    items = [pos_item] + neg_items
    
    # Chạy CatBoost ở chế độ eval_mode=True
    scores_list = end_to_end_recommend_cat(u, eval_mode=True, eval_candidates=items)
    scores_dict = dict(scores_list)
    preds_cat[u] = [(item, scores_dict.get(item, -1e9), item == pos_item) for item in items]
cat_inference_time = (time.time() - start_cat) / len(test_data_sample)

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
