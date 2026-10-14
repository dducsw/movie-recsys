# 🌲 Movie Recommendation Training & Tuning (LightGBM & CatBoost)

Tài liệu này hướng dẫn chi tiết về pipeline huấn luyện và tối ưu hóa mô hình xếp hạng phim dựa trên hai thuật toán Gradient Boosting mạnh mẽ nhất hiện nay: **LightGBM** và **CatBoost**, kết hợp bộ tối ưu siêu tham số tự động **Optuna**.

---

## 📂 Cấu trúc Thư mục

Thư mục nằm tại `evaluation/ml_training/` trong dự án:

```
evaluation/ml_training/
├── README.md                      # Tài liệu này
├── configs.yaml                   # File cấu hình tập trung (paths, features, model hyperparameters)
├── requirements.txt               # Thư viện phục vụ huấn luyện (lightgbm, catboost, optuna, ...)
├── models.joblib                  # Model weights & encoders đã được huấn luyện và đóng gói
├── ranking_comparison_results.png # Biểu đồ so sánh trực quan hiệu năng giữa LightGBM & CatBoost
├── catboost_info/                 # Log quá trình huấn luyện của CatBoost
└── src/                           # Mã nguồn chi tiết
    ├── data_loader.py             # Đọc và merge dữ liệu phim, crawler TMDB và MovieLens
    ├── features.py                # Xây dựng đặc trưng người dùng, phim và tương tác chéo
    ├── models.py                  # Định nghĩa và huấn luyện LightGBM & CatBoost
    ├── train.py                   # Script thực thi phân chia, huấn luyện, đánh giá và so sánh
    ├── evaluate.py                # Cài đặt các metrics xếp hạng (Precision@K, Recall@K, NDCG@K)
    ├── tune.py                    # Tối ưu hóa siêu tham số tự động bằng Optuna
    └── recommend.py               # Suy luận và sinh đề xuất Top-K cá nhân hóa
```

---

## 🎯 Định nghĩa Bài toán (Problem Formulation)

Hệ thống tiếp cận bài toán gợi ý dưới dạng **Dự đoán Điểm đánh giá (Rating Regression)** kết hợp **Xếp hạng danh sách (Item Ranking)**:

$$\hat{r}_{u,m} = f(u, m, X_{u,m})$$

Trong đó:
*   $u$: Người dùng (User).
*   $m$: Bộ phim (Movie).
*   $X_{u,m}$: Vector đặc trưng kết hợp (User features, Movie features, Cross features).
*   $\hat{r}_{u,m}$: Điểm đánh giá dự đoán (sau đó được sắp xếp giảm dần để tạo danh sách Top-$K$).

---

## ⚔️ So sánh Mô hình: LightGBM vs. CatBoost

| Đặc tính | LightGBM | CatBoost |
| :--- | :--- | :--- |
| **Chiến lược phát triển cây** | Leaf-wise (ưu tiên node giảm sai số lớn nhất) | Symmetric Trees (cây đối xứng cân bằng) |
| **Xử lý biến phân loại** | Integer Encoding / Fisher exact split | Target Encoding tự động kèm Ordered Boosting |
| **Tốc độ huấn luyện** | Cực nhanh trên CPU/GPU | Tối ưu hóa tuyệt vời trên GPU (CUDA) |
| **Khả năng khái quát** | Cần tinh chỉnh regularization tránh overfitting | Rất ít khi bị overfitting nhờ Ordered Boosting |

---

## ⚙️ Cấu hình Siêu tham số (`configs.yaml`)

Tất cả các tham số được quản lý tập trung tại `configs.yaml`:
*   `paths`: Đường dẫn tương đối tới `data/crawler/movies_crawled.csv` và `data/ml-latest-small/`.
*   `split`: Phương pháp chia `time_by_user` với `n_holdout_per_user: 5`.
*   `features`: Giới hạn `top_n_directors: 200`, `top_n_actors: 500`, `top_n_keywords: 300`.
*   `lgbm` & `catboost`: Learning rate, số leaves/depth, regularization, early stopping.
*   `recommend`: Kích thước danh sách Top-$K$ (`top_k: 10`).

---

## 🚀 Hướng dẫn Thực thi Chi tiết

Chuyển vào thư mục `evaluation/ml_training/`:

```bash
cd evaluation/ml_training
```

### 1. Cài đặt Thư viện Phụ trợ (Nếu cần)
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện và So sánh Mô hình (`train.py`)
Thực hiện đọc dữ liệu, trích xuất đặc trưng, chia tập Train/Val/Test theo thời gian, huấn luyện đồng thời LightGBM và CatBoost, sau đó in báo cáo so sánh:

```bash
python src/train.py
```
**Kết quả đầu ra:**
*   Chỉ số hồi quy: RMSE, MAE.
*   Chỉ số xếp hạng: Hit Ratio@10, NDCG@10, MRR.
*   Biểu đồ so sánh được lưu tại `ranking_comparison_results.png`.
*   Mô hình được lưu vào `models.joblib`.

### 3. Tự động Tinh chỉnh Siêu tham số bằng Optuna (`tune.py`)
Tìm kiếm bộ siêu tham số tối ưu cho LightGBM hoặc CatBoost:

```bash
python src/tune.py
```

### 4. Sinh Đề xuất Top-K cho Người dùng (`recommend.py`)
Sử dụng mô hình đã huấn luyện để gợi ý Top-10 phim cho một `userId` cụ thể:

```bash
python src/recommend.py --user_id 1 --top_k 10
```
