# 🧠 Machine Learning in Recommendation Systems

Tài liệu này hướng dẫn chi tiết về việc áp dụng các thuật toán Học máy truyền thống (Traditional Machine Learning) trong xây dựng Hệ gợi ý (Recommendation Systems - RecSys). Tài liệu bao gồm lý thuyết toán học, trích dẫn các nghiên cứu khoa học kinh điển, cấu trúc mã nguồn và hướng dẫn thực thi thực nghiệm chi tiết.

---

## 📂 Cấu trúc Thư mục

```
evaluation/machine_learning/
├── README.md                # Tài liệu lý thuyết & hướng dẫn thực thi
├── generate_notebook.py     # Script tự động tạo/làm mới file Jupyter Notebook
└── ml_evaluation.ipynb      # Notebook thực nghiệm toàn bộ các mô hình ML
```

---

## 1. Phân loại Tương tác và Tác vụ Gợi ý

Trước khi áp dụng bất kỳ thuật toán nào, chúng ta cần xác định dạng dữ liệu tương tác đầu vào:
*   **Phản hồi tường minh (Explicit Feedback):** Người dùng trực tiếp đánh giá chất lượng sản phẩm qua điểm số (ví dụ: số sao 0.5 - 5.0 trong tập dữ liệu MovieLens).
    *   *Tác vụ:* Dự đoán điểm xếp hạng (Rating Prediction).
*   **Phản hồi ngầm định (Implicit Feedback):** Hành vi gián tiếp của người dùng như click, lượt xem, thời gian đọc, lịch sử mua hàng. Dữ liệu này chỉ mang tính tích cực (chỉ biết người dùng thích chứ không rõ họ ghét gì).
    *   *Tác vụ:* Đề xuất danh sách Top-K (Top-K Recommendation / Item Ranking).

---

## 2. Các Thuật toán Kinh điển và Cơ sở Nghiên cứu Khoa học

### A. Neighborhood-based Collaborative Filtering (Lọc cộng tác dựa trên lân cận)
Đây là phương pháp tiếp cận trực quan nhất, chia làm hai loại:
1.  **User-based Collaborative Filtering:** Gợi ý cho người dùng $u$ những bộ phim được yêu thích bởi các người dùng khác có sở thích tương tự với $u$.
2.  **Item-based Collaborative Filtering:** Gợi ý cho người dùng $u$ những bộ phim tương tự với những bộ phim mà $u$ đã từng đánh giá cao trong quá khứ.

Để tính toán độ tương đồng giữa hai thực thể (người dùng hoặc phim), ta sử dụng các công thức:
*   **Pearson Correlation Coefficient (Hệ số tương quan Pearson):** Khắc phục được sự thiên vị điểm số của mỗi người dùng (ví dụ: có người khó tính điểm trung bình chỉ 2.0, người dễ tính điểm trung bình là 4.0).
    $$sim(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}}$$
    Trong đó $I_{uv}$ là tập hợp các phim được đánh giá bởi cả $u$ và $v$, $\bar{r}_u$ là điểm đánh giá trung bình của người dùng $u$.

*   **Cosine Similarity (Độ tương đồng Cosine):**
    $$sim(i, j) = \cos(\mathbf{d}_i, \mathbf{d}_j) = \frac{\mathbf{d}_i \cdot \mathbf{d}_j}{\|\mathbf{d}_i\|_2 \|\mathbf{d}_j\|_2}$$

---

### B. Matrix Factorization (Phân rã ma trận - MF)
*   **Bài báo tiêu biểu:** [*Matrix Factorization Techniques for Recommender Systems*](https://ieeexplore.ieee.org/document/5197422) (Yehuda Koren, Robert Bell, Chris Volinsky - IEEE Computer, 2009).

#### Nguyên lý hoạt động
Phương pháp này phân rã ma trận tương tác User-Item thưa thớt $R \in \mathbb{R}^{M \times N}$ thành tích của hai ma trận có hạng thấp (low-rank): ma trận đặc trưng người dùng $P \in \mathbb{R}^{M \times K}$ và ma trận đặc trưng vật phẩm $Q \in \mathbb{R}^{N \times K}$ (với $K \ll \min(M, N)$ là số lượng thuộc tính ẩn - latent factors).

$$\hat{r}_{u,i} = q_i^T p_u$$

Để tăng độ chính xác, mô hình bổ sung thêm các hệ số chệch (biases):
$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$
Trong đó:
*   $\mu$: Điểm đánh giá trung bình toàn bộ hệ thống.
*   $b_u$: Độ chệch của người dùng $u$ (xu hướng chấm điểm cao hay thấp của user đó).
*   $b_i$: Độ chệch của bộ phim $i$ (chất lượng chung của bộ phim so với mặt bằng chung).

#### Hàm mục tiêu tối ưu (Loss Function)
Mô hình được huấn luyện bằng cách tối thiểu hóa sai số bình phương trên các tương tác đã biết, kết hợp với kỹ thuật chính quy hóa $L_2$ để tránh quá khớp (overfitting):

$$\min_{P, Q, b} \sum_{(u,i) \in K} (r_{u,i} - \mu - b_u - b_i - q_i^T p_u)^2 + \lambda \left( \|p_u\|_2^2 + \|q_i\|_2^2 + b_u^2 + b_i^2 \right)$$

---

### C. Factorization Machines (FM)
*   **Bài báo tiêu biểu:** [*Factorization Machines*](https://ieeexplore.ieee.org/document/5694074) (Steffen Rendle - IEEE ICDM, 2010).

#### Lý do ra đời
Matrix Factorization truyền thống chỉ hoạt động trên cặp ID User - Item. FM kết hợp hiệu quả thông tin phụ (side information) như thể loại, đạo diễn, diễn viên vào cùng một mô hình phân rã ma trận:

$$\hat{y}(x) = w_0 + \sum_{j=1}^d w_j x_j + \sum_{j=1}^d \sum_{l=j+1}^d \langle v_j, v_l \rangle x_j x_l$$

Tương tác bậc hai giữa các đặc trưng có thể được tính toán trong thời gian tuyến tính $\mathcal{O}(k \cdot d)$ nhờ kỹ thuật biến đổi đại số của Steffen Rendle, hoạt động xuất sắc trên dữ liệu siêu thưa thớt.

---

## 3. Các Chỉ số Đánh giá Hiệu năng

1. **Sai số Dự đoán (Regression Metrics):**
   * **MAE (Mean Absolute Error):** Đo độ lệch trung bình tuyệt đối.
   * **RMSE (Root Mean Squared Error):** Phạt nặng các lỗi dự đoán sai lệch lớn.

2. **Chất lượng Xếp hạng (Ranking Metrics):**
   * **Precision@K:** Tỷ lệ phim đúng trong Top-K đề xuất.
   * **Recall@K:** Tỷ lệ phim đúng được tìm thấy so với tổng số phim user thích.
   * **NDCG@K:** Đánh giá độ chính xác có tính đến vị trí của phim trong danh sách.

---

## 🚀 Hướng dẫn Chạy Thực nghiệm Chi tiết

### Bước 1: Chuẩn bị Môi trường
Đảm bảo đã cài đặt các thư viện:
```bash
pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn jupyter nbformat
```

### Bước 2: Khởi tạo hoặc Tái tạo Notebook (Tùy chọn)
Nếu bạn muốn sinh lại file notebook mẫu chuẩn từ script:
```bash
python evaluation/machine_learning/generate_notebook.py
```

### Bước 3: Chạy Thực nghiệm trên Jupyter
Khởi chạy Jupyter Notebook hoặc mở file trực tiếp trong VS Code / IDE:
```bash
jupyter notebook evaluation/machine_learning/ml_evaluation.ipynb
```

**Các bước được thực hiện tuần tự trong notebook:**
1. Tải và tiền xử lý dữ liệu `data/ml-latest-small/ratings.csv` và `movies.csv`.
2. Phân chia Train/Test theo tỷ lệ 80/20.
3. Huấn luyện các mô hình Lọc cộng tác:
   - **User-Based CF** (Cosine & Pearson)
   - **Item-Based CF** (Cosine & Pearson)
4. Huấn luyện các mô hình Phân rã ma trận:
   - **SVD** & **SVD++** (với `scikit-surprise`)
   - **NMF** (Non-negative Matrix Factorization)
5. Xây dựng và huấn luyện mô hình **Factorization Machines (FM)** từ đầu bằng NumPy + SGD kết hợp đặc trưng Genres.
6. Xuất bảng so sánh tổng hợp các chỉ số RMSE, MAE, Precision@10, Recall@10, NDCG@10 và vẽ biểu đồ trực quan.
