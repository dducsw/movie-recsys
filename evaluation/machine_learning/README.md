# Machine Learning in Recommendation Systems

Tài liệu này hướng dẫn chi tiết về việc áp dụng các thuật toán Học máy truyền thống (Traditional Machine Learning) trong xây dựng Hệ gợi ý (Recommendation Systems - RecSys). Tài liệu tập trung vào phân tích lý thuyết toán học, trích dẫn các bài báo khoa học nổi tiếng và các chỉ số đánh giá cốt lõi.

---

## 1. Phân loại Tương tác và Tác vụ gợi ý

Trước khi áp dụng bất kỳ thuật toán nào, chúng ta cần xác định dạng dữ liệu tương tác đầu vào:
*   **Phản hồi tường minh (Explicit Feedback):** Người dùng trực tiếp đánh giá chất lượng sản phẩm qua điểm số (ví dụ: số sao 0.5 - 5.0 trong tập dữ liệu MovieLens).
    *   *Tác vụ:* Dự đoán điểm xếp hạng (Rating Prediction).
*   **Phản hồi ngầm định (Implicit Feedback):** Hành vi gián tiếp của người dùng như click, lượt xem, thời gian đọc, lịch sử mua hàng. Dữ liệu này chỉ mang tính tích cực (chỉ biết người dùng thích chứ không rõ họ ghét gì).
    *   *Tác vụ:* Đề xuất danh sách Top-K (Top-K Recommendation / Item Ranking).

---

## 2. Các thuật toán kinh điển và Cơ sở nghiên cứu khoa học

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

Để tăng độ chính xác, mô hình thường bổ sung thêm các hệ số chệch (biases):
$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$
Trong đó:
*   $\mu$: Điểm đánh giá trung bình toàn bộ hệ thống.
*   $b_u$: Độ chệch của người dùng $u$ (xu hướng chấm điểm cao hay thấp của user đó).
*   $b_i$: Độ chệch của bộ phim $i$ (chất lượng chung của bộ phim so với mặt bằng chung).

#### Hàm mục tiêu tối ưu (Loss Function)
Mô hình được huấn luyện bằng cách tối thiểu hóa sai số bình phương trên các tương tác đã biết, kết hợp với kỹ thuật chính quy hóa $L_2$ để tránh quá khớp (overfitting):

$$\min_{P, Q, b} \sum_{(u,i) \in K} (r_{u,i} - \mu - b_u - b_i - q_i^T p_u)^2 + \lambda \left( \|p_u\|_2^2 + \|q_i\|_2^2 + b_u^2 + b_i^2 \right)$$

#### Thuật toán tối ưu hóa
1.  **Stochastic Gradient Descent (SGD):** Cập nhật các tham số lặp đi lặp lại cho từng rating. Dễ triển khai và hội tụ nhanh trên tập dữ liệu lớn.
2.  **Alternating Least Squares (ALS):** Cố định $P$ để tối ưu $Q$, sau đó cố định $Q$ để tối ưu $P$. Rất phù hợp cho xử lý song song và hoạt động tốt trên dữ liệu phản hồi ngầm định (Implicit Feedback).

---

### C. Factorization Machines (FM)
*   **Bài báo tiêu biểu:** [*Factorization Machines*](https://ieeexplore.ieee.org/document/5694074) (Steffen Rendle - IEEE ICDM, 2010).

#### Lý do ra đời
Matrix Factorization truyền thống chỉ hoạt động tốt trên ID của User và Item. Tuy nhiên trong thực tế, chúng ta có rất nhiều đặc trưng phụ (side information) khác như: tuổi, giới tính của người dùng; đạo diễn, diễn viên của bộ phim; thời gian, thiết bị truy cập (ngữ cảnh). FM ra đời nhằm kết hợp hiệu quả thông tin phụ này vào mô hình phân rã ma trận.

#### Công thức toán học
FM mô hình hóa tất cả các mối quan hệ tương tác giữa các cặp biến đặc trưng đầu vào $x \in \mathbb{R}^d$ bằng cách sử dụng các vector ẩn bậc thấp:

$$\hat{y}(x) = w_0 + \sum_{j=1}^d w_j x_j + \sum_{j=1}^d \sum_{l=j+1}^d \langle v_j, v_l \rangle x_j x_l$$

Trong đó:
*   $w_0 \in \mathbb{R}$: Tham số chệch toàn cục.
*   $w_j \in \mathbb{R}$: Trọng số biểu thị ảnh hưởng tuyến tính của đặc trưng thứ $j$.
*   $v_j \in \mathbb{R}^k$: Vector ẩn biểu thị đặc trưng thứ $j$ trong không gian $k$ chiều. Phép tích vô hướng $\langle v_j, v_l \rangle$ giúp ước lượng tương tác giữa đặc trưng $j$ và đặc trưng $l$.

#### Điểm mạnh
*   Tương tác bậc hai giữa các đặc trưng có thể được tính toán trong thời gian tuyến tính $\mathcal{O}(k \cdot d)$ nhờ kỹ thuật biến đổi đại số toán học của Steffen Rendle, thay vì $\mathcal{O}(d^2)$ thông thường.
*   Hoạt động cực kỳ hiệu quả trên các ma trận đặc trưng siêu thưa thớt (highly sparse data).

---

## 3. Các chỉ số đánh giá chất lượng (Evaluation Metrics)

### A. Đánh giá dự đoán điểm số (Regression Metrics)
Thích hợp cho tác vụ Explicit Feedback (dự đoán rating):
*   **Mean Absolute Error (MAE):**
    $$\text{MAE} = \frac{1}{|T|} \sum_{(u,i) \in T} |r_{u,i} - \hat{r}_{u,i}|$$
*   **Root Mean Squared Error (RMSE):** Phạt nặng hơn các lỗi dự đoán lớn.
    $$\text{RMSE} = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{u,i} - \hat{r}_{u,i})^2}$$
    *(Trong đó $T$ là tập dữ liệu kiểm thử - Test set).*

### B. Đánh giá xếp hạng danh sách (Ranking Metrics)
Thích hợp cho tác vụ Implicit Feedback hoặc Đề xuất danh sách Top-K:
*   **Precision@K:** Tỷ lệ sản phẩm được gợi ý thực sự có liên quan (relevant) trong top $K$ sản phẩm được đề xuất.
    $$\text{Precision@K} = \frac{|\text{Các bộ phim được gợi ý có liên quan trong Top K}|}{K}$$
*   **Recall@K:** Tỷ lệ sản phẩm có liên quan được hệ thống tìm thấy trong top $K$ đề xuất.
    $$\text{Recall@K} = \frac{|\text{Các bộ phim được gợi ý có liên quan trong Top K}|}{|\text{Tất cả các bộ phim có liên quan của người dùng}|}$$
*   **NDCG@K (Normalized Discounted Cumulative Gain):** Chỉ số đo lường chất lượng xếp hạng có tính đến vị trí của sản phẩm trong danh sách đề xuất. Sản phẩm có liên quan nằm ở vị trí càng cao thì điểm NDCG càng lớn.
    $$\text{DCG@K} = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$
    $$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
    *(Trong đó $\text{IDCG@K}$ là giá trị DCG lý tưởng nhất khi danh sách được sắp xếp hoàn hảo theo mức độ liên quan).*

---

## 4. Hướng dẫn thiết lập Evaluation Pipeline trong Code

Để đảm bảo kết quả đánh giá mô hình khách quan và chính xác, chúng ta cần tuân thủ quy trình sau:

```
[ Dữ liệu thô (MovieLens) ]
         │
         ▼
[ Tiền xử lý dữ liệu ] (Tách tập Train/Test theo tỷ lệ 80/20 hoặc theo thời gian)
         │
         ├───► Train Set (80%) ───► [ Huấn luyện mô hình (MF, FM, KNN) ]
         │                                      │
         ▼                                      ▼
   Test Set (20%) ───────────────► [ Tạo dự đoán hoặc xếp hạng Top-K ]
                                                │
                                                ▼
                                   [ Tính toán Metrics ]
                                   (RMSE, MAE, Recall@K, NDCG@K)
```

### Các thư viện Python phổ biến hỗ trợ:
*   **Surprise:** Thư viện Python chuyên biệt cho các mô hình Lọc cộng tác cổ điển (SVD, KNN, NMF, SlopeOne). Rất dễ sử dụng và tối ưu hiệu năng tốt.
*   **LightFM:** Thư viện tuyệt vời hỗ trợ cả Matrix Factorization và Factorization Machines trên cả dữ liệu Explicit lẫn Implicit.
