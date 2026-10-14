# 🚀 Deep Learning in Recommendation Systems

Tài liệu này hướng dẫn chi tiết về việc áp dụng các kiến trúc Học sâu (Deep Learning) trong xây dựng Hệ gợi ý (Recommendation Systems - RecSys). Học sâu giúp giải quyết các mối quan hệ tương tác phi tuyến phức tạp và tích hợp hiệu quả thông tin phụ đa dạng.

---

## 📂 Cấu trúc Thư mục

```
evaluation/deep_learning/
├── README.md                      # Tài liệu lý thuyết & hướng dẫn thực thi
├── generate_dl_notebook.py        # Script tự động tạo/làm mới file Jupyter Notebook
├── dl_evaluation.ipynb            # Notebook thực nghiệm toàn bộ các mô hình DL
├── best_neumf_weights.weights.h5  # Trọng số checkpoint tối ưu của mô hình NeuMF
├── best_dfm_weights.weights.h5    # Trọng số checkpoint tối ưu của mô hình DeepFM
└── best_wd_weights.weights.h5     # Trọng số checkpoint tối ưu của mô hình Wide & Deep
```

---

## 1. Tại sao Sử dụng Deep Learning trong RecSys?

Mặc dù các phương pháp Học máy truyền thống (như SVD, FM) rất hiệu quả, chúng vẫn có một số giới hạn:
1.  **Mô hình hóa phi tuyến:** Các thuật toán truyền thống thường biểu diễn tương tác bằng tích vô hướng tuyến tính (Linear Dot Product). Học sâu sử dụng các hàm kích hoạt phi tuyến (ReLU, LeakyReLU) để học các tương tác bậc cao và phức tạp hơn giữa User và Item.
2.  **Side Information đa phương tiện:** Học sâu cho phép biểu diễn các đặc trưng dạng hình ảnh, văn bản (mô tả sản phẩm, review) dưới dạng các vector Embedding dày đặc.
3.  **Hành vi tuần tự (Sequential Dynamics):** Dễ dàng áp dụng các mô hình chuỗi (RNN, Transformer) để nắm bắt sự thay đổi sở thích của người dùng theo thời gian.

---

## 2. Các Kiến trúc Học sâu Tiêu biểu

### A. Neural Collaborative Filtering (NCF / NeuMF)
*   **Bài báo tiêu biểu:** [*Neural Collaborative Filtering*](https://dl.acm.org/doi/10.1145/3038912.3052569) (Xiangnan He et al. - WWW, 2017).

NeuMF kết hợp hai thành phần:
1.  **Generalized Matrix Factorization (GMF):** Sử dụng phép nhân Hadamard (phép nhân từng phần tử) giữa User Embedding và Item Embedding:
    $$\phi^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G$$
2.  **Multi-Layer Perceptron (MLP):** Nối (concatenate) User Embedding và Item Embedding, sau đó truyền qua nhiều tầng Dense với hàm kích hoạt ReLU để học tương tác phi tuyến:
    $$\phi^{MLP} = a_L \left( \mathbf{W}_L^T \cdot ... a_1(\mathbf{W}_1^T [\mathbf{p}_u^M, \mathbf{q}_i^M] + \mathbf{b}_1)... + \mathbf{b}_L \right)$$

Đầu ra của GMF và MLP được gộp lại ở tầng cuối cùng:
$$\hat{y}_{u,i} = \sigma \left( \mathbf{h}^T [\phi^{GMF}, \phi^{MLP}] \right)$$

---

### B. AutoRec (Autoencoders for Collaborative Filtering)
*   **Bài báo tiêu biểu:** [*AutoRec: Autoencoders Meet Collaborative Filtering*](https://dl.acm.org/doi/10.1145/2740908.2742726) (Suvash Sedhain et al. - WWW, 2015).

AutoRec sử dụng mạng Autoencoder tự mã hóa để nén vector đánh giá thưa thớt của user thành biểu diễn ẩn ở tầng nghẽn (bottleneck), sau đó khôi phục lại để dự đoán các điểm đánh giá bị khuyết. Điểm mấu chốt là hàm mất mát (loss function) chỉ tính toán trên các điểm đánh giá quan sát được (Masked MSE Loss):

$$\min_{\theta} \sum_{u} \sum_{i \in \mathcal{O}_u} (r_{u,i} - h(\mathbf{r}^{(u)}; \theta)_i)^2 + \frac{\lambda}{2} \left( \|\mathbf{W}\|_F^2 + \|\mathbf{V}\|_F^2 \right)$$

---

### C. Wide & Deep Learning
*   **Bài báo tiêu biểu:** [*Wide & Deep Learning for Recommender Systems*](https://dl.acm.org/doi/10.1145/2988450.2988454) (Heng-Tze Cheng et al. - ACM DLRS, 2016).

Kết hợp hài hòa giữa:
*   **Wide Component (Ghi nhớ):** Mô hình hồi quy tuyến tính trên các đặc trưng thô và đặc trưng giao cắt (cross-product features), tối ưu cho các quy luật xuất hiện thường xuyên.
*   **Deep Component (Khái quát hóa):** Mạng Feed-Forward Neural Network sâu đi kèm các vector Embedding, tối ưu cho việc khái quát các tương tác mới chưa từng xuất hiện.

---

## 3. Giao thức Đánh giá Chuẩn Học thuật (Evaluation Protocol)

*   **Time-Based Leave-One-Out (LOO):** Với mỗi người dùng, tương tác cuối cùng theo mốc thời gian (`timestamp`) được giữ lại làm tập kiểm thử (Ground Truth positive).
*   **Negative Sampling (1 + 99):** Với mỗi mẫu kiểm thử dương, hệ thống bốc ngẫu nhiên **99 bộ phim mà người dùng chưa từng tương tác** để tạo danh sách 100 ứng viên.
*   **Chỉ số Xếp hạng:**
    *   **Hit Ratio@10 (HR@10):** Đạt 1 nếu bộ phim đúng nằm trong Top 10 đề xuất; ngược lại bằng 0.
    *   **NDCG@10:** Đo lường độ chính xác có tính đến vị trí xếp hạng của bộ phim đúng.
    *   **MRR (Mean Reciprocal Rank):** Nghịch đảo thứ hạng của bộ phim đúng đầu tiên.

---

## 🚀 Hướng dẫn Thực thi Chi tiết

### Bước 1: Chuẩn bị Thư viện
```bash
pip install tensorflow keras pandas numpy matplotlib seaborn jupyter nbformat
```

### Bước 2: Khởi tạo/Làm mới Notebook (Tùy chọn)
```bash
python evaluation/deep_learning/generate_dl_notebook.py
```

### Bước 3: Mở và Chạy Thực nghiệm trên Jupyter
```bash
jupyter notebook evaluation/deep_learning/dl_evaluation.ipynb
```

**Các bước thực hiện trong notebook:**
1. Chuẩn bị dữ liệu từ `data/ml-latest-small/` và phân chia LOO.
2. Thực hiện Negative Sampling tỷ lệ 1:4 cho tập Train và 1:99 cho tập Test.
3. Xây dựng và huấn luyện mô hình **NCF (GMF, MLP, NeuMF)** bằng TensorFlow/Keras.
4. Xây dựng và huấn luyện mô hình **User-based AutoRec** với Masked Loss.
5. Xây dựng và huấn luyện mô hình **Wide & Deep**.
6. Tải các trọng số checkpoint tối ưu có sẵn (`best_neumf_weights.weights.h5`, `best_wd_weights.weights.h5`) để đánh giá nhanh.
7. So sánh hiệu năng giữa các kiến trúc bằng biểu đồ so sánh $HR@10$ và $NDCG@10$.
