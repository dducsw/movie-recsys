# Deep Learning in Recommendation Systems

Tài liệu này hướng dẫn chi tiết về việc áp dụng các kiến trúc Học sâu (Deep Learning) trong xây dựng Hệ gợi ý (Recommendation Systems - RecSys). Học sâu giúp giải quyết các mối quan hệ tương tác phi tuyến phức tạp và tích hợp hiệu quả thông tin phụ đa dạng.

---

## 1. Tại sao sử dụng Deep Learning trong RecSys?

Mặc dù các phương pháp Học máy truyền thống (như SVD, FM) rất hiệu quả, chúng vẫn có một số giới hạn:
1.  **Mô hình hóa phi tuyến:** Các thuật toán truyền thống thường biểu diễn tương tác bằng tích vô hướng tuyến tính (Linear Dot Product). Học sâu sử dụng các hàm kích hoạt phi tuyến (activation functions) để học các tương tác phức tạp hơn giữa User và Item.
2.  **Side Information đa phương tiện:** Học sâu cho phép biểu diễn các đặc trưng dạng hình ảnh, văn bản (mô tả sản phẩm, review), và âm thanh dưới dạng các vector Embedding dày đặc.
3.  **Hành vi tuần tự (Sequential Dynamics):** Dễ dàng áp dụng các mô hình chuỗi (RNN, Transformer) để nắm bắt sự thay đổi sở thích của người dùng theo thời gian.

---

## 2. Các kiến trúc học sâu tiêu biểu và Cơ sở khoa học

### A. Neural Collaborative Filtering (NCF)
*   **Bài báo tiêu biểu:** [*Neural Collaborative Filtering*](https://dl.acm.org/doi/10.1145/3038912.3052569) (Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua - WWW, 2017).

#### Kiến trúc mô hình (NeuMF)
NCF đề xuất một khung kiến trúc chung kết hợp giữa:
1.  **Generalized Matrix Factorization (GMF):** Sử dụng phép nhân Hadamard (phép nhân phần tử) để mô phỏng Matrix Factorization truyền thống.
2.  **Multi-Layer Perceptron (MLP):** Sử dụng các tầng liên kết đầy đủ (dense layers) kết hợp hàm kích hoạt ReLU để học các tương tác phi tuyến tính phức tạp.

```
       [ Output Layer (Dự đoán \hat{y}_{u,i}) ]
                          ▲
                          │ (Liên kết đầy đủ)
              [ NeuMF Layer (Concatenate) ]
              /                         \
             /                           \
     [ GMF Layer ]                   [ MLP Layers ]
          ▲                                ▲
     (Element-wise)                   (Multi-layer)
      /         \                      /         \
 [User Emb]  [Item Emb]           [User Emb]  [Item Emb]
     ▲           ▲                     ▲           ▲
     │           │                     │           │
[ User ID ]  [ Item ID ]          [ User ID ]  [ Item ID ]
```

Hai thành phần GMF và MLP học các không gian embedding hoàn toàn độc lập, sau đó đầu ra được gộp lại ở tầng cuối cùng (NeuMF Layer):

$$\phi^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G$$
$$\phi^{MLP} = a_L \left( \mathbf{W}_L^T \cdot ... a_1(\mathbf{W}_1^T [\mathbf{p}_u^M, \mathbf{q}_i^M] + \mathbf{b}_1)... + \mathbf{b}_L \right)$$
$$\hat{y}_{u,i} = \sigma \left( \mathbf{h}^T [\phi^{GMF}, \phi^{MLP}] \right)$$

*Trong đó $\odot$ là phép nhân element-wise, $\sigma$ là hàm sigmoid, và $\mathbf{h}^T$ là trọng số của tầng đầu ra.*

#### Hàm mục tiêu (Loss Function)
Vì NCF nhắm đến dữ liệu tương tác ngầm định (Implicit Feedback), mô hình sử dụng hàm **Binary Cross-Entropy Loss** (hay còn gọi là log loss):

$$\mathcal{L} = -\sum_{(u,i) \in \mathcal{Y} \cup \mathcal{Y}^-} y_{u,i} \log \hat{y}_{u,i} + (1 - y_{u,i}) \log (1 - \hat{y}_{u,i})$$
*Trong đó $\mathcal{Y}$ là tập các tương tác dương (positive), và $\mathcal{Y}^-$ là tập tương tác âm (negative) được lấy mẫu ngẫu nhiên (Negative Sampling).*

---

### B. AutoRec (Autoencoders for Collaborative Filtering)
*   **Bài báo tiêu biểu:** [*AutoRec: Autoencoders Meet Collaborative Filtering*](https://dl.acm.org/doi/10.1145/2740908.2742726) (Suvash Sedhain, Aditya Krishna Menon, Srivatsan Sridharan, Alex Smola - WWW, 2015).

#### Nguyên lý hoạt động
AutoRec là mô hình hệ gợi ý dựa trên mạng Autoencoder tự mã hóa. Mục tiêu của Autoencoder là nén vector đầu vào thưa thớt thành một biểu diễn ẩn không gian thấp (tầng BottleNeck), sau đó khôi phục lại đầu ra đầy đủ (dự đoán các điểm ratings bị khuyết).

Có hai phiên bản:
1.  **Item-based AutoRec (I-AutoRec):** Đầu vào là vector đánh giá của tất cả user cho một phim $i$ ($\mathbf{r}^{(i)} \in \mathbb{R}^M$).
2.  **User-based AutoRec (U-AutoRec):** Đầu vào là vector đánh giá của một user $u$ cho tất cả phim ($\mathbf{r}^{(u)} \in \mathbb{R}^N$).

Hàm khôi phục điểm đánh giá của I-AutoRec:
$$h(\mathbf{r}^{(i)}; \theta) = f \left( \mathbf{W} \cdot g(\mathbf{V}\mathbf{r}^{(i)} + \mu) + \mathbf{b} \right)$$
Trong đó $\mathbf{V}$ là ma trận trọng số từ Input sang Hidden layer, $\mathbf{W}$ là ma trận trọng số từ Hidden sang Output layer.

#### Hàm mục tiêu tối ưu
Chỉ tối ưu hóa sai số trên các điểm đánh giá đã biết (quan sát được):
$$\min_{\theta} \sum_{i=1}^N \sum_{u \in \mathcal{O}_i} (r_{u,i} - h_{u}(\mathbf{r}^{(i)}; \theta))^2 + \frac{\lambda}{2} \left( \|\mathbf{W}\|_F^2 + \|\mathbf{V}\|_F^2 \right)$$

---

### C. Wide & Deep Learning
*   **Bài báo tiêu biểu:** [*Wide & Deep Learning for Recommender Systems*](https://dl.acm.org/doi/10.1145/2988450.2988454) (Heng-Tze Cheng et al. - ACM DLRS, 2016).

#### Ý tưởng cốt lõi
Mô hình đề xuất sự kết hợp của hai thành phần:
1.  **Wide Component (Memorization - Trí nhớ):** Sử dụng mô hình hồi quy tuyến tính trên các đặc trưng thô và các đặc trưng giao cắt (cross-product transformations). Rất tốt trong việc ghi nhớ các quy luật tương tác đặc trưng xuất hiện thường xuyên trong dữ liệu lịch sử.
2.  **Deep Component (Generalization - Khái quát hóa):** Sử dụng mạng Feed-Forward Neural Network sâu đi kèm các vector Embedding để học các mối quan hệ đặc trưng chưa từng xuất hiện trong lịch sử tương tác.

```
                  [ Output prediction ]
                            ▲
                            │
               [ Joint Training (Log-odds) ]
              /                             \
     [ Wide Component ]              [ Deep Component ]
      (Linear Model)                (Deep Neural Network)
           ▲                                ▲
    (Cross-product)                   (Continuous &
    sparse features                   embedding vectors)
```

Công thức dự đoán của mô hình:
$$P(Y = 1 | \mathbf{x}) = \sigma \left( \mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T a^{(L)} + b \right)$$

---

### D. Self-Attentive Sequential Recommendation (SASRec)
*   **Bài báo tiêu biểu:** [*Self-Attentive Sequential Recommendation*](https://ieeexplore.ieee.org/document/8594844) (Wang-Cheng Kang, Julian McAuley - IEEE ICDM, 2018).

#### Ý tưởng cốt lõi
Hầu hết các hệ gợi ý tuần tự trước đây sử dụng mạng RNN (như GRU4Rec) hoặc CNN để mô hình hóa hành vi. Tuy nhiên, RNN khó học được các mối quan hệ dài hạn (long-term dependency). SASRec áp dụng cơ chế tự chú ý (Self-Attention) từ Transformer để mô hình hóa chuỗi tương tác của người dùng.

#### Cách thức hoạt động
*   Mô hình gán quyền số chú ý (Attention weights) khác nhau cho các sản phẩm người dùng đã xem trong quá khứ để đưa ra quyết định đề xuất sản phẩm tiếp theo.
*   Cơ chế Self-Attention giúp mô hình tập trung vào cả sở thích ngắn hạn (các sản phẩm vừa click gần đây) và sở thích dài hạn (các sản phẩm click từ lâu nhưng có liên quan mật thiết).
*   Tốc độ huấn luyện nhanh hơn RNN rất nhiều vì cơ chế Attention cho phép xử lý song song toàn bộ chuỗi tương tác.

---

## 3. Các chỉ số đánh giá chất lượng (Evaluation Metrics)

Khi đánh giá hệ gợi ý học sâu với Phản hồi ngầm định, chúng ta thường sử dụng phương pháp **Leave-One-Out**:
Với mỗi người dùng, ta chọn ra tương tác cuối cùng (hoặc một tương tác dương ngẫu nhiên) để làm tập test. Sau đó trộn tương tác dương này với $N$ tương tác âm (negative items) khác để tạo danh sách kiểm thử. Nhiệm vụ của mô hình là xếp hạng tương tác dương này càng cao càng tốt trong danh sách.

*   **Hit Ratio@K (HR@K):** Đo lường xem sản phẩm test thực tế (positive item) có xuất hiện trong danh sách Top-K đề xuất hay không. Nếu có thì tính là 1 (Hit), ngược lại là 0.
    $$\text{HR@K} = \frac{\#\text{Hits}}{\#\text{Users}}$$
*   **NDCG@K:** Tương tự như trong Machine Learning truyền thống, đo lường vị trí xuất hiện của sản phẩm test. Nếu sản phẩm test xuất hiện ở vị trí đầu tiên, điểm NDCG sẽ cao hơn nhiều so với khi nó xuất hiện ở cuối danh sách Top-K.
*   **MRR (Mean Reciprocal Rank):** Đo lường nghịch đảo vị trí xếp hạng của sản phẩm test đầu tiên.
    $$\text{MRR} = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{rank_u}$$

---

## 4. Cấu trúc Triển khai PyTorch pipeline (Khái niệm)

Một pipeline cơ bản cho mô hình NCF trên PyTorch thường bao gồm các thành phần sau:

```python
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_gmf, latent_dim_mlp, mlp_hidden_dims):
        super(NeuMF, self).__init__()
        
        # GMF Embeddings
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, latent_dim_gmf)
        
        # MLP Embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)
        
        # MLP Layers
        mlp_layers = []
        input_dim = latent_dim_mlp * 2
        for dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=0.2))
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final NeuMF prediction layer
        final_input_dim = latent_dim_gmf + mlp_hidden_dims[-1]
        self.prediction_layer = nn.Linear(final_input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        # GMF Branch
        user_gmf = self.user_embed_gmf(user_indices)
        item_gmf = self.item_embed_gmf(item_indices)
        phi_gmf = user_gmf * item_gmf # Element-wise product
        
        # MLP Branch
        user_mlp = self.user_embed_mlp(user_indices)
        item_mlp = self.item_embed_mlp(item_indices)
        phi_mlp = torch.cat([user_mlp, item_mlp], dim=-1) # Concatenation
        phi_mlp = self.mlp(phi_mlp)
        
        # Concatenate GMF and MLP outputs
        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        output = self.prediction_layer(fusion)
        return self.sigmoid(output).squeeze()
```
Tập tin [eda_analysis.ipynb](../../eda_analysis.ipynb) đã chứng minh rằng các phân tích sâu sắc từ dữ liệu EDA là tiền đề quan trọng để xác định số chiều Latent Dimensions, tỷ lệ lấy mẫu âm (Negative Sampling ratio), cũng như xử lý tính thưa thớt của ma trận trước khi áp dụng cấu trúc PyTorch phía trên.
