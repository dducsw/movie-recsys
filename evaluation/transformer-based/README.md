# Transformer-based & Generative Recommendation Systems

Tài liệu này hướng dẫn chi tiết về các hệ thống gợi ý dựa trên kiến trúc Transformer và xu hướng công nghệ đột phá của Hệ gợi ý tạo sinh (Generative Recommendation) kết hợp Mô hình ngôn ngữ lớn (LLMs) trong giai đoạn 2025 - 2026.

---

## 1. Sự chuyển dịch sang Transformer trong Gợi ý tuần tự (Sequential Recommendation)

Các phương pháp Collaborative Filtering truyền thống coi tương tác của người dùng là tĩnh (static). Tuy nhiên, sở thích của người dùng thay đổi liên tục theo thời gian. Hệ gợi ý tuần tự giải quyết vấn đề này bằng cách mô hình hóa hành vi người dùng dưới dạng một chuỗi thời gian: $S^u = (i_1^u, i_2^u, ..., i_{|S^u|}^u)$.

Sự ra đời của cơ chế **Self-Attention** (Tự chú ý) từ Transformer đã thay thế hoàn toàn các kiến trúc cũ như RNN (GRU4Rec) hay CNN (Caser) nhờ khả năng nắm bắt phụ thuộc dài hạn tốt hơn và hỗ trợ tính toán song song vượt trội.

### A. SASRec (Self-Attentive Sequential Recommendation)
*   **Bài báo tiêu biểu:** [*Self-Attentive Sequential Recommendation*](https://ieeexplore.ieee.org/document/8594844) (Wang-Cheng Kang, Julian McAuley - IEEE ICDM, 2018).

#### Kiến trúc mô hình
SASRec áp dụng kiến trúc Transformer Decoder một chiều (Unidirectional). Nó sử dụng cơ chế tự chú ý nhân quả (Causal Masking) để đảm bảo tại vị trí bước thời gian $t$, mô hình chỉ chú ý đến các vật phẩm ở các bước từ $1$ đến $t$ nhằm tránh rò rỉ thông tin tương lai:

$$\mathbf{S} = \text{Self-Attention}(\mathbf{E}) = \text{softmax} \left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} + \mathbf{M} \right) \mathbf{V}$$

Trong đó $\mathbf{M}$ là ma trận mặt nạ (masking matrix) với $M_{i,j} = -\infty$ nếu $i < j$ (ngăn cản việc chú ý đến tương lai) và $M_{i,j} = 0$ nếu ngược lại.

---

### B. BERT4Rec (Bidirectional Encoder Representations from Transformer)
*   **Bài báo tiêu biểu:** [*BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*](https://dl.acm.org/doi/10.1145/3357384.3357818) (Fei Sun, Jun Liu, Jianwu Wu, Changhua Pei, Xiao Lin, Ao Han, Linjian Bim - ACM CIKM, 2019).

#### Kiến trúc mô hình
Trái ngược với SASRec, BERT4Rec sử dụng kiến trúc Transformer Encoder hai chiều (Bidirectional). Tác giả lập luận rằng việc giới hạn mô hình chỉ nhìn về bên trái (quá khứ) giống SASRec là quá gò bó. Trong hành vi thực tế, sở thích của người dùng có thể bị ảnh hưởng chéo bởi các sản phẩm xung quanh.

BERT4Rec sử dụng phương pháp huấn luyện **Masked Language Model (Cloze Task)**: Mô hình ẩn ngẫu nhiên một số vật phẩm trong chuỗi lịch sử bằng ký tự đặc biệt `[mask]` và huấn luyện mạng neural dự đoán chính xác các vật phẩm bị ẩn đó dựa trên ngữ cảnh cả hai phía (trước và sau).

```
[ Input: i_1 ── i_2 ── [mask] ── i_4 ── [mask] ]
                 │         │       │       │
                 ▼         ▼       ▼       ▼
       [ Bidirectional Transformer Encoder ]
                 │         │       │       │
                 ▼         ▼       ▼       ▼
[ Output:       i_2       i_3     i_4     i_5    ]
```

---

## 2. Xu hướng 2026: Generative Recommendation & LLM-based RecSys

Bắt đầu từ năm 2024 và bùng nổ mạnh mẽ vào năm 2026, các hệ gợi ý đang chuyển dịch từ mô hình phân biệt truyền thống (Discriminative Models - tính điểm tương đồng vector) sang **Mô hình tạo sinh (Generative Models)** tận dụng sức mạnh của LLMs.

### A. Khung kiến trúc "Recommendation as Language Processing" (P5)
*   **Bài báo tiêu biểu:** [*Recommendation as Language Processing: Two-headed Giant or One Unified Model?*](https://arxiv.org/abs/2203.13366) (Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Zhu, Wang-Cheng Kang, Yongfeng Zhang - NeurIPS, 2022).

#### Ý tưởng cốt lõi
P5 đề xuất một mô hình Transformer dạng Text-to-Text duy nhất (như T5) để giải quyết tất cả các bài toán gợi ý bằng cách chuyển đổi chúng thành các câu lệnh ngôn ngữ tự nhiên (prompts).

```
[ Input Text Prompt ]
"Given the user history: Toy Story, Jumanji, Aladdin.
 What is the next movie the user is likely to watch?"
        │
        ▼
[ Unified Text-to-Text Transformer ]
        │
        ▼
[ Output Text ]
"The Lion King"
```

P5 thống nhất 5 tác vụ gợi ý phổ biến:
1.  **Sequential Recommendation:** Dự đoán sản phẩm tiếp theo.
2.  **Rating Prediction:** Dự đoán số sao người dùng sẽ chấm.
3.  **Explanation Generation:** Sinh câu văn giải thích lý do gợi ý.
4.  **Review Summarization:** Tóm tắt các đánh giá của người dùng.
5.  **Direct Retrieval:** Tìm kiếm sản phẩm phù hợp trực tiếp qua mô tả văn bản.

---

### B. LLM Instruction Tuning (TALLRec)
*   **Bài báo tiêu biểu:** [*TALLRec: An Effective and Efficient Tuning Framework for Aligning Large Language Models with Recommendation*](https://arxiv.org/abs/2305.00447) (Baochuan Li et al. - ACM RecSys, 2023).

#### Cơ chế hoạt động
Mặc dù các LLM thương mại lớn (như GPT-4) có khả năng gợi ý tốt dạng zero-shot, chúng lại có chi phí vận hành cực kỳ cao và không được tối ưu cho các tập dữ liệu người dùng cụ thể. TALLRec đề xuất tinh chỉnh hiệu quả tham số (Parameter-Efficient Fine-Tuning - PEFT như LoRA) trên các mô hình ngôn ngữ mã nguồn mở (như LLaMA-3, Mistral) bằng các tập dữ liệu chỉ dẫn gợi ý (recommendation instructions).

Quy trình gồm 2 giai đoạn:
1.  **Pre-tuning:** Giúp LLM làm quen với các khái niệm và danh mục sản phẩm.
2.  **Instruction Tuning:** Tinh chỉnh LLM để hiểu các câu lệnh như *"Dựa trên lịch sử xem phim của người dùng, phim X có phù hợp hay không? Trả lời Yes hoặc No"*.

---

### C. Generative Retrieval (Truy xuất tạo sinh)
*   **Ý tưởng đột phá 2025 - 2026:** Loại bỏ kiến trúc 2 giai đoạn truyền thống (Retrieval bằng Vector Search + Ranking bằng Multi-task MLP). Thay vào đó, một mô hình Generative Transformer duy nhất sẽ trực tiếp sinh ra các mã định danh của vật phẩm gợi ý.

#### Cách mã hóa sản phẩm (Semantic IDs)
Để Generative Transformer sinh ra sản phẩm chính xác, các sản phẩm cần được mã hóa thành các chuỗi ký tự có tính ngữ nghĩa hệ thống (Semantic IDs) thay vì ID số ngẫu nhiên.
*   *Ví dụ:* Bộ phim **Toy Story (1995)** có thể được gán mã ngữ nghĩa là `[Genre:Animation].[Subgenre:Children].[Year:1990s].[ID:01]`.
*   Mô hình Transformer sẽ sinh ra chuỗi mã này từng token một (Autoregressive Generation) dựa trên lịch sử tương tác của người dùng.

---

## 3. Đa phương tiện (Multimodal RecSys) & RAG trong Hệ gợi ý

### A. Multimodal Transformer (Gợi ý đa phương tiện)
Trong kỷ nguyên 2026, hệ gợi ý không chỉ dựa vào ID hay thuộc tính phân loại (categorical attributes). Các Transformer đa phương tiện (ví dụ: CLIP, Image/Video Transformers) được sử dụng để:
*   Trích xuất đặc trưng sâu từ ảnh bìa phim (movie posters), video trailer, và các bài review tự do của người dùng.
*   Ánh xạ tất cả các đặc trưng đa phương tiện này về chung một không gian Embedding với User.
*   **Giải quyết triệt để vấn đề Khởi đầu lạnh (Cold Start):** Khi có một phim mới chưa từng có tương tác rating, mô hình vẫn có thể gợi ý chính xác cho người dùng nhờ vào sự tương đồng về hình ảnh trailer hoặc mô tả nội dung phim.

### B. Retrieval-Augmented Generation (RAG) cho RecSys
RAG được ứng dụng để tăng tính chính xác và khả năng lý giải (explainability) của hệ gợi ý:
1.  **Retrieve:** Sử dụng một mô hình lọc nhanh (như Vector DB chứa embeddings của phim) để tìm ra top 50 phim liên quan nhất đến sở thích hiện tại của người dùng.
2.  **Augment:** Đưa danh sách 50 phim này kèm lịch sử chi tiết của user vào Prompt làm ngữ cảnh (context).
3.  **Generate:** LLM đóng vai trò là Ranker cuối cùng để chọn ra top 5 phim tốt nhất và sinh ra một đoạn văn bản giải thích sinh động lý do tại sao người dùng nên xem phim này (ví dụ: *"Chúng tôi gợi ý phim Interstellar cho bạn vì bạn rất thích phim Inception của đạo diễn Christopher Nolan và yêu thích chủ đề khoa học vũ trụ"*).

---

## 4. Các chỉ số đánh giá trong kỷ nguyên Generative RecSys

Bên cạnh các chỉ số xếp hạng truyền thống như NDCG@K và Hit Ratio@K (HR@K), các mô hình RecSys thế hệ mới yêu cầu thêm các chỉ số đánh giá văn bản và tính thuyết phục:

### A. Đánh giá tính lý giải (Explainability Metrics)
Đo lường chất lượng của văn bản giải thích lý do gợi ý được sinh ra bởi mô hình:
*   **BLEU (Bilingual Evaluation Understudy):** Đo mức độ trùng khớp các cụm từ (n-grams) giữa văn bản sinh ra và câu văn mẫu của người dùng thực tế.
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Đo mức độ đầy đủ của thông tin trong câu giải thích dựa trên tập tham chiếu.

### B. Đánh giá bằng LLM làm Giám khảo (LLM-as-a-Judge)
*   Sử dụng một LLM mạnh hơn (như GPT-4) để tự động chấm điểm các câu giải thích được sinh ra theo thang điểm 1-5 dựa trên các tiêu chí: Tính cá nhân hóa (Personalization), Tính chân thực (Factualness), và Khả năng thuyết phục (Persuasiveness).
