# Movie Recommendation System (Movie RecSys)

A repository dedicated to researching, experimenting with, and building movie recommendation systems using the MovieLens dataset.

## 👥 Team Members

| Name | Department | Institution |
| :--- | :--- | :--- |
| **Le Dinh Duc** | Computer Science | Ho Chi Minh City University of Technology (HCMUT) |
| **Huynh Le Duy Khanh** | Information Technology | Ho Chi Minh City University of Science (HCMUS) |

---
## 🎯 Development Roadmap

### 📌 Phase 1: Algorithm Exploration & Modeling (Current)
This phase focuses on Exploratory Data Analysis (EDA) and implementing recommendation algorithms ranging from classic approaches to state-of-the-art models on the MovieLens dataset:

* **Machine Learning Approaches**
  * **Memory-Based Collaborative Filtering**: User-Based CF, Item-Based CF (using Cosine, Pearson correlation similarity).
  * **Model-Based Collaborative Filtering**: Matrix Factorization (Singular Value Decomposition (SVD), SVD++, Non-Negative Matrix Factorization (NMF), Alternating Least Squares (ALS)).
  * **Content-Based Filtering**: TF-IDF and Cosine Similarity on movie metadata (genres, tags).
  * **Clustering-Based**: K-Means Clustering, Co-Clustering.

* **Deep Learning Approaches**
  * **Neural Collaborative Filtering (NCF)**: General Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and NeuMF.
  * **Autoencoder-Based**: AutoRec (U-AutoRec & I-AutoRec), Variational Autoencoders (Multi-VAE, Multi-DAE).
  * **Feature-Interaction Models**: Wide & Deep, DeepFM (Deep Factorization Machine).

* **Transformer & Sequence-Based Approaches**
  * **Sequential Recommendation**: SASRec (Self-Attentive Sequential Recommendation), BERT4Rec (Bidirectional Sequential Recommendation).
  * **Transformer-Based Ranking**: BST (Behavior Sequence Transformer).

* **Knowledge Graph-Based Approaches**
  * **Path & Propagation Models**: RippleNet (User preference propagation).
  * **GNN-Based Models**: KGCN (Knowledge Graph Convolutional Networks), KGAT (Knowledge Graph Attention Network).

---

### 🚀 ML Pipeline (3-Stage Recommendation Engine)
Chúng tôi đã xây dựng và kiểm nghiệm thành công một Pipeline gợi ý hoàn chỉnh gồm 3 tầng theo tiêu chuẩn công nghiệp:
1. **Retrieval (Lọc thô)**: Kết hợp Implicit ALS và Content-Based TF-IDF.
2. **Ranking (Xếp hạng chi tiết)**: Dùng LightGBM LambdaRanker với Feature Engineering.
3. **Re-ranking (Đa dạng hóa)**: Áp dụng thuật toán MMR (Maximal Marginal Relevance).

👉 Chi tiết kiến trúc, các quyết định thiết kế và cách chạy được mô tả chi tiết tại [README.md của ML Pipeline](./evaluation/ml_pipeline/README.md).

---

### 📌 Phase 2: Production System Architecture
Transitioning from offline models to a production-grade, real-time recommendation application:
* **Backend API**: FastAPI for serving recommendation results.
* **Real-Time Streaming**: Apache Kafka for ingesting user interactions and streaming real-time logs.
* **Frontend UI**: React for displaying recommendations and capturing user feedback.
* **Feature Store**: Feast for storing and serving real-time user/item features.
* **MLOps Pipeline**: Kubeflow for automating training, evaluation, deployment, and monitoring.

---

## 📚 Research & References

- **Scientific Papers**: Academic research papers located in the research documentation.
- **Reference Repositories**: Community implementations and baseline configurations.



