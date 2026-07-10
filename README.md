# Movie Recommendation System (Movie RecSys)

A repository dedicated to researching, experimenting with, and building movie recommendation systems using the MovieLens dataset.

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



