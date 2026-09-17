# MovieNex — End-to-End Movie Recommendation Platform

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" /></a>
  <a href="https://lightgbm.readthedocs.io/"><img src="https://img.shields.io/badge/LightGBM-FF7F00?style=for-the-badge&logo=python&logoColor=white" alt="LightGBM" /></a>
  <a href="https://qdrant.tech/"><img src="https://img.shields.io/badge/Qdrant-DC2626?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" /></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" /></a>
  <a href="https://redis.io/"><img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis" /></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" /></a>
</p>

<<<<<<< HEAD
Một nền tảng gợi ý phim hoàn chỉnh đạt chuẩn công nghiệp (Production-Grade) kết hợp kiến trúc **3-Stage Recommendation Engine (Retrieval, Ranking, Re-ranking)**, tìm kiếm tương đồng vector đa chiều với **Qdrant**, giao diện streaming web hiện đại (**MovieNex**) và Trợ lý AI đàm thoại thông minh được xây dựng bằng **LangGraph**.
=======
MovieNex is an end-to-end movie recommendation platform designed around a standard 3-stage recommendation architecture: multi-channel candidate retrieval, gradient boosted decision tree ranking, and diversity-focused re-ranking. 

The project includes an online inference backend (FastAPI), a responsive web client (React), a shared core package (`recsys_core`) that prevents training-serving skew, and an automated continuous training (CT) pipeline with MLflow tracking.
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)

---

## 📸 Giao diện & Kiến trúc Chatbot AI (Preview)

<<<<<<< HEAD
### 1. Ứng dụng Web MovieNex
![MovieNex Web Application](assest/image.png)

### 2. Đồ thị Trạng thái Chatbot AI (LangGraph StateGraph)
![LangGraph Chatbot Workflow](assest/chatbot_graph.png)

---

## ✨ Tính năng Cốt lõi (Key Features)

- **Giao diện Streaming Hiện đại (MovieNex Client)**
  - Giao diện lấy cảm hứng từ Netflix/Prime Video xây dựng bằng **React 18 (Vite)** & **Tailwind CSS**.
  - Các danh mục động: *Trending Now*, *Dành riêng cho bạn (AI 3-Stage)*, *Top Đánh giá*, *Khám phá Thể loại*, và *Danh sách Yêu thích*.
  - Hỗ trợ chuyển đổi Dark / Light mode, modal khảo sát Onboarding giải quyết Cold Start, và ghi nhận sự kiện tương tác thời gian thực.

- **Kiến trúc Gợi ý 3 Lớp Chuẩn Công nghiệp (3-Stage RecSys)**
  - **Stage 1 — Candidate Retrieval (Lọc Thô)**: Kết hợp đa kênh giữa Lọc cộng tác ngầm định (Implicit ALS), Lọc nội dung dựa trên từ khóa (TF-IDF), và Tìm kiếm Vector dày đặc (Dense Vector Search qua **Qdrant**).
  - **Stage 2 — Scoring & Ranking (Xếp Hạng Chi Tiết)**: Mô hình **LightGBM LambdaRanker** & **CatBoost** tối ưu trực tiếp chỉ số NDCG dựa trên 8 đặc trưng kết hợp (User, Item, Cross interactions).
  - **Stage 3 — Re-ranking & Diversity (Đa Dạng Hóa)**: Thuật toán **Maximal Marginal Relevance (MMR)** cân bằng giữa độ chính xác và tính phong phú của thể loại, loại bỏ hiện tượng bong bóng lọc (Filter Bubble).

- **Trợ lý Ảo AI Đàm thoại Thông minh (Conversational Movie Assistant)**
  - Xây dựng trên **LangGraph StateGraph** với 4 node xử lý tuần tự kết hợp rẽ nhánh điều kiện: `detect_intent` $\rightarrow$ `query_movies` $\rightarrow$ `enrich_movies` $\rightarrow$ `generate_answer`.
  - Multi-provider LLM Factory: Ưu tiên **Google Gemini 2.5 Flash**, hỗ trợ fallback sang **OpenRouter**, **ZhipuAI** và bộ phân loại Heuristic ngoại tuyến.
  - Quản lý bộ nhớ hội thoại tự động lưu trong **Redis** với TTL 7 ngày.

- **Pipeline Xử lý Đặc trưng & MLOps Liên tục**
  - **Batch Pipeline** (`pipeline/batch_pipeline.py`): Trích xuất TF-IDF, huấn luyện Implicit ALS, tính ma trận Item-Item cosine similarity, đồng bộ Redis, Qdrant và Feast Feature Store.
  - **Streaming Pipeline** (`pipeline/stream_pipeline.py`): Lắng nghe Redis Stream `user-events-stream`, cập nhật điểm Trending và User Profile thời gian thực.
  - **1-Click Continuous Training (CT)** (`evaluation/train_pipeline.py`): Huấn luyện tự động, kiểm tra Quality Gate, ghi log thí nghiệm lên **MLflow Tracking Server** và lưu artifact lên **SeaweedFS S3**.

---

## 🏗️ Kiến trúc Tổng thể Hệ thống

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        UI["MovieNex Frontend<br/>(React + Vite + Tailwind CSS)"]
        ChatUI["AI Assistant Drawer<br/>(LangGraph Conversational Agent)"]
    end

    subgraph Gateway ["API & Service Layer"]
        API["FastAPI Backend Gateway (Port 8000)"]
        Agent["LangGraph Chatbot Workflow"]
=======
![MovieNex Web Application](assets/image.png)

---

## 🧭 System Overview

The application satisfies real-time serving latency constraints ($\le 50\,\text{ms}$) while supporting offline continuous training:

| Layer | Primary Tech | Responsibility |
| :--- | :--- | :--- |
| **Web Client** | React 19, Vite, Vanilla CSS | User interface, personalized discovery shelves, user preference collection. |
| **API Gateway** | FastAPI, Uvicorn, Pydantic v2 | Request routing, auth, telemetry ingestion, recommendation orchestration. |
| **Stage 1: Retrieval** | Implicit ALS, TF-IDF, Qdrant | High-recall candidate generation ($200\text{--}250$ items) under $15\,\text{ms}$. |
| **Stage 2: Ranking** | LightGBM (LambdaRank) | Listwise scoring across 8 standardized user, item, and cross features. |
| **Stage 3: Re-ranking** | Maximal Marginal Relevance (MMR) | Diversity optimization ($\lambda = 0.7$) to avoid genre-clustering filter bubbles. |
| **Data & Cache** | PostgreSQL 15, Redis 7 | User catalog, relational interactions, sub-millisecond feature lookup. |
| **Storage & Tracking** | SeaweedFS (S3), MLflow | Object storage for model binaries, metrics logging, and model registry. |
| **Core Package** | `libs/recsys_core` | Shared feature extractors, ranking metrics, and re-ranking algorithms. |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph ClientLayer ["Client Layer"]
        UI["MovieNex Web Client<br/>(React + Vite)"]
    end

    subgraph ServiceLayer ["API & Service Layer"]
        API["FastAPI Backend Gateway"]
        Agent["Conversational Assistant<br/>(LangGraph + Gemini)"]
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)
    end

    subgraph RecSysEngine ["3-Stage Recommendation Engine"]
        direction TB
<<<<<<< HEAD
        R1["Stage 1: Candidate Retrieval<br/>(Implicit ALS + TF-IDF + Qdrant Embeddings)"]
        R2["Stage 2: Scoring & Ranking<br/>(LightGBM LambdaRanker + Feature Store)"]
        R3["Stage 3: Diversity & Re-ranking<br/>(Maximal Marginal Relevance - MMR)"]
        R1 -->|Candidates: ~200 items| R2
        R2 -->|Ranked items| R3
    end

    subgraph Storage ["Data, Cache & Storage Layer"]
        DB[("PostgreSQL (Port 5435)<br/>Relational Database: movie_db")]
        Cache[("Redis (Port 6379)<br/>Feature Cache, Session Memory & Stream")]
        VectorDB[("Qdrant (Port 6333)<br/>Vector Database: movie_embeddings")]
        S3[("SeaweedFS (Port 8333)<br/>S3 Model Storage")]
        MLflow["MLflow Registry (Port 5000)<br/>Experiment Tracking"]
        Prometheus["Prometheus (Port 9090)<br/>Telemetry Monitoring"]
=======
        R1["Stage 1: Retrieval<br/>(iALS + TF-IDF + Qdrant HNSW)"]
        R2["Stage 2: Ranking<br/>(LightGBM LambdaRanker)"]
        R3["Stage 3: Re-ranking<br/>(Maximal Marginal Relevance)"]
        R1 -->|~200 Candidates| R2
        R2 -->|Ranked Scores| R3
    end

    subgraph DataStorage ["Data & Storage Layer"]
        DB[("PostgreSQL<br/>Catalog & Accounts")]
        Cache[("Redis<br/>Online Feature Store")]
        VectorDB[("Qdrant<br/>Plot Vector Embeddings")]
        S3[("SeaweedFS<br/>S3 Model Storage")]
        MLflow["MLflow<br/>Experiment Tracking"]
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)
    end

    UI <-->|"REST API (HTTP/JSON)"| API
    ChatUI <-->|"Chat API"| API
    API <--> Agent
    API -->|"Trigger 3-Stage Recs"| R1
    R3 -->|"Personalized Top-10"| API

    API <--> DB
    API <--> Cache
    R1 <--> VectorDB
    R1 <--> Cache
    R2 <--> Cache
    R2 <--> S3
    MLflow -.->|"Artifacts"| S3
```

---

<<<<<<< HEAD
## 🤖 Đồ thị Trạng thái Trợ lý AI (LangGraph StateGraph)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([__start__]):::first
    detect_intent(detect_intent)
    query_movies(query_movies)
    enrich_movies(enrich_movies)
    generate_answer(generate_answer)
    __end__([__end__]):::last
    __start__ --> detect_intent;
    detect_intent -.-> generate_answer;
    detect_intent -.-> query_movies;
    enrich_movies --> generate_answer;
    query_movies -.-> enrich_movies;
    query_movies -.-> generate_answer;
    generate_answer --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
=======
## 🎯 3-Stage Recommendation Pipeline

| Stage | Method / Algorithm | Key Inputs | SLA Budget | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Candidate Retrieval** | • Implicit ALS (latent factors)<br/>• TF-IDF Cosine Similarity<br/>• Dense Vector Search (Qdrant) | User history, movie genre metadata, plot synopsis embeddings | $< 15\,\text{ms}$ | ~200 candidate IDs |
| **2. Feature Ranking** | LightGBM LambdaRanker (`objective="lambdarank"`) | 8 standardized features:<br/>`popularity`, `vote_average`, `genre_overlap`, `release_year`, `user_activity`, `user_bias`, `als_score`, `cb_score` | $< 10\,\text{ms}$ | Scored & sorted candidates |
| **3. Diversity Re-ranking** | Maximal Marginal Relevance (MMR) | Stage 2 scores, genre vectors, trade-off parameter $\lambda = 0.7$ | $< 5\,\text{ms}$ | Final top-10 recommendation list |

---

## 📁 Repository Structure

```
movie-recsys/
├── apps/
│   ├── api/                        # FastAPI serving gateway (auth, recs, chatbot)
│   └── web/                        # React + Vite streaming user interface
├── libs/
│   └── recsys_core/                # Shared core library (features, metrics, MMR, fusion)
├── pipelines/
│   ├── training/                   # Continuous training pipeline, configs, integration tests
│   ├── feature_store/              # Feast feature definitions and materialization
│   ├── batch/                      # Offline batch pipelines and artifact synchronization
│   └── stream/                     # Nearline Redis stream processing
├── notebooks/
│   ├── 01_eda/                     # Exploratory Data Analysis
│   ├── 02_classical_ml/            # Classical benchmarks (SVD, ALS, NMF)
│   ├── 03_deep_learning/           # Neural CF, Autoencoders, Sequential models
│   └── 04_pipeline_prototypes/     # 3-stage prototyping and comparison experiments
├── docs/                           # Technical documentation and ADRs
├── assets/                         # Architecture diagrams and UI screenshots
├── Makefile                        # Common developer commands
└── docker-compose.yml              # Local infrastructure orchestration
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)
```

---

## 📁 Cấu trúc Thư mục Dự án

| Thư mục / Tệp tin | Mô tả chi tiết | Tài liệu |
| :--- | :--- | :--- |
| 📂 [`data/`](./data/README.md) | Dữ liệu crawler TMDB (2000-2026), bộ giả lập simulator, nạp PostgreSQL và tập dữ liệu MovieLens 100k | [README](./data/README.md) |
| ├── 📂 [`data/simulator/`](./data/simulator/README.md) | Bộ giả lập tương tác người dùng nâng cao (Time Decay, Gamma distribution, phễu hành vi, OLTP generator) | [README](./data/simulator/README.md) |
| 📂 [`docker/`](./docker/README.md) | Tệp cấu hình hạ tầng cho PostgreSQL, Redis, Qdrant, SeaweedFS S3, MLflow, Prometheus | [README](./docker/README.md) |
| 📂 [`docs/`](./docs/) | Tài liệu kỹ thuật chi tiết về kiến trúc, KPIs kinh doanh, thuật toán RecSys, ML pipeline và DevOps | [Thư mục Docs](./docs/) |
| 📂 [`evaluation/`](./evaluation/README.md) | Trung tâm nghiên cứu & đánh giá toàn diện mô hình học máy và học sâu | [README](./evaluation/README.md) |
| ├── 📂 [`evaluation/machine_learning/`](./evaluation/machine_learning/README.md) | Thực nghiệm ML truyền thống (User/Item CF, SVD, SVD++, NMF, Factorization Machines) | [README](./evaluation/machine_learning/README.md) |
| ├── 📂 [`evaluation/deep_learning/`](./evaluation/deep_learning/README.md) | Thực nghiệm Deep Learning (NCF/NeuMF, AutoRec, Wide & Deep) với checkpoint weights | [README](./evaluation/deep_learning/README.md) |
| ├── 📂 [`evaluation/ml_pipeline/`](./evaluation/ml_pipeline/README.md) | Bộ 6 notebooks liên kết thực thi trọn vẹn pipeline 3 tầng và đánh giá End-to-End | [README](./evaluation/ml_pipeline/README.md) |
| ├── 📂 [`evaluation/ml_training/`](./evaluation/ml_training/README.md) | Huấn luyện mô hình xếp hạng LightGBM và CatBoost kết hợp Optuna hyperparameter tuning | [README](./evaluation/ml_training/README.md) |
| └── 📄 [`evaluation/train_pipeline.py`](./evaluation/train_pipeline.py) | Script 1-Click Continuous Training tự động hóa huấn luyện, đánh giá LOO và ghi log MLflow | — |
| 📂 [`pipeline/`](./pipeline/README.md) | Batch feature pipeline, Streaming consumer (Redis stream), và Feast feature store materialization | [README](./pipeline/README.md) |
| 📂 [`web_app/`](./web_app/README.md) | Ứng dụng Full-stack MovieNex hoàn chỉnh | [README](./web_app/README.md) |
| ├── 📂 [`web_app/backend/`](./web_app/backend/README.md) | FastAPI Gateway (MVC + Services, Qdrant vector search, LangGraph chatbot, Prometheus telemetry) | [README](./web_app/backend/README.md) |
| └── 📂 [`web_app/frontend/`](./web_app/frontend/README.md) | Giao diện streaming React 18 (Vite, Tailwind CSS, Lucide icons, Dark/Light theme, Chatbot drawer) | [README](./web_app/frontend/README.md) |
| 📄 [`docker-compose.yml`](./docker-compose.yml) | Điều phối toàn bộ các dịch vụ hạ tầng, cơ sở dữ liệu và container ứng dụng | — |
| 📄 [`start-webapp.sh`](./start-webapp.sh) | Script khởi động 1-Click tự động cho Linux / macOS | — |
| 📄 [`start-webapp.ps1`](./start-webapp.ps1) | Script khởi động 1-Click tự động cho Windows PowerShell | — |

---

## ⚡ Hướng dẫn Khởi chạy Nhanh (Quick Start)

### 1. Chuẩn bị Mã nguồn & Môi trường

<<<<<<< HEAD
```bash
# Clone repository
git clone https://github.com/dducsw/movie-recsys.git
cd movie-recsys

# Tạo file cấu hình môi trường từ mẫu
=======
### 1. Prerequisites
- Docker & Docker Compose
- Python 3.10+ (tested on Python 3.12)
- Node.js 18+ and npm

### 2. Environment Configuration
```bash
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)
cp .env.example .env
```
Chỉnh sửa file `.env` nếu bạn muốn tích hợp LLM Key (Google Gemini, OpenRouter hoặc ZhipuAI) và TMDB API Key.

<<<<<<< HEAD
---

### 2. Khởi chạy WebApp 1-Click (Khuyên Dùng)

#### Cách A: Trên Linux / macOS
```bash
chmod +x start-webapp.sh
./start-webapp.sh
```

#### Cách B: Trên Windows (PowerShell)
```powershell
.\start-webapp.ps1
```
*(Script sẽ tự động khởi động Docker containers, tạo virtualenv backend, cài đặt dependencies npm và khởi chạy song song Backend + Frontend)*.

---

### 3. Khởi chạy Thủ công Từng Thành phần

Nếu bạn muốn kiểm soát chi tiết từng dịch vụ:

#### Bước 3.1: Khởi động Hạ tầng Lưu trữ (Docker Compose)
```bash
docker compose up -d postgres redis qdrant seaweedfs prometheus
```

#### Bước 3.2: Khởi tạo Dữ liệu Phim & Người dùng (Lần đầu chạy)
```bash
# Import dữ liệu phim vào PostgreSQL
python data/simulator/import_movies_to_db.py --csv_path data/crawler/movies_crawled.csv

# Sinh dữ liệu người dùng và lịch sử tương tác OLTP vào PostgreSQL
python data/simulator/generate_oltp_data.py --num_users 100

# Chạy Batch Feature Pipeline để tính similarity và đẩy embeddings lên Qdrant & Redis
python pipeline/batch_pipeline.py
```

#### Bước 3.3: Khởi chạy Backend FastAPI
```bash
cd web_app/backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
* API Server: http://localhost:8000
* Swagger Docs: http://localhost:8000/docs
* Prometheus Telemetry: http://localhost:8000/metrics

#### Bước 3.4: Khởi chạy Frontend React (Vite)
Mở một terminal mới:
=======
### 3. Launch Core Infrastructure
```bash
# Start PostgreSQL, Redis, Qdrant, SeaweedFS, and MLflow
make infra
# or: docker compose up -d
```

### 4. Run Tests & Validation
```bash
# Run test suite across apps, core library, and training pipeline
make test
```

### 5. Run Continuous Training
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)
```bash
# Execute the automated 1-click training and evaluation pipeline
make train
```
<<<<<<< HEAD
* Giao diện người dùng: **http://localhost:5173**

---

### 4. Khởi chạy Toàn bộ trong Docker (Containerized Mode)
Nếu muốn chạy cả Frontend và Backend hoàn toàn bên trong Docker mà không cần cài đặt Node.js hay Python cục bộ:
```bash
docker compose --profile app up -d --build
```

---

### 5. Huấn luyện Mô hình Học máy 1-Click (Continuous Training)
Để tái huấn luyện mô hình xếp hạng LightGBM mới nhất với dữ liệu tương tác hiện tại và ghi log vào MLflow:
```bash
# Đảm bảo MLflow đang chạy
docker compose up -d mlflow postgres seaweedfs

# Thực thi pipeline huấn luyện
python evaluation/train_pipeline.py
```
* Dashboard MLflow: **http://localhost:5000**

---

## 🌐 Danh mục Cổng Truy cập Mặc định

| Dịch vụ | Địa chỉ URL | Cổng | Mô tả |
| :--- | :--- | :--- | :--- |
| **MovieNex Web Client** | [http://localhost:5173](http://localhost:5173) | `5173` | Giao diện streaming người dùng |
| **FastAPI Backend API** | [http://localhost:8000](http://localhost:8000) | `8000` | REST API cho ứng dụng và hệ gợi ý |
| **Swagger UI Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | `8000` | Tài liệu API tương tác trực tiếp |
| **Qdrant Vector Dashboard** | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | `6333` | Quản lý Vector Collections và Search |
| **MLflow Experiment Tracking** | [http://localhost:5000](http://localhost:5000) | `5000` | Quản lý thí nghiệm và Model Registry |
| **Prometheus Telemetry** | [http://localhost:9090](http://localhost:9090) | `9090` | Thu thập metrics độ trễ và lưu lượng |
| **PostgreSQL Database** | `localhost:5435` | `5435` | CSDL quan hệ chính (`movie_db`) |
| **Redis Cache** | `localhost:6379` | `6379` | Online Feature Cache & Streams |
| **SeaweedFS S3 Storage** | [http://localhost:8333](http://localhost:8333) | `8333` | Kho lưu trữ Object tương thích AWS S3 |
=======

### 6. Start Applications

| Component | Directory | Commands | Default Port |
| :--- | :--- | :--- | :--- |
| **Backend API** | `apps/api` | `pip install -r requirements.txt`<br/>`uvicorn main:app --reload --port 8000` | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Frontend UI** | `apps/web` | `npm install`<br/>`npm run dev` | [http://localhost:5173](http://localhost:5173) |
| **MLflow UI** | Orchestrated | `make infra` | [http://localhost:5000](http://localhost:5000) |
| **Qdrant Dashboard** | Orchestrated | `make infra` | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) |

*(On Windows, you can also run `.\start-webapp.ps1` for automated local setup).*

---

## 🧪 Testing & Code Quality

The project maintains a test suite covering unit logic, integration flows, and API contracts:

| Test Scope | Target Path | Coverage Focus |
| :--- | :--- | :--- |
| **Core Algorithms** | `libs/recsys_core/tests/` | Feature extraction, ranking metrics ($HR$, $NDCG$, $ILD$), MMR re-ranking, RRF fusion. |
| **Training Pipeline** | `pipelines/training/tests/` | Retrieval factor generation, LightGBM schema compatibility, quality gate evaluation. |
| **API Endpoints** | `apps/api/tests/` | Route authentication, recommendation serving contracts, cold-start handling. |

To run the complete suite:
```bash
make test
```

---

## 📚 Technical Documentation

Detailed architectural and engineering guides are available in the [`docs/`](./docs) directory:

| Document | Topic | Description |
| :--- | :--- | :--- |
| 📘 [**System Architecture**](./docs/architecture.md) | Architecture & Serving | Service boundaries, latency budgets ($\le 50\,\text{ms}$), data flow, and caching strategy. |
| 📗 [**ML Pipeline Specification**](./docs/ml-pipeline.md) | 3-Stage Pipeline | Retrieval setup, feature engineering, LightGBM training, and LOO validation. |
| 📙 [**Applied RecSys & AI**](./docs/recsys-applied.md) | Algorithms & Math | Formulation of iALS, FM, LightGBM LambdaRank, MMR, and conversational agent. |
| 📓 [**Production & MLOps**](./docs/deployment.md) | Deployment & Ops | Container configuration, runbooks, healthchecks, and model registry lifecycle. |
| 📕 [**Business Requirements**](./docs/business-requirements.md) | Product & Metrics | User personas, journey maps, North Star metrics, and offline-to-online metric alignment. |
| 📔 [**Architecture Decisions**](./docs/architecture-decisions.md) | ADR Records | Rationale for monorepo layout, shared core library, and training-serving skew control. |

---

## 👥 Team & Engineering Scope

| Contributor | Focus Area | Affiliation |
| :--- | :--- | :--- |
| **Le Dinh Duc** | Big Data Engineer | Computer Science, Ho Chi Minh City University of Technology (HCMUT) |
| **Huynh Le Duy Khanh** | Data Scientist | Information Technology, Ho Chi Minh City University of Science (HCMUS) |

### 🎯 Key Engineering Contributions (Junior MLE Scope)

| Component | Area | Implementation Details |
| :--- | :--- | :--- |
| **`libs/recsys_core`** | Shared Core Package | Packaged reusable feature transformations, ranking metrics ($HR@K$, $NDCG@K$, $ILD$), and re-ranking algorithms (RRF, MMR) to eliminate training-serving skew. |
| **`pipelines/training`** | Continuous Training (CT) | Implemented 1-click training pipeline with Time-Based Leave-One-Out split, negative sampling, LightGBM LambdaRank training from YAML configs, and MLflow logging. |
| **Metric Quality Gates** | MLOps & Validation | Automated latency ($< 30\,\text{ms}$) and metric quality gates preventing regression before model artifact publishing. |
| **Serving Integration** | FastAPI Serving Gateway | Harmonized feature schemas across offline training and live serving in `apps/api/app/services/recsys.py`. |
| **Testing Infrastructure** | CI / Automation | Authored multi-layer automated test suite (46 tests) integrated into GitHub Actions CI. |
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)

---

## 📚 Tài liệu Kỹ thuật Chi tiết (`docs/`)

<<<<<<< HEAD
| Tài liệu | Lĩnh vực | Nội dung cốt lõi |
| :--- | :--- | :--- |
| 📘 [**System Architecture**](./docs/architecture.md) | Kiến trúc Hạ tầng | Quy trình phục vụ $\le 50\text{ms}$ p95, mô hình luồng dữ liệu Hybrid Lambda/Kappa, Redis feature cache, và Qdrant vector engine. |
| 📗 [**Business Requirements & KPIs**](./docs/business-requirements.md) | Nghiệp vụ & Chỉ số | Hành trình người dùng, chỉ số North Star, công thức toán học ($HR@K$, $NDCG@K$, $ILD$), và chiến lược Cold-Start. |
| 📙 [**Applied RecSys & Conversational AI**](./docs/recsys-applied.md) | Thuật toán & Trợ lý AI | Cơ sở toán học của iALS, Factorization Machines, LightGBM vs. CatBoost, MMR diversity, và cỗ máy trạng thái LangGraph. |
| 📕 [**ML Pipeline Specification**](./docs/ml-pipeline.md) | Kỹ thuật ML 3 Lớp | Stage 1 Multi-channel Retrieval, Stage 2 Feature Engineering & Ranking, Stage 3 MMR, và giao thức Time-Based LOO. |
| 📓 [**Production Deployment & MLOps**](./docs/deployment.md) | DevOps & Giám sát | Docker Compose orchestration, MLflow model registry, Quality Gates tự động, zero-downtime hot reload, và phát hiện data drift. |

---

## 👥 Nhóm Tác giả & Đóng góp

| Họ và Tên | Vai trò | Chuyên ngành | Trường Đại học |
| :--- | :--- | :--- | :--- |
| **Lê Đình Đức** | Big Data & RecSys Engineer | Khoa học Máy tính | Đại học Bách Khoa ĐHQG-HCM (HCMUT) |
| **Huỳnh Lê Duy Khánh** | Data Scientist / Co-Developer | Công nghệ Thông tin | Đại học Khoa học Tự nhiên ĐHQG-HCM (HCMUS) |

---

## 📄 Bản quyền (License)

Dự án được phân phối dưới giấy phép **MIT License** — xem tệp [LICENSE](LICENSE) để biết thêm thông tin chi tiết.
=======
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
>>>>>>> 7671402 (refactor folder structure and add 46 unit test for ml pipeline)
