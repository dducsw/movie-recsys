# MovieNex Recommendation System Architecture

Document mô tả chi tiết kiến trúc tổng thể, sơ đồ hệ thống và quy trình xử lý dữ liệu của hệ thống Movie Recommender System (MovieNex).

---

## 1. System Overview (Tổng quan Hệ thống)

MovieNex là hệ thống gợi ý phim 3 giai đoạn (3-Stage Recommendation Architecture) kết hợp giữa Offline Batch Processing, Real-time Event Streaming, Vector Database Search và LLM Chatbot Agent.

```mermaid
flowchart TD
    subgraph Frontend ["Frontend (React + Vite)"]
        UI[User Interface / Web App]
        Card[MovieCard + Impression Tracking]
        ChatUI[Chatbot Interface]
    end

    subgraph Backend ["Backend API (FastAPI MVC)"]
        AuthCtrl[Auth Controller]
        RecCtrl[Recsys Controller]
        ChatCtrl[Chatbot Controller]
        EventCtrl[Events Controller]
        
        RecEngine[3-Stage Recsys Engine]
        ChatGraph[LangGraph Chatbot Workflow]
        KafkaProducer[Async Kafka Event Producer]
    end

    subgraph Storage ["Data & Storage Layer"]
        PG[(PostgreSQL 15)]
        Redis[(Redis 7 Cache / Profile)]
        Qdrant[(Qdrant Vector DB)]
        S3[(SeaweedFS S3 Storage)]
        MLflow[MLflow Server]
    end

    subgraph Pipeline ["Big Data & Processing Pipeline"]
        Kafka{{Kafka Broker KRaft}}
        SparkBatch[Spark Batch ETL & ALS]
        SparkStream[Spark Structured Streaming]
    end

    UI --> AuthCtrl & RecCtrl & ChatCtrl & EventCtrl
    Card --> EventCtrl
    EventCtrl --> KafkaProducer
    KafkaProducer --> Kafka

    RecCtrl --> RecEngine
    RecEngine -->|1. Candidate Retrieval| Redis & Qdrant
    RecEngine -->|2. Ranking| S3
    ChatCtrl --> ChatGraph
    ChatGraph --> Redis

    Kafka --> SparkStream
    SparkStream -->|Update Trending| Redis

    SparkBatch -->|Sync Models & Features| S3 & Redis & Qdrant
    SparkBatch --> MLflow
    PG <--> AuthCtrl & RecCtrl
```

---

## 2. Recommendation Engine (3-Stage Engine)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as FastAPI Backend
    participant Redis as Redis Cache
    participant Qdrant as Qdrant Vector DB
    participant Model as LightGBM Ranker (S3)
    participant MMR as MMR Re-ranker

    User->>API: GET /api/recommendations (movie_ids)
    
    rect rgb(240, 240, 240)
        note over API,Qdrant: Stage 1: Candidate Retrieval (Pool ~100 movies)
        API->>Redis: Get precomputed similar & trending movies
        API->>Qdrant: Query nearest vectors using user/item centroid
        Qdrant-->>API: Return candidate embeddings & retrieval scores
    end

    rect rgb(230, 245, 230)
        note over API,Model: Stage 2: Feature Scoring & Ranking
        API->>Model: Build feature matrix (popularity, genres, ALS, user_bias)
        Model-->>API: Return LightGBM prediction scores
    end

    rect rgb(230, 230, 245)
        note over API,MMR: Stage 3: Diversity & Re-ranking
        API->>MMR: Maximal Marginal Relevance (lambda=0.7)
        MMR-->>API: Top N diverse recommendations
    end

    API-->>User: JSON Recommendation Response
```

---

## 3. Storage & Infrastructure Stack

| Services | Công nghệ | Vai trò / Nhiệm vụ |
|---|---|---|
| Database | PostgreSQL 15 | Lưu trữ người dùng, preferences, watchlist, ratings |
| Online Store | Redis 7 | Dynamic cache, realtime trending score, user profile & chat history |
| Vector DB | Qdrant | Vector embedding search cho content similarity |
| Object Storage | SeaweedFS (S3) | Lưu trữ model artifacts (LightGBM, CatBoost) |
| Streaming Broker | Apache Kafka (KRaft) | Nhận interaction events (click, impression, rating) |
| Compute | Apache Spark 3.5 | Batch processing, ALS factor calculation, Structured Streaming |
| Experiment | MLflow | Quản lý phiên bản model và tracking metrics |
