# MovieNex System & Infrastructure Architecture

This document provides the architectural specification for **MovieNex**, an industrial-grade, full-stack movie recommendation and streaming platform. The architecture is engineered to satisfy low-latency online inference SLAs ($\le 50\,\text{ms}$ p95), resilient real-time interaction ingestion, multi-modal vector search, and scalable offline/nearline model lifecycle management.

---

## 1. High-Level System Architecture

MovieNex implements a **Hybrid Lambda/Kappa-inspired Recommender Architecture** that decouples heavy offline batch training from low-latency online inference and nearline event streaming.

```mermaid
flowchart TD
    subgraph ClientTier ["1. Client & Presentation Layer"]
        SPA["MovieNex Web Client<br/>(React 18 + Vite + Tailwind CSS)"]
        Tracker["Interaction & Telemetry Tracker<br/>(Impressions, Clicks, Dwell Time, Ratings)"]
        SPA --- Tracker
    end

    subgraph APITier ["2. API Gateway & Serving Layer"]
        Gateway["FastAPI Gateway<br/>(Asynchronous ASGI, Pydantic v2, Uvicorn)"]
        AuthSvc["Auth & Session Controller<br/>(JWT + Session State)"]
        RecSvc["Recommendation Service Controller<br/>(3-Stage Orchestration)"]
        ChatAgent["Conversational AI Agent<br/>(LangChain + LangGraph + Gemini API)"]
        EventProducer["Telemetry & Event Producer<br/>(Async Kafka Ingestion)"]

        Gateway --> AuthSvc
        Gateway --> RecSvc
        Gateway --> ChatAgent
        Gateway --> EventProducer
    end

    subgraph EngineTier ["3. 3-Stage Recommendation Engine (Online Inference)"]
        direction TB
        Retrieval["Stage 1: Multi-Channel Candidate Retrieval<br/>• Vector ANN Search (Qdrant)<br/>• Implicit ALS Candidate Pool<br/>• Content-Based TF-IDF Metadata Search"]
        Ranking["Stage 2: Scoring & Feature Ranking<br/>• LightGBM LambdaRanker<br/>• Real-time Feature Assembler"]
        Reranking["Stage 3: Business Logic & Re-ranking<br/>• Maximal Marginal Relevance (MMR)<br/>• Freshness Boost & De-duplication"]

        Retrieval -->|Top ~200 Candidates| Ranking
        Ranking -->|Ranked Scores| Reranking
    end

    subgraph DataTier ["4. Storage, Caching & Vector Tier"]
        Postgres[("PostgreSQL 15<br/>• Relational DB<br/>• User Profiles & Movie Catalog")]
        Redis[("Redis 7 In-Memory<br/>• Sub-ms Feature Store<br/>• Real-time Session Cache")]
        Qdrant[("Qdrant Vector DB<br/>• Dense Movie Embeddings<br/>• HNSW Vector Index")]
        S3[("SeaweedFS S3 Storage<br/>• Model Binaries & Encoders<br/>• Feature Dumps & Datasets")]
        MLflow["MLflow Server<br/>• Experiment Tracking<br/>• Model Registry & Lineage"]
    end

    subgraph AsyncTier ["5. Streaming & Batch MLOps Tier"]
        Kafka{{Apache Kafka Message Bus<br/>• Interaction Event Stream<br/>• Real-time Log Ingestion}}
        StreamWorker["Nearline Stream Processor<br/>• Real-time Sliding-Window Trending<br/>• User Real-time Interaction Vector"]
        BatchTrainer["Batch Training Pipeline<br/>• Spark / Scikit-Learn / LightGBM<br/>• LOO Evaluation & S3 Export"]
    end

    %% Client to API
    ClientTier -->|"HTTPS / REST API & SSE"| APITier

    %% Serving Interactions
    RecSvc --> EngineTier
    EngineTier <--> Redis
    EngineTier <--> Qdrant
    EngineTier <--> S3
    AuthSvc <--> Postgres
    ChatAgent <--> Redis
    ChatAgent <--> Qdrant

    %% Telemetry & Streaming
    EventProducer -->|"Publish Events"| Kafka
    Kafka --> StreamWorker
    StreamWorker -->|"Update Real-time Signals"| Redis
    StreamWorker -->|"Persist Interactions"| Postgres

    %% Batch Training
    Postgres -.->|"Batch Extract"| BatchTrainer
    BatchTrainer -->|"Log Runs & Metrics"| MLflow
    BatchTrainer -->|"Deploy Artifacts"| S3
    BatchTrainer -->|"Sync Vector Index"| Qdrant
```

---

## 2. End-to-End Online Serving Flow

The online recommendation path is optimized for sub-50ms execution. The sequence below details the dynamic execution flow when a user requests their personalized discovery feed:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client Browser
    participant API as FastAPI Gateway
    participant Cache as Redis (Feature Store)
    participant Qdrant as Qdrant Vector DB
    participant S3 as SeaweedFS / Model Memory
    participant LGBM as LightGBM Ranker
    participant MMR as MMR Re-ranker

    User->>API: GET /api/recommendations/for-you?user_id=1024&top_k=10
    
    rect rgb(245, 247, 250)
        note over API,Cache: Step 1: Session & Feature Retrieval (< 5ms)
        API->>Cache: Fetch User Real-time History, Activity Count & Bias
        Cache-->>API: User Context & Recent Interaction Vectors
    end

    rect rgb(240, 248, 255)
        note over API,Qdrant: Step 2: Multi-Channel Retrieval Pool (~15ms)
        par Implicit ALS Pool
            API->>Cache: Retrieve Top-100 ALS Precomputed Candidates
        and Dense Vector Similarity
            API->>Qdrant: Approximate Nearest Neighbor (HNSW) on User Preference Centroid
            Qdrant-->>API: Top-100 Semantic Similarity Candidates
        and Content-Based TF-IDF
            API->>Cache: Retrieve Metadata-matched Candidates (Director, Genres)
        end
        API->>API: Merge & Deduplicate Candidate Sets (Total ~200-250 items)
    end

    rect rgb(245, 255, 245)
        note over API,LGBM: Step 3: Online Feature Assembly & Ranking (~15ms)
        API->>Cache: Multi-get Item Features (Popularity, Vote Avg, Genres, Release Year)
        API->>API: Compute Cross-Features (Genre Overlap, Retrieval Channel Scores)
        API->>LGBM: Execute Batch Prediction (LightGBM C-API / In-Memory Inference)
        LGBM-->>API: Pointwise / Pairwise Relevance Scores
    end

    rect rgb(255, 250, 240)
        note over API,MMR: Step 4: Diversity Re-ranking & Deduplication (~5ms)
        API->>MMR: Compute Maximal Marginal Relevance (Lambda = 0.7)
        MMR-->>API: Top-K Diverse & High-Scoring Item IDs
    end

    API->>Cache: Hydrate Movie Metadata (Titles, Posters, Genres)
    API-->>User: 200 OK (JSON Feed + Telemetry Session Token)
```

---

## 3. Storage & Infrastructure Matrix

| Subsystem | Technology | Storage Type | SLA / Latency | Primary Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Relational Database** | PostgreSQL 15 (Alpine) | Persistent Block Store | $< 10\,\text{ms}$ | Source of truth for users, credentials, movie metadata, and explicit ratings. |
| **Online Feature Store** | Redis 7 (In-Memory) | In-Memory Key-Value | $< 1\,\text{ms}$ | Real-time sliding window telemetry, user preference vectors, session state, fast candidate cache. |
| **Vector Search Engine** | Qdrant | Vector Index (HNSW) | $< 15\,\text{ms}$ | High-dimensional dense embeddings of movie synopsis and plot vectors for semantic retrieval. |
| **Object Store** | SeaweedFS (S3 API) | Distributed Object Store | $< 25\,\text{ms}$ | Serialized machine learning models (`.joblib`, `.json`), TF-IDF matrices, and training datasets. |
| **Experiment Platform** | MLflow Server | SQLite / Postgres backend | N/A (Offline) | Model lineage, metric comparisons ($HR@K$, $NDCG@K$), hyperparameter logging, and model registry. |
| **Message Broker** | Apache Kafka (KRaft) | Distributed Commit Log | $< 5\,\text{ms}$ | High-throughput streaming ingestion of impression, click, and dwell-time events. |

---

## 4. Fault Tolerance & Reliability Design

1. **Graceful Degradation (Fallback Hierarchy)**:
   - If **Qdrant** is unreachable $\rightarrow$ Fallback to cached ALS offline candidate lists.
   - If **LightGBM Model Inference** times out $\rightarrow$ Fallback to pure retrieval score sorting + MMR.
   - If **Redis Feature Store** is degraded $\rightarrow$ Fallback to static global popularity rankings (`Trending Now`) served from PostgreSQL or static cache.
2. **Cold-Start Resilience**:
   - **New Users**: Onboarded via genre selection chips or instant contextual retrieval from initial exploratory clicks without requiring model retraining.
   - **New Movies**: Immediately indexed into Qdrant vector space via metadata TF-IDF embeddings, enabling instant discovery without historical interaction logs.
3. **Stateless Serving Nodes**:
   - The FastAPI backend gateway is completely stateless; model artifacts are loaded in memory upon startup and hot-reloaded via background worker signals when new model versions are registered in MLflow.

