# Production Deployment & MLOps Infrastructure

This document specifies the containerized deployment topology, CI/CD automation, model registry workflow, and telemetry observability stack for the **MovieNex** recommendation platform.

---

## 1. Containerized Service Topology

MovieNex is orchestrated via Docker Compose for local/staging environments and designed for seamless transition to Kubernetes (EKS/GKE).

```mermaid
flowchart TD
    subgraph Ingress ["Edge & Gateway Tier"]
        Nginx["Nginx Reverse Proxy / Load Balancer<br/>(Port: 80 / 443)"]
    end

    subgraph AppCluster ["Application Services (Profile: app)"]
        Frontend["movie-recsys-frontend<br/>(React 18 + Vite static build on Nginx :5173)"]
        Backend["movie-recsys-backend<br/>(FastAPI + Uvicorn Workers :8000)"]
    end

    subgraph DataCluster ["Data & Vector Infrastructure"]
        Postgres[("movie-recsys-db<br/>(PostgreSQL 15 :5435)")]
        Redis[("movie-recsys-redis<br/>(Redis 7 Alpine :6379)")]
        Qdrant[("movie-recsys-qdrant<br/>(Vector Engine :6333 / :6334)")]
    end

    subgraph MLOpsCluster ["Model & Storage Infrastructure"]
        SeaweedFS[("movie-recsys-seaweedfs<br/>(S3 API :8333 / Filer :8888)")]
        S3Init["seaweedfs-init<br/>(Auto Bucket Initializer)"]
        MLflow["movie-recsys-mlflow<br/>(Tracking & Registry :5000)"]
    end

    Nginx --> Frontend
    Nginx --> Backend
    Backend <--> Postgres
    Backend <--> Redis
    Backend <--> Qdrant
    Backend <--> SeaweedFS
    
    MLflow <--> Postgres
    MLflow <--> SeaweedFS
    S3Init -->|Creates Buckets| SeaweedFS
```

---

## 2. CI/CD & Model Promotion Lifecycle

```mermaid
flowchart LR
    subgraph Training ["1. Offline Training"]
        Data[("Feature Store & DB")] --> Train["Batch Pipeline<br/>(train_pipeline.py)"]
        Train --> Eval["LOO Evaluation<br/>(HR@10 & NDCG@10)"]
    end

    subgraph Governance ["2. Registry & Validation"]
        Eval --> Gate{"Pass Quality Gate?<br/>NDCG@10 > Baseline"}
        Gate -->|Yes| Register["MLflow Model Registry<br/>(Promote to 'Staging')"]
        Gate -->|No| Alert["Alert ML Engineer<br/>(Slack/Webhook)"]
    end

    subgraph Release ["3. Zero-Downtime Deployment"]
        Register --> S3Upload["Sync Binaries to SeaweedFS S3"]
        S3Upload --> HotReload["Signal FastAPI Hot Reload<br/>(/api/admin/reload-model)"]
        HotReload --> Serving["Active Live Traffic"]
    end
```

---

## 3. Environment Variables & Secret Configuration

Production environments are configured via `.env` files:

```bash
# Core Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DB=movie_db
DB_HOST=postgres
DB_PORT=5432

# Redis Cache & Online Feature Store
REDIS_HOST=redis
REDIS_PORT=6379

# Qdrant Vector Engine
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Object Storage (SeaweedFS S3-Compatible)
AWS_ACCESS_KEY_ID=seaweedfskey
AWS_SECRET_ACCESS_KEY=seaweedfssecret
SEAWEEDFS_FILER_URL=http://seaweedfs:8888
MLFLOW_S3_ENDPOINT_URL=http://seaweedfs:8333

# External APIs & LLM Providers
TMDB_API_KEY=your_tmdb_api_key
GEMINI_API_KEY=your_gemini_api_key
```

---

## 4. Production Runbook

### 4.1. Starting the Entire Infrastructure
```bash
# Launch storage, cache, vector DB, and MLflow
docker compose up -d postgres redis qdrant seaweedfs seaweedfs-init mlflow

# Launch full application stack (including Frontend & Backend)
docker compose --profile app up -d
```

### 4.2. Healthcheck & Diagnostic Verification
```bash
# Verify PostgreSQL readiness
docker exec -it movie-recsys-db pg_isready -U postgres -d movie_db

# Check Redis responsiveness
docker exec -it movie-recsys-redis redis-cli ping

# Check Qdrant cluster telemetry
curl -f http://localhost:6333/cluster/status

# Check SeaweedFS cluster status
curl -f http://localhost:9333/cluster/status
```

---

## 5. Monitoring & Observability Architecture

```mermaid
flowchart TD
    App["FastAPI Serving Pods"] -->|Prometheus Metrics| Prom["Prometheus Server"]
    App -->|Telemetry Logs| Kafka["Kafka Broker"]
    Kafka --> ELK["OpenSearch / Elasticsearch"]
    Prom --> Grafana["Grafana Dashboards"]
    ELK --> Grafana

    subgraph Dashboards ["Monitored Dimensions"]
        D1["System: QPS, Latency p50/p95/p99, Error Rate"]
        D2["ML: Pointwise Score Distribution, Prediction Drift"]
        D3["Business: Click-Through Rate (CTR), Dwell Time"]
    end

    Grafana --> Dashboards
```

1. **Service Metrics**: End-to-end inference latency, throughput (QPS), and cache hit ratios.
2. **Data & Prediction Drift**: Monitored by logging candidate score distributions. Significant distribution drift triggers automated retraining pipelines.
3. **Rollback Mechanism**: If online metric degradation is observed, FastAPI serving instances fall back to previous stable model versions directly from MLflow within $< 10$ seconds.
