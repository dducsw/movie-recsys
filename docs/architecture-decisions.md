# Architecture Decision Records (ADR)

This document records the architectural and engineering design decisions for **MovieNex**.

---

## ADR-001: Monorepo Layout Restructuring

### Context & Problem Statement
The codebase initially grew from research notebooks with all scripts located in a single `evaluation/` directory. This created several issues:
- Continuous training scripts were mixed with exploratory notebooks.
- Dependency lifecycles between backend serving (`FastAPI`) and training scripts were difficult to manage separately.
- Automated CI testing could not run clean isolation across packages.

### Decision
Reorganize the repository into standard monorepo top-level folders:

| Directory | Scope | Purpose |
| :--- | :--- | :--- |
| `apps/` | Applications | `apps/api` (FastAPI serving) and `apps/web` (React client). |
| `libs/` | Shared Packages | `libs/recsys_core/` shared between pipelines and serving API. |
| `pipelines/` | Data & ML Workflows | Continuous training, feature store ingestion, and batch jobs. |
| `notebooks/` | Research & EDA | Structured notebooks (01 EDA, 02 Classical, 03 Deep Learning, 04 Prototyping). |
| `deploy/` | Deployment | Docker Compose, service configurations, and Prometheus telemetry. |

### Consequences

| Positive | Trade-offs |
| :--- | :--- |
| • Clear boundaries between serving and training.<br/>• Simplified CI test runner setup.<br/>• Easy to publish `libs/recsys_core` as an internal package. | • Required updating legacy import paths and Docker compose file mounts. |

---

## ADR-002: Shared Core Package (`libs/recsys_core/`)

### Context & Problem Statement
Recommendation systems frequently suffer from **Training-Serving Skew** when feature calculations, normalization rules, and metric functions are implemented separately in training scripts and production web services.

### Decision
Extract all shared business and mathematical logic into `libs/recsys_core`:

| Module | Purpose | Key Components |
| :--- | :--- | :--- |
| `features.py` | Schema & Feature Extraction | Standard 8-feature schema, vectorized movie metadata extraction, user profile statistics. |
| `metrics.py` | Offline Evaluation | Standardized calculations for $HR@K$, $NDCG@K$, $MRR$, and Intra-List Diversity ($ILD$). |
| `reranking.py` | Slate Diversity | Maximal Marginal Relevance (MMR) and entropy-based exploration tuning. |
| `fusion.py` | Candidate Blending | Reciprocal Rank Fusion (RRF) and BM25 text relevance. |

### Consequences

| Positive | Trade-offs |
| :--- | :--- |
| • Guarantees identical feature transformations across train and serve time.<br/>• Allows fast unit testing without database or web server dependencies. | • Requires developers to install or link `recsys_core` in their environment (`pip install -e libs/recsys_core`). |

---

## ADR-003: Automated Continuous Training (CT) & Latency Quality Gate

### Context & Problem Statement
Models retrained with new interaction data could accidentally increase inference latency or alter feature schemas, causing production serving outages or SLA violations.

### Decision
Implement a 1-click Continuous Training pipeline (`pipelines/training/train_pipeline.py`) equipped with automated quality gates:
1. **Schema Check**: Verifies the fitted LightGBM model consumes the exact 8 standardized features.
2. **Latency SLA Gate**: Asserts that inference latency under negative-sampling benchmark remains $< 30.0\,\text{ms}$.
3. **Artifact Registry**: Exports validated weights to MLflow and S3 storage with feature importance logs.

### Consequences

| Positive | Trade-offs |
| :--- | :--- |
| • Fully reproducible retraining from `configs.yaml`.<br/>• Automatic safety check prevents deploying slow or invalid models.<br/>• Automated feature importance logging to MLflow. | • Negative sampling protocol evaluates a representative subset ($N=200$ users) to keep training fast. |
