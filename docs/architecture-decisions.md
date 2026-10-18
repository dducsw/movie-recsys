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

---

## ADR-004: Two-Tower Neural Candidate Retrieval (DSSM)

### Context & Problem Statement
Averaging historical item vectors in collaborative filtering loses nonlinear user context, session recency, and cold-start metadata signals. 

### Decision
Introduce a dedicated **Two-Tower Neural Retrieval model** (`libs/recsys_core/src/recsys_core/two_tower.py`):
1. **User Tower**: Projects interaction history with exponential decay weighting, user genre distribution, activity level, and rating bias into a normalized 64-dimensional embedding $\mathbf{u} \in \mathbb{R}^{64}$.
2. **Item Tower**: Projects item semantic plot representations, genres, and metadata into the same 64-dimensional embedding $\mathbf{v} \in \mathbb{R}^{64}$.
3. **Pure NumPy Forward Pass**: Optimized with zero PyTorch runtime overhead in online SLA budget ($< 5\,\text{ms}$).

### Consequences
| Positive | Trade-offs |
| :--- | :--- |
| • Maps user context and content metadata into a unified metric space.<br/>• Seamless blending with iALS and BM25 via Reciprocal Rank Fusion (RRF). | • Requires maintaining dual-tower weights synchronization during CT cycles. |

---

## ADR-005: 11-Dimensional Standardized Feature Parity & Catalog Alignment

### Context & Problem Statement
Initial evaluation revealed two major discrepancies:
1. Low-vote outlier items artificially inflated ranker scores.
2. The movie catalog ID domain diverged from the MovieLens rating ID domain (only 5% ID overlap), causing extreme popularity bias where popularity gain accounted for >2100 while personalization was suppressed.

### Decision
1. **Catalog Domain Alignment**: Synchronize the movie catalog directly with the interaction domain (9,742 movies in `ml-latest-small`, achieving 100% ID coverage).
2. **11-Feature Standardization**: Extend `RANKING_FEATURES` to include `bayesian_rating` ($WR = \frac{v}{v+m}R + \frac{m}{v+m}C$), `genre_affinity_pct`, and `recency_decay_score` with strict training-serving parity across `features.py`, `train_pipeline.py`, and `recsys.py`.

### Consequences
| Positive | Trade-offs |
| :--- | :--- |
| • Personalization signal (`als_score`) feature gain increased by 800% (from 1,330 to 10,664).<br/>• Popularity bias reduced by 78%, restoring balanced recommendation slates.<br/>• 100% schema alignment across offline prototypes, CT pipelines, and online API serving. | • Feature matrix dimension increased from 8 to 11 columns, with negligible (< 0.2ms) latency impact. |
