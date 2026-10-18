# 3-Stage Recommendation Pipeline Specification

This document details the design and implementation of the **3-Stage Recommendation Pipeline** used in MovieNex. The pipeline decouples candidate retrieval, ranking, and diversity re-ranking to balance recommendation quality, catalog coverage, and response latency.

---

## 1. Pipeline Overview

```mermaid
flowchart TD
    RawData[("Interaction Logs & Aligned Catalog Metadata")] --> Prep["Step 1: Time-Based Leave-One-Out (LOO) Split<br/>(Data Drift Audit & PSI Validation)"]

    subgraph Stage1 ["Stage 1: Multi-Channel Candidate Retrieval"]
        direction TB
        Prep --> CB["1A. Content-Based TF-IDF / BM25<br/>(Genres, Cast, Director)"]
        Prep --> ALS["1B. Implicit ALS (iALS)<br/>(Interaction Confidence c = 1 + α·r)"]
        Prep --> VEC["1C. Vector Search (Qdrant)<br/>(Plot Synopsis Embeddings)"]
        Prep --> TT["1D. Two-Tower Neural Retrieval<br/>(DSSM User & Item Embeddings 64-d)"]
        
        CB -->|Top Candidates| Union["Candidate Fusion (RRF) & Deduplication<br/>(Top 200–300 Candidates)"]
        ALS -->|Top Candidates| Union
        VEC -->|Top Candidates| Union
        TT -->|Top Candidates| Union
    end

    subgraph Stage2 ["Stage 2: Scoring & Feature Ranking"]
        direction TB
        Union --> FE["Feature Transformation Engine (11 Dimensions)<br/>• Item Stats (popularity, vote_average, bayesian_rating, release_year)<br/>• User Context (user_activity, user_bias)<br/>• Cross & Decay (genre_overlap, genre_affinity_pct, recency_decay_score)<br/>• Retrieval Proxies (als_score, cb_score)"]
        FE --> LGBM["LightGBM LambdaRanker / CatBoost YetiRank<br/>(Pairwise NDCG Optimization with Mixed Negatives)"]
        LGBM --> RankedList["Scored & Sorted Candidates"]
    end

    subgraph Stage3 ["Stage 3: Diversity & Re-ranking"]
        direction TB
        RankedList --> MMR["Maximal Marginal Relevance (MMR)<br/>(Dynamic User Entropy-based λ in [0.4, 0.9])"]
        MMR --> Freshness["Nearline Signal Blending<br/>• Remove previously watched titles<br/>• Real-time Dwell Time & Impression Feedback"]
        Freshness --> FinalRecs["Final Top-10 Recommendations"]
    end

    FinalRecs --> Eval["Offline Metric Evaluation<br/>(HR@10, NDCG@10, ILD, Latency SLA < 15ms)"]
```

---

## 2. Stage Breakdown & Details

### Stage Summary

| Stage | Method | Input | Latency Budget | Target Output |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Retrieval** | TF-IDF / BM25, iALS, Qdrant HNSW, Two-Tower DSSM | User interaction history & item embeddings | $\le 15\,\text{ms}$ | 200–300 candidate IDs |
| **Stage 2: Ranking** | LightGBM LambdaRanker / CatBoost YetiRank | 11-feature tabular representation | $\le 10\,\text{ms}$ | Ranked candidate list |
| **Stage 3: Re-ranking** | Maximal Marginal Relevance (MMR) | Ranking scores & genre representation | $\le 5\,\text{ms}$ | Top-10 diverse slate |

---

### Step 1: Data Preparation, Drift Audit & Catalog Alignment
- **Time-Based Leave-One-Out (LOO)**: For each user, the latest interaction by timestamp is assigned to the test set, preventing temporal data leakage.
- **Data & Feature Drift Monitoring**: Audits distribution drift via Population Stability Index (PSI) and checks null rate thresholds before model training.
- **100% Catalog Domain Alignment**: Aligns movie catalog IDs directly with interaction domains (`ml-latest-small` catalog with 9,742 movies for 100% ID coverage) to eliminate ID divergence and popularity bias.

---

### Step 2: Multi-Channel Retrieval (Stage 1)

| Channel | Method | Description |
| :--- | :--- | :--- |
| **Content-Based** | TF-IDF + Cosine / BM25 | Builds token matrices from genre, director, and metadata keywords. Measures similarity with user preference profiles. |
| **Collaborative Filtering** | Implicit ALS (iALS) | Models implicit feedback with Hu-Koren-Volinsky confidence $c_{ui} = 1 + \alpha \cdot r_{ui}$ ($\alpha=40$), trained with `implicit.als` (SVD fallback offline). Computes latent dot products $\mathbf{x}_u^T \mathbf{y}_i$. |
| **Dense Semantic Search** | Qdrant HNSW Index | Queries pinned `all-MiniLM-L6-v2` 384-dim sentence embeddings of title+genres+overview to retrieve thematically relevant titles. |
| **Two-Tower Neural Retrieval** | DSSM Architecture (`two_tower.py`) | User Tower maps interaction recency decay and genre affinity; Item Tower projects semantic and metadata features into a shared 64-dim metric space. |

---

### Step 3: Feature Engineering & Ranking (Stage 2)

Candidates from Stage 1 are transformed into an **11-dimensional standardized feature vector** ensuring 100% parity across training and online serving (`libs/recsys_core/src/recsys_core/features.py`):

| Feature Name | Category | Description | Normalization / Scale |
| :--- | :--- | :--- | :--- |
| `popularity` | Item | Total user interaction volume for movie $i$. | Continuous value. |
| `vote_average` | Item | Average review score from dataset. | Scale $0.0\text{--}10.0$. |
| `bayesian_rating` | Item | IMDb Bayesian weighted rating $WR = \frac{v}{v+m} R + \frac{m}{v+m} C$ ($m=25, C=5.0$) taming low-vote anomalies. | Scale $0.0\text{--}10.0$. |
| `genre_overlap` | Cross | Count of overlapping genres between candidate and user profile. | Non-negative integer. |
| `genre_affinity_pct` | Cross | Genre overlap ratio normalized by user genre set size $\frac{\text{overlap}}{\max(\|U_{\text{genres}}\|, 1)}$. | Continuous value in $[0.0, 1.0]$. |
| `release_year` | Item | Release year of the movie. | Integer year (fallback: 2010). |
| `recency_decay_score` | Temporal | Exponential decay centered on 2026: $\exp(-0.046 \times (2026 - \text{year}))$. | Continuous value in $[0.0, 1.0]$. |
| `user_activity` | User | Number of interactions by user $u$. | Log-transformed: $\log(1 + N_{\text{ratings}})$. |
| `user_bias` | User | Mean rating of user $u$ minus catalog mean rating. | Zero-centered continuous value. |
| `als_score` | Retrieval | Dot product between user and item ALS latent factors. | Continuous score ($\approx -1.0\text{--}1.0$). |
| `cb_score` | Retrieval | Content relevance proxy score. | Continuous value in $[0.0, 1.0]$. |

#### Ranking Hyperparameters (`pipelines/training/configs.yaml`)

| Parameter | Value | Purpose |
| :--- | :--- | :--- |
| `objective` | `lambdarank` | Optimizes listwise ranking directly. |
| `metric` | `ndcg` | Evaluates normalized discounted cumulative gain. |
| `eval_at` | `[5, 10]` | Evaluation cut-offs for early stopping. |
| `n_estimators` | `100` | Maximum tree boosting iterations. |
| `learning_rate` | `0.05` | Shrinkage rate per tree update. |
| `num_leaves` | `15` | Controls model capacity and prevents overfitting. |
| `min_child_samples` | `5` | Minimum observations required per leaf. |

---

### Step 4: Diversity Re-ranking (Stage 3)

To prevent redundant recommendations (e.g., all superhero sequels), **Maximal Marginal Relevance (MMR)** selects items sequentially:

$$\text{MMR}(u, i, S) = \operatorname{argmax}_{i \in C \setminus S} \left[ \lambda \cdot \text{Score}(u, i) - (1 - \lambda) \cdot \max_{j \in S} \text{Sim}(i, j) \right]$$

- $C$: Ranked candidates from Stage 2.
- $S$: Slate of already selected items.
- $\text{Sim}(i, j)$: Jaccard similarity across genre vectors.
- $\lambda = 0.7$: Tuned trade-off parameter between relevance ($0.7$) and diversity ($0.3$).

---

## 3. Implementation Notebooks & Scripts

| Notebook / Script | Location | Description |
| :--- | :--- | :--- |
| Data Preparation | [`notebooks/04_pipeline_prototypes/01_data_preparation.ipynb`](../notebooks/04_pipeline_prototypes/01_data_preparation.ipynb) | LOO temporal split and interaction matrix assembly. |
| Content-Based Retrieval | [`notebooks/04_pipeline_prototypes/02_content_based_retrieval.ipynb`](../notebooks/04_pipeline_prototypes/02_content_based_retrieval.ipynb) | TF-IDF feature extraction and cosine candidate generation. |
| Collaborative Filtering | [`notebooks/04_pipeline_prototypes/03_collaborative_filtering.ipynb`](../notebooks/04_pipeline_prototypes/03_collaborative_filtering.ipynb) | Implicit ALS model tuning and factor export. |
| LightGBM Ranker | [`notebooks/04_pipeline_prototypes/04_lightgbm_ranker.ipynb`](../notebooks/04_pipeline_prototypes/04_lightgbm_ranker.ipynb) | Feature preparation and LambdaRank model training. |
| MMR Re-ranking | [`notebooks/04_pipeline_prototypes/05_mmr_reranking.ipynb`](../notebooks/04_pipeline_prototypes/05_mmr_reranking.ipynb) | MMR diversity tuning and Intra-List Distance (ILD) checks. |
| End-to-End Pipeline | [`notebooks/04_pipeline_prototypes/06_end_to_end_pipeline.ipynb`](../notebooks/04_pipeline_prototypes/06_end_to_end_pipeline.ipynb) | Complete 3-stage validation against baseline models. |
| Model Comparison | [`notebooks/04_pipeline_prototypes/07_catboost_vs_lightgbm.ipynb`](../notebooks/04_pipeline_prototypes/07_catboost_vs_lightgbm.ipynb) | Offline comparison between LightGBM and CatBoost YetiRank. |
| Continuous Training Script | [`pipelines/training/train_pipeline.py`](../pipelines/training/train_pipeline.py) | 1-click script for retraining, SLA verification, and MLflow logging. |
