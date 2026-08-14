# Movie Recommendation System with LightGBM & CatBoost

An end-to-end machine learning pipeline for **movie recommendation** that formulates recommendation as a **rating regression problem**: given a user and a movie, the system predicts the rating that the user is likely to give.

The predicted ratings are then used to generate personalized **Top-K recommendations**, evaluated using ranking metrics such as **Precision@K, Recall@K, and NDCG@K**.

The project provides a complete workflow including:

* Data loading and preprocessing
* User, movie, and interaction feature engineering
* Time-based train/validation/test splitting
* LightGBM and CatBoost model training
* Hyperparameter optimization with Optuna
* Regression and ranking evaluation
* Model comparison
* Top-K recommendation inference
* Model ensembling

---

## 1. Project Structure

```text
movie_recommender/
│
├── data/
│   ├── crawler/
│   │   └── movies_crawled.csv
│   │
│   └── ml-latest-small/
│       ├── clicks.csv
│       ├── movies.csv
│       └── ratings.csv
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── recommend.py
│
├── configs.yaml
├── requirements.txt
└── README.md
```

### Main Components

| File               | Description                                                    |
| ------------------ | -------------------------------------------------------------- |
| `data_loader.py`   | Loads and merges movie, rating, and crawler data               |
| `features.py`      | Generates user, movie, and user-movie interaction features     |
| `models.py`        | Defines LightGBM and CatBoost models                           |
| `train.py`         | Performs splitting, training, evaluation, and model comparison |
| `evaluate.py`      | Implements Precision@K, Recall@K, and NDCG@K                   |
| `tune.py`          | Performs hyperparameter optimization with Optuna               |
| `recommend.py`     | Generates personalized Top-K recommendations                   |
| `configs.yaml`     | Central configuration file                                     |
| `requirements.txt` | Python dependencies                                            |

---

## 2. Problem Formulation

The system treats recommendation as a **rating prediction problem**.

For a user $u$ and movie $m$, the model predicts:

$$
\hat{r}*{u,m} = f(u,m,X*{u,m})
$$

where:

* $u$ is the user
* $m$ is the movie
* $X_{u,m}$ represents user, movie, and interaction features
* $\hat{r}_{u,m}$ is the predicted rating

The predicted ratings are then used to rank candidate movies:

$$
m_1, m_2, \ldots, m_K
$$

with the highest predicted scores appearing first.

This allows the same model to be evaluated from both a **regression perspective** and a **recommendation/ranking perspective**.

---

## 3. Models

The pipeline compares two gradient-boosting models.

### LightGBM

LightGBM is a highly efficient gradient-boosting framework based on decision trees.

Advantages:

* Fast training and inference
* Efficient memory usage
* Good performance on tabular data
* Supports large datasets
* Leaf-wise tree growth

However, its leaf-wise growth strategy can cause overfitting, particularly on small datasets, so appropriate regularization and hyperparameter tuning are important.

### CatBoost

CatBoost is another gradient-boosting algorithm that provides strong support for categorical features.

Advantages:

* Native categorical feature handling
* Strong performance on tabular datasets
* Reduced need for extensive categorical preprocessing
* Robust performance with relatively little tuning

CatBoost may require more computational resources than LightGBM, but can provide competitive or better recommendation performance.

---

## 4. Prerequisites

Python **3.8+** is recommended.

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd movie_recommender
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 5. Data Preparation

Place the datasets under the `data/` directory according to the project structure.

The rating dataset should contain the following columns:

```text
userId,movieId,rating,timestamp
```

The expected file is:

```text
data/ml-latest-small/ratings.csv
```

For example:

```csv
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
1,6,4.0,964982224
2,1,5.0,978824268
```

The movie dataset should provide information such as:

```text
movieId,title,genres
```

Additional movie metadata can be loaded from:

```text
data/crawler/movies_crawled.csv
```

The exact features generated from these datasets are controlled by `src/features.py` and `configs.yaml`.

---

## 6. Pipeline

The complete pipeline consists of three main stages:

```text
                    ┌──────────────────┐
                    │   Raw Datasets   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Preparation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature          │
                    │ Engineering      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Time-based Split │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │   LightGBM    │             │   CatBoost    │
      └───────┬───────┘             └───────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    ┌──────────────────┐
                    │    Evaluation    │
                    │ RMSE / MAE       │
                    │ P@K / R@K / NDCG │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Recommendation   │
                    │     Top-K        │
                    └──────────────────┘
```

---

# 7. Step 1 — Hyperparameter Tuning

Hyperparameter tuning is optional but recommended.

Run:

```bash
python src/tune.py
```

The tuning process uses **Optuna** to search for suitable hyperparameters for LightGBM and CatBoost.

The best parameters are saved to:

```text
best_params.json
```

The number of optimization trials can be configured in:

```text
src/tune.py
```

For example:

```python
N_TRIALS = 50
```

Increasing the number of trials may improve the resulting hyperparameters but will increase training time.

If tuning is skipped, the training pipeline uses the default parameters specified in `configs.yaml`.

---

# 8. Step 2 — Training and Evaluation

Run:

```bash
python src/train.py
```

The training pipeline performs the following operations:

1. Loads the datasets
2. Merges movie and rating information
3. Generates features
4. Performs a time-based per-user split
5. Trains LightGBM
6. Trains CatBoost
7. Calculates regression metrics
8. Calculates ranking metrics
9. Compares the two models
10. Saves the trained models and related data

---

## 9. Time-Based Data Splitting

The system uses a **time-based per-user split** instead of a random split.

For each user, ratings are ordered chronologically:

```text
Old ratings ───────────────────────────► Recent ratings

       Train              Validation       Test
   ├───────────────┤    ├──────────┤    ├─────────┤
```

This better simulates a real recommendation scenario because the model learns from a user's historical behavior and predicts their future preferences.

The number of held-out interactions can be configured through:

```yaml
split:
  n_holdout_per_user: ...
```

This approach also helps reduce temporal data leakage.

---

# 10. Evaluation

The system evaluates the models from two perspectives.

## Regression Metrics

### RMSE

Root Mean Squared Error:

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
$$

Lower RMSE is better.

### MAE

Mean Absolute Error:

$$
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
$$

Lower MAE is better.

---

## Ranking Metrics

Because the ultimate goal is recommendation, regression metrics alone are not sufficient.

The system also evaluates:

### Precision@K

Measures the proportion of recommended movies that are relevant:

$$
Precision@K =
\frac{#\text{ relevant items in Top-K}}{K}
$$

Higher is better.

### Recall@K

Measures how many relevant items are successfully recommended:

$$
Recall@K =
\frac{#\text{ relevant items in Top-K}}
{#\text{ relevant items}}
$$

Higher is better.

### NDCG@K

Normalized Discounted Cumulative Gain gives higher importance to relevant items appearing near the top of the recommendation list.

$$
NDCG@K =
\frac{DCG@K}{IDCG@K}
$$

Higher is better, with a maximum value of 1.

---

# 11. Model Comparison

After training, the pipeline reports results for both models.

Example:

```text
================ Model Comparison ================

                RMSE      MAE      P@10    R@10    NDCG@10
LightGBM        0.82      0.61     0.32    0.18     0.29
CatBoost        0.79      0.59     0.35    0.20     0.32

Best Model: CatBoost
```

The project can select the best model based on **validation RMSE**, while ranking metrics are used to assess recommendation quality.

It is important to consider both objectives because a model with the lowest RMSE does not necessarily produce the best Top-K recommendation list.

---

# 12. Saved Model

After training, the pipeline saves the trained models and supporting information into:

```text
models.joblib
```

This file can contain:

* LightGBM model
* CatBoost model
* Feature information
* Training configuration
* Other objects required for inference

This allows the recommendation script to load the trained models without retraining them.

---

# 13. Step 3 — Generate Recommendations

After training, recommendations can be generated for a specific user.

Run:

```bash
python src/recommend.py <user_id>
```

For example:

```bash
python src/recommend.py 42
```

The system generates candidate movies, predicts the user's expected rating for each movie, ranks them by predicted score, and returns the Top-K movies.

Example output:

```text
   movieId                    title                    score
0       50       Usual Suspects, The                4.851234
1      318       Shawshank Redemption, The          4.782910
2      111       Taxi Driver                        4.650123
...
```

The number of returned recommendations can be configured using:

```yaml
recommend:
  top_k: 10
```

---

# 14. Model Ensembling

By default, the recommendation pipeline can combine the predictions of LightGBM and CatBoost.

For a user-movie pair $(u,m)$:

$$
\hat{r}_{ensemble}
==================

\frac{
\hat{r}*{LGBM}+
\hat{r}*{CAT}
}{2}
$$

The resulting score is then used for ranking.

Ensembling can improve recommendation stability because the two models may learn different patterns from the same feature set.

---

# 15. Configuration

The main configuration is centralized in:

```text
configs.yaml
```

Important configuration options include:

### Data Paths

Specify the locations of:

* Ratings
* Movies
* Crawled movie metadata
* Other datasets

### Data Splitting

Configure:

```yaml
split:
  n_holdout_per_user: ...
```

This determines how many recent interactions are reserved for validation and testing.

### Feature Engineering

The feature configuration can control parameters such as:

```yaml
features:
  top_n_directors: ...
  top_n_actors: ...
  genre_list:
    - Action
    - Adventure
    - Animation
    - Comedy
    - Drama
```

These settings determine which movie metadata and categorical information are included in the model.

### Recommendation

Configure:

```yaml
recommend:
  top_k: 10
```

to control the number of recommended movies.

---

# 16. Complete Workflow

A typical execution sequence is:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

```text
data/
├── crawler/
│   └── movies_crawled.csv
└── ml-latest-small/
    ├── clicks.csv
    ├── movies.csv
    └── ratings.csv
```

### 3. Tune hyperparameters

```bash
python src/tune.py
```

### 4. Train and evaluate models

```bash
python src/train.py
```

### 5. Generate recommendations

```bash
python src/recommend.py 42
```

---

# 17. Key Design Decisions

### Time-Based Splitting

Rather than randomly splitting interactions, the system uses chronological user-level splitting to better represent real-world recommendation.

### Feature-Based Recommendation

The models combine:

* User features
* Movie features
* User-movie interaction features
* Movie metadata
* Historical behavior

### Regression + Ranking Evaluation

The system does not rely solely on RMSE/MAE. Since the final application is a recommender system, Top-K ranking metrics are also reported.

### Multiple Models

Both LightGBM and CatBoost are trained to provide a direct comparison between two powerful gradient-boosting approaches.

### Ensemble Inference

The recommendation system can average model predictions to obtain a more robust final ranking.

---

# 18. Expected Results

The exact performance depends on the dataset, feature engineering, data split, and hyperparameters.

In general:

* **LightGBM** provides fast training and inference.
* **CatBoost** can perform strongly with categorical and structured movie features.
* **Hyperparameter tuning** can improve both regression and ranking performance.
* **Ensembling** may produce more stable recommendations than either individual model.

The final model should therefore be selected based not only on RMSE/MAE but also on the actual **Top-K recommendation quality**.

---

# 19. Future Improvements

Potential extensions include:

* Learning-to-rank objectives instead of pure rating regression
* Pairwise ranking losses
* Incorporating implicit feedback such as clicks
* User sequence modeling
* Collaborative filtering features
* Matrix-factorization embeddings
* Transformer-based recommendation models
* Candidate generation + ranking architecture
* More advanced negative sampling
* Online recommendation evaluation
* Diversity and novelty metrics
* Cold-start handling
* Real-time recommendation serving

---

## License

This project is intended for research, experimentation, and educational purposes.
