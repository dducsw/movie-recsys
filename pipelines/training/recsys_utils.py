"""
pipelines/training/recsys_utils.py
-----------------------------------
Notebook-only prototype helpers.

Everything algorithmically meaningful here now delegates to the production
library ``recsys_core`` — the single source of truth shared by training and
serving.  This module stays as a thin compatibility layer so the generated
notebooks (``notebooks/...``) keep their original import surface
(``from recsys_utils import ...``).

Notebook-only extensions that the core library deliberately does not carry
(``BM25.transform_from_bow`` / ``BM25.score_from_bow`` and the BM25-based
``cb_score`` in :func:`extract_user_features`) live here as small subclasses /
overrides on top of the core implementations.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    """Set random seed for reproducibility across random and numpy."""
    random.seed(seed)
    np.random.seed(seed)

try:
    from recsys_core.features import (
        RECENCY_DECAY_HALFLIFE_YEARS,
        build_ranking_row,
        extract_movie_release_year,
        reference_year,
    )
    from recsys_core.fusion import BM25 as _CoreBM25
    from recsys_core.fusion import reciprocal_rank_fusion as _core_rrf
    from recsys_core.metrics import (
        calculate_beyond_accuracy_metrics as _core_beyond_accuracy,
    )
    from recsys_core.metrics import (
        rmse_mae,
    )
    from recsys_core.reranking import calculate_user_lambda
except ImportError:  # pragma: no cover - notebooks may run without the package installed
    _root = os.path.dirname(os.path.abspath(__file__))
    while _root and not os.path.exists(os.path.join(_root, "libs", "recsys_core", "src")):
        _parent = os.path.dirname(_root)
        if _parent == _root:
            raise
        _root = _parent
    _core_src = os.path.join(_root, "libs", "recsys_core", "src")
    if _core_src not in sys.path:
        sys.path.insert(0, _core_src)
    from recsys_core.features import (
        RECENCY_DECAY_HALFLIFE_YEARS,
        build_ranking_row,
        extract_movie_release_year,
        reference_year,
    )
    from recsys_core.fusion import BM25 as _CoreBM25
    from recsys_core.fusion import reciprocal_rank_fusion as _core_rrf
    from recsys_core.metrics import (
        calculate_beyond_accuracy_metrics as _core_beyond_accuracy,
    )
    from recsys_core.metrics import (
        rmse_mae,
    )
    from recsys_core.reranking import calculate_user_lambda

# Decay rate of the recency feature (exp(-RECENCY_DECAY_RATE * years_ago)),
# shared by the inline notebook feature blocks so notebooks stay in exact
# parity with the core formula.
RECENCY_DECAY_RATE = np.log(2.0) / RECENCY_DECAY_HALFLIFE_YEARS

# Re-export surface for ``from recsys_utils import *`` in the generated notebooks.
__all__ = [
    "RECENCY_DECAY_HALFLIFE_YEARS",
    "RECENCY_DECAY_RATE",
    "BM25",
    "calculate_beyond_accuracy_metrics",
    "calculate_user_lambda",
    "evaluate_explicit",
    "evaluate_implicit_loo",
    "extract_user_features",
    "reciprocal_rank_fusion",
    "reference_year",
    "sample_train_data_implicit",
    "set_seed",
    "split_data_explicit",
    "split_data_implicit_leave_one_out",
]


# ==========================================
# 1. DATA SPLITTING
# ==========================================

def split_data_explicit(ratings_df, test_size=0.2, random_state=42):
    """
    Random train/test split for explicit feedback evaluation (rating prediction).
    """
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def split_data_implicit_leave_one_out(ratings_df, user_col='userId', item_col='movieId', timestamp_col='timestamp', seed=42):
    """
    Split dataset using Leave-One-Out protocol (last user interaction as test set).
    Uses unobserved catalog items as the negative pool (full-catalog evaluation) to prevent sampling bias.

    Returns:
        train_ratings: DataFrame containing training set
        test_data: list of tuples (user_id, positive_item_id, list of all negative_item_ids)
        user_interacted_items: dict {user_id: set of interacted item_ids}
    """
    set_seed(seed)

    # Sắp xếp theo timestamp
    ratings_sorted = ratings_df.sort_values(by=[user_col, timestamp_col])

    # Lấy chỉ mục dòng cuối cùng (tương tác cuối) của mỗi user
    last_indices = ratings_sorted.groupby(user_col).tail(1).index

    test_ratings = ratings_df.loc[last_indices].copy()
    train_ratings = ratings_df.drop(last_indices).copy()

    unique_items = ratings_df[item_col].unique()
    all_items_set = set(unique_items)

    # Dict lưu các sản phẩm user đã tương tác (cả train và test) để tránh lấy mẫu trùng lặp
    user_interacted_items = ratings_df.groupby(user_col)[item_col].apply(set).to_dict()

    test_data = []

    for _, row in test_ratings.iterrows():
        user = row[user_col]
        pos_item = row[item_col]
        interacted = user_interacted_items.get(user, set())

        # Full-catalog: lấy TẤT CẢ phim user chưa tương tác làm negatives
        neg_items = list(all_items_set - interacted)

        test_data.append((user, pos_item, neg_items))

    return train_ratings, test_data, user_interacted_items

# ==========================================
# 2. DYNAMIC NEGATIVE SAMPLING
# ==========================================

def sample_train_data_implicit(train_df, user_interacted_items, num_items, num_negatives=4, user_col='user_idx', item_col='movie_idx', seed=42):
    """
    Perform dynamic negative sampling for implicit training (pointwise).
    For each positive interaction, uniformly sample `num_negatives` unobserved items.
    Used for deep ranking architectures (NeuMF, Wide & Deep, DeepFM).
    """
    set_seed(seed)
    user_input, item_input, labels = [], [], []
    train_users = train_df[user_col].values
    train_items = train_df[item_col].values

    for u, i in zip(train_users, train_items):
        # Positive interaction (label 1)
        user_input.append(u)
        item_input.append(i)
        labels.append(1.0)

        # Negative interactions (label 0)
        interacted = user_interacted_items.get(u, set())
        for _ in range(num_negatives):
            neg_item = random.randint(0, num_items - 1)
            while neg_item in interacted:
                neg_item = random.randint(0, num_items - 1)
            user_input.append(u)
            item_input.append(neg_item)
            labels.append(0.0)

    return np.array(user_input), np.array(item_input), np.array(labels)

# ==========================================
# 3. EVALUATION METRICS (delegated to recsys_core)
# ==========================================

def evaluate_explicit(predictions):
    """
    Calculate RMSE and MAE for explicit rating predictions.
    predictions: list of tuples (true_rating, est_rating)
    """
    y_true = np.array([float(p[0]) for p in predictions])
    y_pred = np.array([float(p[1]) for p in predictions])
    return rmse_mae(y_true, y_pred)

def evaluate_implicit_loo(predictions_dict, k=10):
    """
    Evaluate Top-K ranking quality on Leave-One-Out test instances.
    predictions_dict: dict {user_id: [(item_id, est_score, is_positive), ...]}

    Returns:
        mean_hr: Hit Ratio@K
        mean_ndcg: NDCG@K
        mean_mrr: Mean Reciprocal Rank
    """
    hits = []
    ndcgs = []
    mrrs = []

    for uid, items in predictions_dict.items():
        # Sắp xếp theo score dự đoán giảm dần
        items.sort(key=lambda x: x[1], reverse=True)

        # Trích xuất danh sách item sau khi sắp xếp
        ranked_items = [item[0] for item in items]
        # Tìm item dương thực tế
        pos_items = [item[0] for item in items if item[2]]
        if not pos_items:
            continue
        pos_item = pos_items[0]
        # Vị trí của phim dương trong danh sách gợi ý (1-indexed)
        rank = ranked_items.index(pos_item) + 1

        # Hit Ratio @ K và NDCG @ K
        if rank <= k:
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)

        # MRR
        mrrs.append(1.0 / rank)

    if not hits:
        return 0.0, 0.0, 0.0
    return np.mean(hits), np.mean(ndcgs), np.mean(mrrs)

# ==========================================
# 4. BEYOND ACCURACY METRICS (delegated to recsys_core)
# ==========================================

def calculate_beyond_accuracy_metrics(recommendations, train_df, movies_df, movie_features=None, k=10, item_col='movieId'):
    """
    Calculate Intra-List Diversity, Novelty, and Catalog Coverage.
    (Delegates to ``recsys_core.metrics.calculate_beyond_accuracy_metrics``.)
    """
    return _core_beyond_accuracy(
        recommendations, train_df, movies_df,
        movie_features=movie_features, k=k, item_col=item_col,
    )

# ==========================================
# 5. RETRIEVAL & FUSION ALGORITHMS (delegated to recsys_core)
# ==========================================

class BM25(_CoreBM25):
    """
    Okapi BM25 ranking algorithm for sparse content-based candidate retrieval.

    Inherits the core implementation (shared with serving) and adds the
    notebook-only Bag-of-Words query helpers used by notebook 04.
    """

    def transform_from_bow(self, query_bow):
        """
        Tính BM25 cho toàn bộ corpus từ query dạng Bag-of-Words (sparse vector)
        """
        q_indices = self._bow_indices(query_bow)
        if len(q_indices) == 0:
            return np.zeros(self.tf.shape[0])

        tf_q = self.tf[:, q_indices].toarray()
        idf_q = self.idf[q_indices]

        denom = tf_q + self.k1 * (1.0 - self.b + self.b * self.doc_len[:, np.newaxis] / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return np.asarray(scores.sum(axis=1)).ravel()

    def score_from_bow(self, query_bow, doc_idx):
        """
        Tính BM25 cho 1 tài liệu cụ thể từ query dạng Bag-of-Words (sparse vector)
        """
        q_indices = self._bow_indices(query_bow)
        if len(q_indices) == 0:
            return 0.0

        tf_q = self.tf[doc_idx, q_indices].toarray().ravel()
        idf_q = self.idf[q_indices]

        doc_len = self.doc_len[doc_idx]
        denom = tf_q + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return scores.sum()

    @staticmethod
    def _bow_indices(query_bow):
        """Extract non-zero term indices from a sparse or dense BoW query vector."""
        import scipy.sparse as sp
        if sp.issparse(query_bow):
            q_indices = query_bow.nonzero()[1] if query_bow.ndim > 1 else query_bow.nonzero()[0]
        else:
            q_indices = np.where(np.asarray(query_bow).ravel() > 0)[0]
        return q_indices


def reciprocal_rank_fusion(als_candidates, bm25_candidates, k=60):
    """
    Merge ranked candidate lists from multiple retrievers using Reciprocal Rank
    Fusion (RRF).  (Delegates to ``recsys_core.fusion.reciprocal_rank_fusion``.)
    """
    return _core_rrf(als_candidates, bm25_candidates, k=k)


# ==========================================
# 6. RANKING FEATURE EXTRACTION (shared parity via recsys_core)
# ==========================================

def _bayesian_rating(vote_avg, vote_cnt):
    """Bayesian shrinkage of vote_average toward 5.0 with a prior weight of 25 votes."""
    vote_avg = float(vote_avg)
    vote_cnt = float(vote_cnt)
    return float((vote_cnt / (vote_cnt + 25.0)) * vote_avg + (25.0 / (vote_cnt + 25.0)) * 5.0)


def extract_user_features(user_id, movie_ids, train_ratings, movies_df, users_df, user_to_idx, movie_to_idx, als_model, bm25, is_train=False, target_mid=None):
    """
    Extract the 11 canonical ranking features for candidate movie_ids of user_id.

    The 9 shared features come from :func:`recsys_core.features.build_ranking_row`
    (single source of train/serve parity: canonical feature names — including
    ``retrieval_score`` — and the shared recency-decay formula).  ``cb_score``
    keeps the notebook-only BM25 semantics and overrides the genre-affinity
    value produced by the core row.  Prevents target leakage by excluding
    ``target_mid`` from the liked history when ``is_train=True``.
    """
    if not movie_ids:
        if is_train:
            return pd.DataFrame()
        return pd.DataFrame(), []

    # Đảm bảo index của movies_df và users_df đã được set
    movies_df_idx = movies_df if movies_df.index.name == 'movieId' else movies_df.set_index('movieId')
    users_df_idx = users_df if users_df.index.name == 'user_id' else users_df.set_index('user_id')

    user = users_df_idx.loc[user_id]

    # 1. Lịch sử xem (liked_ids)
    train_pos = train_ratings[(train_ratings['userId'] == user_id) & (train_ratings['rating'] >= 3.5)]
    liked_ids = train_pos['movieId'].tolist()

    fav_m_list = [int(m) for m in str(user.get("favorite_movies", "")).split("|") if str(m).isdigit()]

    # 2. Thể loại yêu thích (lấy từ lịch sử xem thực tế, tránh tĩnh)
    user_fav_genres = set()
    for lmid in liked_ids:
        if lmid in movies_df_idx.index:
            g_str = movies_df_idx.loc[lmid, "genres"]
            if pd.notna(g_str):
                user_fav_genres.update(str(g_str).split("|"))

    # Fallback nếu lịch sử trống
    if not user_fav_genres:
        for fid in fav_m_list:
            if fid in movies_df_idx.index:
                g_str = movies_df_idx.loc[fid, "genres"]
                if pd.notna(g_str):
                    user_fav_genres.update(str(g_str).split("|"))

    # Loại target ra khỏi lịch sử khi train (tránh target leakage)
    if is_train and target_mid is not None and target_mid in liked_ids:
        liked_ids = [lid for lid in liked_ids if lid != target_mid]

    # Giới hạn 20 phim gần nhất
    liked_ids = liked_ids[-20:]

    # 3. Tính cb_score (BM25 soup query; dùng liked_ids hoặc favorite_movies nếu cold start)
    user_bm25_scores = None
    query_mids = liked_ids if liked_ids else fav_m_list
    if query_mids:
        liked_soups = [movies_df_idx.loc[lid, 'soup'] for lid in query_mids if lid in movies_df_idx.index and 'soup' in movies_df_idx.columns]
        if liked_soups:
            user_bm25_scores = bm25.transform(" ".join(liked_soups))

    # 4. Shared profile passed to the unified feature builder
    u_idx = user_to_idx.get(user_id, None)
    profile = {
        "genres": user_fav_genres,
        "als_vec": als_model.user_factors[u_idx] if u_idx is not None else None,
        "activity": user.get('activity_level', 1.0),
        "bias": user.get('user_bias', 0.0),
    }

    rows = []
    valid_mids = []
    for mid in movie_ids:
        if mid not in movies_df_idx.index:
            continue
        valid_mids.append(mid)
        movie = movies_df_idx.loc[mid]
        m_idx = movie_to_idx.get(mid, None)

        meta = {
            "popularity": movie.get('popularity', 1.0),
            "vote_average": movie.get('vote_average', 5.0),
            "bayesian_rating": _bayesian_rating(
                movie.get('vote_average', 5.0),
                movie.get('vote_count', movie.get('popularity', 10.0)),
            ),
            "genres": set(str(movie['genres']).split('|')),
            "release_year": extract_movie_release_year(movie),
            "als_vec": als_model.item_factors[m_idx] if m_idx is not None else None,
        }
        row = build_ranking_row(meta, profile)

        # cb_score kiểu notebook: điểm tương đồng nội dung BM25, không phải affinity
        row["cb_score"] = float(user_bm25_scores[m_idx]) if (user_bm25_scores is not None and m_idx is not None) else 0.0
        rows.append(row)

    df_res = pd.DataFrame(rows)
    if is_train:
        return df_res
    return df_res, valid_mids
