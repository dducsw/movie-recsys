import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed=42):
    """Thiết lập random seed để tái lập kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)

# ==========================================
# 1. PHÂN CHIA DỮ LIỆU (DATA SPLITTING)
# ==========================================

def split_data_explicit(ratings_df, test_size=0.2, random_state=42):
    """
    Chia dữ liệu ngẫu nhiên cho luồng Explicit Feedback (dự đoán ratings).
    """
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def split_data_implicit_leave_one_out(ratings_df, user_col='userId', item_col='movieId', timestamp_col='timestamp', seed=42):
    """
    Chia tập dữ liệu theo phương pháp Leave-One-Out (tương tác cuối cùng của mỗi user làm Test).
    Với mỗi tương tác dương ở tập Test, trộn thêm 99 bộ phim ngẫu nhiên mà user chưa từng xem.
    
    Trả về:
        train_ratings: DataFrame chứa tập train
        test_data: list of tuples (user_id, positive_item_id, list of 99 negative_item_ids)
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
        
        # Danh sách phim user chưa xem
        non_interacted = list(all_items_set - interacted)
        
        # Lấy mẫu ngẫu nhiên 99 phim âm
        k_neg = min(99, len(non_interacted))
        neg_items = random.sample(non_interacted, k_neg)
        
        test_data.append((user, pos_item, neg_items))
        
    return train_ratings, test_data, user_interacted_items

# ==========================================
# 2. LẤY MẪU ÂM ĐỘNG (DYNAMIC NEGATIVE SAMPLING)
# ==========================================

def sample_train_data_implicit(train_df, user_interacted_items, num_items, num_negatives=4, user_col='user_idx', item_col='movie_idx', seed=42):
    """
    Lấy mẫu âm động cho tập Train (pointwise).
    Với mỗi tương tác dương, lấy ngẫu nhiên `num_negatives` tương tác âm.
    Dùng cho các thuật toán học sâu như NeuMF, Wide & Deep, DeepFM.
    """
    set_seed(seed)
    user_input, item_input, labels = [], [], []
    train_users = train_df[user_col].values
    train_items = train_df[item_col].values
    
    for u, i in zip(train_users, train_items):
        # Tương tác dương (nhãn 1)
        user_input.append(u)
        item_input.append(i)
        labels.append(1.0)
        
        # Tương tác âm (nhãn 0)
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
# 3. ĐO LƯỜNG CHỈ SỐ ĐÁNH GIÁ (METRICS)
# ==========================================

def evaluate_explicit(predictions):
    """
    Tính RMSE và MAE cho luồng Explicit (Rating Prediction).
    predictions: list of tuples (true_rating, est_rating)
    """
    y_true = np.array([p[0] for p in predictions])
    y_pred = np.array([p[1] for p in predictions])
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae

def evaluate_implicit_loo(predictions_dict, k=10):
    """
    Đánh giá chất lượng xếp hạng Top-K trên tập Leave-One-Out (1 positive + 99 negatives).
    predictions_dict: dict {user_id: [(item_id, est_score, is_positive), ...]}
    
    Trả về:
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
        pos_item = [item[0] for item in items if item[2]][0]
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
        
    return np.mean(hits), np.mean(ndcgs), np.mean(mrrs)

# ==========================================
# 4. CHỈ SỐ ĐÁNH GIÁ NÂNG CAO (BEYOND ACCURACY)
# ==========================================

def calculate_beyond_accuracy_metrics(recommendations, train_df, movies_df, movie_features=None, k=10, item_col='movieId'):
    """
    Tính Diversity, Novelty và Coverage của hệ gợi ý.
    
    Args:
        recommendations: dict {user_id: list of top-k recommended item_ids}
        train_df: DataFrame tập train để tính độ phổ biến (popularity)
        movies_df: DataFrame chứa danh sách phim (movieId, genres)
        movie_features: DataFrame chứa vector TF-IDF của genres (index là movieId). Nếu None sẽ tự tính.
        k: số lượng phim gợi ý được xem xét
        item_col: tên cột movie ID
        
    Trả về:
        mean_diversity: Độ đa dạng thể loại trong danh sách Top-K (càng cao càng tốt)
        mean_novelty: Mức độ mới lạ (self-information) của phim được gợi ý (càng cao càng tốt)
        coverage: Tỷ lệ phủ của kho phim (càng cao càng tốt)
    """
    # 1. Tính toán độ phổ biến (Popularity) của phim trong tập train
    item_counts = train_df[item_col].value_counts().to_dict()
    total_ratings = len(train_df)
    
    # Tránh chia cho 0, gán xác suất tối thiểu là 1 tương tác
    item_popularity = {iid: (count / total_ratings) for iid, count in item_counts.items()}
    
    # 2. Xây dựng vector đặc trưng genres nếu chưa được truyền vào
    if movie_features is None:
        movies_copy = movies_df.copy()
        movies_copy['genres_clean'] = movies_copy['genres'].str.replace('|', ' ', regex=False)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        tfidf_matrix = tfidf.fit_transform(movies_copy['genres_clean'])
        movie_features = pd.DataFrame(tfidf_matrix.toarray(), index=movies_copy[item_col])
        
    novelty_scores = []
    diversity_scores = []
    all_recommended_items = set()
    
    for uid, top_k in recommendations.items():
        top_k = list(top_k)[:k]
        if not top_k:
            continue
            
        all_recommended_items.update(top_k)
        
        # --- Tính Novelty ---
        # Novelty(i) = -log2(popularity(i))
        user_novelty = []
        for iid in top_k:
            pop = item_popularity.get(iid, 1 / total_ratings)
            user_novelty.append(-np.log2(pop))
        novelty_scores.append(np.mean(user_novelty))
        
        # --- Tính Diversity (1 - Cosine Similarity) ---
        valid_items = [iid for iid in top_k if iid in movie_features.index]
        if len(valid_items) > 1:
            feats = movie_features.loc[valid_items].values
            sim_matrix = cosine_similarity(feats)
            n_items = len(valid_items)
            diffs = []
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    diffs.append(1.0 - sim_matrix[i, j])
            diversity_scores.append(np.mean(diffs) if diffs else 0.0)
        else:
            diversity_scores.append(0.0)
            
    # --- Tính Coverage ---
    total_items = movies_df[item_col].nunique()
    coverage = len(all_recommended_items) / total_items if total_items > 0 else 0.0
    
    mean_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
    mean_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
    
    return mean_diversity, mean_novelty, coverage
