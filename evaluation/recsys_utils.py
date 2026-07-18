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

# ==========================================
# 5. THUẬT TOÁN HỖ TRỢ NÂNG CẤP (BM25, RRF, DYNAMIC LAMBDA)
# ==========================================

class BM25:
    """
    Thuật toán BM25 (Okapi BM25) phục vụ lọc thô dựa trên nội dung (Content-Based Retrieval).
    Sử dụng CountVectorizer của scikit-learn để tối ưu hóa hiệu năng tính toán.
    """
    def __init__(self, b=0.75, k1=1.5):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
        self.b = b
        self.k1 = k1
        self.tf = None
        self.doc_len = None
        self.avg_doc_len = None
        self.idf = None

    def fit(self, corpus):
        """
        corpus: list of strings (mô tả nội dung phim)
        """
        self.tf = self.vectorizer.fit_transform(corpus) # shape (N, V)
        self.doc_len = np.array(self.tf.sum(axis=1)).flatten()
        self.avg_doc_len = self.doc_len.mean()
        
        N = self.tf.shape[0]
        # Tính Document Frequency (DF) cho từng từ
        df = np.bincount(self.tf.indices, minlength=self.tf.shape[1])
        # Công thức IDF của BM25
        self.idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        return self

    def transform(self, query_str):
        """
        Trả về mảng điểm số BM25 cho tất cả tài liệu trong corpus tương ứng với query_str
        """
        q_vec = self.vectorizer.transform([query_str]).toarray()[0]
        q_indices = np.where(q_vec > 0)[0]
        if len(q_indices) == 0:
            return np.zeros(self.tf.shape[0])
            
        # Trích xuất term frequency cho các từ trong query
        tf_q = self.tf[:, q_indices].toarray() # shape (N, len(q_indices))
        idf_q = self.idf[q_indices] # shape (len(q_indices),)
        
        # Công thức BM25: score = sum( idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / L_avg)) )
        denom = tf_q + self.k1 * (1.0 - self.b + self.b * self.doc_len[:, np.newaxis] / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return scores.sum(axis=1)

    def score(self, query_str, doc_idx):
        """
        Trả về điểm số BM25 của 1 tài liệu doc_idx cụ thể đối với query_str
        """
        q_vec = self.vectorizer.transform([query_str]).toarray()[0]
        q_indices = np.where(q_vec > 0)[0]
        if len(q_indices) == 0:
            return 0.0
            
        tf_q = self.tf[doc_idx, q_indices].toarray()[0]
        idf_q = self.idf[q_indices]
        
        doc_len = self.doc_len[doc_idx]
        denom = tf_q + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return scores.sum()

    def transform_from_bow(self, query_bow):
        """
        Tính BM25 cho toàn bộ corpus từ query dạng Bag-of-Words (sparse vector)
        """
        import scipy.sparse as sp
        if sp.issparse(query_bow):
            q_indices = query_bow.nonzero()[1]
        else:
            q_indices = np.where(query_bow > 0)[0]
            
        if len(q_indices) == 0:
            return np.zeros(self.tf.shape[0])
            
        tf_q = self.tf[:, q_indices].toarray()
        idf_q = self.idf[q_indices]
        denom = tf_q + self.k1 * (1.0 - self.b + self.b * self.doc_len[:, np.newaxis] / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return scores.sum(axis=1)

    def score_from_bow(self, query_bow, doc_idx):
        """
        Tính BM25 cho 1 tài liệu cụ thể từ query dạng Bag-of-Words (sparse vector)
        """
        import scipy.sparse as sp
        if sp.issparse(query_bow):
            q_indices = query_bow.nonzero()[1]
        else:
            q_indices = np.where(query_bow > 0)[0]
            
        if len(q_indices) == 0:
            return 0.0
            
        tf_q = self.tf[doc_idx, q_indices].toarray()[0]
        idf_q = self.idf[q_indices]
        
        doc_len = self.doc_len[doc_idx]
        denom = tf_q + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return scores.sum()


def reciprocal_rank_fusion(als_candidates, bm25_candidates, k=60):
    """
    Gộp danh sách xếp hạng từ hai nguồn bằng thuật toán Reciprocal Rank Fusion (RRF).
    als_candidates: list of item_ids (đã sắp xếp từ tốt nhất đến kém nhất)
    bm25_candidates: list of item_ids (đã sắp xếp từ tốt nhất đến kém nhất)
    k: hằng số làm mượt (smoothing constant)
    
    Trả về:
        sorted_candidates: list of (item_id, rrf_score) được xếp hạng giảm dần
    """
    rrf_scores = {}
    
    # Đăng ký thứ hạng từ nguồn ALS (Collaborative Filtering)
    for rank, item_id in enumerate(als_candidates, start=1):
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))
        
    # Đăng ký thứ hạng từ nguồn BM25 (Content-Based)
    for rank, item_id in enumerate(bm25_candidates, start=1):
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))
        
    # Sắp xếp theo rrf_score giảm dần
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates


def calculate_user_lambda(user_liked_genres, all_genres, base_min=0.4, base_max=0.9):
    """
    Tính toán lambda động cho người dùng dựa trên Shannon Entropy của các thể loại đã thích.
    user_liked_genres: list các thể loại từ lịch sử phim user đã xem/thích.
    all_genres: danh sách tất cả các thể loại độc bản trong hệ thống để chuẩn hóa entropy.
    """
    if not user_liked_genres:
        return 0.7 # fallback mặc định
        
    from collections import Counter
    import math
    
    counts = Counter(user_liked_genres)
    total = len(user_liked_genres)
    
    # Tính Shannon Entropy
    entropy = 0.0
    for g, cnt in counts.items():
        p = cnt / total
        entropy -= p * math.log2(p)
        
    # Entropy cực đại khi thể loại phân bố đều trên tất cả các thể loại độc bản
    max_entropy = math.log2(len(all_genres)) if len(all_genres) > 1 else 1.0
    norm_entropy = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0
    
    # Entropy càng cao (sở thích đa dạng) -> lambda càng thấp (đa dạng hóa nhiều hơn)
    # Entropy càng thấp (sở thích chuyên biệt) -> lambda càng cao (đa dạng hóa ít hơn)
    dynamic_lambda = base_max - (base_max - base_min) * norm_entropy
    return dynamic_lambda


def extract_user_features(user_id, movie_ids, train_ratings, movies_df, users_df, user_to_idx, movie_to_idx, als_model, bm25, is_train=False, target_mid=None):
    """
    Trích xuất đặc trưng thống nhất cho danh sách movie_ids của user_id (Inference / Eval mode).
    Đảm bảo loại bỏ target_mid khỏi liked_ids khi is_train=True để tránh leakage.
    """
    import pandas as pd
    import numpy as np
    
    if not movie_ids:
        return pd.DataFrame()
        
    # Đảm bảo index của movies_df và users_df đã được set
    if movies_df.index.name != 'movieId':
        movies_df_idx = movies_df.set_index('movieId')
    else:
        movies_df_idx = movies_df
        
    if users_df.index.name != 'user_id':
        users_df_idx = users_df.set_index('user_id')
    else:
        users_df_idx = users_df
        
    user = users_df_idx.loc[user_id]
    
    # 1. Lịch sử xem (liked_ids)
    train_pos = train_ratings[(train_ratings['userId'] == user_id) & (train_ratings['rating'] >= 3.5)]
    liked_ids = train_pos['movieId'].tolist()
    
    # 2. Thể loại yêu thích (lấy từ lịch sử xem thực tế, tránh tĩnh)
    user_fav_genres = set()
    for lmid in liked_ids:
        if lmid in movies_df_idx.index:
            g_str = movies_df_idx.loc[lmid, "genres"]
            if pd.notna(g_str):
                user_fav_genres.update(str(g_str).split("|"))
                
    # Fallback nếu lịch sử trống
    if not user_fav_genres:
        fav_m_str = str(user.get("favorite_movies", ""))
        fav_m_list = [int(m) for m in fav_m_str.split("|") if str(m).isdigit()]
        for fid in fav_m_list:
            if fid in movies_df_idx.index:
                g_str = movies_df_idx.loc[fid, "genres"]
                if pd.notna(g_str):
                    user_fav_genres.update(str(g_str).split("|"))
    
    if is_train and target_mid is not None and target_mid in liked_ids:
        liked_ids = [lid for lid in liked_ids if lid != target_mid]
        
    # Giới hạn 20 phim gần nhất
    liked_ids = liked_ids[-20:]
    
    # 3. Tính cb_score (dùng liked_ids hoặc fallback về favorite_movies nếu cold start)
    user_bm25_scores = None
    query_mids = liked_ids if liked_ids else [int(m) for m in str(user.get("favorite_movies", "")).split("|") if str(m).isdigit()]
    if query_mids:
        liked_soups = [movies_df_idx.loc[lid, 'soup'] for lid in query_mids if lid in movies_df_idx.index and 'soup' in movies_df_idx.columns]
        if liked_soups:
            query = " ".join(liked_soups)
            user_bm25_scores = bm25.transform(query)
            
    # 4. Tạo features list
    features = []
    als_user_factors = als_model.user_factors
    als_item_factors = als_model.item_factors
    u_idx = user_to_idx.get(user_id, None)
    
    for mid in movie_ids:
        if mid not in movies_df_idx.index:
            continue
        movie = movies_df_idx.loc[mid]
        
        movie_genres = set(str(movie['genres']).split('|'))
        genre_overlap = len(user_fav_genres.intersection(movie_genres))
        
        try:
            release_year = int(str(movie['release_date'])[:4])
        except:
            release_year = 2010
            
        m_idx = movie_to_idx.get(mid, None)
        als_score = als_user_factors[u_idx].dot(als_item_factors[m_idx]) if (u_idx is not None and m_idx is not None) else 0.0
        cb_score = user_bm25_scores[m_idx] if (user_bm25_scores is not None and m_idx is not None) else 0.0
        
        features.append({
            'popularity': movie['popularity'],
            'vote_average': movie['vote_average'],
            'genre_overlap': genre_overlap,
            'release_year': release_year,
            'user_activity': user['activity_level'],
            'user_bias': user['user_bias'],
            'als_score': als_score,
            'cb_score': cb_score
        })
        
    return pd.DataFrame(features)
