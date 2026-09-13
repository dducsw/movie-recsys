"""
recsys_core.fusion
------------------
Candidate fusion algorithms (Reciprocal Rank Fusion) and sparse text retrieval (BM25).
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np


def reciprocal_rank_fusion(
    als_candidates: List[int],
    bm25_candidates: List[int],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Merge ranked candidate lists from multiple retrievers using Reciprocal Rank Fusion (RRF).

    Args:
        als_candidates: list of item_ids (ordered best to worst).
        bm25_candidates: list of item_ids (ordered best to worst).
        k: smoothing constant (default: 60).

    Returns:
        sorted_candidates: list of (item_id, rrf_score) sorted in descending order.
    """
    rrf_scores: Dict[int, float] = {}

    for rank, item_id in enumerate(als_candidates, start=1):
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))

    for rank, item_id in enumerate(bm25_candidates, start=1):
        rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (1.0 / (k + rank))

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class BM25:
    """
    Okapi BM25 ranking algorithm for sparse content-based candidate retrieval.
    Leverages scikit-learn CountVectorizer for sparse matrix optimizations.
    """
    def __init__(self, b: float = 0.75, k1: float = 1.5):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
        self.b = b
        self.k1 = k1
        self.tf = None
        self.doc_len = None
        self.avg_doc_len = None
        self.idf = None

    def fit(self, corpus: List[str]):
        """Fit BM25 parameters on list of document strings."""
        self.tf = self.vectorizer.fit_transform(corpus)
        self.doc_len = np.array(self.tf.sum(axis=1)).flatten()
        self.avg_doc_len = self.doc_len.mean()

        N = self.tf.shape[0]
        df = np.bincount(self.tf.indices, minlength=self.tf.shape[1])
        self.idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        return self

    def transform(self, query_str: str) -> np.ndarray:
        """Return BM25 score array for all documents in fitted corpus."""
        q_vec = self.vectorizer.transform([query_str]).toarray()[0]
        q_indices = np.where(q_vec > 0)[0]
        if len(q_indices) == 0:
            return np.zeros(self.tf.shape[0])

        tf_q = self.tf[:, q_indices].toarray()
        idf_q = self.idf[q_indices]

        denom = tf_q + self.k1 * (1.0 - self.b + self.b * self.doc_len[:, np.newaxis] / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return scores.sum(axis=1)

    def score(self, query_str: str, doc_idx: int) -> float:
        """Return BM25 score of document doc_idx for query_str."""
        q_vec = self.vectorizer.transform([query_str]).toarray()[0]
        q_indices = np.where(q_vec > 0)[0]
        if len(q_indices) == 0:
            return 0.0

        tf_q = self.tf[doc_idx, q_indices].toarray()[0]
        idf_q = self.idf[q_indices]

        doc_len = self.doc_len[doc_idx]
        denom = tf_q + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
        scores = (tf_q * (self.k1 + 1.0) / denom) * idf_q
        return float(scores.sum())
