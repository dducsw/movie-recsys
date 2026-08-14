import numpy as np
import pandas as pd
from collections import defaultdict

def _dcg(scores):
    return sum(s / np.log2(i+2) for i, s in enumerate(scores))

def evaluate_ranking(eval_df: pd.DataFrame, k: int = 10,
                     relevance_threshold: float = 4.0) -> dict:
    """
    For each user, rank candidate movies by predicted rating, take top-K,
    and compute Precision@K, Recall@K, NDCG@K vs. ground-truth high-rated items.
    """
    precisions, recalls, ndcgs = [], [], []
    for uid, g in eval_df.groupby("userId"):
        if len(g) < 2: continue
        g = g.sort_values("pred", ascending=False)
        topk = g.head(k)
        rel_pred = (topk["rating"] >= relevance_threshold).astype(int).tolist()
        n_rel    = int((g["rating"] >= relevance_threshold).sum())
        n_rec    = len(rel_pred)

        # precision@k, recall@k
        precisions.append(sum(rel_pred) / max(1, n_rec))
        recalls.append(sum(rel_pred) / max(1, n_rel))

        # NDCG@k: ideal ranking = sort actual ratings desc
        ideal = sorted(g["rating"].tolist(), reverse=True)[:k]
        idcg = _dcg([1 if r >= relevance_threshold else 0 for r in ideal])
        dcg  = _dcg(rel_pred)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}":    float(np.mean(recalls))    if recalls    else 0.0,
        f"NDCG@{k}":      float(np.mean(ndcgs))      if ndcgs      else 0.0,
        "n_users_evaluated": len(precisions),
    }