"""
pipeline/feature/batch/metrics.py
----------------------------------
Offline evaluation metrics calculation (Hit Ratio @ 10 and NDCG @ 10).
"""

import os
import math
import logging
from collections import defaultdict
from typing import Dict, List, Any
import pandas as pd

from pipeline.feature.config import RATINGS_CSV

logger = logging.getLogger("BatchMetrics")


def compute_offline_metrics(similarities: Dict[int, List[Dict[str, Any]]], ratings_csv: str = RATINGS_CSV) -> Dict[str, float]:
    """Compute Hit Ratio @ 10 and NDCG @ 10 on user interaction history using Leave-One-Out split."""
    if not os.path.exists(ratings_csv):
        return {"HR@10": 0.0, "NDCG@10": 0.0}
    try:
        df = pd.read_csv(ratings_csv)
        if "movieid" in df.columns:
            df.rename(columns={"movieid": "movieId", "userid": "userId"}, inplace=True)

        if "timestamp" in df.columns:
            df = df.sort_values(["userId", "timestamp"])

        user_groups = df.groupby("userId")
        hits = 0
        ndcgs = 0.0
        total_eval_users = 0

        for user_id, group in user_groups:
            if len(group) < 2:
                continue

            target_movie = int(group.iloc[-1]["movieId"])
            train_movies = set(group.iloc[:-1]["movieId"].astype(int))

            candidate_scores = defaultdict(float)
            for prev_m in train_movies:
                for sim in similarities.get(prev_m, []):
                    c_id = sim["movieId"]
                    if c_id not in train_movies:
                        candidate_scores[c_id] += sim["score"]

            if not candidate_scores:
                continue

            top_recs = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)[:10]

            total_eval_users += 1
            if target_movie in top_recs:
                hits += 1
                rank = top_recs.index(target_movie) + 1
                ndcgs += 1.0 / math.log2(rank + 1)

        hr_10 = round(hits / total_eval_users, 4) if total_eval_users > 0 else 0.0
        ndcg_10 = round(ndcgs / total_eval_users, 4) if total_eval_users > 0 else 0.0
        logger.info(f"--- Offline Evaluation Metrics ({total_eval_users} users evaluated) --- -> HR@10: {hr_10}, NDCG@10: {ndcg_10}")
        return {"HR@10": hr_10, "NDCG@10": ndcg_10}
    except Exception as e:
        logger.warning(f"Could not compute offline evaluation metrics: {e}")
        return {"HR@10": 0.0, "NDCG@10": 0.0}
