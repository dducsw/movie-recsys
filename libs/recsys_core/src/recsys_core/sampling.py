"""
recsys_core.sampling
--------------------
Negative sampling strategies for ranking training and evaluation.

Uniform sampling is fast but too easy; popularity-weighted sampling
produces harder negatives that improve listwise discrimination.
The CT pipeline mixes both (default 50/50).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_negatives(
    rated_ids: set[int],
    all_ids: np.ndarray,
    popularity: pd.Series | None = None,
    n_neg: int = 10,
    hard_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``n_neg`` unrated item ids, mixing uniform + popularity-weighted.

    Args:
        rated_ids: Item ids the user already rated (excluded).
        all_ids: Full catalog ids.
        popularity: Optional Series indexed by item id with interaction counts.
        n_neg: Number of negatives to draw.
        hard_ratio: Fraction drawn from popularity distribution (hard negatives).
        rng: Optional numpy Generator for reproducibility.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pool = np.array([m for m in all_ids if m not in rated_ids])
    if len(pool) == 0 or n_neg <= 0:
        return np.array([], dtype=int)

    n_neg = min(n_neg, len(pool))
    n_hard = round(n_neg * hard_ratio)
    n_uniform = n_neg - n_hard

    sampled: list[np.ndarray] = []
    if n_uniform > 0:
        sampled.append(rng.choice(pool, size=n_uniform, replace=False))

    if n_hard > 0:
        if popularity is not None and len(popularity) > 0:
            weights = np.array(
                [float(popularity.get(int(m), 1.0)) for m in pool], dtype=np.float64
            )
            weights = weights / weights.sum()
            # Exclude already uniform-sampled to avoid duplicates
            chosen = set(sampled[0].tolist()) if sampled else set()
            mask = np.array([int(m) not in chosen for m in pool])
            cand, w = pool[mask], weights[mask]
            if len(cand) > 0:
                k = min(n_hard, len(cand))
                w = w / w.sum()
                sampled.append(rng.choice(cand, size=k, replace=False, p=w))
        if sum(len(s) for s in sampled) < n_neg:  # top-up uniform
            chosen = set().union(*[set(s.tolist()) for s in sampled]) if sampled else set()
            rest = np.array([m for m in pool if m not in chosen])
            need = n_neg - sum(len(s) for s in sampled)
            if len(rest) > 0:
                sampled.append(
                    rng.choice(rest, size=min(need, len(rest)), replace=False)
                )

    result = np.concatenate(sampled) if sampled else np.array([], dtype=int)
    return rng.permutation(result)[:n_neg]
