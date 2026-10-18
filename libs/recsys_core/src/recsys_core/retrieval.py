"""
recsys_core.retrieval
---------------------
Implicit-feedback collaborative filtering (true iALS).

Implements the Hu-Koren-Volinsky confidence model::

    c_ui = 1 + alpha * r_ui
    p_ui = 1 if r_ui > 0 else 0

Uses the ``implicit`` library when available, with a deterministic
TruncatedSVD fallback so unit tests and offline environments still work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IALSConfig:
    """Hyperparameters for implicit ALS training."""

    factors: int = 32
    regularization: float = 0.01
    iterations: int = 20
    alpha: float = 40.0  # confidence scaling for c = 1 + alpha * r
    random_state: int = 42


def build_confidence_matrix(
    ratings_df: pd.DataFrame,
    alpha: float = 40.0,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Build user-item confidence matrix for ``implicit``.

    Args:
        ratings_df: DataFrame with ``userId``, ``movieId``, ``rating`` columns.
        alpha: Confidence scaling factor.

    Returns:
        Tuple of (confidence CSR matrix shaped [n_users, n_items],
        unique user ids, unique item ids).
    """
    from scipy.sparse import csr_matrix

    user_ids = ratings_df["userId"].astype("category")
    item_ids = ratings_df["movieId"].astype("category")
    unique_users: np.ndarray = user_ids.cat.categories.to_numpy()
    unique_items: np.ndarray = item_ids.cat.categories.to_numpy()

    # Confidence: c = 1 + alpha * r  (Hu-Koren-Volinsky, 2008)
    confidence = 1.0 + float(alpha) * ratings_df["rating"].to_numpy(dtype=np.float32)
    # implicit ALS expects [users x items]
    matrix = csr_matrix(
        (confidence, (user_ids.cat.codes.to_numpy(), item_ids.cat.codes.to_numpy())),
        dtype=np.float32,
    )
    return matrix, unique_users, unique_items


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def train_ials_factors(
    ratings_df: pd.DataFrame,
    config: IALSConfig | None = None,
) -> dict[int, np.ndarray]:
    """Train item latent factors with true iALS, fallback to SVD.

    Returns:
        Mapping of ``movieId -> L2-normalized factor vector``.
    """
    if ratings_df is None or ratings_df.empty:
        return {}
    cfg = config or IALSConfig()

    conf_matrix, _, unique_items = build_confidence_matrix(ratings_df, alpha=cfg.alpha)

    # ── Primary: implicit.als.AlternatingLeastSquares ──
    try:
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=cfg.factors,
            regularization=cfg.regularization,
            iterations=cfg.iterations,
            random_state=cfg.random_state,
        )
        model.fit(conf_matrix)
        item_factors = np.asarray(model.item_factors, dtype=np.float32)
        logger.info("Trained true iALS factors for %d items.", len(unique_items))
        return {
            int(mid): _l2_normalize(item_factors[idx])
            for idx, mid in enumerate(unique_items)
        }
    except ImportError:
        logger.warning("`implicit` not installed, falling back to TruncatedSVD.")
    except Exception as exc:  # noqa: BLE001 - fallback must never crash training
        logger.warning("iALS training failed (%s), falling back to SVD.", exc)

    # ── Fallback: TruncatedSVD on confidence-weighted interactions ──
    from sklearn.decomposition import TruncatedSVD

    n_comp = min(cfg.factors, min(conf_matrix.shape) - 1)
    if n_comp < 1:
        factors = np.ones((len(unique_items), cfg.factors), dtype=np.float32)
    else:
        svd = TruncatedSVD(n_components=n_comp, random_state=cfg.random_state)
        # conf is [users x items] -> transpose for item factors [items x k]
        factors = svd.fit_transform(conf_matrix.T).astype(np.float32)
        if factors.shape[1] < cfg.factors:  # pad for stable downstream dim
            factors = np.pad(
                factors, ((0, 0), (0, cfg.factors - factors.shape[1])), mode="constant"
            )
    logger.info("Trained SVD fallback factors for %d items.", len(unique_items))
    return {
        int(mid): _l2_normalize(factors[idx]) for idx, mid in enumerate(unique_items)
    }
