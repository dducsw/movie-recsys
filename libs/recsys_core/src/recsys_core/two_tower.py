"""
recsys_core.two_tower
----------------------
Two-Tower (DSSM) Neural Retrieval Representation for MovieRecSys.

User Tower:
    Projects user interaction history, genre affinity vector, and activity into
    a normalized latent embedding u in R^d.

Item Tower:
    Projects item semantic plot embedding, metadata features (popularity, vote, year)
    into the same normalized latent embedding v in R^d.

Candidate Scoring:
    score(u, v) = u^T v (Cosine similarity in shared latent space).
Designed with pure NumPy inference for zero PyTorch/GPU overhead in online SLA budget (< 15ms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoTowerConfig:
    embedding_dim: int = 64
    content_dim: int = 384
    num_genres: int = 19
    random_state: int = 42


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


class TwoTowerRetrieval:
    """
    Lightweight Two-Tower Neural Projector for Candidate Retrieval.
    Maps User Context & Item Metadata into shared d-dimensional metric space.
    """

    def __init__(self, config: TwoTowerConfig | None = None) -> None:
        self.config = config or TwoTowerConfig()
        rng = np.random.default_rng(self.config.random_state)

        # User projection weight: [content_dim + num_genres + 2] -> embedding_dim
        user_input_dim = self.config.content_dim + self.config.num_genres + 2
        scale_u = np.sqrt(2.0 / (user_input_dim + self.config.embedding_dim))
        self.user_proj = (rng.standard_normal((user_input_dim, self.config.embedding_dim)) * scale_u).astype(np.float32)
        self.user_bias = np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Item projection weight: [content_dim + num_genres + 2] -> embedding_dim
        item_input_dim = self.config.content_dim + self.config.num_genres + 2
        scale_i = np.sqrt(2.0 / (item_input_dim + self.config.embedding_dim))
        self.item_proj = (rng.standard_normal((item_input_dim, self.config.embedding_dim)) * scale_i).astype(np.float32)
        self.item_bias = np.zeros(self.config.embedding_dim, dtype=np.float32)

    def encode_user(
        self,
        liked_vectors: list[np.ndarray] | np.ndarray | None,
        genre_distribution: np.ndarray | None = None,
        activity_level: float = 1.0,
        rating_bias: float = 0.0,
    ) -> np.ndarray:
        """
        User Tower forward pass:
        Combines recent item embeddings with user genre preferences and activity.
        """
        # 1. Aggregate recent item embeddings
        if liked_vectors is not None and len(liked_vectors) > 0:
            arr = np.asarray(liked_vectors, dtype=np.float32)
            if arr.ndim == 1:
                content_repr = arr
            else:
                # Exponential recency weighting: recent interactions have higher weights
                weights = np.exp(np.linspace(-0.5, 0.0, len(arr)))
                weights /= weights.sum()
                content_repr = np.sum(arr * weights[:, np.newaxis], axis=0)
        else:
            content_repr = np.zeros(self.config.content_dim, dtype=np.float32)

        if len(content_repr) < self.config.content_dim:
            content_repr = np.pad(content_repr, (0, self.config.content_dim - len(content_repr)))
        elif len(content_repr) > self.config.content_dim:
            content_repr = content_repr[:self.config.content_dim]

        # 2. Genre distribution vector
        if genre_distribution is None or len(genre_distribution) != self.config.num_genres:
            genre_vec = np.zeros(self.config.num_genres, dtype=np.float32)
        else:
            genre_vec = np.asarray(genre_distribution, dtype=np.float32)

        # 3. Dense context concat [content_dim (384) + genres (19) + activity (1) + bias (1)]
        feat_vector = np.concatenate([
            content_repr,
            genre_vec,
            np.array([np.log1p(max(0.0, activity_level)), rating_bias], dtype=np.float32)
        ])

        # 4. Dense projection + ReLU + L2 Normalization
        hidden = np.dot(feat_vector, self.user_proj) + self.user_bias
        activated = np.maximum(hidden, 0.0)  # ReLU
        return _l2_normalize(activated)

    def encode_item(
        self,
        content_vector: np.ndarray | None,
        genre_vector: np.ndarray | None = None,
        popularity: float = 0.0,
        vote_average: float = 5.0,
    ) -> np.ndarray:
        """
        Item Tower forward pass:
        Combines item content embedding with metadata and genre profile.
        """
        if content_vector is None:
            content_repr = np.zeros(self.config.content_dim, dtype=np.float32)
        else:
            content_repr = np.asarray(content_vector, dtype=np.float32)
            if len(content_repr) < self.config.content_dim:
                content_repr = np.pad(content_repr, (0, self.config.content_dim - len(content_repr)))
            elif len(content_repr) > self.config.content_dim:
                content_repr = content_repr[:self.config.content_dim]

        if genre_vector is None or len(genre_vector) != self.config.num_genres:
            g_vec = np.zeros(self.config.num_genres, dtype=np.float32)
        else:
            g_vec = np.asarray(genre_vector, dtype=np.float32)

        norm_pop = float(np.log1p(max(0.0, popularity)))
        norm_vote = float(vote_average / 10.0)

        feat_vector = np.concatenate([
            content_repr,
            g_vec,
            np.array([norm_pop, norm_vote], dtype=np.float32)
        ])

        hidden = np.dot(feat_vector, self.item_proj) + self.item_bias
        activated = np.maximum(hidden, 0.0)  # ReLU
        return _l2_normalize(activated)

    def score(self, user_vec: np.ndarray, item_vec: np.ndarray) -> float:
        """Compute cosine similarity score between user and item representations."""
        return float(np.dot(user_vec, item_vec))
