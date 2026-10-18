"""Tests for recsys_core.two_tower (Two-Tower Retrieval DSSM)."""

import numpy as np
from recsys_core.two_tower import TwoTowerConfig, TwoTowerRetrieval


def test_two_tower_dimensions_and_normalization():
    config = TwoTowerConfig(embedding_dim=32, content_dim=64, num_genres=19)
    model = TwoTowerRetrieval(config)

    # User Tower
    dummy_liked = [np.ones(64, dtype=np.float32), np.zeros(64, dtype=np.float32)]
    u_vec = model.encode_user(
        liked_vectors=dummy_liked,
        genre_distribution=np.ones(19, dtype=np.float32),
        activity_level=5.0,
        rating_bias=0.2
    )

    assert u_vec.shape == (32,)
    assert abs(float(np.linalg.norm(u_vec)) - 1.0) < 1e-4

    # Item Tower
    i_vec = model.encode_item(
        content_vector=np.ones(64, dtype=np.float32),
        genre_vector=np.ones(19, dtype=np.float32),
        popularity=25.0,
        vote_average=8.5
    )

    assert i_vec.shape == (32,)
    assert abs(float(np.linalg.norm(i_vec)) - 1.0) < 1e-4

    # Cosine Similarity score within [-1.0, 1.0]
    score = model.score(u_vec, i_vec)
    assert -1.0 <= score <= 1.0
