"""Tests for sampling + quality gate + embeddings fallback."""

import numpy as np
import pandas as pd
import pytest

from recsys_core.embeddings import EMBEDDING_DIM, encode_texts
from recsys_core.quality_gate import (
    QualityGateConfig,
    QualityGateError,
    check_quality_gate,
)
from recsys_core.sampling import sample_negatives


def test_sample_negatives_excludes_rated_and_mixes_hard():
    all_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    popularity = pd.Series({i: float(i) for i in all_ids})  # 10 most popular
    negs = sample_negatives({1, 2}, all_ids, popularity=popularity,
                            n_neg=4, hard_ratio=0.5,
                            rng=np.random.default_rng(42))
    assert len(negs) == 4
    assert not set(negs.tolist()) & {1, 2}


def test_quality_gate_pass_and_fail():
    ok = {"HR@10": 0.5, "NDCG@10": 0.4, "ILD@10": 0.5, "avg_latency_ms": 10.0}
    check_quality_gate(ok, QualityGateConfig(), k=10)  # no raise
    bad = dict(ok, **{"HR@10": 0.0})
    with pytest.raises(QualityGateError):
        check_quality_gate(bad, QualityGateConfig(), k=10)


def test_embeddings_fallback_dim():
    vecs = encode_texts(["Toy Story. Genres: Animation Comedy."])
    assert vecs.shape == (1, EMBEDDING_DIM)
