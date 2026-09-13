"""
Unit tests for recsys_core.fusion
Testing Reciprocal Rank Fusion (RRF) algorithm and BM25 sparse retrieval.
"""

import pytest
import numpy as np
from recsys_core.fusion import reciprocal_rank_fusion, BM25


def test_reciprocal_rank_fusion_empty():
    fused = reciprocal_rank_fusion([], [])
    assert fused == []


def test_reciprocal_rank_fusion_single_nonempty_channel():
    fused = reciprocal_rank_fusion([10, 20, 30], [])
    assert len(fused) == 3
    # Returns list of (item_id, score)
    assert fused[0][0] == 10
    assert fused[1][0] == 20
    assert fused[2][0] == 30
    assert fused[0][1] == pytest.approx(1.0 / (60 + 1))


def test_reciprocal_rank_fusion_consensus():
    # Channel 1: [10, 20]
    # Channel 2: [20, 30]
    # Item 20 appears in both channels (rank 2 in ch1, rank 1 in ch2)
    # Score 20: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
    # Score 10: 1/(60+1) = 0.016393
    # Score 30: 1/(60+2) = 0.016129
    # Item 20 should win rank 1
    fused = reciprocal_rank_fusion([10, 20], [20, 30], k=60)
    assert len(fused) == 3
    assert fused[0][0] == 20
    assert fused[0][1] > fused[1][1]
    items_in_order = [item for item, score in fused]
    assert items_in_order == [20, 10, 30]


def test_bm25_fit_transform():
    corpus = [
        "the matrix sci-fi action keanu reeves neo",
        "toy story animation family comedy pixar",
        "the godfather crime drama al pacino mafia"
    ]
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(corpus)

    scores = bm25.transform("sci-fi neo action")
    assert len(scores) == 3
    # Doc 0 should have the highest BM25 score
    assert np.argmax(scores) == 0
    assert scores[0] > 0.0

    # Query with no terms in vocabulary
    empty_scores = bm25.transform("zzzzqwerty xyz")
    assert np.all(empty_scores == 0.0)


def test_bm25_score_single_doc():
    corpus = [
        "interstellar space sci-fi christopher nolan",
        "inception dream mind sci-fi christopher nolan"
    ]
    bm25 = BM25().fit(corpus)
    score_doc0 = bm25.score("space", 0)
    score_doc1 = bm25.score("space", 1)
    assert score_doc0 > 0.0
    assert score_doc1 == 0.0
