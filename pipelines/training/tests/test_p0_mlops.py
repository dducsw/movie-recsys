"""P0 MLOps tests: validation fail-fast + parity + versioning."""

import numpy as np
import pandas as pd
import pytest

from recsys_core.features import TFIDF_MAX_FEATURES, build_ranking_row


def test_tfidf_constant_shared():
    assert TFIDF_MAX_FEATURES == 50


def test_validate_rejects_bad_data():
    from pipelines.training.validate import validate_movies, validate_ratings

    with pytest.raises(ValueError):
        validate_movies(pd.DataFrame())
    with pytest.raises(ValueError):
        validate_ratings(pd.DataFrame([{"userId": 1, "movieId": 2, "rating": 9.0}]))


def test_parity_builder_deterministic():
    meta = {"popularity": 10.0, "vote_average": 8.0, "release_year": 2020.0,
            "genres": {"Action"}, "als_vec": np.array([1.0, 0.0])}
    profile = {"genres": {"Action", "Drama"}, "activity": 1.0,
               "bias": 0.0, "als_vec": np.array([1.0, 0.0])}
    a = build_ranking_row(meta, profile)
    b = build_ranking_row(dict(meta), dict(profile))
    assert a == b


def test_versioned_save_and_prune(tmp_path):
    from pipelines.training import versioning
    import glob
    import os

    # save_versioned only writes immutable *candidates* — no promotion.
    paths = [versioning.save_versioned({"v": 1}, str(tmp_path), "lgb_ranker")
             for _ in range(5)]
    versioned = glob.glob(os.path.join(str(tmp_path), "lgb_ranker_*.pkl"))
    assert len(versioned) == 5  # candidates are never auto-pruned
    assert not os.path.exists(os.path.join(str(tmp_path), "lgb_ranker.pkl"))
    assert versioning.read_champion(str(tmp_path), "lgb_ranker") is None

    # Explicit promotion publishes the + prunes to KEEP_LAST (champion kept).
    promoted = versioning.promote(str(tmp_path), "lgb_ranker", paths[-1])
    assert os.path.exists(os.path.join(str(tmp_path), "lgb_ranker.pkl"))
    assert versioning.read_champion(str(tmp_path), "lgb_ranker") == promoted
    versioned = glob.glob(os.path.join(str(tmp_path), "lgb_ranker_*.pkl"))
    assert len(versioned) <= versioning.KEEP_LAST
    assert promoted in [os.path.basename(v) for v in versioned]  # champion never pruned
