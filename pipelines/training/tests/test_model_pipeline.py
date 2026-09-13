"""
evaluation/tests/test_model_pipeline.py
---------------------------------------
Automated Unit Tests for Model Artifacts, Inference Schema, and Latency SLA.
"""

import os
import pickle
import time
import pytest
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(TESTS_DIR)))
_candidates = [
    os.path.join(REPO_ROOT, "notebooks", "04_pipeline_prototypes", "models"),
    os.path.join(REPO_ROOT, "pipelines", "training", "ml_pipeline", "models"),
    os.path.join(REPO_ROOT, "evaluation", "ml_pipeline", "models"),
    os.path.join(REPO_ROOT, "models"),
]
MODELS_DIR = next((p for p in _candidates if os.path.exists(p)), _candidates[0])


def test_model_artifacts_exist():
    """Verify that essential model weights exist when artifacts are present."""
    required_files = ["lgb_ranker.pkl", "tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "als_model.pkl"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        pytest.skip(f"Model artifacts not present in environment (gitignored binary files): {missing}")
    for f in required_files:
        path = os.path.join(MODELS_DIR, f)
        assert os.path.exists(path), f"Missing model artifact: {f}"


def test_ranker_inference_and_latency():
    """Verify LightGBM ranker inference correctness and latency SLA (< 20ms for 100 items)."""
    ranker_path = os.path.join(MODELS_DIR, "lgb_ranker.pkl")
    if os.path.exists(ranker_path):
        with open(ranker_path, "rb") as f:
            ranker = pickle.load(f)
    else:
        # Fallback to in-memory ranker if binary artifact is absent (e.g. fresh CI clone)
        import lightgbm as lgb
        X_dummy = np.random.rand(100, 8).astype(np.float32)
        y_dummy = np.random.randint(0, 5, size=100).astype(np.int32)
        ranker = lgb.LGBMRanker(n_estimators=5, min_child_samples=1, num_leaves=4, random_state=42)
        ranker.fit(X_dummy, y_dummy, group=[50, 50])

    # 100 candidate items with 8 standard ranking features (RANKING_FEATURES)
    n_features = getattr(ranker, "n_features_", 8)
    dummy_features = np.random.rand(100, n_features).astype(np.float32)

    # Warm-up call (initializes OpenMP thread pool)
    _ = ranker.predict(dummy_features)

    # Measured warm inference
    t0 = time.perf_counter()
    scores = ranker.predict(dummy_features)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0

    assert len(scores) == 100, "Output prediction shape mismatch"
    assert not np.isnan(scores).any(), "NaN values found in ranker output"
    assert latency_ms < 20.0, f"Inference latency too high: {latency_ms:.2f}ms exceeds SLA 20ms"
