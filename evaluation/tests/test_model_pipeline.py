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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "evaluation", "ml_pipeline", "models")


def test_model_artifacts_exist():
    """Verify that essential model weights exist."""
    required_files = ["lgb_ranker.pkl", "tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "als_model.pkl"]
    for f in required_files:
        path = os.path.join(MODELS_DIR, f)
        assert os.path.exists(path), f"Missing model artifact: {f}"


def test_ranker_inference_and_latency():
    """Verify LightGBM ranker inference correctness and latency SLA (< 20ms for 100 items)."""
    ranker_path = os.path.join(MODELS_DIR, "lgb_ranker.pkl")
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)

    # 100 candidate items with 4 features: [popularity, vote_average, log_pop, vote_norm]
    dummy_features = np.random.rand(100, 4).astype(np.float32)

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
