"""
pipelines/training/tests/conftest.py
------------------------------------
Test configuration and sys.path setup for training pipeline tests.
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(TESTS_DIR)
PIPELINES_DIR = os.path.dirname(TRAINING_DIR)
REPO_ROOT = os.path.dirname(PIPELINES_DIR)
ML_TRAINING_SRC = os.path.join(TRAINING_DIR, "ml_training", "src")
CORE_SRC = os.path.join(REPO_ROOT, "libs", "recsys_core", "src")

for path in [REPO_ROOT, TRAINING_DIR, ML_TRAINING_SRC, CORE_SRC]:
    if path not in sys.path and os.path.exists(path):
        sys.path.insert(0, path)
