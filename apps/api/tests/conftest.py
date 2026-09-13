"""
apps/api/tests/conftest.py
--------------------------
Pytest bootstrap for API test suite.
Configures sys.path with API root, repo root, and recsys_core shared library.
"""

import os
import sys

API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(API_DIR))
CORE_SRC = os.path.join(REPO_ROOT, "libs", "recsys_core", "src")

for p in [API_DIR, REPO_ROOT, CORE_SRC]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)
