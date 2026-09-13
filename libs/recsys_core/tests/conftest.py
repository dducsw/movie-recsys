"""
libs/recsys_core/tests/conftest.py
----------------------------------
Add recsys_core src directory to sys.path during test discovery and execution.
"""

import os
import sys

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(os.path.dirname(_tests_dir), "src")

if _src_dir not in sys.path and os.path.exists(_src_dir):
    sys.path.insert(0, _src_dir)
