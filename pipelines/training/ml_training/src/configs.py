"""
evaluation/ml_training/src/configs.py
-------------------------------------
Loads configuration dictionary from configs.yaml with directory resolution fallback.
"""

import os
import yaml

_src_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_src_dir)

_candidates = [
    os.path.join(_parent_dir, "configs.yaml"),
    os.path.join(_src_dir, "configs.yaml"),
    "configs.yaml",
]

CFG = {}
for path in _candidates:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f)
        break
