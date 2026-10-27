"""Single implementation of repo-root discovery for artifact path resolution.

Previously reimplemented three times (``services/recsys.py``,
``services/ml_training_service.py``, ``services/feature_store.py``) — the
walk direction and marker contract must stay identical everywhere.
"""

import os


def find_repo_root(*markers: str) -> str:
    """Walk up from this module's directory until one contains a marker.

    Args:
        markers: file or directory *names* relative to a candidate root
            (e.g. ``".git"``, ``"docker-compose.yml"``).

    Returns the first directory containing any marker; the filesystem root
    itself when no marker ever matches (same fallback as the old walkers).
    """
    root = os.path.dirname(os.path.abspath(__file__))
    while root and root != os.path.dirname(root):
        if any(os.path.exists(os.path.join(root, marker)) for marker in markers):
            return root
        root = os.path.dirname(root)
    return root
