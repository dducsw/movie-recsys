"""
recsys_core.envinfo
-------------------
Environment provenance helpers shared by every training entrypoint.

Previously reimplemented verbatim in ``train.py``, ``tune.py``,
``train_pipeline.py`` and ``versioning.py`` — a single copy here makes the
provenance contract traceable in exactly one place.
"""

import subprocess


def git_sha(short: bool = False) -> str:
    """Commit SHA of the working tree, or ``"nogit"`` when git is unavailable."""
    try:
        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        return out.decode().strip()
    except Exception:
        return "nogit"
