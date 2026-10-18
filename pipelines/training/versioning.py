"""Versioned artifact saving: {name}_{timestamp}_{gitsha}.pkl + latest + champion."""

from __future__ import annotations

import glob
import logging
import os
import pickle
import subprocess
import time

logger = logging.getLogger(__name__)

KEEP_LAST = 3


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def save_versioned(obj: object, models_dir: str, base_name: str) -> str:
    """Pickle obj to versioned file, refresh latest copy + champion pointer."""
    os.makedirs(models_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000000) % 1000000:06d}"
    versioned = f"{base_name}_{stamp}_{_git_sha()}.pkl"
    versioned_path = os.path.join(models_dir, versioned)
    with open(versioned_path, "wb") as f:
        pickle.dump(obj, f)
    latest_path = os.path.join(models_dir, f"{base_name}.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(obj, f)
    with open(os.path.join(models_dir, f"{base_name}.champion.txt"), "w",
              encoding="utf-8") as f:
        f.write(versioned)
    prune_old(models_dir, base_name)
    logger.info("Saved %s (latest + champion).", versioned)
    return versioned_path


def prune_old(models_dir: str, base_name: str, keep: int = KEEP_LAST) -> None:
    """Keep newest `keep` versioned files, delete the rest."""
    pattern = os.path.join(models_dir, f"{base_name}_*.pkl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for old in files[keep:]:
        try:
            os.remove(old)
        except OSError:
            pass
