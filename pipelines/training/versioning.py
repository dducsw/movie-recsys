"""
pipelines.training.versioning
-----------------------------
Versioned artifact storage with an explicit **promotion** step.

Layout inside ``models_dir``::

    {base_name}_{timestamp}_{gitsha}.pkl     immutable versioned artifact
    {base_name}.pkl                          "latest" pointer (last promotion)
    {base_name}.champion.txt                 name of the promoted version

The lifecycle is deliberately two-phase:

1. :func:`save_versioned` writes an *immutable candidate* artifact.  It never
   touches the ``latest``/``champion`` pointers.
2. :func:`promote` moves the pointers.  This must only be called **after** the
   quality gate has passed.

That ordering is what guarantees that a gate-failing model can never end up
serving traffic.  :func:`rollback` moves the pointers back to a previous
version.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
import time
from typing import Any, Mapping, Optional

from recsys_core.envinfo import git_sha

logger = logging.getLogger(__name__)

KEEP_LAST = 3


def _git_sha() -> str:
    """Short SHA (used in file names) — see :mod:`recsys_core.envinfo`."""
    return git_sha(short=True)


def _champion_pointer_path(models_dir: str, base_name: str) -> str:
    return os.path.join(models_dir, f"{base_name}.champion.txt")


def _write_pointer(models_dir: str, base_name: str, versioned_name: str) -> None:
    with open(_champion_pointer_path(models_dir, base_name), "w", encoding="utf-8") as f:
        f.write(versioned_name)


def save_versioned(obj: object, models_dir: str, base_name: str) -> str:
    """Pickle ``obj`` to an immutable versioned file and return its path.

    This does **not** promote the artifact: the ``latest`` and ``champion``
    pointers are untouched.  Call :func:`promote` once the quality gate passes.
    """
    os.makedirs(models_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000000) % 1000000:06d}"
    versioned = f"{base_name}_{stamp}_{_git_sha()}.pkl"
    versioned_path = os.path.join(models_dir, versioned)
    with open(versioned_path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved candidate artifact %s (not promoted).", versioned)
    return versioned_path


def promote(models_dir: str, base_name: str, versioned_path: str) -> str:
    """Promote a previously saved candidate to be the serving artifact.

    Writes both the ``latest`` copy and the ``champion`` pointer, then prunes
    old versions (never the champion).

    Returns the versioned file name (not the full path).
    """
    if not os.path.exists(versioned_path):
        raise FileNotFoundError(f"Cannot promote missing artifact: {versioned_path}")

    versioned_name = os.path.basename(versioned_path)
    latest_path = os.path.join(models_dir, f"{base_name}.pkl")
    with open(versioned_path, "rb") as src, open(latest_path, "wb") as dst:
        dst.write(src.read())

    _write_pointer(models_dir, base_name, versioned_name)
    prune_old(models_dir, base_name)
    logger.info("Promoted %s -> champion.", versioned_name)
    return versioned_name


def read_champion(models_dir: str, base_name: str) -> Optional[str]:
    """Return the name of the currently promoted version, or ``None``."""
    pointer = _champion_pointer_path(models_dir, base_name)
    if not os.path.exists(pointer):
        return None
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            name = f.read().strip()
        return name or None
    except OSError:
        return None


def save_metrics(
    models_dir: str,
    base_name: str,
    versioned_name: str,
    metrics: Mapping[str, Any],
) -> str:
    """Persist the metrics of a promoted version next to the artifact.

    The sidecar lets the *next* CT run compare the challenger against the
    incumbent instead of only checking absolute thresholds.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{base_name}.{versioned_name}.metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(metrics), f, indent=2, default=str)
    return path


def load_champion_metrics(models_dir: str, base_name: str) -> Optional[dict[str, Any]]:
    """Load the metrics of the currently promoted version, or ``None``."""
    champion = read_champion(models_dir, base_name)
    if not champion:
        return None
    path = os.path.join(models_dir, f"{base_name}.{champion}.metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def rollback(models_dir: str, base_name: str) -> Optional[str]:
    """Re-point ``latest``/``champion`` to the previous version.

    The champion pointer keeps a single name, so the rollback target is the
    newest versioned artifact that is *not* the current champion.  Returns the
    restored version name, or ``None`` when there is nothing to roll back to.
    """
    current = read_champion(models_dir, base_name)
    versions = list_versions(models_dir, base_name)
    candidates = [v for v in versions if v != current]
    if not candidates:
        logger.warning("Rollback requested for %s but no previous version exists.", base_name)
        return None

    target = candidates[0]
    promote(models_dir, base_name, os.path.join(models_dir, target))
    logger.warning("Rolled back %s from %s to %s.", base_name, current, target)
    return target


def list_versions(models_dir: str, base_name: str) -> list[str]:
    """Return versioned artifact names for ``base_name``, newest first."""
    pattern = os.path.join(models_dir, f"{base_name}_*.pkl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]


def prune_old(models_dir: str, base_name: str, keep: int = KEEP_LAST) -> None:
    """Keep the newest ``keep`` versioned files; never delete the champion."""
    pattern = os.path.join(models_dir, f"{base_name}_*.pkl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    champion = read_champion(models_dir, base_name)
    for old in files[keep:]:
        if champion and os.path.basename(old) == champion:
            continue
        try:
            os.remove(old)
        except OSError:
            pass
