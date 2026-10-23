"""
recsys_core.encoding
--------------------
Canonical train/predict categorical data preparation.

This module is the *single source of truth* for how categorical features are
encoded for LightGBM and CatBoost — both pipelines and the serving layer call
these helpers, so the two sides can never drift apart.

LightGBM
    Categorical columns must be ``pandas.Categorical`` whose vocabulary comes
    from the **training split** (``category_maps``).  Values are matched by
    value, therefore they must be left untouched: training on ints and
    predicting on strings would silently turn every row into "all categories
    missing".

CatBoost
    Categorical columns are cast to ``str`` and their positions within the
    feature list are passed explicitly as ``cat_features``.
"""

from typing import Any, Mapping, Optional, Sequence

import pandas as pd


def prepare_lgb_prediction_data(df, cols: Sequence[str], category_maps: Mapping[str, Any]):
    """Build a LightGBM-ready frame using the *training* category vocabulary.

    Args:
        df: raw candidate/evaluation frame.
        cols: exact feature order (must equal the training column order).
        category_maps: mapping ``column -> Index of allowed training categories``.

    Returns:
        A new frame containing only ``cols``, in that order, with the
        categorical columns re-encoded against the persisted vocabulary.
    """
    X = df[list(cols)].copy()
    for c, categories in category_maps.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=categories)
    return X


def prepare_catboost_prediction_data(df, cols: Sequence[str], cats: Sequence[str]):
    """Build a CatBoost-ready frame: categorical columns cast to ``str``."""
    X = df[list(cols)].copy()
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X


def catboost_feature_positions(cols: Sequence[str], cats: Sequence[str]) -> list[int]:
    """Categorical column positions within ``cols`` (for ``cat_features=``)."""
    return [list(cols).index(c) for c in cats]


def best_iteration_of(model) -> Optional[int]:
    """Best iteration of an early-stopped LightGBM Booster, or ``None``.

    ``None`` lets LightGBM use every tree; passing ``0`` would be an error.
    """
    try:
        best = int(getattr(model, "best_iteration", 0) or 0)
    except Exception:
        return None
    return best if best > 0 else None


class InvalidCategoryMapError(ValueError):
    """Raised when a persisted category map cannot be restored safely."""


def validate_category_maps(category_maps: Optional[Mapping[str, Any]]) -> bool:
    """Return True when ``category_maps`` is a usable {column: Index/ndarray} map."""
    if not isinstance(category_maps, dict) or not category_maps:
        return False
    for _, categories in category_maps.items():
        if not hasattr(categories, "__len__"):
            return False
    return True
