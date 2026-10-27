"""
pipelines/feature_store/feast_repo/parity_check.py
---------------------------------------------------
Offline↔online feature parity check for the Feast Redis online store.

Silent skew between the offline (PostgreSQL) feature table and the online
(Feast → Redis) store is one of the most common causes of production model
degradation that never shows up in offline eval — the model is scored with
different numbers than it was trained against.  This check samples N entity
rows, reads the same features from both stores, and fails loudly on a
mismatch rate above the tolerance.

Run standalone (needs the ``feast`` / ``psycopg2`` deps and a live stack):
    python -m pipelines.feature_store.feast_repo.parity_check [--sample-size 100]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipelines.feature_store.feature.config import (  # noqa: E402
    POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER,
)

logger = logging.getLogger("FeastParityCheck")

FEATURES = [
    "popularity", "vote_average", "vote_count", "release_year",
]
MATCH_TOLERANCE = 1e-3  # float32 online store rounds; allow small absolute drift
MAX_MISMATCH_RATE = 0.01


def _offline_values(movie_ids: list[int]) -> Dict[int, dict[str, Any]]:
    """Read the same feature definitions the Feast SQL source computes."""
    import pandas as pd
    import psycopg2

    ids_csv = ",".join(str(int(m)) for m in movie_ids)
    with psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=3,
    ) as conn:
        df = pd.read_sql(
            f"""
            SELECT movieid AS "movieId",
                   popularity::float4 AS popularity,
                   vote_average::float4 AS vote_average,
                   COALESCE(vote_count, 0)::int8 AS vote_count,
                   COALESCE(NULLIF(SUBSTRING(release_date, 1, 4), ''), '2010')::int8 AS release_year
            FROM movies
            WHERE movieid IN ({ids_csv})
            """, conn,
        )
    return {int(r["movieId"]): r for _, r in df.iterrows()}


def check_parity(sample_size: int = 100, seed: int = 42) -> bool:
    """Compare Feast online values vs the offline SQL source for a sample."""
    from feast import FeatureStore

    repo_path = os.path.dirname(os.path.abspath(__file__))
    store = FeatureStore(repo_path=repo_path)

    ids = _sample_movie_ids(sample_size, seed)
    offline = _offline_values(ids)

    online_rows = store.get_online_features(
        features=[f"movie_stats:{f}" for f in FEATURES],
        entity_rows=[{"movieId": mid} for mid in ids],
    ).to_df()
    online = {int(r["movieId"]): r for _, r in online_rows.iterrows()}

    mismatches = []
    compared = 0
    for mid in ids:
        if mid not in offline or mid not in online:
            mismatches.append((mid, "missing", None, None))
            continue
        o, n = offline[mid], online[mid]
        for f in FEATURES:
            off, on = float(o[f]), float(n.get(f) or 0.0)
            if abs(off - on) > MATCH_TOLERANCE * max(1.0, abs(off)):
                mismatches.append((mid, f, off, on))
            compared += 1

    rate = len(mismatches) / max(compared, 1)
    logger.info("Parity check: %d entities, %d feature cells, mismatch rate %.4f",
                len(ids), compared, rate)
    if mismatches:
        for mid, f, off, on in mismatches[:10]:
            logger.warning("Parity mismatch movie=%s feature=%s offline=%s online=%s",
                           mid, f, off, on)

    ok = rate <= MAX_MISMATCH_RATE
    if not ok:
        logger.error("Parity FAIL: mismatch rate %.4f > %.4f — the online store "
                     "is serving features the offline source does not agree with.",
                     rate, MAX_MISMATCH_RATE)
    else:
        logger.info("Parity OK.")
    return ok


def _sample_movie_ids(sample_size: int, seed: int) -> list[int]:
    """Sample movieIds that exist in the offline source."""
    import pandas as pd
    import psycopg2

    with psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=3,
    ) as conn:
        ids = pd.read_sql("SELECT movieid AS \"movieId\" FROM movies",
                          conn)["movieId"]
    rng = random.Random(seed)
    pool = list(ids.astype(int).unique())
    rng.shuffle(pool)
    return pool[:sample_size]


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Feast offline↔online parity check")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    ok = check_parity(sample_size=args.sample_size)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
