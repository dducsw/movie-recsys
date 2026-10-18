"""CRAG-style entity linking: mention -> catalog movieId (no extra LLM call)."""

from __future__ import annotations

import difflib
import logging
import re

logger = logging.getLogger(__name__)

_QUOTED = re.compile(r"['\"“”]([^'\"“”]{2,80})['\"“”]")


def extract_mentions(text: str) -> list[str]:
    """Quoted spans + Capitalized phrases as candidate title mentions."""
    if not text:
        return []
    out: list[str] = []
    for m in _QUOTED.findall(text):
        cand = m.strip()
        if cand and cand not in out:
            out.append(cand)
    for m in re.findall(r"\b[A-ZÀ-Þ][\w'’-]*(?:\s+[A-ZÀ-Þ0-9][\w'’-]*){0,4}\b", text):
        cand = m.strip()
        if len(cand) >= 3 and cand not in out and len(out) < 10:
            out.append(cand)
    return out[:10]


def link_titles(mentions: list[str], catalog: list[dict],
                cutoff: float = 0.6) -> list[dict]:
    """Fuzzy-match mentions to catalog rows. Returns matched rows (deduped)."""
    if not mentions or not catalog:
        return []
    titles = [str(c.get("title", "")) for c in catalog]
    lowered = [t.lower() for t in titles]
    matched: list[dict] = []
    seen: set = set()
    for mention in mentions:
        m = mention.lower().strip()
        if not m:
            continue
        if m in lowered:
            row = catalog[lowered.index(m)]
            mid = row.get("movieId") or row.get("movieid")
            if mid not in seen:
                seen.add(mid)
                matched.append(row)
            continue
        best = difflib.get_close_matches(mention, titles, n=1, cutoff=cutoff)
        if best:
            row = catalog[titles.index(best[0])]
            mid = row.get("movieId") or row.get("movieid")
            if mid not in seen:
                seen.add(mid)
                matched.append(row)
    return matched
