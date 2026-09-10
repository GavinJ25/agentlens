"""
Exact match rate metric.

Hashes the text output of every run and measures what fraction of runs
produce bit-for-bit identical output. A perfectly deterministic agent at
temperature=0 should score 1.0. Any variation signals non-determinism.
"""

import hashlib
import logging
from collections import Counter
from typing import Any

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    """
    Return a stable SHA-256 hex digest of a UTF-8 encoded text string.

    Args:
        text: Agent output string to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute exact match rate across all runs for one prompt.

    The majority output hash is determined first, then the fraction of
    runs matching that majority is returned as ``exact_match_rate``.
    This key matches the threshold name in config.yaml.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``exact_match_rate`` (float): fraction of runs producing the
                                          majority output (0.0 – 1.0)
          - ``unique_outputs``   (int):   number of distinct output hashes
          - ``majority_hash``    (str):   hash of the most common output
          - ``hash_counts``      (dict):  {hash: count} for all observed hashes
          - ``per_run``          (list):  [{run_index, hash, is_majority}]
    """
    if not results:
        logger.warning("No results supplied to exact_match.compute.")
        return {
            "exact_match_rate": 0.0,
            "unique_outputs": 0,
            "majority_hash": "",
            "hash_counts": {},
            "per_run": [],
        }

    hashes = [_text_hash(r.text) for r in results]
    counts: Counter = Counter(hashes)
    majority_hash, majority_count = counts.most_common(1)[0]
    exact_match_rate = majority_count / len(results)

    per_run = [
        {
            "run_index": r.run_index,
            "hash": h,
            "is_majority": h == majority_hash,
        }
        for r, h in zip(results, hashes)
    ]

    result = {
        "exact_match_rate": float(exact_match_rate),
        "unique_outputs": len(counts),
        "majority_hash": majority_hash,
        "hash_counts": dict(counts),
        "per_run": per_run,
    }

    logger.debug(
        "Exact match | prompt_id=%s rate=%.4f unique=%d",
        results[0].prompt_id,
        exact_match_rate,
        len(counts),
    )
    return result