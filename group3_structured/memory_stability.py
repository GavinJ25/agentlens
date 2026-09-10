"""
Memory and belief state stability metric.

Tests whether the agent arrives at the same beliefs and slot fills across
repeated runs by extracting key-value assertions from the output text or
reasoning trace. Assertions are detected via pattern matching on common
statement forms ("X is Y", "X = Y", "the X is Y", etc.).

This is a structural test — it does not require the agent to expose an
explicit memory API. If your agent does expose structured memory or belief
state in ``metadata``, this module also reads from ``metadata.memory`` or
``metadata.slots`` if present, preferring those over text extraction.
"""

import logging
import re
from typing import Any

import numpy as np

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# Patterns for extracting key=value style assertions from text
_RE_ASSERTION_KV = re.compile(
    r"\b([\w\s]{2,30}?)\s*(?:is|are|=|:)\s*([^\n.]{1,80})",
    re.IGNORECASE,
)
_RE_CLEAN_KEY = re.compile(r"\s+")


def _extract_slots_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """
    Extract belief slot fills from RunResult metadata if available.

    Looks for ``metadata["memory"]`` or ``metadata["slots"]`` — both are
    accepted as flat string-keyed dicts. The first found takes precedence.

    Args:
        metadata: The metadata dict from a RunResult instance.

    Returns:
        Dict of {slot_name: value_string}. Empty dict if not present.
    """
    for key in ("memory", "slots", "beliefs"):
        candidate = metadata.get(key)
        if isinstance(candidate, dict):
            return {str(k): str(v) for k, v in candidate.items()}
    return {}


def _extract_slots_from_text(text: str) -> dict[str, str]:
    """
    Extract key-value assertions from free text using regex heuristics.

    Normalises keys to lowercase with underscores. When the same key
    appears multiple times the last occurrence wins (most recent assertion).

    Args:
        text: Raw text or reasoning trace from a RunResult.

    Returns:
        Dict of {normalised_key: value_string}.
    """
    slots: dict[str, str] = {}
    for match in _RE_ASSERTION_KV.finditer(text):
        raw_key = match.group(1).strip()
        value = match.group(2).strip().rstrip(".,;")
        # Normalise key: lowercase, replace spaces with underscores
        key = _RE_CLEAN_KEY.sub("_", raw_key.lower())
        if 2 <= len(key) <= 40:  # Discard very short or very long keys
            slots[key] = value
    return slots


def _slot_overlap(ref: dict[str, str], candidate: dict[str, str]) -> dict[str, Any]:
    """
    Compare two slot-fill dicts and return overlap statistics.

    Args:
        ref:       Slot fills from the reference run (run_index=0).
        candidate: Slot fills from the run being evaluated.

    Returns:
        Dict with keys: key_jaccard, value_match_rate, keys_added,
        keys_removed, value_mismatches.
    """
    ref_keys = set(ref.keys())
    cand_keys = set(candidate.keys())
    shared_keys = ref_keys & cand_keys
    all_keys = ref_keys | cand_keys

    key_jaccard = len(shared_keys) / len(all_keys) if all_keys else 1.0

    value_matches = sum(
        1 for k in shared_keys
        if ref[k].lower().strip() == candidate[k].lower().strip()
    )
    value_match_rate = value_matches / len(shared_keys) if shared_keys else 1.0

    value_mismatches = [
        {"key": k, "reference": ref[k], "candidate": candidate[k]}
        for k in shared_keys
        if ref[k].lower().strip() != candidate[k].lower().strip()
    ]

    return {
        "key_jaccard": round(key_jaccard, 6),
        "value_match_rate": round(value_match_rate, 6),
        "keys_added": sorted(cand_keys - ref_keys),
        "keys_removed": sorted(ref_keys - cand_keys),
        "value_mismatches": value_mismatches,
    }


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute memory and belief state stability across all runs for one prompt.

    Extracts slot fills from each run (preferring metadata.memory/slots
    over text extraction) and compares all runs to run_index=0.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``key_jaccard_mean``      (float): mean key-set Jaccard similarity
          - ``value_match_rate_mean`` (float): mean exact value match rate
                                               over shared keys
          - ``slot_source``           (str):   "metadata" or "text_extraction"
          - ``reference_slot_count``  (int):   number of slots in reference run
          - ``per_run``               (list):  per-run overlap dicts
    """
    if not results:
        logger.warning("No results supplied to memory_stability.compute.")
        return {
            "key_jaccard_mean": 0.0,
            "value_match_rate_mean": 0.0,
            "slot_source": "none",
            "reference_slot_count": 0,
            "per_run": [],
        }

    # Determine slot extraction strategy from reference run
    ref_meta_slots = _extract_slots_from_metadata(results[0].metadata)
    using_metadata = bool(ref_meta_slots)
    slot_source = "metadata" if using_metadata else "text_extraction"

    def _get_slots(r: RunResult) -> dict[str, str]:
        if using_metadata:
            return _extract_slots_from_metadata(r.metadata)
        source = r.reasoning_trace if r.reasoning_trace.strip() else r.text
        return _extract_slots_from_text(source)

    ref_slots = ref_meta_slots if using_metadata else _get_slots(results[0])

    logger.debug(
        "Memory stability | prompt_id=%s source=%s ref_slots=%d",
        results[0].prompt_id,
        slot_source,
        len(ref_slots),
    )

    per_run: list[dict[str, Any]] = []
    key_jaccards: list[float] = []
    value_match_rates: list[float] = []

    for r in results:
        slots = _get_slots(r)
        overlap = _slot_overlap(ref_slots, slots)
        overlap["run_index"] = r.run_index
        overlap["slot_count"] = len(slots)
        per_run.append(overlap)
        key_jaccards.append(overlap["key_jaccard"])
        value_match_rates.append(overlap["value_match_rate"])

    result = {
        "key_jaccard_mean": float(np.mean(key_jaccards)),
        "value_match_rate_mean": float(np.mean(value_match_rates)),
        "slot_source": slot_source,
        "reference_slot_count": len(ref_slots),
        "per_run": per_run,
    }

    logger.debug(
        "Memory stability | prompt_id=%s key_jaccard=%.4f value_match=%.4f",
        results[0].prompt_id,
        result["key_jaccard_mean"],
        result["value_match_rate_mean"],
    )
    return result