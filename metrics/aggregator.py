"""
Score aggregation utilities.

Combines per-prompt metric outputs into a summary dict per group, and
combines per-group summaries into the final suite-wide report structure.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Keys that should never be averaged (they are lists, strings, or counts
# that don't make sense as a numeric mean across prompts)
_NON_AGGREGATE_KEYS = {"per_run", "per_prompt"}


def aggregate_group(
    per_prompt_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Combine per-prompt metric dicts into a single group-level summary.

    For every numeric key present across prompt score dicts, computes the
    mean and std-dev across prompts. Non-numeric keys (e.g. ``per_run``)
    are dropped from the summary but the full per-prompt detail is
    preserved under ``by_prompt``.

    Args:
        per_prompt_scores: Dict mapping prompt_id to a metric compute()
                            output dict.

    Returns:
        Dict with keys:
          - ``summary``    (dict): {metric_name: {mean, std}} across prompts
          - ``by_prompt``  (dict): the original per-prompt scores, unmodified
          - ``prompt_count`` (int): number of prompts aggregated
    """
    if not per_prompt_scores:
        logger.warning("aggregate_group received no per-prompt scores.")
        return {"summary": {}, "by_prompt": {}, "prompt_count": 0}

    # Collect all numeric keys across all prompts
    numeric_keys: set[str] = set()
    for scores in per_prompt_scores.values():
        for key, value in scores.items():
            if key in _NON_AGGREGATE_KEYS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    summary: dict[str, dict[str, float]] = {}
    for key in sorted(numeric_keys):
        values = [
            scores[key]
            for scores in per_prompt_scores.values()
            if key in scores and isinstance(scores[key], (int, float))
        ]
        if not values:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "summary": summary,
        "by_prompt": per_prompt_scores,
        "prompt_count": len(per_prompt_scores),
    }


def aggregate_suite(
    group_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Combine all group-level aggregated results into the final suite report.

    Args:
        group_results: Dict mapping group name (e.g. "g1_text") to the
                        output of aggregate_group() for that group.

    Returns:
        Dict with keys:
          - ``groups``        (dict): the input, passed through unmodified
          - ``overall_passed``(bool): True if every group's threshold checks
                                      passed (requires "threshold_check" key
                                      to be present per group, set by reporter)
          - ``group_count``   (int):  number of groups included
    """
    overall_passed = True
    for group_name, group_data in group_results.items():
        check = group_data.get("threshold_check")
        if check is not None and not check.get("passed", True):
            overall_passed = False
            logger.warning("Group '%s' failed threshold checks.", group_name)

    return {
        "groups": group_results,
        "overall_passed": overall_passed,
        "group_count": len(group_results),
    }