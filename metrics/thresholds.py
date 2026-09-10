"""
Pass/fail threshold evaluation.

Reads cutoff values from config.yaml under the thresholds section and
evaluates a group's score dict against them. A score "passes" if its
value meets or exceeds the configured threshold (higher is better) by
default. Metrics listed under thresholds.lower_is_better use the
opposite direction: the value must be at or below the cutoff to pass.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    """
    Return the thresholds dict from config, excluding non-threshold keys.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Dict mapping threshold name to cutoff float, e.g.
        {"rouge_l": 0.85, "bert_score_f1": 0.90, ...}.
    """
    raw = cfg.get("thresholds", {})
    return {k: v for k, v in raw.items() if isinstance(v, (int, float))}


def evaluate(scores: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate a score dict against configured thresholds.

    Only keys present in both ``scores`` and ``thresholds`` are evaluated.
    Keys in scores that have no matching threshold (e.g. ``per_run``,
    ``num_clusters``) are ignored for pass/fail purposes but preserved
    in the returned dict for reference.

    Args:
        scores: A single metric's compute() output, e.g. the dict returned
                by group1_text.rouge.compute().
        cfg:    Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``passed``      (bool): True if all evaluated metrics meet threshold
          - ``checks``      (list): [{metric, value, threshold, passed}]
          - ``checked_count`` (int): number of metrics evaluated
    """
    thresholds = get_thresholds(cfg)
    lower_is_better: set[str] = set(cfg.get("thresholds", {}).get("lower_is_better", []))
    checks: list[dict[str, Any]] = []

    for metric_name, cutoff in thresholds.items():
        if metric_name not in scores:
            continue
        value = scores[metric_name]
        if not isinstance(value, (int, float)):
            logger.warning(
                "Threshold metric '%s' has non-numeric value %r — skipping.",
                metric_name, value,
            )
            continue
        if metric_name in lower_is_better:
            passed = value <= cutoff
        else:
            passed = value >= cutoff
        checks.append({
            "metric": metric_name,
            "value": round(float(value), 6),
            "threshold": cutoff,
            "passed": passed,
        })
        if not passed:
            direction = "<=" if metric_name in lower_is_better else ">="
            logger.warning(
                "Threshold FAILED | %s = %.4f (need %s %.4f)",
                metric_name, value, direction, cutoff,
            )

    result = {
        "passed": all(c["passed"] for c in checks) if checks else True,
        "checks": checks,
        "checked_count": len(checks),
    }
    return result