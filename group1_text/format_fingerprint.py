"""
Format fingerprint metric.

Captures structural features of each run's text output — section count,
bullet/list depth, heading count, prose-to-list ratio — without doing any
semantic comparison. Detects format drift (e.g. the agent switching from
prose to bullet points) independently of content changes.
"""

import logging
import re
from typing import Any

import numpy as np

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# Patterns for structural feature extraction
_RE_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_RE_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_RE_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_RE_SENTENCE = re.compile(r"[.!?]+\s")


def _fingerprint(text: str) -> dict[str, Any]:
    """
    Extract structural features from a single text output.

    Args:
        text: Agent output string to fingerprint.

    Returns:
        Dict of integer/float structural counts and ratios.
    """
    headings = len(_RE_HEADING.findall(text))
    bullets = len(_RE_BULLET.findall(text))
    numbered = len(_RE_NUMBERED.findall(text))
    code_blocks = len(_RE_CODE_BLOCK.findall(text))
    list_items = bullets + numbered
    sentences = len(_RE_SENTENCE.findall(text)) + 1  # +1 for last sentence
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    total_lines = len(text.splitlines())
    list_lines = bullets + numbered
    prose_ratio = (total_lines - list_lines) / total_lines if total_lines > 0 else 1.0

    return {
        "headings": headings,
        "bullets": bullets,
        "numbered_items": numbered,
        "code_blocks": code_blocks,
        "list_items": list_items,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "prose_ratio": round(prose_ratio, 4),
    }


def _feature_std(per_run: list[dict[str, Any]], feature: str) -> float:
    """
    Compute std-dev of a single feature across all run fingerprints.

    Args:
        per_run:  List of fingerprint dicts.
        feature:  Key name of the feature to aggregate.

    Returns:
        Standard deviation as a float.
    """
    values = [r[feature] for r in per_run]
    return float(np.std(values))


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute format fingerprints for all runs and measure structural variance.

    A high std-dev on any structural feature indicates the agent is
    switching formats across identical inputs (e.g. sometimes using
    bullet lists, sometimes prose paragraphs).

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``per_run``         (list):  fingerprint dict per run
          - ``std``             (dict):  std-dev of each feature across runs
          - ``format_drift``    (bool):  True if any feature std-dev > 0.5
          - ``most_variable``   (str):   name of the highest-variance feature
    """
    if not results:
        logger.warning("No results supplied to format_fingerprint.compute.")
        return {}

    per_run = []
    for r in results:
        fp = _fingerprint(r.text)
        fp["run_index"] = r.run_index
        per_run.append(fp)

    features = ["headings", "bullets", "numbered_items", "code_blocks",
                "list_items", "sentences", "paragraphs", "prose_ratio"]

    std_by_feature = {f: _feature_std(per_run, f) for f in features}
    most_variable = max(std_by_feature, key=lambda k: std_by_feature[k])
    format_drift = any(v > 0.5 for v in std_by_feature.values())

    result = {
        "per_run": per_run,
        "std": std_by_feature,
        "format_drift": format_drift,
        "most_variable": most_variable,
    }

    logger.debug(
        "Format fingerprint | prompt_id=%s drift=%s most_variable=%s (std=%.3f)",
        results[0].prompt_id,
        format_drift,
        most_variable,
        std_by_feature[most_variable],
    )
    return result