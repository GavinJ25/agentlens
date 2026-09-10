"""
Reasoning step consistency metric.

Analyses the internal reasoning trace (or text fallback) for consistency
across runs along three dimensions:

  1. Step count stability    — does the agent use the same number of steps?
  2. Topic order consistency — do the same topics appear in the same order?
  3. Entity overlap          — are the same named entities mentioned?

This gives a richer picture than decision_path.py (which measures edit
distance on step strings) by focusing on what the agent is reasoning
*about* rather than the exact wording of each step.
"""

import logging
import re
from typing import Any

import numpy as np

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# Lightweight entity heuristics — capitalised words not at sentence start
_RE_ENTITY = re.compile(r"(?<!\.\s)(?<!\n)(?<!\A)\b([A-Z][a-zA-Z0-9_\-]+)\b")
# Step boundary markers
_RE_STEP_BOUNDARY = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]\s+|step\s+\d+[.:]?\s+|→\s+|[-*]\s+)",
    re.IGNORECASE,
)


def _split_steps(text: str) -> list[str]:
    """
    Split a reasoning trace into individual step strings.

    Args:
        text: Reasoning trace or text output from a RunResult.

    Returns:
        List of non-empty step strings in document order.
    """
    parts = _RE_STEP_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _extract_entities(text: str) -> set[str]:
    """
    Extract a set of candidate named entities using capitalisation heuristics.

    This is intentionally lightweight — no NER model dependency. Swap for
    spaCy ``ents`` if the agent's domain requires precision.

    Args:
        text: Step text or full trace.

    Returns:
        Set of unique candidate entity strings (lowercased for comparison).
    """
    return {m.group(1).lower() for m in _RE_ENTITY.finditer(text)}


def _topic_order_similarity(steps_a: list[str], steps_b: list[str]) -> float:
    """
    Measure topic order consistency between two step sequences using the
    longest common subsequence (LCS) ratio of their extracted entity sets.

    Args:
        steps_a: Step list from the reference run.
        steps_b: Step list from the candidate run.

    Returns:
        LCS-based similarity in [0, 1]. 1.0 means same topic order.
    """
    topics_a = [frozenset(_extract_entities(s)) for s in steps_a]
    topics_b = [frozenset(_extract_entities(s)) for s in steps_b]

    # LCS on topic sets — two steps "match" if their entity overlap is ≥ 0.5
    m, n = len(topics_a), len(topics_b)
    if m == 0 or n == 0:
        return 1.0 if m == n else 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if topics_a[i - 1] and topics_b[j - 1]:
                inter = len(topics_a[i - 1] & topics_b[j - 1])
                union = len(topics_a[i - 1] | topics_b[j - 1])
                jaccard = inter / union if union > 0 else 0.0
            else:
                jaccard = 1.0 if topics_a[i - 1] == topics_b[j - 1] else 0.0
            dp[i][j] = dp[i - 1][j - 1] + jaccard if jaccard >= 0.5 else max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    return float(lcs / max(m, n))


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute reasoning step consistency across all runs for one prompt.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``step_count_mean``       (float): mean step count across runs
          - ``step_count_std``        (float): std-dev of step counts
          - ``topic_order_similarity``(float): mean topic-order similarity vs reference
          - ``entity_overlap_mean``   (float): mean Jaccard entity overlap vs reference
          - ``using_trace``           (bool):  True if reasoning_trace was used
          - ``per_run``               (list):  [{run_index, step_count,
                                                 topic_order_sim, entity_overlap}]
    """
    if not results:
        logger.warning("No results supplied to reasoning_steps.compute.")
        return {
            "step_count_mean": 0.0,
            "step_count_std": 0.0,
            "topic_order_similarity": 0.0,
            "entity_overlap_mean": 0.0,
            "using_trace": False,
            "per_run": [],
        }

    using_trace = bool(results[0].reasoning_trace.strip())
    get_text = (lambda r: r.reasoning_trace) if using_trace else (lambda r: r.text)

    if not using_trace:
        logger.debug(
            "prompt_id=%s: reasoning_trace empty — using text for reasoning_steps.",
            results[0].prompt_id,
        )

    ref_steps = _split_steps(get_text(results[0]))
    ref_entities = _extract_entities(get_text(results[0]))

    per_run: list[dict[str, Any]] = []
    step_counts: list[int] = []
    topic_sims: list[float] = []
    entity_overlaps: list[float] = []

    for r in results:
        steps = _split_steps(get_text(r))
        entities = _extract_entities(get_text(r))

        topic_sim = _topic_order_similarity(ref_steps, steps)
        if ref_entities or entities:
            inter = len(ref_entities & entities)
            union = len(ref_entities | entities)
            entity_overlap = inter / union if union > 0 else 1.0
        else:
            entity_overlap = 1.0

        step_counts.append(len(steps))
        topic_sims.append(topic_sim)
        entity_overlaps.append(entity_overlap)

        per_run.append({
            "run_index": r.run_index,
            "step_count": len(steps),
            "topic_order_sim": round(topic_sim, 6),
            "entity_overlap": round(entity_overlap, 6),
        })

    result = {
        "step_count_mean": float(np.mean(step_counts)),
        "step_count_std": float(np.std(step_counts)),
        "topic_order_similarity": float(np.mean(topic_sims)),
        "entity_overlap_mean": float(np.mean(entity_overlaps)),
        "using_trace": using_trace,
        "per_run": per_run,
    }

    logger.debug(
        "Reasoning steps | prompt_id=%s steps_mean=%.1f topic_sim=%.4f entity_overlap=%.4f",
        results[0].prompt_id,
        result["step_count_mean"],
        result["topic_order_similarity"],
        result["entity_overlap_mean"],
    )
    return result