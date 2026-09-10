"""
Decision path determinacy metric.

Extracts the sequence of decision steps from each run's reasoning_trace
and measures consistency across runs using Levenshtein edit distance and
a normalised similarity score.

Decision steps are detected by line-level pattern matching on common
reasoning markers (numbered steps, "Step N:", "→", "Therefore", etc.).
If no reasoning trace is available, the metric operates on the text
field instead and logs a notice.
"""

import logging
import re
from typing import Any

import numpy as np

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# Patterns that signal a decision or reasoning step
_STEP_PATTERNS = [
    re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE),       # 1. or 1) item
    re.compile(r"^\s*step\s+\d+", re.MULTILINE | re.IGNORECASE),  # Step N:
    re.compile(r"^\s*→\s+\S", re.MULTILINE),              # → action
    re.compile(r"^\s*(therefore|thus|so|hence|first|then|finally)[,:]?\s+\S",
               re.MULTILINE | re.IGNORECASE),
]


def _extract_steps(text: str) -> list[str]:
    """
    Extract decision step strings from a reasoning trace or text output.

    Collects lines matching any step pattern, normalises whitespace, and
    returns them in document order. Duplicates are preserved because the
    same action appearing twice is meaningful.

    Args:
        text: Raw reasoning_trace or text string from a RunResult.

    Returns:
        Ordered list of normalised step strings. Empty list if none found.
    """
    lines = text.splitlines()
    steps: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _STEP_PATTERNS:
            if pattern.match(line):
                steps.append(re.sub(r"\s+", " ", stripped))
                break
    return steps


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """
    Compute Levenshtein edit distance between two lists of step strings.

    Each element is treated as an atomic token; equality is exact string
    match after normalisation.

    Args:
        seq_a: Decision step list from one run.
        seq_b: Decision step list from another run.

    Returns:
        Integer edit distance (insertions + deletions + substitutions).
    """
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


def _normalised_similarity(dist: int, len_a: int, len_b: int) -> float:
    """
    Convert a raw edit distance to a normalised similarity in [0, 1].

    Normalises by the maximum possible distance (max(len_a, len_b)) so
    that long and short traces are comparable.

    Args:
        dist:  Raw Levenshtein distance.
        len_a: Length of the first sequence.
        len_b: Length of the second sequence.

    Returns:
        Similarity score where 1.0 = identical, 0.0 = completely different.
    """
    max_len = max(len_a, len_b, 1)
    return 1.0 - (dist / max_len)


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute decision path consistency across all runs for one prompt.

    Compares each run's extracted step sequence against the reference
    (run_index=0) using Levenshtein distance and normalised similarity.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``decision_path_similarity`` (float): mean normalised similarity
                                                   vs reference path
          - ``decision_path_sim_std``   (float): std-dev of similarities
          - ``mean_edit_distance``      (float): mean raw edit distance
          - ``reference_step_count``    (int):   steps in reference run
          - ``using_trace``             (bool):  True if reasoning_trace was used,
                                                  False if fell back to text
          - ``per_run``                 (list):  [{run_index, step_count,
                                                   edit_distance, similarity}]
    """
    if not results:
        logger.warning("No results supplied to decision_path.compute.")
        return {
            "decision_path_similarity": 0.0,
            "decision_path_sim_std": 0.0,
            "mean_edit_distance": 0.0,
            "reference_step_count": 0,
            "using_trace": False,
            "per_run": [],
        }

    # Prefer reasoning_trace; fall back to text if trace is empty
    using_trace = bool(results[0].reasoning_trace.strip())
    source_text = (
        (lambda r: r.reasoning_trace) if using_trace
        else (lambda r: r.text)
    )
    if not using_trace:
        logger.debug(
            "prompt_id=%s: reasoning_trace is empty — using text field for decision path.",
            results[0].prompt_id,
        )

    ref_steps = _extract_steps(source_text(results[0]))
    ref_len = len(ref_steps)

    per_run: list[dict[str, Any]] = []
    similarities: list[float] = []
    edit_distances: list[int] = []

    for r in results:
        steps = _extract_steps(source_text(r))
        dist = _levenshtein(ref_steps, steps)
        sim = _normalised_similarity(dist, ref_len, len(steps))
        similarities.append(sim)
        edit_distances.append(dist)

        per_run.append({
            "run_index": r.run_index,
            "step_count": len(steps),
            "edit_distance": dist,
            "similarity": round(sim, 6),
        })

    result = {
        "decision_path_similarity": float(np.mean(similarities)),
        "decision_path_sim_std": float(np.std(similarities)),
        "mean_edit_distance": float(np.mean(edit_distances)),
        "reference_step_count": ref_len,
        "using_trace": using_trace,
        "per_run": per_run,
    }

    logger.debug(
        "Decision path | prompt_id=%s similarity=%.4f edit_dist=%.2f ref_steps=%d",
        results[0].prompt_id,
        result["decision_path_similarity"],
        result["mean_edit_distance"],
        ref_len,
    )
    return result