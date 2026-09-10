"""
Confidence calibration metric.

Measures whether the agent's stated confidence correlates with its actual
consistency rate across runs. A well-calibrated agent that says "I am 90%
confident" should be consistent with the reference ~90% of the time.

Confidence is extracted from the agent's output using two strategies,
tried in order:

  1. metadata["confidence"] — a float in [0, 1] if the agent exposes it
  2. Regex extraction from the text field — patterns like "confidence: 0.9",
     "I am 90% confident", "certainty: high", etc.

If no confidence signal is found, the prompt is skipped and logged.

Calibration is measured using:
  - Expected Calibration Error (ECE): mean |confidence - accuracy| per bin
  - Brier score: mean squared error between confidence and binary correctness
"""

import logging
import re
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# Regex patterns for extracting stated confidence from text
_RE_CONF_PATTERNS = [
    re.compile(r"confidence[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE),
    re.compile(r"confidence[:\s]+([0-9]*\.[0-9]+)", re.IGNORECASE),
    re.compile(r"i(?:'m| am)\s+([0-9]+(?:\.[0-9]+)?)\s*%\s+confident", re.IGNORECASE),
    re.compile(r"certainty[:\s]+(high|medium|low)", re.IGNORECASE),
    re.compile(r"probability[:\s]+([0-9]*\.[0-9]+)", re.IGNORECASE),
]

# Verbal confidence → numeric mapping
_VERBAL_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}

_N_CALIBRATION_BINS = 10


def _extract_confidence(result: RunResult) -> float | None:
    """
    Extract a stated confidence value from a RunResult.

    Tries metadata["confidence"] first, then regex extraction from text.

    Args:
        result: A single RunResult instance.

    Returns:
        Confidence as a float in [0, 1], or None if not found.
    """
    # Strategy 1: structured metadata
    meta_conf = result.metadata.get("confidence")
    if meta_conf is not None:
        try:
            val = float(meta_conf)
            return max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            pass

    # Strategy 2: regex on text
    for pattern in _RE_CONF_PATTERNS:
        match = pattern.search(result.text)
        if match:
            raw = match.group(1).strip().lower()
            if raw in _VERBAL_MAP:
                return _VERBAL_MAP[raw]
            try:
                val = float(raw)
                # Handle percentage values > 1
                if val > 1.0:
                    val = val / 100.0
                return max(0.0, min(1.0, val))
            except ValueError:
                continue

    return None


def _binary_correct(
    sc: rouge_scorer.RougeScorer,
    reference: str,
    candidate: str,
    threshold: float,
) -> int:
    """
    Return 1 if the candidate's ROUGE-L vs reference meets the threshold.

    Args:
        sc:        Shared RougeScorer instance.
        reference: Reference output (run_index=0).
        candidate: Output being evaluated.
        threshold: ROUGE-L threshold from config.

    Returns:
        1 if ROUGE-L >= threshold, else 0.
    """
    score = sc.score(reference, candidate)["rougeL"].fmeasure
    return int(score >= threshold)


def _expected_calibration_error(
    confidences: list[float],
    accuracies: list[int],
    n_bins: int = _N_CALIBRATION_BINS,
) -> float:
    """
    Compute Expected Calibration Error (ECE) over confidence/accuracy pairs.

    Partitions (confidence, accuracy) pairs into n_bins equal-width bins
    and computes the weighted mean of |mean_confidence - mean_accuracy|.

    Args:
        confidences: List of stated confidence values in [0, 1].
        accuracies:  List of binary correctness values (0 or 1).
        n_bins:      Number of calibration bins.

    Returns:
        ECE as a float in [0, 1]. Lower is better.
    """
    if not confidences:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, acc in zip(confidences, accuracies):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, acc))

    ece = 0.0
    n = len(confidences)
    for bucket in bins:
        if not bucket:
            continue
        mean_conf = np.mean([b[0] for b in bucket])
        mean_acc = np.mean([b[1] for b in bucket])
        ece += (len(bucket) / n) * abs(mean_conf - mean_acc)

    return float(ece)


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Measure alignment between stated agent confidence and actual consistency.

    Extracts confidence from each run, determines binary correctness using
    ROUGE-L vs the reference run, then computes ECE and Brier score.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``ece``                (float): Expected Calibration Error (lower = better)
          - ``brier_score``        (float): mean squared error of confidence vs correctness
          - ``confidence_mean``    (float): mean stated confidence across runs
          - ``accuracy_mean``      (float): fraction of runs that were "correct"
                                            (ROUGE-L >= threshold)
          - ``coverage``           (float): fraction of runs where confidence was found
          - ``per_run``            (list):  [{run_index, confidence, correct,
                                              rouge_l, confidence_found}]
    """
    if len(results) < 2:
        logger.warning(
            "prompt_id=%s has fewer than 2 runs — calibration skipped.",
            results[0].prompt_id if results else "unknown",
        )
        return {
            "ece": 0.0,
            "brier_score": 0.0,
            "confidence_mean": 0.0,
            "accuracy_mean": 0.0,
            "coverage": 0.0,
            "per_run": [],
        }

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    threshold: float = cfg["thresholds"]["rouge_l"]
    reference_text = results[0].text

    confidences: list[float] = []
    accuracies: list[int] = []
    per_run: list[dict[str, Any]] = []
    found_count = 0

    for r in results[1:]:
        rouge_l = sc.score(reference_text, r.text)["rougeL"].fmeasure
        correct = int(rouge_l >= threshold)
        conf = _extract_confidence(r)
        confidence_found = conf is not None

        if confidence_found:
            confidences.append(conf)
            accuracies.append(correct)
            found_count += 1

        per_run.append({
            "run_index": r.run_index,
            "confidence": conf,
            "confidence_found": confidence_found,
            "rouge_l": round(rouge_l, 6),
            "correct": correct,
        })

    n_evaluated = len(results) - 1
    coverage = found_count / n_evaluated if n_evaluated > 0 else 0.0

    if not confidences:
        logger.warning(
            "prompt_id=%s: no confidence signals found in any run — "
            "calibration metrics are 0.0. Ensure your agent exposes confidence "
            "in metadata['confidence'] or text output.",
            results[0].prompt_id,
        )
        return {
            "ece": 0.0,
            "brier_score": 0.0,
            "confidence_mean": 0.0,
            "accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
            "coverage": 0.0,
            "per_run": per_run,
        }

    brier = float(np.mean([
        (c - a) ** 2 for c, a in zip(confidences, accuracies)
    ]))
    ece = _expected_calibration_error(confidences, accuracies)

    result = {
        "ece": round(ece, 6),
        "brier_score": round(brier, 6),
        "confidence_mean": round(float(np.mean(confidences)), 6),
        "accuracy_mean": round(float(np.mean(accuracies)), 6),
        "coverage": round(coverage, 6),
        "per_run": per_run,
    }

    logger.debug(
        "Calibration | prompt_id=%s ece=%.4f brier=%.4f coverage=%.4f",
        results[0].prompt_id,
        ece,
        brier,
        coverage,
    )
    return result