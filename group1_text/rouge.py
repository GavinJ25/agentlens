"""
ROUGE-1, ROUGE-2, and ROUGE-L determinacy metrics.

Treats run_index=0 as the reference (gold) run for each prompt and scores
every other run against it. Reports the mean and std-dev of each ROUGE
variant across all non-reference runs, aggregated over all prompts.
"""

import logging
from typing import Any

from rouge_score import rouge_scorer

from agent.schema import RunResult

logger = logging.getLogger(__name__)

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


def _score_pair(
    scorer: rouge_scorer.RougeScorer,
    reference: str,
    hypothesis: str,
) -> dict[str, float]:
    """
    Score one hypothesis against one reference, returning F1 for each type.

    Args:
        scorer:     Initialised RougeScorer instance.
        reference:  Gold run text.
        hypothesis: Run text being evaluated.

    Returns:
        Dict of {rouge1: float, rouge2: float, rougeL: float} F1 scores.
    """
    scores = scorer.score(reference, hypothesis)
    return {key: scores[key].fmeasure for key in _ROUGE_TYPES}


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute mean ROUGE-1, ROUGE-2, and ROUGE-L across all runs per prompt.

    Run index 0 is used as the reference for each prompt. All remaining
    runs are scored against it. The returned dict contains per-prompt
    breakdowns and overall means suitable for threshold comparison.

    The key ``rouge_l`` in the returned dict matches the threshold name
    defined in config.yaml under thresholds.rouge_l.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``rouge_l``      (float): mean ROUGE-L F1 across all run pairs
          - ``rouge1``       (float): mean ROUGE-1 F1
          - ``rouge2``       (float): mean ROUGE-2 F1
          - ``rouge_l_std``  (float): std-dev of ROUGE-L scores
          - ``per_run``      (list):  per-run score dicts
    """
    import numpy as np

    if len(results) < 2:
        logger.warning(
            "prompt_id=%s has fewer than 2 runs — ROUGE metrics skipped.",
            results[0].prompt_id if results else "unknown",
        )
        return {"rouge_l": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rouge_l_std": 0.0, "per_run": []}

    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)
    reference = results[0].text
    per_run: list[dict[str, float]] = []

    for run in results[1:]:
        pair_scores = _score_pair(scorer, reference, run.text)
        pair_scores["run_index"] = run.run_index
        per_run.append(pair_scores)

    rouge1_scores = [r["rouge1"] for r in per_run]
    rouge2_scores = [r["rouge2"] for r in per_run]
    rougel_scores = [r["rougeL"] for r in per_run]

    result = {
        "rouge_l": float(np.mean(rougel_scores)),
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rouge_l_std": float(np.std(rougel_scores)),
        "per_run": per_run,
    }

    logger.debug(
        "ROUGE | prompt_id=%s rouge_l=%.4f rouge1=%.4f rouge2=%.4f",
        results[0].prompt_id,
        result["rouge_l"],
        result["rouge1"],
        result["rouge2"],
    )
    return result