"""
BLEU and METEOR determinacy metrics.

BLEU measures modified n-gram precision (good for exact phrasing consistency).
METEOR additionally accounts for stemming and synonym matches (better recall).

Both metrics treat run_index=0 as the reference for each prompt.
"""

import logging
from typing import Any

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _bleu(reference: str, hypothesis: str) -> float:
    """
    Compute sentence-level BLEU-4 with add-1 smoothing.

    Args:
        reference:  Gold run text.
        hypothesis: Run text being evaluated.

    Returns:
        BLEU-4 score as a float in [0, 1].
    """
    ref_tokens = [word_tokenize(reference.lower())]
    hyp_tokens = word_tokenize(hypothesis.lower())
    smoother = SmoothingFunction().method1
    return float(sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother))


def _meteor(reference: str, hypothesis: str) -> float:
    """
    Compute METEOR score between reference and hypothesis.

    Args:
        reference:  Gold run text.
        hypothesis: Run text being evaluated.

    Returns:
        METEOR score as a float in [0, 1].
    """
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    return float(meteor_score([ref_tokens], hyp_tokens))


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute mean BLEU-4 and METEOR scores across all runs for one prompt.

    Run index 0 is used as the reference. Requires NLTK data packages
    ``punkt`` and ``wordnet`` to be downloaded (handled in requirements
    setup — run ``python -m nltk.downloader punkt wordnet`` once).

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``bleu``      (float): mean BLEU-4 across all non-reference runs
          - ``meteor``    (float): mean METEOR score
          - ``bleu_std``  (float): std-dev of BLEU scores
          - ``meteor_std``(float): std-dev of METEOR scores
          - ``per_run``   (list):  per-run score dicts
    """
    if len(results) < 2:
        logger.warning(
            "prompt_id=%s has fewer than 2 runs — BLEU/METEOR skipped.",
            results[0].prompt_id if results else "unknown",
        )
        return {"bleu": 0.0, "meteor": 0.0, "bleu_std": 0.0, "meteor_std": 0.0, "per_run": []}

    reference = results[0].text
    per_run: list[dict[str, Any]] = []

    for run in results[1:]:
        per_run.append({
            "run_index": run.run_index,
            "bleu": _bleu(reference, run.text),
            "meteor": _meteor(reference, run.text),
        })

    bleu_scores = [r["bleu"] for r in per_run]
    meteor_scores = [r["meteor"] for r in per_run]

    result = {
        "bleu": float(np.mean(bleu_scores)),
        "meteor": float(np.mean(meteor_scores)),
        "bleu_std": float(np.std(bleu_scores)),
        "meteor_std": float(np.std(meteor_scores)),
        "per_run": per_run,
    }

    logger.debug(
        "BLEU/METEOR | prompt_id=%s bleu=%.4f meteor=%.4f",
        results[0].prompt_id,
        result["bleu"],
        result["meteor"],
    )
    return result