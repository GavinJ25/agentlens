"""
BERTScore determinacy metric.

BERTScore computes token-level contextual similarity between texts using
a pretrained BERT model. It is more sensitive than ROUGE to paraphrase
and semantic equivalence, making it a strong signal for output stability
even when exact wording varies across runs.

Run index 0 is used as the reference for each prompt. The mean F1 across
all non-reference runs is returned under ``bert_score_f1``, matching the
threshold key in config.yaml.
"""

import logging
from typing import Any

import numpy as np
from bert_score import score as bert_score_fn

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# BERTScore model — rescale_with_baseline improves human correlation.
_BERT_MODEL = "distilbert-base-uncased"


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute BERTScore F1 for all non-reference runs against run_index=0.

    BERTScore is computed in a single batched call for efficiency. The
    ``lang`` parameter is set to ``"en"``; swap to ``"multilingual"``
    or another code if your agent outputs another language.

    The key ``bert_score_f1`` matches the threshold in config.yaml.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``bert_score_f1``      (float): mean BERTScore F1
          - ``bert_score_precision``(float): mean BERTScore Precision
          - ``bert_score_recall``  (float): mean BERTScore Recall
          - ``bert_score_f1_std``  (float): std-dev of F1 across runs
          - ``per_run``            (list):  [{run_index, precision, recall, f1}]
    """
    if len(results) < 2:
        logger.warning(
            "prompt_id=%s has fewer than 2 runs — BERTScore skipped.",
            results[0].prompt_id if results else "unknown",
        )
        return {
            "bert_score_f1": 0.0,
            "bert_score_precision": 0.0,
            "bert_score_recall": 0.0,
            "bert_score_f1_std": 0.0,
            "per_run": [],
        }

    reference_text = results[0].text
    candidates = [r.text for r in results[1:]]
    references = [reference_text] * len(candidates)

    logger.debug(
        "Computing BERTScore | prompt_id=%s n_candidates=%d",
        results[0].prompt_id,
        len(candidates),
    )

    try:
        precision_t, recall_t, f1_t = bert_score_fn(
            candidates,
            references,
            model_type=_BERT_MODEL,
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )
    except Exception as exc:
        logger.error(
            "BERTScore computation failed for prompt_id=%s: %s",
            results[0].prompt_id,
            exc,
        )
        return {
            "bert_score_f1": 0.0,
            "bert_score_precision": 0.0,
            "bert_score_recall": 0.0,
            "bert_score_f1_std": 0.0,
            "per_run": [],
        }

    precision = precision_t.numpy()
    recall = recall_t.numpy()
    f1 = f1_t.numpy()

    per_run = [
        {
            "run_index": results[i + 1].run_index,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i in range(len(candidates))
    ]

    result = {
        "bert_score_f1": float(np.mean(f1)),
        "bert_score_precision": float(np.mean(precision)),
        "bert_score_recall": float(np.mean(recall)),
        "bert_score_f1_std": float(np.std(f1)),
        "per_run": per_run,
    }

    logger.debug(
        "BERTScore | prompt_id=%s f1=%.4f precision=%.4f recall=%.4f",
        results[0].prompt_id,
        result["bert_score_f1"],
        result["bert_score_precision"],
        result["bert_score_recall"],
    )
    return result