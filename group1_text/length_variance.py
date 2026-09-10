"""
Output length variance metric.

Tracks token and character length distributions across runs for the same
prompt. High variance signals non-deterministic verbosity — the agent is
choosing different levels of detail across identical inputs.
"""

import logging
from typing import Any

import numpy as np

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _token_count(text: str) -> int:
    """
    Approximate token count by whitespace splitting.

    A full tokenizer is not used here to avoid a heavy dependency inside
    a lightweight metric. For subword-accurate counts, swap this for a
    tiktoken or transformers tokenizer.

    Args:
        text: Agent output string.

    Returns:
        Number of whitespace-delimited tokens.
    """
    return len(text.split())


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute length statistics (chars and tokens) across all runs.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``char_mean``    (float): mean character length across runs
          - ``char_std``     (float): std-dev of character lengths
          - ``char_cv``      (float): coefficient of variation (std/mean)
          - ``token_mean``   (float): mean whitespace-token count
          - ``token_std``    (float): std-dev of token counts
          - ``token_cv``     (float): coefficient of variation for tokens
          - ``char_min``     (int)
          - ``char_max``     (int)
          - ``token_min``    (int)
          - ``token_max``    (int)
          - ``per_run``      (list):  [{run_index, char_len, token_count}]
    """
    if not results:
        logger.warning("No results supplied to length_variance.compute.")
        return {}

    char_lens = np.array([len(r.text) for r in results], dtype=float)
    token_counts = np.array([_token_count(r.text) for r in results], dtype=float)

    def _cv(arr: np.ndarray) -> float:
        mean = float(np.mean(arr))
        return float(np.std(arr) / mean) if mean > 0 else 0.0

    per_run = [
        {
            "run_index": r.run_index,
            "char_len": int(len(r.text)),
            "token_count": int(_token_count(r.text)),
        }
        for r in results
    ]

    result = {
        "char_mean": float(np.mean(char_lens)),
        "char_std": float(np.std(char_lens)),
        "char_cv": _cv(char_lens),
        "token_mean": float(np.mean(token_counts)),
        "token_std": float(np.std(token_counts)),
        "token_cv": _cv(token_counts),
        "char_min": int(np.min(char_lens)),
        "char_max": int(np.max(char_lens)),
        "token_min": int(np.min(token_counts)),
        "token_max": int(np.max(token_counts)),
        "per_run": per_run,
    }

    logger.debug(
        "Length variance | prompt_id=%s char_mean=%.1f char_cv=%.4f token_mean=%.1f token_cv=%.4f",
        results[0].prompt_id,
        result["char_mean"],
        result["char_cv"],
        result["token_mean"],
        result["token_cv"],
    )
    return result