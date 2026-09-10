"""
Cosine similarity determinacy metric.

Measures pairwise cosine similarity between all run embeddings for the
same prompt. Because embeddings are L2-normalised by the embedder, cosine
similarity reduces to the dot product, making all-pairs computation cheap.

The mean pairwise similarity is returned under the key ``cosine_sim``,
matching the threshold name in config.yaml.
"""

import logging
from typing import Any

import numpy as np

from agent.schema import RunResult
from group2_embeddings.embedder import load_embeddings

logger = logging.getLogger(__name__)


def _pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the upper-triangle pairwise cosine similarities for a matrix
    of L2-normalised row vectors.

    Because vectors are unit-normalised, the dot product equals cosine
    similarity. Only the strictly upper triangle (i < j) is returned to
    avoid self-similarity and duplicate pairs.

    Args:
        matrix: 2-D float32 array of shape (n, dim) with unit-norm rows.

    Returns:
        1-D array of n*(n-1)/2 cosine similarity values.
    """
    dot = matrix @ matrix.T  # (n, n)
    n = matrix.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    return dot[upper_idx]


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute pairwise cosine similarity across all run embeddings.

    The key ``cosine_sim`` in the returned dict matches the threshold
    defined in config.yaml under thresholds.cosine_sim.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``cosine_sim``      (float): mean pairwise cosine similarity
          - ``cosine_sim_std``  (float): std-dev of pairwise similarities
          - ``cosine_sim_min``  (float): minimum pairwise similarity
          - ``cosine_sim_max``  (float): maximum pairwise similarity
          - ``n_pairs``         (int):   number of pairs evaluated
    """
    if len(results) < 2:
        logger.warning(
            "prompt_id=%s has fewer than 2 runs — cosine_sim skipped.",
            results[0].prompt_id if results else "unknown",
        )
        return {
            "cosine_sim": 0.0,
            "cosine_sim_std": 0.0,
            "cosine_sim_min": 0.0,
            "cosine_sim_max": 0.0,
            "n_pairs": 0,
        }

    matrix = load_embeddings(results, cfg)
    similarities = _pairwise_cosine(matrix)

    result = {
        "cosine_sim": float(np.mean(similarities)),
        "cosine_sim_std": float(np.std(similarities)),
        "cosine_sim_min": float(np.min(similarities)),
        "cosine_sim_max": float(np.max(similarities)),
        "n_pairs": int(len(similarities)),
    }

    logger.debug(
        "Cosine sim | prompt_id=%s mean=%.4f std=%.4f min=%.4f",
        results[0].prompt_id,
        result["cosine_sim"],
        result["cosine_sim_std"],
        result["cosine_sim_min"],
    )
    return result