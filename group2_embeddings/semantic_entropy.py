"""
Semantic entropy determinacy metric.

Clusters run embeddings by cosine similarity and computes Shannon entropy
over the resulting cluster distribution. Low entropy means the agent
reliably produces outputs in the same semantic neighbourhood — high
entropy signals scattered, inconsistent outputs.

Inspired by the semantic entropy approach from Kuhn et al. (2023)
"Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
in Natural Language Generation".

This module uses embedding vectors from group2_embeddings.embedder rather
than NLI entailment, making it fast and dependency-light while still
capturing semantic rather than surface variation.
"""

import logging
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from agent.schema import RunResult
from group2_embeddings.embedder import load_embeddings

logger = logging.getLogger(__name__)


def _cluster_embeddings(
    matrix: np.ndarray,
    distance_threshold: float = 0.15,
) -> np.ndarray:
    """
    Hierarchically cluster L2-normalised embedding vectors by cosine distance.

    Cosine distance = 1 - cosine_similarity. A threshold of 0.15 means
    vectors with cosine similarity ≥ 0.85 are placed in the same cluster,
    consistent with the ``cosine_sim`` threshold in config.yaml.

    Args:
        matrix:             2-D array of shape (n_runs, dim), unit-norm rows.
        distance_threshold: Maximum cosine distance within a cluster.

    Returns:
        1-D integer array of cluster labels, one per row of matrix.
    """
    if matrix.shape[0] == 1:
        return np.array([0])

    # Cosine distance for unit vectors: 1 - dot(a, b)
    dot = matrix @ matrix.T
    # Clip to [0, 2] to guard against floating-point drift above 1.0
    distance_matrix = np.clip(1.0 - dot, 0.0, 2.0)

    # Convert square distance matrix to condensed form for scipy
    n = distance_matrix.shape[0]
    condensed = distance_matrix[np.triu_indices(n, k=1)]

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")
    return labels - 1  # fcluster is 1-indexed; shift to 0-indexed


def _shannon_entropy(labels: np.ndarray) -> float:
    """
    Compute Shannon entropy (bits) over a discrete label distribution.

    Args:
        labels: Integer cluster label per run.

    Returns:
        Entropy in bits. Zero means all runs are in the same cluster.
    """
    n = len(labels)
    counts = np.bincount(labels)
    probs = counts[counts > 0] / n
    return float(-np.sum(probs * np.log2(probs)))


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute semantic entropy over clustered run embeddings for one prompt.

    Embeddings are loaded from the cache written by embedder.py. Runs are
    clustered by cosine distance using a threshold derived from the
    ``thresholds.cosine_sim`` config value (distance = 1 - similarity).

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``semantic_entropy``   (float): Shannon entropy over cluster sizes (bits)
          - ``num_clusters``       (int):   number of distinct semantic clusters
          - ``majority_cluster_size`` (int): size of the largest cluster
          - ``consistency_ratio``  (float): fraction of runs in majority cluster
          - ``distance_threshold`` (float): cosine distance threshold used
          - ``per_run``            (list):  [{run_index, cluster_id}]
    """
    if not results:
        logger.warning("No results supplied to semantic_entropy.compute.")
        return {
            "semantic_entropy": 0.0,
            "num_clusters": 0,
            "majority_cluster_size": 0,
            "consistency_ratio": 0.0,
            "distance_threshold": 0.0,
            "per_run": [],
        }

    if len(results) == 1:
        return {
            "semantic_entropy": 0.0,
            "num_clusters": 1,
            "majority_cluster_size": 1,
            "consistency_ratio": 1.0,
            "distance_threshold": 0.0,
            "per_run": [{"run_index": results[0].run_index, "cluster_id": 0}],
        }

    cosine_sim_threshold: float = cfg["thresholds"].get("cosine_sim", 0.92)
    distance_threshold: float = round(1.0 - cosine_sim_threshold, 6)

    matrix = load_embeddings(results, cfg)
    labels = _cluster_embeddings(matrix, distance_threshold=distance_threshold)

    entropy = _shannon_entropy(labels)
    cluster_counts = np.bincount(labels)
    num_clusters = int(len(cluster_counts))
    majority_cluster_size = int(np.max(cluster_counts))
    consistency_ratio = float(majority_cluster_size / len(results))

    per_run = [
        {"run_index": r.run_index, "cluster_id": int(labels[i])}
        for i, r in enumerate(results)
    ]

    result = {
        "semantic_entropy": entropy,
        "num_clusters": num_clusters,
        "majority_cluster_size": majority_cluster_size,
        "consistency_ratio": consistency_ratio,
        "distance_threshold": distance_threshold,
        "per_run": per_run,
    }

    logger.debug(
        "Semantic entropy | prompt_id=%s entropy=%.4f clusters=%d consistency=%.4f",
        results[0].prompt_id,
        entropy,
        num_clusters,
        consistency_ratio,
    )
    return result