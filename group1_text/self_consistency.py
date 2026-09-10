"""
Self-consistency scoring metric.

Clusters outputs by semantic similarity (using lightweight exact-match
grouping on normalized text as a fast proxy) and measures how often the
agent lands in the majority cluster. Inspired by the Self-Consistency
Chain-of-Thought method (Wang et al., 2022).

For deeper semantic clustering, group2_embeddings/semantic_entropy.py
uses embedding vectors — this module operates on raw text only.
"""

import logging
import re
from collections import Counter
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """
    Lightly normalize text for grouping — lowercase, collapse whitespace,
    strip punctuation-only differences.

    Args:
        text: Raw agent output string.

    Returns:
        Normalized string for cluster key comparison.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _rouge_l(scorer_instance: rouge_scorer.RougeScorer, a: str, b: str) -> float:
    """
    Compute ROUGE-L F1 between two strings.

    Args:
        scorer_instance: Shared RougeScorer to avoid re-instantiation.
        a: First string.
        b: Second string.

    Returns:
        ROUGE-L F1 score as a float in [0, 1].
    """
    return scorer_instance.score(a, b)["rougeL"].fmeasure


def _assign_clusters(texts: list[str], threshold: float = 0.85) -> list[int]:
    """
    Greedily assign texts to clusters based on ROUGE-L similarity.

    The first text starts cluster 0. Each subsequent text joins the
    cluster of the first text it exceeds the threshold against, or
    starts a new cluster if none match.

    Args:
        texts:     List of normalized output strings.
        threshold: ROUGE-L similarity threshold for cluster membership.

    Returns:
        List of integer cluster IDs, one per text, in input order.
    """
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    cluster_reps: list[str] = []
    assignments: list[int] = []

    for text in texts:
        assigned = False
        for cid, rep in enumerate(cluster_reps):
            if _rouge_l(sc, rep, text) >= threshold:
                assignments.append(cid)
                assigned = True
                break
        if not assigned:
            assignments.append(len(cluster_reps))
            cluster_reps.append(text)

    return assignments


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute self-consistency score for a set of runs on one prompt.

    Clusters outputs by ROUGE-L similarity, identifies the majority
    cluster, and returns the fraction of runs in that cluster as the
    self-consistency score.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``self_consistency``   (float): fraction of runs in majority cluster
          - ``num_clusters``       (int):   number of distinct output clusters
          - ``majority_cluster_id``(int):   ID of the most common cluster
          - ``cluster_entropy``    (float): Shannon entropy over cluster sizes
          - ``per_run``            (list):  [{run_index, cluster_id}]
    """
    if not results:
        logger.warning("No results supplied to self_consistency.compute.")
        return {
            "self_consistency": 0.0,
            "num_clusters": 0,
            "majority_cluster_id": -1,
            "cluster_entropy": 0.0,
            "per_run": [],
        }

    normalized = [_normalize(r.text) for r in results]
    cluster_ids = _assign_clusters(normalized)

    counts: Counter = Counter(cluster_ids)
    majority_cluster_id, majority_count = counts.most_common(1)[0]
    self_consistency = majority_count / len(results)

    # Shannon entropy over cluster distribution
    probs = np.array([c / len(results) for c in counts.values()])
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    per_run = [
        {"run_index": r.run_index, "cluster_id": cid}
        for r, cid in zip(results, cluster_ids)
    ]

    result = {
        "self_consistency": float(self_consistency),
        "num_clusters": len(counts),
        "majority_cluster_id": int(majority_cluster_id),
        "cluster_entropy": entropy,
        "per_run": per_run,
    }

    logger.debug(
        "Self-consistency | prompt_id=%s score=%.4f clusters=%d entropy=%.4f",
        results[0].prompt_id,
        self_consistency,
        len(counts),
        entropy,
    )
    return result