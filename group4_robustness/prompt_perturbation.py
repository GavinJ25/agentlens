"""
Prompt perturbation robustness metric.

Loads paraphrase variants from prompts/perturbations.json, runs the agent
once per variant per original prompt, and measures how much the output
changes relative to the canonical run from outputs/runs/.

A deterministic agent should be invariant to semantics-preserving surface
changes. High output delta on paraphrases signals that the agent is
sensitive to phrasing rather than meaning.

perturbations.json schema:
{
    "<prompt_id>": {
        "original": "...",
        "variants": ["variant_1", "variant_2", ...]
    }
}
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from agent.client import call_agent
from agent.schema import RunResult
from runner.cache import load_run

logger = logging.getLogger(__name__)

_PERTURBATIONS_PATH = Path("prompts/perturbations.json")


def _load_perturbations() -> dict[str, dict[str, Any]]:
    """
    Load prompt perturbation variants from prompts/perturbations.json.

    Returns:
        Dict mapping prompt_id to {"original": str, "variants": list[str]}.

    Raises:
        FileNotFoundError: If perturbations.json does not exist.
    """
    if not _PERTURBATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Perturbations file not found at {_PERTURBATIONS_PATH}. "
            "Create prompts/perturbations.json before running G4."
        )
    return json.loads(_PERTURBATIONS_PATH.read_text(encoding="utf-8"))


def _rouge_l(scorer_instance: rouge_scorer.RougeScorer, ref: str, hyp: str) -> float:
    """
    Compute ROUGE-L F1 between a reference and hypothesis string.

    Args:
        scorer_instance: Shared RougeScorer instance.
        ref: Reference text (canonical run output).
        hyp: Hypothesis text (perturbed run output).

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    return scorer_instance.score(ref, hyp)["rougeL"].fmeasure


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Run perturbed prompt variants against the agent and measure output delta.

    For each prompt_id in results, loads the canonical reference output
    (run_index=0 from outputs/runs/), then runs each paraphrase variant
    once and scores the output against the reference via ROUGE-L.

    Args:
        results: Canonical RunResult objects (one per prompt_id, run_index=0
                 is used as reference). Typically the first run from the
                 G1/G2/G3 batch.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``perturbation_rouge_l_mean`` (float): mean ROUGE-L across all
                                                    variants and prompts
          - ``perturbation_rouge_l_std``  (float): std-dev
          - ``invariance_rate``           (float): fraction of variants scoring
                                                    above thresholds.rouge_l
          - ``per_prompt``               (list):  per-prompt breakdown dicts
    """
    perturbations = _load_perturbations()
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    threshold: float = cfg["thresholds"]["rouge_l"]

    all_scores: list[float] = []
    per_prompt: list[dict[str, Any]] = []

    # Build prompt_id → reference text map from run_index=0
    ref_map: dict[str, str] = {}
    for r in results:
        if r.run_index == 0:
            ref_map[r.prompt_id] = r.text

    for prompt_id, entry in perturbations.items():
        if prompt_id not in ref_map:
            logger.warning(
                "prompt_id=%s found in perturbations.json but not in results — skipping.",
                prompt_id,
            )
            continue

        reference_text = ref_map[prompt_id]
        variants: list[str] = entry.get("variants", [])
        variant_scores: list[dict[str, Any]] = []

        for idx, variant_text in enumerate(variants):
            try:
                variant_result = call_agent(
                    prompt_id=f"{prompt_id}_perturb_{idx}",
                    run_index=idx,
                    prompt_text=variant_text,
                    cfg=cfg,
                )
                score = _rouge_l(sc, reference_text, variant_result.text)
                all_scores.append(score)
                variant_scores.append({
                    "variant_index": idx,
                    "rouge_l": round(score, 6),
                    "above_threshold": score >= threshold,
                })
                logger.debug(
                    "Perturbation | prompt_id=%s variant=%d rouge_l=%.4f",
                    prompt_id, idx, score,
                )
            except Exception as exc:
                logger.error(
                    "Perturbation run failed | prompt_id=%s variant=%d: %s",
                    prompt_id, idx, exc,
                )

        prompt_mean = float(np.mean([v["rouge_l"] for v in variant_scores])) if variant_scores else 0.0
        per_prompt.append({
            "prompt_id": prompt_id,
            "variant_count": len(variants),
            "rouge_l_mean": prompt_mean,
            "variants": variant_scores,
        })

    invariance_rate = (
        float(sum(1 for s in all_scores if s >= threshold) / len(all_scores))
        if all_scores else 0.0
    )

    result = {
        "perturbation_rouge_l_mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "perturbation_rouge_l_std": float(np.std(all_scores)) if all_scores else 0.0,
        "invariance_rate": invariance_rate,
        "per_prompt": per_prompt,
    }

    logger.info(
        "Prompt perturbation complete | rouge_l_mean=%.4f invariance_rate=%.4f",
        result["perturbation_rouge_l_mean"],
        result["invariance_rate"],
    )
    return result