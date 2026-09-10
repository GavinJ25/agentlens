"""
Context window stress test metric.

Loads growing-distractor prompts from prompts/context_stress.json and runs
the agent at each distractor size, measuring when output consistency
drops below the ROUGE-L threshold. Identifies the context length at which
the agent's attention or retrieval degrades ("the breaking point").

context_stress.json schema:
{
    "<prompt_id>": {
        "core_task": "...",
        "levels": [
            {"label": "baseline", "prefix": ""},
            {"label": "1k_tokens", "prefix": "<distractor text ~1k tokens>"},
            {"label": "2k_tokens", "prefix": "<distractor text ~2k tokens>"},
            ...
        ]
    }
}

The agent receives: prefix + "\n\n" + core_task at each level.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from agent.client import call_agent
from agent.schema import RunResult

logger = logging.getLogger(__name__)

_CONTEXT_STRESS_PATH = Path("prompts/context_stress.json")


def _load_stress_prompts() -> dict[str, dict[str, Any]]:
    """
    Load context stress prompt levels from prompts/context_stress.json.

    Returns:
        Dict mapping prompt_id to {"core_task": str, "levels": list[dict]}.

    Raises:
        FileNotFoundError: If context_stress.json does not exist.
    """
    if not _CONTEXT_STRESS_PATH.exists():
        raise FileNotFoundError(
            f"Context stress file not found at {_CONTEXT_STRESS_PATH}. "
            "Create prompts/context_stress.json before running G4."
        )
    return json.loads(_CONTEXT_STRESS_PATH.read_text(encoding="utf-8"))


def _build_stressed_prompt(prefix: str, core_task: str) -> str:
    """
    Combine a distractor prefix with the core task string.

    Args:
        prefix:    Distractor context (empty string for baseline).
        core_task: The actual task the agent should solve.

    Returns:
        Combined prompt string with a double newline separator.
    """
    if not prefix:
        return core_task
    return f"{prefix}\n\n{core_task}"


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Run context stress levels and measure ROUGE-L degradation per level.

    Uses the baseline level (empty prefix) output as the reference for
    each prompt. Subsequent levels with growing distractors are scored
    against this baseline.

    Args:
        results: Canonical RunResult objects — used only to determine which
                 prompt_ids are active. The stress test runs fresh agent
                 calls for every level including baseline.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``context_stress_rouge_l_mean`` (float): mean ROUGE-L across all
                                                      prompts at max stress level
          - ``breaking_point_label``        (str | None): label of first level
                                                           where ROUGE-L drops
                                                           below threshold; None
                                                           if threshold never breached
          - ``per_prompt``                  (list): per-prompt stress curves
    """
    stress_prompts = _load_stress_prompts()
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    threshold: float = cfg["thresholds"]["rouge_l"]

    active_ids = {r.prompt_id for r in results}
    per_prompt: list[dict[str, Any]] = []
    max_level_scores: list[float] = []
    all_breaking_points: list[str | None] = []

    for prompt_id, entry in stress_prompts.items():
        if prompt_id not in active_ids:
            logger.debug(
                "prompt_id=%s not in active results — skipping context stress.",
                prompt_id,
            )
            continue

        core_task: str = entry["core_task"]
        levels: list[dict[str, Any]] = entry["levels"]
        baseline_text: str | None = None
        level_curve: list[dict[str, Any]] = []
        breaking_point: str | None = None

        for level_idx, level in enumerate(levels):
            label: str = level.get("label", str(level_idx))
            prefix: str = level.get("prefix", "")
            full_prompt = _build_stressed_prompt(prefix, core_task)

            try:
                run_result = call_agent(
                    prompt_id=f"{prompt_id}_stress_{label}",
                    run_index=level_idx,
                    prompt_text=full_prompt,
                    cfg=cfg,
                )
            except Exception as exc:
                logger.error(
                    "Context stress run failed | prompt_id=%s level=%s: %s",
                    prompt_id, label, exc,
                )
                continue

            # Baseline is always the first level
            if baseline_text is None:
                baseline_text = run_result.text
                rouge_l = 1.0
            else:
                rouge_l = sc.score(
                    baseline_text, run_result.text
                )["rougeL"].fmeasure

            # Record first breach
            if breaking_point is None and rouge_l < threshold and level_idx > 0:
                breaking_point = label

            approx_tokens = len(prefix.split()) if prefix else 0
            level_curve.append({
                "label": label,
                "approx_prefix_tokens": approx_tokens,
                "rouge_l": round(rouge_l, 6),
                "above_threshold": rouge_l >= threshold,
            })

            logger.debug(
                "Context stress | prompt_id=%s level=%s rouge_l=%.4f",
                prompt_id, label, rouge_l,
            )

        # Score at the maximum stress level
        if level_curve:
            max_level_scores.append(level_curve[-1]["rouge_l"])

        all_breaking_points.append(breaking_point)
        per_prompt.append({
            "prompt_id": prompt_id,
            "breaking_point_label": breaking_point,
            "curve": level_curve,
        })

    # Determine overall breaking point (first label where any prompt breaks)
    overall_breaking = next(
        (bp for bp in all_breaking_points if bp is not None), None
    )

    result = {
        "context_stress_rouge_l_mean": float(np.mean(max_level_scores)) if max_level_scores else 0.0,
        "breaking_point_label": overall_breaking,
        "per_prompt": per_prompt,
    }

    logger.info(
        "Context stress complete | max_level_rouge_l=%.4f breaking_point=%s",
        result["context_stress_rouge_l_mean"],
        overall_breaking,
    )
    return result