"""
Temperature sensitivity sweep metric.

Runs the agent across a configurable range of temperature values for each
canonical prompt and measures how output consistency (ROUGE-L vs the
temperature=0 reference) decays as temperature increases.

The sweep characterises the agent's stochasticity profile: where does
determinism break down and at what rate? Results are written as a decay
curve suitable for plotting.

Temperature values swept: [0.0, 0.1, 0.2, ..., 1.0] by default.
Each temperature gets runs.n / 4 runs (minimum 2) to keep costs bounded.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from agent.client import call_agent
from agent.schema import RunResult

logger = logging.getLogger(__name__)

_DEFAULT_TEMPS = [round(t * 0.1, 1) for t in range(11)]   # 0.0 → 1.0
_MIN_RUNS_PER_TEMP = 2


def _runs_per_temp(cfg: dict[str, Any]) -> int:
    """
    Determine how many runs to execute per temperature value.

    Uses runs.n // 4, clamped to a minimum of _MIN_RUNS_PER_TEMP.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Integer number of runs per temperature.
    """
    return max(cfg["runs"]["n"] // 4, _MIN_RUNS_PER_TEMP)


def _mean_rouge_l(
    sc: rouge_scorer.RougeScorer,
    reference: str,
    candidates: list[str],
) -> float:
    """
    Compute mean ROUGE-L F1 between a reference and a list of candidates.

    Args:
        sc:         Shared RougeScorer instance.
        reference:  Gold reference text (temperature=0 output).
        candidates: List of agent outputs at one temperature value.

    Returns:
        Mean ROUGE-L F1 across all candidates.
    """
    scores = [sc.score(reference, c)["rougeL"].fmeasure for c in candidates]
    return float(np.mean(scores))


def _run_at_temperature(
    prompt_id: str,
    prompt_text: str,
    temperature: float,
    n_runs: int,
    cfg: dict[str, Any],
) -> list[RunResult]:
    """
    Execute n_runs agent calls at a specific temperature for one prompt.

    Temporarily overrides runs.temperature in a copy of cfg so the rest
    of the config remains untouched.

    Args:
        prompt_id:   Canonical prompt identifier.
        prompt_text: Prompt string to send to the agent.
        temperature: Temperature value to use for this batch.
        n_runs:      Number of runs to execute.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        List of RunResult objects, one per completed run.
    """
    temp_cfg = deepcopy(cfg)
    temp_cfg["runs"]["temperature"] = temperature

    parallel: int = cfg["runs"].get("parallel", 4)
    results: list[RunResult] = []

    with ThreadPoolExecutor(max_workers=min(parallel, n_runs)) as pool:
        futures = {
            pool.submit(
                call_agent,
                f"{prompt_id}_temp{temperature}",
                i,
                prompt_text,
                temp_cfg,
            ): i
            for i in range(n_runs)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                run_idx = futures[future]
                logger.error(
                    "Temp sweep run failed | prompt_id=%s temp=%.1f run=%d: %s",
                    prompt_id, temperature, run_idx, exc,
                )

    return results


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Sweep temperature from 0.0 to 1.0 and measure ROUGE-L decay per prompt.

    Uses run_index=0 from the provided results list as the reference
    (temperature=0 canonical output) for each prompt_id.

    Args:
        results:  Canonical RunResult objects — one per prompt_id (run_index=0
                  used as reference). Pass the first run from the G1 batch.
        cfg:      Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``decay_curve``         (list): [{temperature, rouge_l_mean}] averaged
                                            across all prompts
          - ``breakpoint_temp``     (float): lowest temperature where mean ROUGE-L
                                             drops below thresholds.rouge_l
          - ``temp_rouge_l_at_zero``(float): mean ROUGE-L at temperature=0.0
                                             (sanity check — should be ~1.0)
          - ``per_prompt``          (list):  per-prompt decay curves
    """
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    threshold: float = cfg["thresholds"]["rouge_l"]
    n_runs = _runs_per_temp(cfg)
    temperatures = _DEFAULT_TEMPS

    # Build reference map from provided results (run_index=0)
    ref_map: dict[str, str] = {
        r.prompt_id: r.text for r in results if r.run_index == 0
    }

    prompt_text_map: dict[str, str] = {
        r.prompt_id: r.metadata.get("prompt_text", "")
        for r in results if r.run_index == 0
    }

    per_prompt: list[dict[str, Any]] = []
    # temp → list of per-prompt rouge_l scores (for averaging across prompts)
    temp_scores: dict[float, list[float]] = {t: [] for t in temperatures}

    for prompt_id, reference_text in ref_map.items():
        prompt_text = prompt_text_map[prompt_id]
        prompt_curve: list[dict[str, Any]] = []

        for temp in temperatures:
            temp_results = _run_at_temperature(
                prompt_id, prompt_text, temp, n_runs, cfg
            )
            if not temp_results:
                logger.warning(
                    "No results for prompt_id=%s temp=%.1f — skipping point.",
                    prompt_id, temp,
                )
                continue

            candidates = [r.text for r in temp_results]
            mean_rl = _mean_rouge_l(sc, reference_text, candidates)
            temp_scores[temp].append(mean_rl)
            prompt_curve.append({
                "temperature": temp,
                "rouge_l_mean": round(mean_rl, 6),
                "n_runs": len(temp_results),
            })
            logger.debug(
                "Temp sweep | prompt_id=%s temp=%.1f rouge_l=%.4f",
                prompt_id, temp, mean_rl,
            )

        per_prompt.append({
            "prompt_id": prompt_id,
            "curve": prompt_curve,
        })

    # Aggregate decay curve across all prompts
    decay_curve = [
        {
            "temperature": t,
            "rouge_l_mean": round(float(np.mean(temp_scores[t])), 6)
            if temp_scores[t] else None,
        }
        for t in temperatures
    ]

    # Find the lowest temperature where mean ROUGE-L first drops below threshold
    breakpoint_temp: float | None = None
    for point in decay_curve:
        if point["rouge_l_mean"] is not None and point["rouge_l_mean"] < threshold:
            breakpoint_temp = point["temperature"]
            break

    rouge_at_zero = next(
        (p["rouge_l_mean"] for p in decay_curve if p["temperature"] == 0.0),
        None,
    )

    result = {
        "decay_curve": decay_curve,
        "breakpoint_temp": breakpoint_temp,
        "temp_rouge_l_at_zero": rouge_at_zero,
        "per_prompt": per_prompt,
    }

    logger.info(
        "Temperature sweep complete | breakpoint_temp=%s rouge_at_zero=%s",
        breakpoint_temp,
        rouge_at_zero,
    )
    return result