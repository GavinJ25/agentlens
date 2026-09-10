"""
Multi-run regression metric.

Compares current run outputs against a saved baseline
(outputs/results/baseline.json) to detect drift after agent updates.

Workflow:
  1. First run: no baseline exists → current scores ARE the baseline.
     The file is written to outputs/results/baseline.json automatically.
  2. Subsequent runs: current scores are diffed against the baseline.
     Any prompt whose ROUGE-L drops more than the configured tolerance
     is flagged as a regression.

This module is designed to be used as a CI gate: a non-zero
``regression_count`` in the output signals a failing check.

Tolerance is read from thresholds.rouge_l in config — a drop below that
absolute value (not a delta) constitutes a regression.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rouge_score import rouge_scorer

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _baseline_path(cfg: dict[str, Any]) -> Path:
    """
    Resolve the baseline file path from config.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Path to outputs/results/baseline.json.
    """
    return Path(cfg["output"]["dir"]) / "results" / "baseline.json"


def _load_baseline(path: Path) -> dict[str, float] | None:
    """
    Load baseline ROUGE-L scores from disk.

    Args:
        path: Path to the baseline JSON file.

    Returns:
        Dict mapping prompt_id to baseline ROUGE-L float, or None if
        the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("rouge_l_by_prompt", {})


def _save_baseline(
    scores: dict[str, float],
    path: Path,
    cfg: dict[str, Any],
) -> None:
    """
    Persist the current run's ROUGE-L scores as the new baseline.

    Args:
        scores: Dict mapping prompt_id to mean ROUGE-L float.
        path:   Destination file path.
        cfg:    Full config dict (stored in baseline for traceability).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rouge_l_by_prompt": scores,
        "runs_n": cfg["runs"]["n"],
        "temperature": cfg["runs"]["temperature"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Baseline written to %s", path)


def _compute_rouge_l_per_prompt(
    results: list[RunResult],
    prompt_ids: list[str],
) -> dict[str, float]:
    """
    Compute mean ROUGE-L (run_index=0 as reference) per prompt_id from
    a flat list of RunResult objects.

    Args:
        results:    All RunResult objects across all prompt_ids.
        prompt_ids: Ordered list of prompt_ids to score.

    Returns:
        Dict mapping prompt_id to mean ROUGE-L float.
    """
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    by_prompt: dict[str, list[RunResult]] = {pid: [] for pid in prompt_ids}
    for r in results:
        if r.prompt_id in by_prompt:
            by_prompt[r.prompt_id].append(r)

    scores: dict[str, float] = {}
    for pid, runs in by_prompt.items():
        if len(runs) < 2:
            scores[pid] = 1.0 if len(runs) == 1 else 0.0
            continue
        ref = sorted(runs, key=lambda r: r.run_index)[0].text
        rls = [
            sc.score(ref, r.text)["rougeL"].fmeasure
            for r in runs
        ]
        scores[pid] = float(np.mean(rls))
    return scores


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compare current run ROUGE-L scores against the saved baseline.

    On first run (no baseline file), writes the current scores as baseline
    and returns regression_count=0. On subsequent runs, diffs against the
    baseline and flags prompts that have regressed.

    Args:
        results: All RunResult objects from the current batch (all prompt_ids,
                 all run indices).
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``regression_count``     (int):   number of prompts that regressed
          - ``is_first_run``         (bool):  True if no baseline existed
          - ``threshold``            (float): rouge_l threshold from config
          - ``per_prompt``           (list):  [{prompt_id, current_rouge_l,
                                               baseline_rouge_l, delta,
                                               regressed}]
    """
    baseline_path = _baseline_path(cfg)
    threshold: float = cfg["thresholds"]["rouge_l"]

    prompt_ids = sorted({r.prompt_id for r in results})
    current_scores = _compute_rouge_l_per_prompt(results, prompt_ids)

    existing_baseline = _load_baseline(baseline_path)
    is_first_run = existing_baseline is None

    if is_first_run:
        logger.info(
            "No baseline found at %s — writing current scores as baseline.",
            baseline_path,
        )
        _save_baseline(current_scores, baseline_path, cfg)

    baseline_scores: dict[str, float] = existing_baseline or current_scores

    per_prompt: list[dict[str, Any]] = []
    regression_count = 0

    for pid in prompt_ids:
        current = current_scores.get(pid, 0.0)
        baseline = baseline_scores.get(pid, current)
        delta = round(current - baseline, 6)
        regressed = (not is_first_run) and (current < threshold)

        if regressed:
            regression_count += 1
            logger.warning(
                "REGRESSION detected | prompt_id=%s current=%.4f baseline=%.4f delta=%.4f",
                pid, current, baseline, delta,
            )

        per_prompt.append({
            "prompt_id": pid,
            "current_rouge_l": round(current, 6),
            "baseline_rouge_l": round(baseline, 6),
            "delta": delta,
            "regressed": regressed,
        })

    result = {
        "regression_count": regression_count,
        "is_first_run": is_first_run,
        "threshold": threshold,
        "per_prompt": per_prompt,
    }

    logger.info(
        "Regression check complete | regressions=%d is_first_run=%s",
        regression_count,
        is_first_run,
    )
    return result