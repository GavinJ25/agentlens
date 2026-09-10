"""
Entry point for the agent determinacy test suite.

Reads config.yaml, runs the batch executor, then runs each enabled test
group in order, writing per-group scores and a final aggregated report.

Usage:
    python run_suite.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from agent.schema import RunResult
from metrics.aggregator import aggregate_group, aggregate_suite
from metrics.reporter import print_console_report, write_group_scores, write_report
from metrics.thresholds import evaluate
from runner.batch import run_batch
from runner.cache import load_all_runs
from runner.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_prompts(path: Path = Path("prompts/core.json")) -> dict[str, str]:
    """
    Load the canonical prompt set from prompts/core.json.

    Args:
        path: Path to the prompts file.

    Returns:
        Dict mapping prompt_id to prompt text.

    Raises:
        FileNotFoundError: If the prompts file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prompts file not found at {path}. "
            "Create prompts/core.json with at least one prompt before running the suite."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_group1(
    all_results: dict[str, list[RunResult]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Run all group1_text metrics for every prompt and aggregate the results.

    Args:
        all_results: Dict mapping prompt_id to its list of RunResult objects.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        Aggregated group result dict with a threshold_check key attached.
    """
    from group1_text import (
        rouge, bleu_meteor, exact_match,
        length_variance, format_fingerprint, self_consistency,
    )

    per_prompt: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for prompt_id, results in all_results.items():
        scores: dict[str, Any] = {}
        for module in (rouge, bleu_meteor, exact_match,
                       length_variance, format_fingerprint, self_consistency):
            try:
                scores.update(module.compute(results, cfg))
            except Exception as exc:
                logger.error(
                    "G1 metric %s failed for prompt_id=%s: %s",
                    module.__name__, prompt_id, exc,
                )
                errors.append({"module": module.__name__, "error": type(exc).__name__, "prompt_id": prompt_id})
        per_prompt[prompt_id] = scores

    aggregated = aggregate_group(per_prompt)
    aggregated["threshold_check"] = _evaluate_group_thresholds(per_prompt, cfg)
    if errors:
        aggregated["_errors"] = errors
    return aggregated


def _run_group2(
    all_results: dict[str, list[RunResult]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Run all group2_embeddings metrics for every prompt and aggregate the results.

    Args:
        all_results: Dict mapping prompt_id to its list of RunResult objects.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        Aggregated group result dict with a threshold_check key attached.
    """
    from group2_embeddings import cosine_sim, bert_score, semantic_entropy

    per_prompt: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for prompt_id, results in all_results.items():
        scores: dict[str, Any] = {}
        for module in (cosine_sim, bert_score, semantic_entropy):
            try:
                scores.update(module.compute(results, cfg))
            except Exception as exc:
                logger.error(
                    "G2 metric %s failed for prompt_id=%s: %s",
                    module.__name__, prompt_id, exc,
                )
                errors.append({"module": module.__name__, "error": type(exc).__name__, "prompt_id": prompt_id})
        per_prompt[prompt_id] = scores

    aggregated = aggregate_group(per_prompt)
    aggregated["threshold_check"] = _evaluate_group_thresholds(per_prompt, cfg)
    if errors:
        aggregated["_errors"] = errors
    return aggregated


def _run_group3(
    all_results: dict[str, list[RunResult]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Run all group3_structured metrics for every prompt and aggregate the results.

    Args:
        all_results: Dict mapping prompt_id to its list of RunResult objects.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        Aggregated group result dict with a threshold_check key attached.
    """
    from group3_structured import (
        schema_validator, tool_sequence, decision_path,
        reasoning_steps, memory_stability,
    )

    per_prompt: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for prompt_id, results in all_results.items():
        scores: dict[str, Any] = {}
        for module in (schema_validator, tool_sequence, decision_path,
                       reasoning_steps, memory_stability):
            try:
                scores.update(module.compute(results, cfg))
            except Exception as exc:
                logger.error(
                    "G3 metric %s failed for prompt_id=%s: %s",
                    module.__name__, prompt_id, exc,
                )
                errors.append({"module": module.__name__, "error": type(exc).__name__, "prompt_id": prompt_id})
        per_prompt[prompt_id] = scores

    aggregated = aggregate_group(per_prompt)
    aggregated["threshold_check"] = _evaluate_group_thresholds(per_prompt, cfg)
    if errors:
        aggregated["_errors"] = errors
    return aggregated


def _run_group4(
    all_results: dict[str, list[RunResult]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Run all group4_robustness metrics. Unlike G1-G3, these call the agent
    live and are not run per-prompt in the same way — each module manages
    its own prompt iteration internally.

    Args:
        all_results: Dict mapping prompt_id to its list of RunResult objects,
                     used as reference/canonical data by G4 modules.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        Aggregated group result dict with a threshold_check key attached.
    """
    from group4_robustness import (
        prompt_perturbation, temperature_sweep, context_stress,
        regression, calibration,
    )

    flat_results = [r for results in all_results.values() for r in results]
    scores: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    try:
        scores["prompt_perturbation"] = prompt_perturbation.compute(flat_results, cfg)
    except Exception as exc:
        logger.error("G4 prompt_perturbation failed: %s", exc)
        errors.append({"module": "prompt_perturbation", "error": type(exc).__name__, "prompt_id": None})

    try:
        scores["temperature_sweep"] = temperature_sweep.compute(flat_results, cfg)
    except Exception as exc:
        logger.error("G4 temperature_sweep failed: %s", exc)
        errors.append({"module": "temperature_sweep", "error": type(exc).__name__, "prompt_id": None})

    try:
        scores["context_stress"] = context_stress.compute(flat_results, cfg)
    except Exception as exc:
        logger.error("G4 context_stress failed: %s", exc)
        errors.append({"module": "context_stress", "error": type(exc).__name__, "prompt_id": None})

    try:
        scores["regression"] = regression.compute(flat_results, cfg)
    except Exception as exc:
        logger.error("G4 regression failed: %s", exc)
        errors.append({"module": "regression", "error": type(exc).__name__, "prompt_id": None})

    # Calibration runs per-prompt like G1-G3
    calibration_per_prompt: dict[str, Any] = {}
    for prompt_id, results in all_results.items():
        try:
            calibration_per_prompt[prompt_id] = calibration.compute(results, cfg)
        except Exception as exc:
            logger.error("G4 calibration failed for prompt_id=%s: %s", prompt_id, exc)
            errors.append({"module": "calibration", "error": type(exc).__name__, "prompt_id": prompt_id})
    scores["calibration"] = calibration_per_prompt

    # Build threshold check from the modules that produce gate-able scores.
    # Calibration is per-prompt, so run it through the same path as G1-G3.
    threshold_check = _evaluate_group_thresholds(calibration_per_prompt, cfg)

    # Regression gate: any regression_count > 0 is a failure.
    reg = scores.get("regression", {})
    regression_count = reg.get("regression_count", 0)
    is_first_run = reg.get("is_first_run", True)
    if not is_first_run and regression_count > 0:
        threshold_check["passed"] = False
        threshold_check["checks"].append({
            "metric": "regression_count",
            "value": regression_count,
            "threshold": 0,
            "passed": False,
        })
        threshold_check["checked_count"] = threshold_check.get("checked_count", 0) + 1

    result = {
        "summary": scores,
        "by_prompt": calibration_per_prompt,
        "prompt_count": len(all_results),
        "threshold_check": threshold_check,
    }
    if errors:
        result["_errors"] = errors
    return result


def _evaluate_group_thresholds(
    per_prompt: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate threshold pass/fail using the mean of each metric across prompts.

    Args:
        per_prompt: Dict mapping prompt_id to its score dict.
        cfg:        Full config dict loaded from config.yaml.

    Returns:
        Output of metrics.thresholds.evaluate() applied to mean scores.
    """
    import numpy as np

    if not per_prompt:
        return {"passed": True, "checks": [], "checked_count": 0}

    numeric_keys: set[str] = set()
    for scores in per_prompt.values():
        for key, value in scores.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.add(key)

    mean_scores: dict[str, float] = {}
    for key in numeric_keys:
        values = [s[key] for s in per_prompt.values() if key in s]
        if values:
            mean_scores[key] = float(np.mean(values))

    return evaluate(mean_scores, cfg)


def main() -> int:
    """
    Run the full determinacy test suite end to end.

    Returns:
        Process exit code — 0 if all enabled groups passed their thresholds
        (or output.fail_fast is False and execution completed), 1 otherwise.
    """
    cfg = load_config()
    prompts = _load_prompts()
    groups_cfg = cfg.get("groups", {})
    fail_fast: bool = cfg["output"].get("fail_fast", False)

    logger.info("Starting determinacy test suite | %d prompts loaded", len(prompts))

    # Step 1: batch run (skips cached runs automatically)
    all_results = run_batch(prompts, cfg)

    if not any(all_results.values()):
        logger.error("No run results were produced. Aborting suite.")
        return 1

    group_results: dict[str, dict[str, Any]] = {}

    # Step 2: Group 1 — text metrics
    if groups_cfg.get("g1_text", True):
        logger.info("Running Group 1 — text metrics")
        g1 = _run_group1(all_results, cfg)
        group_results["g1_text"] = g1
        write_group_scores("g1_text", g1, cfg)
        if fail_fast and not g1["threshold_check"]["passed"]:
            logger.error("fail_fast enabled — stopping after g1_text failure.")
            _finalize(group_results, cfg)
            return 1

    # Step 3: Group 2 — embeddings (depends on G1 having run, but reads from cache directly)
    if groups_cfg.get("g2_embeddings", True):
        logger.info("Running Group 2 — embedding metrics")
        g2 = _run_group2(all_results, cfg)
        group_results["g2_embeddings"] = g2
        write_group_scores("g2_embeddings", g2, cfg)
        if fail_fast and not g2["threshold_check"]["passed"]:
            logger.error("fail_fast enabled — stopping after g2_embeddings failure.")
            _finalize(group_results, cfg)
            return 1

    # Step 4: Group 3 — structured metrics
    if groups_cfg.get("g3_structured", True):
        logger.info("Running Group 3 — structured metrics")
        g3 = _run_group3(all_results, cfg)
        group_results["g3_structured"] = g3
        write_group_scores("g3_structured", g3, cfg)
        if fail_fast and not g3["threshold_check"]["passed"]:
            logger.error("fail_fast enabled — stopping after g3_structured failure.")
            _finalize(group_results, cfg)
            return 1

    # Step 5: Group 4 — robustness (opt-in, separate live runs)
    if groups_cfg.get("g4_robustness", False):
        logger.info("Running Group 4 — robustness metrics (live agent calls)")
        g4 = _run_group4(all_results, cfg)
        group_results["g4_robustness"] = g4
        write_group_scores("g4_robustness", g4, cfg)

    return _finalize(group_results, cfg)


def _finalize(group_results: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> int:
    """
    Aggregate all group results, print the console report, write report.json,
    and return the appropriate process exit code.

    Args:
        group_results: Dict mapping group name to its aggregated result.
        cfg:            Full config dict loaded from config.yaml.

    Returns:
        0 if overall_passed is True, 1 otherwise.
    """
    suite_report = aggregate_suite(group_results)
    print_console_report(suite_report)
    write_report(suite_report, cfg)

    if suite_report["overall_passed"]:
        logger.info("Suite PASSED.")
        return 0
    logger.error("Suite FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())