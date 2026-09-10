"""
Batch runner for the agent determinacy test suite.

Executes the agent N times per prompt using a thread pool for concurrency.
Results are written to outputs/runs/ via runner.cache. Already-cached runs
are skipped automatically — delete outputs/runs/ to force a full re-run.

This is the only module (outside group4_robustness) that calls agent.client.
All metric groups read from the cache instead.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from agent.client import call_agent
from agent.schema import RunResult
from runner.cache import is_cached, save_run

logger = logging.getLogger(__name__)


def _run_one(
    prompt_id: str,
    run_index: int,
    prompt_text: str,
    cfg: dict[str, Any],
) -> RunResult:
    """
    Execute a single agent call and persist the result.

    Skips the call if a cached result already exists on disk.

    Args:
        prompt_id:   Prompt identifier matching a key in prompts/core.json.
        run_index:   Zero-based index of this run within the batch.
        prompt_text: Raw prompt string sent to the agent.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        The RunResult, either freshly obtained or loaded from cache.
    """
    if is_cached(prompt_id, run_index, cfg):
        from runner.cache import load_run
        return load_run(prompt_id, run_index, cfg)

    result = call_agent(
        prompt_id=prompt_id,
        run_index=run_index,
        prompt_text=prompt_text,
        cfg=cfg,
    )
    save_run(result, cfg)
    return result


def run_batch(
    prompts: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, list[RunResult]]:
    """
    Run the agent N times for each prompt in the given dict.

    Uses a thread pool sized to runs.parallel from config. Already-cached
    runs are skipped without calling the agent. Progress is shown via tqdm.

    Args:
        prompts: Dict mapping prompt_id → prompt_text, typically loaded
                 from prompts/core.json.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict mapping prompt_id → list of RunResult (length N, ordered by
        run_index). Prompts where all runs failed are included with an
        empty list and a warning logged.
    """
    n: int = cfg["runs"]["n"]
    parallel: int = cfg["runs"].get("parallel", 4)
    output_dir = Path(cfg["output"]["dir"]) / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = len(prompts) * n
    results: dict[str, list[RunResult | None]] = {
        pid: [None] * n for pid in prompts
    }

    logger.info(
        "Starting batch: %d prompts × %d runs = %d total calls (parallel=%d)",
        len(prompts),
        n,
        total_tasks,
        parallel,
    )

    futures = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for prompt_id, prompt_text in prompts.items():
            for run_index in range(n):
                future = pool.submit(
                    _run_one,
                    prompt_id,
                    run_index,
                    prompt_text,
                    cfg,
                )
                futures[future] = (prompt_id, run_index)

        with tqdm(total=total_tasks, desc="Batch runs", unit="run") as pbar:
            for future in as_completed(futures):
                prompt_id, run_index = futures[future]
                try:
                    result = future.result()
                    results[prompt_id][run_index] = result
                    logger.debug(
                        "Completed prompt_id=%s run_index=%d latency_ms=%.1f",
                        prompt_id,
                        run_index,
                        result.latency_ms,
                    )
                except Exception as exc:
                    logger.error(
                        "Run failed for prompt_id=%s run_index=%d: %s",
                        prompt_id,
                        run_index,
                        exc,
                    )
                finally:
                    pbar.update(1)

    # Filter out None slots (failed runs) and warn
    clean: dict[str, list[RunResult]] = {}
    for prompt_id, run_list in results.items():
        completed = [r for r in run_list if r is not None]
        if len(completed) < n:
            logger.warning(
                "prompt_id=%s: only %d/%d runs completed",
                prompt_id,
                len(completed),
                n,
            )
        clean[prompt_id] = completed

    logger.info(
        "Batch complete. %d/%d runs succeeded across %d prompts.",
        sum(len(v) for v in clean.values()),
        total_tasks,
        len(prompts),
    )

    return clean