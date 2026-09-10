"""
Run cache for the agent determinacy test suite.

Provides helpers to check whether a (prompt_id, run_index) pair has already
been executed and saved, and to load existing RunResult objects from disk.

The cache key is the output file path: outputs/runs/{prompt_id}_{run_index}.json.
No content hashing is done at this layer — the presence of the file is sufficient
to skip a re-run. To force a full re-run, delete the outputs/runs/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Any

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _runs_dir(cfg: dict[str, Any]) -> Path:
    """
    Resolve the outputs/runs/ directory from config.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Path to the runs output directory.
    """
    return Path(cfg["output"]["dir"]) / "runs"


def run_path(prompt_id: str, run_index: int, cfg: dict[str, Any]) -> Path:
    """
    Return the canonical file path for a single run result.

    Args:
        prompt_id:  Prompt identifier string.
        run_index:  Zero-based run index within the batch.
        cfg:        Full config dict loaded from config.yaml.

    Returns:
        Path of the form outputs/runs/{prompt_id}_{run_index}.json.
    """
    return _runs_dir(cfg) / f"{prompt_id}_{run_index}.json"


def is_cached(prompt_id: str, run_index: int, cfg: dict[str, Any]) -> bool:
    """
    Check whether a run result already exists on disk.

    Args:
        prompt_id:  Prompt identifier string.
        run_index:  Zero-based run index within the batch.
        cfg:        Full config dict loaded from config.yaml.

    Returns:
        True if the output file exists and is non-empty, False otherwise.
    """
    path = run_path(prompt_id, run_index, cfg)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        logger.debug("Cache hit: %s", path)
    return exists


def save_run(result: RunResult, cfg: dict[str, Any]) -> Path:
    """
    Persist a RunResult to disk as JSON.

    Creates the outputs/runs/ directory if it does not exist.

    Args:
        result: The RunResult to save.
        cfg:    Full config dict loaded from config.yaml.

    Returns:
        Path where the result was written.
    """
    dest = run_path(result.prompt_id, result.run_index, cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.debug("Saved run result to %s", dest)
    return dest


def load_run(prompt_id: str, run_index: int, cfg: dict[str, Any]) -> RunResult:
    """
    Load a RunResult from disk.

    Args:
        prompt_id:  Prompt identifier string.
        run_index:  Zero-based run index within the batch.
        cfg:        Full config dict loaded from config.yaml.

    Returns:
        Deserialised RunResult instance.

    Raises:
        FileNotFoundError: If the run file does not exist.
        ValueError: If the file content cannot be parsed as a RunResult.
    """
    path = run_path(prompt_id, run_index, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Run result not found: {path}. "
            "Run runner.batch.run_batch() first."
        )
    raw = path.read_text(encoding="utf-8")
    return RunResult.model_validate_json(raw)


def load_all_runs(prompt_id: str, cfg: dict[str, Any]) -> list[RunResult]:
    """
    Load all cached RunResult objects for a given prompt_id.

    Loads run indices 0 through runs.n-1. Skips any index whose file is
    missing and logs a warning, so partial batches are tolerated.

    Args:
        prompt_id: Prompt identifier string.
        cfg:       Full config dict loaded from config.yaml.

    Returns:
        List of RunResult instances in run_index order.
    """
    n: int = cfg["runs"]["n"]
    results: list[RunResult] = []
    for i in range(n):
        if is_cached(prompt_id, i, cfg):
            try:
                results.append(load_run(prompt_id, i, cfg))
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Failed to load run %s index %d: %s", prompt_id, i, exc
                )
        else:
            logger.warning(
                "Missing run file for prompt_id=%s run_index=%d — skipping",
                prompt_id,
                i,
            )
    return results