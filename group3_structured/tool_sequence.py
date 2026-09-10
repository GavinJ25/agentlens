"""
Tool call sequence determinacy metric.

Measures whether the agent invokes the same tools, in the same order,
with the same arguments across repeated identical runs. Three levels of
comparison are reported:

  1. Name-only match  — same tool names in same order
  2. Signature match  — same names + same argument keys (not values)
  3. Full match       — same names + identical serialised arguments

The key ``tool_seq_match`` in the returned dict matches the threshold
in config.yaml under thresholds.tool_seq_match.
"""

import hashlib
import json
import logging
from typing import Any

from agent.schema import RunResult, ToolCall

logger = logging.getLogger(__name__)


def _name_sequence(tool_calls: list[ToolCall]) -> tuple[str, ...]:
    """
    Extract an ordered tuple of tool names from a list of ToolCall objects.

    Args:
        tool_calls: Ordered list of ToolCall instances from one run.

    Returns:
        Tuple of tool name strings in call order.
    """
    return tuple(tc.name for tc in tool_calls)


def _signature_sequence(tool_calls: list[ToolCall]) -> tuple[tuple[str, frozenset], ...]:
    """
    Extract an ordered sequence of (tool_name, frozenset_of_arg_keys) tuples.

    Args:
        tool_calls: Ordered list of ToolCall instances from one run.

    Returns:
        Tuple of (name, frozenset(arg_keys)) pairs in call order.
    """
    return tuple(
        (tc.name, frozenset(tc.arguments.keys()))
        for tc in tool_calls
    )


def _full_hash(tool_calls: list[ToolCall]) -> str:
    """
    Produce a stable hash of the complete serialised tool call sequence.

    Arguments are JSON-serialised with sorted keys to ensure stability
    regardless of dict insertion order.

    Args:
        tool_calls: Ordered list of ToolCall instances from one run.

    Returns:
        SHA-256 hex digest (64 characters).
    """
    payload = json.dumps(
        [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _levenshtein_name(seq_a: tuple, seq_b: tuple) -> int:
    """
    Compute Levenshtein edit distance between two tool-name sequences.

    Used to quantify how far apart two runs are when sequences differ,
    providing a richer signal than a binary match/no-match.

    Args:
        seq_a: Tool name tuple from run A.
        seq_b: Tool name tuple from run B.

    Returns:
        Minimum number of insertions, deletions, or substitutions needed
        to transform seq_a into seq_b.
    """
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Compute tool call sequence consistency across all runs for one prompt.

    Run index 0 is used as the reference. The ``tool_seq_match`` key
    (full-sequence hash match rate) corresponds to the config threshold.

    If no run has any tool calls, all match rates are 1.0 (vacuously true)
    and a debug message is logged.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``tool_seq_match``     (float): fraction of runs with full-match
                                            to reference sequence
          - ``name_match_rate``    (float): fraction matching name-only sequence
          - ``signature_match_rate``(float): fraction matching name+arg-keys
          - ``mean_edit_distance`` (float): mean Levenshtein distance vs reference
          - ``reference_length``   (int):   number of tool calls in reference run
          - ``per_run``            (list):  per-run comparison detail dicts
    """
    if not results:
        logger.warning("No results supplied to tool_sequence.compute.")
        return {
            "tool_seq_match": 1.0,
            "name_match_rate": 1.0,
            "signature_match_rate": 1.0,
            "mean_edit_distance": 0.0,
            "reference_length": 0,
            "per_run": [],
        }

    ref = results[0]
    ref_names = _name_sequence(ref.tool_calls)
    ref_sigs = _signature_sequence(ref.tool_calls)
    ref_hash = _full_hash(ref.tool_calls)
    ref_len = len(ref.tool_calls)

    if ref_len == 0:
        logger.debug(
            "prompt_id=%s run_index=0 has no tool calls — "
            "tool_seq_match will be vacuously 1.0 if all runs also have none.",
            ref.prompt_id,
        )

    per_run: list[dict[str, Any]] = []
    name_matches = 0
    sig_matches = 0
    full_matches = 0
    edit_distances: list[int] = []

    for r in results:
        r_names = _name_sequence(r.tool_calls)
        r_sigs = _signature_sequence(r.tool_calls)
        r_hash = _full_hash(r.tool_calls)

        name_ok = r_names == ref_names
        sig_ok = r_sigs == ref_sigs
        full_ok = r_hash == ref_hash
        edit_dist = _levenshtein_name(ref_names, r_names)

        name_matches += int(name_ok)
        sig_matches += int(sig_ok)
        full_matches += int(full_ok)
        edit_distances.append(edit_dist)

        per_run.append({
            "run_index": r.run_index,
            "tool_count": len(r.tool_calls),
            "name_sequence": list(r_names),
            "name_match": name_ok,
            "signature_match": sig_ok,
            "full_match": full_ok,
            "edit_distance": edit_dist,
        })

    n = len(results)
    result = {
        "tool_seq_match": float(full_matches / n),
        "name_match_rate": float(name_matches / n),
        "signature_match_rate": float(sig_matches / n),
        "mean_edit_distance": float(sum(edit_distances) / n),
        "reference_length": ref_len,
        "per_run": per_run,
    }

    logger.debug(
        "Tool sequence | prompt_id=%s full_match=%.4f name_match=%.4f edit_dist=%.2f",
        results[0].prompt_id,
        result["tool_seq_match"],
        result["name_match_rate"],
        result["mean_edit_distance"],
    )
    return result