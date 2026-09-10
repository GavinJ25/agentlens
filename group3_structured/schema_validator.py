"""
Schema validation determinacy metric.

For agents that return structured JSON in their text output, this module
validates every run against the schema inferred from run_index=0. It
measures what fraction of runs produce output that is valid JSON and
structurally consistent with the reference run.

If the agent does not return JSON, all runs will score as non-JSON and
the metric returns a validity_rate of 0.0 with a clear log warning —
it does not raise an exception, allowing the rest of the suite to proceed.
"""

import json
import logging
from typing import Any

from agent.schema import RunResult

logger = logging.getLogger(__name__)


def _try_parse_json(text: str) -> tuple[bool, Any]:
    """
    Attempt to parse a string as JSON.

    Strips markdown code fences (```json ... ```) before parsing, since
    agents commonly wrap structured output in fenced blocks.

    Args:
        text: Raw agent output string.

    Returns:
        Tuple of (success: bool, parsed_value: Any | None).
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove opening fence (```json or ```) and closing fence (```)
        inner = lines[1:-1] if len(lines) > 2 else lines[1:]
        stripped = "\n".join(inner).strip()
    try:
        return True, json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False, None


def _extract_schema(obj: Any, depth: int = 0, max_depth: int = 4) -> Any:
    """
    Recursively extract the structural schema of a parsed JSON value.

    Returns a skeleton that replaces leaf values with their Python type
    name, preserving keys and list structure up to max_depth.

    Args:
        obj:       Parsed JSON value (dict, list, str, int, float, bool, None).
        depth:     Current recursion depth (internal use).
        max_depth: Maximum depth before collapsing to type name.

    Returns:
        A schema skeleton: dicts with type-name leaves, lists with one
        representative element schema, or a type-name string for scalars.
    """
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {k: _extract_schema(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_extract_schema(obj[0], depth + 1, max_depth)]
    return type(obj).__name__


def _schema_match(ref_schema: Any, candidate_schema: Any) -> bool:
    """
    Check whether a candidate schema matches the reference schema.

    Dict key sets must match exactly. List schemas match if both are lists.
    Scalar type names must match.

    Args:
        ref_schema:       Schema extracted from run_index=0.
        candidate_schema: Schema extracted from the run being evaluated.

    Returns:
        True if schemas are structurally compatible.
    """
    if type(ref_schema) != type(candidate_schema):
        return False
    if isinstance(ref_schema, dict):
        if ref_schema.keys() != candidate_schema.keys():
            return False
        return all(
            _schema_match(ref_schema[k], candidate_schema[k])
            for k in ref_schema
        )
    if isinstance(ref_schema, list):
        return isinstance(candidate_schema, list)
    return ref_schema == candidate_schema


def compute(results: list[RunResult], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Validate JSON structure consistency across all runs for one prompt.

    Uses run_index=0 as the reference schema. Each subsequent run is
    checked for (a) valid JSON parsability and (b) structural schema match
    against the reference.

    Args:
        results: All RunResult objects for one prompt_id, ordered by run_index.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        Dict with keys:
          - ``validity_rate``    (float): fraction of runs producing valid JSON
          - ``schema_match_rate``(float): fraction of valid-JSON runs matching
                                          the reference schema (0.0 if no JSON)
          - ``reference_is_json``(bool):  whether run_index=0 parsed as JSON
          - ``per_run``          (list):  [{run_index, is_valid_json,
                                            schema_matches, keys_added,
                                            keys_removed}]
    """
    if not results:
        logger.warning("No results supplied to schema_validator.compute.")
        return {
            "validity_rate": 0.0,
            "schema_match_rate": 0.0,
            "reference_is_json": False,
            "per_run": [],
        }

    ref_ok, ref_parsed = _try_parse_json(results[0].text)
    ref_schema = _extract_schema(ref_parsed) if ref_ok else None

    if not ref_ok:
        logger.warning(
            "prompt_id=%s run_index=0 is not valid JSON — "
            "schema_validator will report 0.0 for all runs.",
            results[0].prompt_id,
        )

    per_run: list[dict[str, Any]] = []
    valid_count = 0
    schema_match_count = 0

    for r in results:
        is_valid, parsed = _try_parse_json(r.text)
        schema_matches = False
        keys_added: list[str] = []
        keys_removed: list[str] = []

        if is_valid:
            valid_count += 1
            if ref_schema is not None:
                candidate_schema = _extract_schema(parsed)
                schema_matches = _schema_match(ref_schema, candidate_schema)
                schema_match_count += int(schema_matches)

                # Surface-level key diff for dict outputs
                if isinstance(ref_parsed, dict) and isinstance(parsed, dict):
                    ref_keys = set(ref_parsed.keys())
                    cand_keys = set(parsed.keys())
                    keys_added = sorted(cand_keys - ref_keys)
                    keys_removed = sorted(ref_keys - cand_keys)

        per_run.append({
            "run_index": r.run_index,
            "is_valid_json": is_valid,
            "schema_matches": schema_matches,
            "keys_added": keys_added,
            "keys_removed": keys_removed,
        })

    n = len(results)
    validity_rate = valid_count / n
    schema_match_rate = schema_match_count / n if ref_ok else 0.0

    result = {
        "validity_rate": float(validity_rate),
        "schema_match_rate": float(schema_match_rate),
        "reference_is_json": ref_ok,
        "per_run": per_run,
    }

    logger.debug(
        "Schema validator | prompt_id=%s validity=%.4f schema_match=%.4f",
        results[0].prompt_id,
        validity_rate,
        schema_match_rate,
    )
    return result