"""
Agent client — the single call interface for the determinacy test suite.

All configuration is read from config.yaml via runner.config_loader.
Nothing in this file is hardcoded. Swap agents by editing config.yaml only.

The client is intentionally thin: it sends one prompt, waits for a response,
and maps it into a RunResult. Pre-processing or post-processing of the agent
payload belongs in an input/output adapter, not here.
"""

import logging
import os
import time
from typing import Any

import httpx

from agent.schema import RunResult, ToolCall

logger = logging.getLogger(__name__)


def _resolve_api_key(api_key_env: str) -> str:
    """
    Resolve the agent API key from an environment variable.

    Args:
        api_key_env: Name of the environment variable holding the key.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the environment variable is not set.
    """
    key = os.environ.get(api_key_env)
    if not key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return key


def _build_headers(cfg: dict[str, Any]) -> dict[str, str]:
    """
    Build HTTP headers for the agent request.

    Merges the Authorization header with any extra_headers from config.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Dict of HTTP headers.
    """
    agent_cfg = cfg["agent"]
    api_key = _resolve_api_key(agent_cfg["api_key_env"])
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(agent_cfg.get("extra_headers", {}))
    return headers


def _parse_tool_calls(raw: list[dict[str, Any]]) -> list[ToolCall]:
    """
    Parse a list of raw tool call dicts from the agent response.

    Expects each item to have at least a 'name' key. 'arguments' and
    'result' are optional and default to empty dict / None respectively.

    Args:
        raw: List of tool call dicts from the agent response body.

    Returns:
        List of ToolCall instances in original order.
    """
    calls = []
    for item in raw:
        try:
            calls.append(
                ToolCall(
                    name=item["name"],
                    arguments=item.get("arguments", {}),
                    result=item.get("result"),
                )
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed tool call entry: %s", type(exc).__name__)
    return calls


def call_agent(
    prompt_id: str,
    run_index: int,
    prompt_text: str,
    cfg: dict[str, Any],
) -> RunResult:
    """
    Send one prompt to the configured agent endpoint and return a RunResult.

    This is the only function in the suite that makes outbound HTTP calls
    to the agent under test. G1-G3 groups read from cached RunResult
    objects; G4 robustness modules call this function directly.

    The agent response body is expected to be JSON with this shape
    (all fields except 'text' are optional):

        {
            "text": "...",
            "tool_calls": [{"name": "...", "arguments": {...}, "result": ...}],
            "reasoning_trace": "...",
            "metadata": {...}
        }

    If your agent returns a different shape, adapt the mapping below
    or write an input adapter that wraps this function.

    Args:
        prompt_id:   Identifier for the prompt (matches prompts/core.json key).
        run_index:   Zero-based index of this run within the batch.
        prompt_text: The raw prompt string to send to the agent.
        cfg:         Full config dict loaded from config.yaml.

    Returns:
        A populated RunResult instance.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses.
        httpx.TimeoutException: If the agent exceeds timeout_s.
    """
    agent_cfg = cfg["agent"]
    runs_cfg = cfg["runs"]

    endpoint: str = agent_cfg["endpoint"]
    timeout_s: float = float(agent_cfg.get("timeout_s", 30))
    temperature: float = float(runs_cfg.get("temperature", 0.0))

    headers = _build_headers(cfg)
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
    }

    logger.debug(
        "Calling agent | prompt_id=%s run_index=%d endpoint=%s",
        prompt_id,
        run_index,
        endpoint,
    )

    t_start = time.monotonic()
    with httpx.Client(timeout=timeout_s) as http:
        response = http.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
    latency_ms = (time.monotonic() - t_start) * 1000.0

    body: dict[str, Any] = response.json()

    text: str = body.get("text", "")
    if not text:
        logger.warning(
            "Agent returned empty 'text' for prompt_id=%s run_index=%d",
            prompt_id,
            run_index,
        )

    tool_calls = _parse_tool_calls(body.get("tool_calls") or [])
    reasoning_trace: str = body.get("reasoning_trace", "")
    metadata: dict[str, Any] = body.get("metadata", {})
    metadata.setdefault("prompt_text", prompt_text)

    result = RunResult(
        prompt_id=prompt_id,
        run_index=run_index,
        text=text,
        tool_calls=tool_calls,
        reasoning_trace=reasoning_trace,
        latency_ms=round(latency_ms, 2),
        metadata=metadata,
    )

    logger.debug(
        "Run complete | prompt_id=%s run_index=%d latency_ms=%.1f",
        prompt_id,
        run_index,
        latency_ms,
    )

    return result