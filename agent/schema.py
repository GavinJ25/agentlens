"""
Shared data models for the agent determinacy test suite.

RunResult is the single record produced by one agent call.
Every test group reads from this schema — do not modify fields.
"""

from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """A single tool invocation made by the agent during a run."""

    name: str
    arguments: dict[str, Any]
    result: Any = None


class RunResult(BaseModel):
    """
    Complete record of one agent run against one prompt.

    Produced by agent.client and persisted to outputs/runs/.
    All test groups consume this schema — never add or remove fields.
    """

    prompt_id: str
    run_index: int
    text: str                        # final text output from agent
    tool_calls: list[ToolCall] = []  # ordered list of tool invocations
    reasoning_trace: str = ""        # scratchpad / CoT if available
    latency_ms: float = 0.0
    metadata: dict[str, Any] = {}