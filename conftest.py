"""
Shared pytest fixtures for the agent determinacy test suite.

Provides a session-scoped config fixture, a loaded prompt set fixture,
and a convenience fixture for accessing cached run results within tests.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from runner.config_loader import load_config

logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    """
    Load config.yaml once per test session.

    Returns:
        Full config dict loaded from config.yaml in the project root.
    """
    return load_config()


@pytest.fixture(scope="session")
def core_prompts() -> dict[str, str]:
    """
    Load the canonical prompt set from prompts/core.json once per session.

    Returns:
        Dict mapping prompt_id to prompt text.
    """
    path = Path("prompts/core.json")
    if not path.exists():
        pytest.skip(f"prompts/core.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_prompt_id(core_prompts: dict[str, str]) -> str:
    """
    Return the first prompt_id from the core prompt set for single-prompt tests.

    Args:
        core_prompts: The loaded core prompts fixture.

    Returns:
        The first prompt_id key in document order.
    """
    return next(iter(core_prompts))