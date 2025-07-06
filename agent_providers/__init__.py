from __future__ import annotations

"""Agent provider factory and utilities."""

from .base import BaseAgentProvider

__all__ = [
    "BaseAgentProvider",
    "get_agent_provider",
]


def get_agent_provider(name: str) -> BaseAgentProvider:  # noqa: D401
    """Return an agent provider instance for *name* (case-insensitive).

    Parameters
    ----------
    name:
        Provider name, e.g. ``"browser-use"``.

    Raises
    ------
    ValueError
        If *name* is unknown.
    """
    normalized = name.lower().replace("_", "-").strip()

    if normalized in {"browser-use", "browser", "browseruse"}:
        from .browser_use_provider import BrowserUseAgentProvider  # lazy import to avoid heavy deps

        return BrowserUseAgentProvider()

    if normalized in {"computer-use", "computer", "computeruse"}:
        from .computer_use_provider import CuaAgentProvider  # lazy import to avoid heavy deps

        return CuaAgentProvider()

    raise ValueError(f"Unknown AGENT_PROVIDER: {name}") 