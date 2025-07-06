from __future__ import annotations

"""Abstract base class for agent providers.

An *agent provider* is responsible for executing a complete conversation turn
based on the incoming *items* list as defined in :pyfunc:`run_full_turn`.

Each provider instance can keep internal state (e.g., a persistent browser
session) so that consecutive calls can share context as required.
"""

import abc
from typing import Dict, List, Callable


class BaseAgentProvider(abc.ABC):
    """Abstract base for concrete agent implementations."""

    @abc.abstractmethod
    async def run_full_turn(self, items: List[Dict[str, str]], start_url: str, step_handler: Callable[[str], None]) -> List[Dict[str, str]]:  # noqa: D401
        """Run a complete conversation turn.

        Parameters
        ----------
        items
            Conversation items including the *current* user message. Each item
            must be a mapping containing at least `{"role": str, "content": str}`.
        start_url
            The starting URL that should be opened the first time the agent
            runs.

        Returns
        -------
        str
            Assistant response for the current turn.
        """

    async def close(self) -> None:  # noqa: D401
        """Optional cleanup coroutine (e.g., close sessions)."""
        # Default is a no-op so subclasses don't *have* to implement it.
        return None 