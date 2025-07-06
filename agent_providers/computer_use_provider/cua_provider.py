from __future__ import annotations

"""Computer-use based agent provider.
"""

from typing import Dict, List, Callable
import asyncio

from agent_providers.base import BaseAgentProvider

from .agent import Agent
from computers.config import *
from computers.default import *
from computers import computers_config

def acknowledge_safety_check_callback(message: str) -> bool:
    response = input(
        f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): "
    ).lower()
    return response.lower().strip() == "y"

class CuaAgentProvider(BaseAgentProvider):
    """An :pyclass:`BaseAgentProvider` implementation using *computer_use*.

    The heavy *computer-use* agent is created lazily once we receive the first
    :pyfunc:`run_full_turn` call – this allows us to inject the *step_handler*
    provided at runtime and avoids expensive browser start-up when unused.
    """

    def __init__(self) -> None:  # noqa: D401
        # Store class for later lazy init (starting playwright is expensive)
        self._computer_cls = computers_config["local-playwright"]  # TODO: env-configurable via env var
        self._computer = None  # Will hold an *entered* computer instance

        # The underlying *Agent* will be instantiated once we get the first
        # call with a valid *step_handler*.
        self._agent: Agent | None = None

        # Whether we already navigated to *start_url* on the first turn.
        self._initial_turn = True

    # ------------------------------------------------------------------
    # BaseAgentProvider interface
    # ------------------------------------------------------------------

    async def run_full_turn(self, items: List[Dict[str, str]], start_url: str, step_handler: Callable[[str], None]) -> List[Dict[str, str]]:  # noqa: D401
        # ------------------------------------------------------------------
        # Extract *current* task (last user message) & build message context
        # ------------------------------------------------------------------
        if not items:
            raise ValueError("'items' list cannot be empty")

        # Last item must be the new user message according to our contract
        last_item = items[-1]
        if last_item.get("role") != "user":
            raise ValueError("Last item in 'items' must be a user message")
        
        # Lazily instantiate computer + heavy agent --------------------------------
        if self._computer is None:
            # Create + enter the computer instance *inside a worker thread* so
            # that the Playwright **sync** API does not block or interfere
            # with the asyncio event-loop.
            self._computer = await asyncio.to_thread(
                lambda: self._computer_cls().__enter__()
            )

        # Lazily instantiate the heavy agent with the runtime step_handler
        if self._agent is None:
            self._agent = Agent(computer=self._computer, acknowledge_safety_check_callback=acknowledge_safety_check_callback, step_handler=step_handler)
        else:
            # Update the handler so callers can change it (e.g. enable voice)
            self._agent.step_handler = step_handler  # type: ignore[attr-defined]

        # Navigate to *start_url* on the very first turn only so that users who
        # rely on screen readers always begin on a deterministic page.
        if start_url and self._initial_turn:
            try:
                # Off-load navigation to the worker thread as it uses the
                # sync Playwright API.
                await asyncio.to_thread(self._computer.goto, start_url)  # type: ignore[attr-defined]
            finally:
                self._initial_turn = False

        # The inner agent is blocking / sync – off-load to a thread so that the
        # event-loop remains responsive.
        output_items: List[Dict[str, str]] = await asyncio.to_thread(
            self._agent.run_full_turn,  # type: ignore[arg-type]
            items,
            print_steps=False,
            show_images=False,
            debug=False,
        )

        return output_items

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: D401
        # Gracefully shut down the computer (and therefore Playwright).
        # We call the context-manager cleanup to ensure browsers are closed
        # and Playwright is stopped.  The three 'None' arguments correspond
        # to exc_type, exc_val and exc_tb when used in a "with" block.
        if self._computer is not None:
            # __exit__ uses sync Playwright – run it in a worker thread.
            try:
                exit_func = getattr(self._computer, "__exit__", None)
                if callable(exit_func):
                    await asyncio.to_thread(exit_func, None, None, None)
            finally:
                self._computer = None

        # Clear agent reference as well so it gets GC'd.
        self._agent = None