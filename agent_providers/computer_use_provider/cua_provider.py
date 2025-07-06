from __future__ import annotations

"""Computer-use based agent provider.
"""

from typing import Dict, List, Optional, Callable

from agent_providers.base import BaseAgentProvider

from computer_use_provider.agent import Agent
from computers.config import *
from computers.default import *
from computers import computers_config


class CuaAgentProvider(BaseAgentProvider):
    """An :pyclass:`BaseAgentProvider` implementation using *computer_use*."""

    def __init__(self, step_handler: Callable[[str], None]) -> None:  # noqa: D401
        ComputerClass = computers_config['local-playwright'] # TODO: make this configurable
        self.computer = ComputerClass()
        self.agent = Agent(computer=self.computer, step_handler=step_handler)
        self.initial_turn = True

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
        
        if start_url and self.initial_turn:
            self.agent.computer.goto(start_url)
            self.initial_turn = False
        
        output_items = self.agent.run_full_turn(
            items,
            print_steps=True,
            show_images=False,
            debug=False,
        )
        
        return output_items

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: D401
        pass