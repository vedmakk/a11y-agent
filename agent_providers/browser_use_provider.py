from __future__ import annotations

"""Browser-use based agent provider.

This provider wraps the original *browser_use* logic that used to live in
``main.py`` and exposes it via the new :pyclass:`~agent_providers.base.BaseAgentProvider`
interface.
"""

from typing import Dict, List, Optional, Callable

from .base import BaseAgentProvider
from agent_providers.system_prompt import get_system_prompt


class BrowserUseAgentProvider(BaseAgentProvider):
    """An :pyclass:`BaseAgentProvider` implementation using *browser_use*."""

    def __init__(self, model: str | None = None) -> None:  # noqa: D401
        # *Heavy* imports are done lazily inside :pymeth:`run_full_turn` so that
        # simply instantiating this class does not pull in playwright etc.
        self._model_name = model or "gpt-4.1"
        self._browser_session = None  # Will hold BrowserSession instance
        self._llm = None  # ChatOpenAI – lazy

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _ensure_llm(self):  # noqa: D401
        if self._llm is None:
            from browser_use.llm import ChatOpenAI  # type: ignore

            self._llm = ChatOpenAI(model=self._model_name)
        return self._llm

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

        current_task: str = str(last_item.get("content", "")).strip()
        if not current_task:
            raise ValueError("Current user message is empty")

        # Build history string excluding the last user message
        history_pairs: List[str] = []
        i = 0
        # Only process until second last item (i < len(items)-1)
        while i < len(items) - 1:
            if (
                items[i].get("role") == "user"
                and i + 1 < len(items) - 1
                and items[i + 1].get("role") == "assistant"
            ):
                history_pairs.append(
                    f"User: {items[i]['content']}\nAgent: {items[i + 1]['content']}"
                )
                i += 2
            else:
                i += 1  # Skip items that do not form proper pairs
        message_context = "\n\n".join(history_pairs) if history_pairs else None

        # ------------------------------------------------------------------
        # Lazy heavy imports (browser_use) – only when method called
        # ------------------------------------------------------------------
        from browser_use import Agent  # type: ignore
        from browser_use.browser.session import BrowserSession  # type: ignore

        llm = self._ensure_llm()

        # Prepare optional navigation action for the *first* run so visually
        # impaired users start on a known page.
        initial_actions = []
        if start_url and self._browser_session is None:
            initial_actions.append({"go_to_url": {"url": start_url, "new_tab": True}})

        extend_system_message = get_system_prompt()

        # Create a persistent BrowserSession on first run
        if self._browser_session is None:
            self._browser_session = BrowserSession(keep_alive=True, initialized=False)

        agent = Agent(
            task=current_task,
            llm=llm,
            message_context=message_context,
            use_vision=True,
            initial_actions=initial_actions,
            extend_system_message=extend_system_message,
            browser_session=self._browser_session,
        )

        history = await agent.run()

        # Attempt to fetch a concise result description
        try:
            result_text: str = history.final_result()  # type: ignore[attr-defined]
        except Exception:
            result_text = str(history)

        # Update session handle (should be the same instance)
        self._browser_session = agent.browser_session  # type: ignore[attr-defined]

        return [{"role": "assistant", "content": result_text}]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:  # noqa: D401
        """Close the persistent browser session (if any)."""
        if self._browser_session is not None:
            try:
                await self._browser_session.stop()
            except Exception:
                # Don't raise on cleanup.
                pass
            finally:
                self._browser_session = None 