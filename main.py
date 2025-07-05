# Standard lib
import argparse
import asyncio
import sys
from typing import Callable, Optional

# Make sure environment variables from `.env` are available **before** we
# instantiate `VoiceIO` or Browser-Use models.
from dotenv import load_dotenv

# Load .env at import time so every subsequent import sees the variables.
load_dotenv()

try:
    from voice_io import VoiceIO  # noqa: E402
except ImportError:
    VoiceIO = None  # type: ignore

# Lazy import browser_use only when needed to avoid heavy imports on --help

def build_step_handler(enable_voice: bool) -> tuple[Callable[[str], None], Optional["VoiceIO"]]:  # type: ignore[name-defined]
    """Return a `(handler, voice_io)` pair depending on *enable_voice*."""

    if enable_voice:
        if VoiceIO is None:
            raise RuntimeError("voice_io dependencies missing – cannot enable --voice")
        voice_io = VoiceIO()

        def _handler(msg: str):
            print(msg)
            try:
                voice_io.speak(msg)
            except Exception as exc:
                print(f"[VoiceIO] Failed to speak: {exc}")

        return _handler, voice_io

    # Text-only fallback
    return print, None


# ---------------------------------------------------------------------------
# Browser-Use helper
# ---------------------------------------------------------------------------


async def run_browser_agent(
    task: str,
    start_url: str,
    browser_session,
) -> tuple[str, "BrowserSession"]:  # type: ignore[name-defined]
    """Execute *task* using a Browser-Use Agent and return (result, session)."""

    from browser_use import Agent  # lazy import
    from browser_use.llm import ChatOpenAI
    from browser_use.browser.session import BrowserSession

    llm = ChatOpenAI(model="gpt-4o")

    # Prepare optional navigation action so visually-impaired users always start
    # on a known page.
    initial_actions = []
    if start_url and browser_session is None:
        initial_actions.append({"go_to_url": {"url": start_url, "new_tab": True}})

    extend_system_message = (
        """
        You are an accessibility assistant and help the user browse the web, get information and get
        things done. You act as a Screen Reader and help the user navigate the web and execute actions.

        You assume that the user can't see the screen, so despite executing the actions to
        fulfill the user's intent, you should describe what you did and what is going on the
        screen, in a way that is accessible to the user.

        Your answers should be concise, brief and to the point, but at the same time - since the
        user can't see the screen - you should also describe what is going on the screen, in a way
        that is accessible to the user.

        Go from important information to less important information (e.g. Menu items, Main content is important, sidebars or footers are less important).

        Important constraints:
        - User your vision capabilities to take screenshots of the screen and also describe the images (remember: the user can't see the screen)
        - YOU DO NOT navigate to another page unless the user asks you to do so.
        - Don't overthink it, always try to execute the user's intent with the least amount of steps and as fast as possible – while replying as swift as possible.
        """
    )

    # Create a persistent BrowserSession on first run
    if browser_session is None:
        browser_session = BrowserSession(keep_alive=True, initialized=False)

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=True,
        initial_actions=initial_actions,
        extend_system_message=extend_system_message,
        browser_session=browser_session,
    )

    history = await agent.run()

    # Attempt to fetch a concise result description
    try:
        result_text = history.final_result()
    except Exception:
        result_text = str(history)

    return result_text, agent.browser_session  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Interactive loop (async)
# ---------------------------------------------------------------------------


async def interactive_loop(args) -> None:  # noqa: C901  – keeps CLI simple
    step_handler, voice_io = build_step_handler(args.voice)

    if args.voice:
        step_handler(
            "Push-to-talk enabled. Hold SPACE to speak, release to send. Say 'exit' to quit."  # type: ignore[arg-type]
        )
    else:
        print("Type your instructions (or 'exit' to quit):")

    shared_session = None  # will hold BrowserSession

    while True:
        try:
            if args.voice and voice_io is not None:
                step_handler("Waiting for input...")
                # push_to_talk is blocking – off-load to thread so we don't block the event-loop
                user_input = await asyncio.to_thread(voice_io.push_to_talk)
                user_input = user_input.strip()
            else:
                user_input = await asyncio.to_thread(lambda: input("› ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        try:
            step_handler(f"Executing input: {user_input}")
            result, shared_session = await run_browser_agent(
                user_input,
                args.start_url,
                shared_session,
            )
            step_handler(result)
        except Exception as exc:
            print(f"[Error] {exc}")
            if args.debug:
                raise

    # Clean-up ------------------------------------------------------------------
    if shared_session is not None:
        try:
            await shared_session.stop()
        except Exception as exc:
            if args.debug:
                raise
            print(f"[Warning] Failed to close browser session cleanly: {exc}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 – keep CLI simple
    parser = argparse.ArgumentParser(description="Accessibility browsing agent (voice optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed output.")
    parser.add_argument(
        "--start-url",
        type=str,
        default="https://bing.com",
        help="Start the browsing session with this URL (only for browser environments).",
    )
    parser.add_argument("--voice", action="store_true", help="Enable voice input and output (requires microphone and speakers).")

    args = parser.parse_args()

    # Ensure playwright is installed – show helpful hint instead of crashing later
    try:
        import playwright  # noqa: F401
    except ImportError:
        print(
            "Playwright Python package not installed. Please run: 'pip install playwright && playwright install'",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(interactive_loop(args))


if __name__ == "__main__":
    main() 