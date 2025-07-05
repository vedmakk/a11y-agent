import argparse
import asyncio
import os
import sys
from typing import Callable, Optional

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


async def run_browser_agent(task: str, start_url: str, step_handler: Callable[[str], None]) -> str:
    """Run a Browser-Use agent for *task* and return the final result text."""

    from browser_use import Agent  # imported lazily to speed up CLI startup
    from browser_use.llm import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o")  # Uses OPENAI_API_KEY env var

    # Optionally pre-navigate to a start URL so that the agent begins on a useful page
    initial_actions = []
    if start_url:
        initial_actions.append({
            "go_to_url": {"url": start_url, "new_tab": True}
        })

    extend_system_message = """
    You are an accessibility assistant and help the user browse the web and get
    things done. You act as a Screen Reader and help the user navigate the web.

    You assume that the user can't see the screen, so despite executing the actions to
    fulfill the user's intent, you should describe what you did and what is going on the 
    screen, in a way that is accessible to the user. Remember to be very concise, brief and to
    to the point.
    """

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=True,
        initial_actions=initial_actions,
        extend_system_message=extend_system_message,
    )

    history = await agent.run()
    # Try to pick a nice final summary if available
    try:
        return history.final_result()
    except Exception:
        return str(history)


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

    step_handler, voice_io = build_step_handler(args.voice)

    if args.voice:
        assert voice_io is not None
        print("[VoiceIO] Push-to-talk enabled. Hold SPACE to speak, release to send. Say 'exit' to quit.")
    else:
        print("Type your instructions (or 'exit' to quit):")

    while True:
        try:
            if args.voice and voice_io is not None:
                user_input = voice_io.push_to_talk().strip()
            else:
                user_input = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        # Run the browser agent for the given input
        try:
            result = asyncio.run(run_browser_agent(user_input, args.start_url, step_handler))
            step_handler(result)
        except Exception as exc:
            print(f"[Error] {exc}")
            if args.debug:
                raise


if __name__ == "__main__":
    # Ensure playwright is installed – show helpful hint instead of crashing later
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("Playwright Python package not installed. Please run: 'pip install playwright && playwright install'", file=sys.stderr)
        sys.exit(1)

    main() 