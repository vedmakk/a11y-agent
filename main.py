# Standard lib
import argparse
import asyncio
import sys
from typing import Callable, Optional

# Make sure environment variables from `.env` are available **before** we
# instantiate `VoiceIO` or Browser-Use models.
from dotenv import load_dotenv

# Load .env at import time; override ensures .env settings win over inherited envs
load_dotenv(override=True)

import os

try:
    from voice_io import VoiceIO  # noqa: E402
except ImportError:
    VoiceIO = None  # type: ignore

# Agent provider factory
from agent_providers import get_agent_provider

def build_step_handler(enable_voice: bool) -> tuple[Callable[[str], None], Optional["VoiceIO"]]:  # type: ignore[name-defined]
    """Return a `(handler, voice_io)` pair depending on *enable_voice*."""

    if enable_voice:
        if VoiceIO is None:
            raise RuntimeError("voice_io dependencies missing – cannot enable --voice")

        # ---------------------------------------------------------------
        # Determine STT and TTS providers from environment variables
        # ---------------------------------------------------------------
        stt_name = os.getenv("VOICE_STT_PROVIDER", "openai").lower()
        tts_name = os.getenv("VOICE_TTS_PROVIDER", "openai").lower()

        # Helper to instantiate provider based on name ----------
        def _make_provider(kind: str, name: str):  # noqa: D401
            if name == "openai":
                from speech_providers.openai_provider import OpenAIProvider  # lazy import

                model_trans = os.getenv("VOICE_OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
                model_tts = os.getenv("VOICE_OPENAI_TTS_MODEL", "tts-1")
                voice = os.getenv("VOICE_OPENAI_VOICE", "alloy")

                # We may share a single OpenAIProvider for both STT and TTS
                return OpenAIProvider(
                    model_transcription=model_trans,
                    model_tts=model_tts,
                    voice=voice,
                )
            elif name == "system":
                if kind == "stt":
                    from speech_providers.system_provider import SystemSTTProvider  # lazy import

                    return SystemSTTProvider()
                else:
                    from speech_providers.system_provider import SystemTTSProvider  # lazy import

                    return SystemTTSProvider()
            else:
                raise ValueError(f"Unknown {kind.upper()} provider: {name}")

        # Instantiate providers (may reuse same instance if both are openai)
        if stt_name == tts_name == "openai":
            shared = _make_provider("stt", "openai")  # type: ignore[assignment]
            stt_provider = shared  # type: ignore[assignment]
            tts_provider = shared  # type: ignore[assignment]
        else:
            stt_provider = _make_provider("stt", stt_name)
            tts_provider = _make_provider("tts", tts_name)

        voice_io = VoiceIO(stt_provider=stt_provider, tts_provider=tts_provider)

        def _handler(msg: str, *, cache: bool = False):
            print(msg)
            try:
                voice_io.speak(msg, cache=cache)
            except Exception as exc:
                print(f"[VoiceIO] Failed to speak: {exc}")

        return _handler, voice_io

    # Text-only fallback
    def plain_handler(msg: str, *, cache: bool = False):  # noqa: D401
        print(msg)

    return plain_handler, None


# ---------------------------------------------------------------------------
# Interactive loop (async)
# ---------------------------------------------------------------------------


async def interactive_loop(args) -> None:  # noqa: C901  – keeps CLI simple
    step_handler, voice_io = build_step_handler(args.voice)

    if args.voice:
        step_handler(
            "Voice mode enabled. Press 'ESC' to skip playback. Hold SPACE to speak, release to send. Say 'exit' to quit"  # type: ignore[arg-type]
            , cache=True
        )
    else:
        print("Type your instructions (or 'exit' to quit):")

    # ---------------------------------------------------------------
    # Initialise chosen agent provider
    # ---------------------------------------------------------------
    agent_provider_name = os.getenv("AGENT_PROVIDER", "browser-use")
    agent_provider = get_agent_provider(agent_provider_name)

    conversation_history: list[dict[str, str]] = []  # Store conversation messages

    step_handler(f"You're currently on {args.start_url}.", cache=True)

    while True:
        try:
            if args.voice and voice_io is not None:
                step_handler("Waiting for input... or say 'exit' to quit.", cache=True)
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
            step_handler("Exiting...", cache=True)
            break

        try:
            step_handler(f"Executing input: {user_input}")
            # Run full turn via selected agent provider
            items = [*conversation_history, {"role": "user", "content": user_input}]
            result_items = await agent_provider.run_full_turn(items, args.start_url, step_handler)

            # computer-use provider returns a list of items, so we need to get the last item and get the content
            readable_result = result_items[-1]["content"] if isinstance(result_items[-1]["content"], str) else result_items[-1]["content"][0]["text"]

            step_handler(readable_result)

            # Update conversation history
            conversation_history.extend([
                {"role": "user", "content": user_input},
                *result_items,
            ])
        except Exception as exc:
            print(f"[Error] {exc}")
            if args.debug:
                raise

    # Clean-up ------------------------------------------------------------------
    try:
        await agent_provider.close()
    except Exception as exc:
        if args.debug:
            raise
        print(f"[Warning] Failed to close agent cleanly: {exc}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 – keep CLI simple
    parser = argparse.ArgumentParser(description="Accessibility browsing agent (voice optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed output.")
    parser.add_argument(
        "--start-url",
        type=str,
        default="https://google.com",
        help="Start the browsing session with this URL.",
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