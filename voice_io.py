from __future__ import annotations

"""Utility for voice input (speech-to-text) and voice output (text-to-speech) using
OpenAI audio endpoints.

Requirements (added to requirements.txt):
    openai>=1.30.0
    sounddevice >= 0.4.6
    soundfile >= 0.12.1
    playsound==1.2.2
    numpy
    scipy

The implementation purposefully keeps dependencies minimal and only imported when
necessary so that the app can still work in text-only mode without the extra
libraries installed.
"""

import os
import subprocess
import sys
import tempfile
import time
from typing import Optional
import threading
import queue

# Lazy imports for heavy / optional deps
try:
    import sounddevice as sd  # type: ignore
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
except Exception:
    sd = None  # type: ignore
    np = None  # type: ignore
    sf = None  # type: ignore

try:
    from playsound import playsound  # type: ignore
except Exception:
    playsound = None  # type: ignore

try:
    import openai  # type: ignore
except Exception:
    openai = None

DEFAULT_SAMPLE_RATE = 16_000


class VoiceIO:
    """High-level helper that records microphone input, transcribes it using
    OpenAI Whisper and speaks text responses using OpenAI TTS.
    """

    def __init__(
        self,
        model_transcription: str = "whisper-1",
        model_tts: str = "tts-1",
        voice: str = "alloy",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        verbose: bool = False,
    ) -> None:
        if openai is None:
            raise ImportError(
                "`openai` python package is required for VoiceIO. Install with `pip install openai`"
            )
        if sd is None or np is None or sf is None:
            raise ImportError(
                "`sounddevice`, `soundfile` and `numpy` are required for VoiceIO. Install with `pip install sounddevice soundfile numpy`"
            )
        # `playsound` is optional because we have OS fallbacks.

        self.sample_rate = sample_rate
        self.model_transcription = model_transcription
        self.model_tts = model_tts
        self.voice_name = voice
        self.verbose = verbose
        # Client is created lazily because `openai.OpenAI()` will read env vars
        # and may raise if missing.
        self._client: Optional[openai.OpenAI] = None

    # ---------- OpenAI helpers ----------

    @property
    def client(self) -> "openai.OpenAI":  # type: ignore[name-defined]
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    # ---------- Recording helpers ----------

    def record_audio(self, duration: int = 5, filename: Optional[str] = None) -> str:
        """Record microphone audio for *duration* seconds.

        Returns path to a temporary WAV file containing the recording.
        """
        if sd is None:
            raise RuntimeError("sounddevice is not available – cannot record audio")

        if self.verbose:
            print(f"[VoiceIO] Recording for {duration} seconds…")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype="int16")
        sd.wait()  # Wait until recording is finished

        # Ensure directory for temp wav exists
        if filename is None:
            fd, filename = tempfile.mkstemp(suffix=".wav", prefix="voiceio_")
            os.close(fd)  # we'll write with soundfile

        sf.write(filename, recording, self.sample_rate)
        if self.verbose:
            print(f"[VoiceIO] Saved recording to {filename}")
        return filename

    # ---------- Speech ↔ Text ----------

    def speech_to_text(self, wav_path: str) -> str:
        if self.verbose:
            print("[VoiceIO] Transcribing audio with OpenAI Whisper…")
        with open(wav_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model=self.model_transcription,
                file=f,
            )
        text = transcription.text.strip()
        if self.verbose:
            print(f"[VoiceIO] Transcription result: {text}")
        return text

    def _text_to_speech_mp3_bytes(self, text: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.model_tts,
            voice=self.voice_name,
            input=text,
        )
        return response.content  # type: ignore[attr-defined]

    def text_to_speech(self, text: str) -> str:
        """Convert *text* to speech (MP3) and write it to a temporary file.

        Returns MP3 file path.
        """
        if self.verbose:
            print(f"[VoiceIO] Generating speech for: {text[:60]}…")
        audio_bytes = self._text_to_speech_mp3_bytes(text)
        fd, filename = tempfile.mkstemp(suffix=".mp3", prefix="voiceio_tts_")
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return filename

    # ---------- Playback ----------

    def play_audio(self, file_path: str) -> None:
        """Play an audio file.

        Primary method is the `playsound` package. If that fails (e.g. missing
        `AppKit` on macOS) we fall back to a platform-specific system command.
        """
        if self.verbose:
            print(f"[VoiceIO] Playing: {file_path}")
        try:
            if playsound is None:
                raise RuntimeError("playsound not available")
            playsound(file_path)
        except Exception as e:
            # Fallback strategies per platform
            if self.verbose:
                print(f"[VoiceIO] playsound failed: {e}. Falling back to system player…")
            if sys.platform == "darwin":
                subprocess.run(["afplay", file_path], check=False)
            elif sys.platform.startswith("linux"):
                # Try common CLI players
                for cmd in (["mpg123", "-q", file_path], ["aplay", file_path]):
                    if subprocess.run(["which", cmd[0]], capture_output=True).returncode == 0:
                        subprocess.run(cmd, check=False)
                        break
                else:
                    raise RuntimeError("No suitable audio player found on Linux") from e
            elif sys.platform.startswith("win"):
                # Use PowerShell on Windows
                subprocess.run([
                    "powershell",
                    "-c",
                    f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();",
                ], check=False)
            else:
                raise

    # ---------- High-level helpers ----------

    def speak(self, text):  # type: ignore[override]
        """Generate speech for *text* (str or list content) and play it."""
        # Normalize possible content structures
        if isinstance(text, list):
            # join text fragments or dicts containing 'text'
            processed_parts = []
            for part in text:
                if isinstance(part, dict) and "text" in part:
                    processed_parts.append(str(part["text"]))
                else:
                    processed_parts.append(str(part))
            text = " ".join(processed_parts)

        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return

        mp3_path = self.text_to_speech(text)
        try:
            self.play_audio(mp3_path)
        finally:
            # Cleanup temp file
            try:
                os.remove(mp3_path)
            except FileNotFoundError:
                pass 

    def play_beep(self):
        self.play_audio("beep.m4a")

    # ---------- Push-to-talk helper ----------

    def push_to_talk(self, hotkey: str = "space") -> str:  # type: ignore[override]
        """Record audio while *hotkey* is held down (default: *space* bar).

        When the user presses and holds the key, a short beep plays, recording starts
        and continues until the key is released. The captured audio is transcribed
        with :py:meth:`speech_to_text` and the temporary WAV file is cleaned up.

        Returns the recognised text (empty string if nothing recognised or the
        recording failed).
        """
        if sd is None:
            raise RuntimeError("sounddevice is not available – cannot record audio")

        try:
            from pynput import keyboard as kb  # lazy import – external optional dep
        except Exception as ex:  # pragma: no cover
            raise ImportError(
                "`pynput` is required for push-to-talk functionality. Install with `pip install pynput`."
            ) from ex

        print("[VoiceIO] Hold SPACE and speak… release to finish (press ESC to cancel).")

        # Synchronisation primitives to track key state
        pressed_evt = threading.Event()
        released_evt = threading.Event()
        cancel_evt = threading.Event()

        def _on_press(key):  # noqa: ANN001 – callback signature
            if key == kb.Key.esc:  # allow cancelling
                cancel_evt.set()
                return False  # stop listener
            if key == kb.Key.space and not pressed_evt.is_set():
                pressed_evt.set()
                # Play start beep (non-blocking)
                try:
                    self.play_beep()
                except Exception:
                    pass

        def _on_release(key):  # noqa: ANN001
            if key == kb.Key.space and pressed_evt.is_set():
                released_evt.set()
                return False  # stop listener once we're done recording

        # Wait for the user to press/hold the hotkey ---------------------------------
        with kb.Listener(on_press=_on_press, on_release=_on_release) as listener:
            listener.join()  # blocks until _on_release stops listener

            if cancel_evt.is_set():
                print("[VoiceIO] Recording cancelled.")
                return ""

        if not pressed_evt.is_set():
            # Should not happen but guard anyway
            return ""

        # ---------------------------------------------------------------------------
        # We recorded the key press but now need to record until key release.
        # Use non-blocking key monitor so we can capture audio frames continuously.
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def _audio_cb(indata, _frames, _time, _status):  # noqa: D401
            """Callback – push frames into queue."""
            if indata.size:
                audio_queue.put(indata.copy())

        # Start a keyboard listener in *background* to detect the release while
        # we run the blocking InputStream below.
        def _monitor_release():  # noqa: D401
            with kb.Listener(on_release=_on_release) as l_release:
                l_release.join()

        monitor_thread = threading.Thread(target=_monitor_release, daemon=True)
        monitor_thread.start()

        # Capture audio ----------------------------------------------------------------
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", callback=_audio_cb):
            # Wait until the release event fires
            while not released_evt.is_set():
                if cancel_evt.is_set():
                    break
                time.sleep(0.01)

        # Play end beep (best-effort)
        try:
            self.play_beep()
        except Exception:
            pass

        # Aggregate frames into a single numpy array
        if audio_queue.empty():
            return ""  # nothing captured

        frames: list[np.ndarray] = []
        while not audio_queue.empty():
            frames.append(audio_queue.get())
        audio_data = np.concatenate(frames, axis=0)

        # Write to temp wav, transcribe, cleanup
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="voiceio_ptt_")
        os.close(fd)
        sf.write(wav_path, audio_data, self.sample_rate)

        try:
            return self.speech_to_text(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass