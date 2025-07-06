from __future__ import annotations

"""Utility for voice input (speech-to-text) and voice output (text-to-speech).

The core I/O logic is provider-agnostic. Concrete STT/TTS implementations live
in `providers/` and are injected at runtime (see `main.py`).

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
import hashlib
from providers.base import STTProvider, TTSProvider

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

# No direct provider imports here – providers are instantiated externally.

DEFAULT_SAMPLE_RATE = 16_000


class VoiceIO:
    """High-level helper that records microphone input, transcribes it using
    an injected STT provider and speaks text responses using an injected TTS
    provider.
    """

    def __init__(
        self,
        *,
        stt_provider: STTProvider,
        tts_provider: TTSProvider,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        verbose: bool = False,
    ) -> None:
        # Core audio deps are mandatory for recording / playback
        if sd is None or np is None or sf is None:
            raise ImportError(
                "`sounddevice`, `soundfile` and `numpy` are required for VoiceIO. Install with `pip install sounddevice soundfile numpy`"
            )

        # Providers (mandatory)
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider

        self.sample_rate = sample_rate
        self.verbose = verbose

        # Simple on-disk TTS cache ----------------------------------------------------
        self._cache_dir = os.path.join(tempfile.gettempdir(), "voiceio_cache")
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
        except Exception:
            # Fallback to cwd if temp dir is not writable
            self._cache_dir = os.path.abspath("voiceio_cache")
            os.makedirs(self._cache_dir, exist_ok=True)

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
        """Transcribe *wav_path* using the configured STT provider."""
        if self.verbose:
            print("[VoiceIO] Transcribing audio…")
        text = self.stt_provider.transcribe(wav_path)  # type: ignore[arg-type]
        if self.verbose:
            print(f"[VoiceIO] Transcription result: {text}")
        return text

    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """Generate speech for *text* via the configured TTS provider.

        Returns the path to the generated audio file (created by the provider).
        """
        if self.verbose:
            print(f"[VoiceIO] Generating speech for: {text[:60]}…")
        return self.tts_provider.synthesize(text, output_path)  # type: ignore[arg-type]

    # ---------- Playback ----------

    def play_audio(self, file_path: str) -> None:
        """Play an audio file.

        Primary method is to spawn a lightweight system player subprocess so that
        we can *optionally* interrupt playback early by pressing the *ESC* key.
        If that is not possible (e.g. `pynput` not installed or a suitable player
        binary is missing) we fall back to the blocking :pypi:`playsound` method
        just like before.
        """
        if self.verbose:
            print(f"[VoiceIO] Playing: {file_path} (press ESC to skip)")

        # ------------------------------------------------------------------
        # Try to enable interrupt-able playback via a dedicated subprocess
        # ------------------------------------------------------------------
        try:
            from pynput import keyboard as kb  # lazy optional dependency
        except Exception:
            kb = None  # type: ignore

        def _spawn_player() -> Optional[subprocess.Popen]:  # noqa: D401
            """Start an OS specific audio player as a subprocess (non-blocking).

            Returns the *Popen* handle if successful, otherwise *None*.
            """
            if sys.platform == "darwin":
                return subprocess.Popen(["afplay", file_path])
            if sys.platform.startswith("linux"):
                for cmd in (["mpg123", "-q", file_path], ["aplay", file_path]):
                    if subprocess.run(["which", cmd[0]], capture_output=True).returncode == 0:
                        return subprocess.Popen(cmd)
                return None
            if sys.platform.startswith("win"):
                # PowerShell one-liner – still runs synchronously but inside our
                # own process so we can terminate it.
                ps_cmd = (
                    f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"
                )
                return subprocess.Popen(["powershell", "-c", ps_cmd])
            # Unsupported OS
            return None

        # Only attempt interruptible mode if we have *both* pynput and a player
        if kb is not None:
            player_proc = _spawn_player()
            if player_proc is not None:
                interrupted = threading.Event()

                def _on_press(key):  # noqa: ANN001
                    if key == kb.Key.esc:
                        # Mark interrupted and kill the subprocess.
                        interrupted.set()
                        try:
                            player_proc.terminate()
                        except Exception:
                            pass
                        finally:
                            self.play_beep()
                        return False  # stop listener

                listener = kb.Listener(on_press=_on_press)
                listener.start()
                # Wait until playback ends or is interrupted
                player_proc.wait()
                listener.stop()

                if interrupted.is_set() and self.verbose:
                    print("[VoiceIO] Playback interrupted by user.")
                return  # Either way we're done – early or not.

        # ------------------------------------------------------------------
        # Fallback: original blocking behaviour with playsound / system players
        # ------------------------------------------------------------------
        try:
            if playsound is None:
                raise RuntimeError("playsound not available")
            playsound(file_path)
        except Exception as e:
            # Fallback strategies per platform (blocking)
            if self.verbose:
                print(f"[VoiceIO] playsound failed: {e}. Falling back to system player…")
            if sys.platform == "darwin":
                subprocess.run(["afplay", file_path], check=False)
            elif sys.platform.startswith("linux"):
                for cmd in (["mpg123", "-q", file_path], ["aplay", file_path]):
                    if subprocess.run(["which", cmd[0]], capture_output=True).returncode == 0:
                        subprocess.run(cmd, check=False)
                        break
                else:
                    raise RuntimeError("No suitable audio player found on Linux") from e
            elif sys.platform.startswith("win"):
                subprocess.run([
                    "powershell",
                    "-c",
                    f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();",
                ], check=False)
            else:
                raise

    # ---------- High-level helpers ----------

    def speak(self, text, *, cache: bool = False):  # type: ignore[override]
        """Generate speech for *text* (str or list content) and play it.

        If *cache* is True we store the generated MP3 in a temp directory and
        reuse it for identical messages (hashing the UTF-8 text). This avoids
        repeated TTS API calls for common prompts such as "Waiting for input…".
        """
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

        # ---------------------------------------------------------------------
        ext = getattr(self.tts_provider, "file_extension", ".mp3")

        cached_path: Optional[str] = None
        if cache:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
            cached_path = os.path.join(self._cache_dir, f"{key}{ext}")
            if os.path.exists(cached_path):
                self.play_audio(cached_path)
                return

        audio_path = self.text_to_speech(text, cached_path if cache else None)
        try:
            self.play_audio(audio_path)
        finally:
            # Cleanup temp file if not cached
            if os.path.exists(audio_path) and (not cache or audio_path != cached_path):
                try:
                    os.remove(audio_path)
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

        print("[VoiceIO] Hold SPACE and speak… release to finish (ESC to cancel).")

        pressed_evt = threading.Event()
        released_evt = threading.Event()
        cancel_evt = threading.Event()

        def _on_press(key):  # noqa: ANN001
            if key == kb.Key.esc:
                cancel_evt.set()
                return False
            if key == kb.Key.space:
                pressed_evt.set()
                return False  # Stop early; we only needed the press.

        # Wait for the first SPACE press ------------------------------------------------
        with kb.Listener(on_press=_on_press) as wait_listener:
            wait_listener.join()

        if cancel_evt.is_set():
            print("[VoiceIO] Recording cancelled.")
            return ""

        if not pressed_evt.is_set():
            return ""  # Shouldn't happen

        # Play start beep --------------------------------------------------------------
        try:
            self.play_beep()
        except Exception:
            pass

        # Prepare listener for release while recording

        def _on_release(key):  # noqa: ANN001
            if key == kb.Key.space:
                released_evt.set()
                return False

        release_listener = kb.Listener(on_release=_on_release)
        release_listener.start()

        # Capture audio between press and release --------------------------------------
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def _audio_cb(indata, _frames, _time, _status):  # noqa: D401
            if indata.size:
                audio_queue.put(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=_audio_cb,
        ):
            while not released_evt.is_set() and not cancel_evt.is_set():
                time.sleep(0.01)

        release_listener.join()

        # Play end beep ----------------------------------------------------------------
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