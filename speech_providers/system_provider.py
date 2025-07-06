from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Optional

try:
    import speech_recognition as sr  # type: ignore
except Exception:  # pragma: no cover
    sr = None  # type: ignore

from .base import STTProvider, TTSProvider

__all__ = ["SystemSTTProvider", "SystemTTSProvider"]


class SystemSTTProvider(STTProvider):
    """Offline speech-to-text provider using *speech_recognition* + Sphinx."""

    def __init__(self) -> None:
        if sr is None:
            raise ImportError(
                "The 'speech_recognition' package (with PocketSphinx) is required for SystemSTTProvider."
            )

    def transcribe(self, wav_path: str) -> str:  # noqa: D401
        recognizer = sr.Recognizer()  # type: ignore[attr-defined]
        with sr.AudioFile(wav_path) as source:  # type: ignore[attr-defined]
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_sphinx(audio)  # type: ignore[attr-defined]
        except Exception:
            return ""


class SystemTTSProvider(TTSProvider):
    """Offline text-to-speech provider using built-in OS capabilities."""

    def __init__(self) -> None:
        # Determine appropriate extension per-platform
        if sys.platform == "darwin":
            self.file_extension = ".aiff"
        elif sys.platform.startswith("linux"):
            self.file_extension = ".wav"
        elif sys.platform.startswith("win"):
            self.file_extension = ".wav"
        else:
            raise RuntimeError("System TTS not supported on this OS")

    def _ensure_cmd(self, candidates: list[list[str]]) -> list[str]:
        """Return the first executable command from *candidates*."""
        for cmd in candidates:
            if subprocess.run(["which", cmd[0]], capture_output=True).returncode == 0:
                return cmd
        raise RuntimeError("No suitable system TTS command found.")

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:  # noqa: D401
        # Decide output path
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=self.file_extension, prefix="voiceio_sys_tts_")
            os.close(fd)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if sys.platform == "darwin":
            subprocess.run(["say", "-o", output_path, text], check=True)
        elif sys.platform.startswith("linux"):
            # Prefer pico2wave, then espeak
            try:
                cmd = self._ensure_cmd([["pico2wave", "-w", output_path, text], ["espeak", "-w", output_path, text]])
                subprocess.run(cmd, check=True)
            except RuntimeError:
                raise RuntimeError("Neither pico2wave nor espeak is available on this system.")
        elif sys.platform.startswith("win"):
            ps_cmd = (
                "Add-Type -AssemblyName System.speech; "
                f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$speak.SetOutputToWaveFile('{output_path}'); "
                f"$speak.Speak('{text}');"
            )
            subprocess.run(["powershell", "-Command", ps_cmd], check=True)
        else:  # pragma: no cover
            raise RuntimeError("Unsupported OS for SystemTTSProvider")

        return output_path 