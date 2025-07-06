from __future__ import annotations

import os
import tempfile
from typing import Optional

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore

from .base import STTProvider, TTSProvider

__all__ = ["OpenAIProvider"]


class OpenAIProvider(STTProvider, TTSProvider):
    """Speech provider that delegates STT and TTS to OpenAI endpoints."""

    file_extension: str = ".mp3"

    def __init__(
        self,
        model_transcription: str = "whisper-1",
        model_tts: str = "tts-1",
        voice: str = "alloy",
    ) -> None:
        if openai is None:
            raise ImportError(
                "The 'openai' python package is required for OpenAIProvider. Install with `pip install openai`."
            )
        self.model_transcription = model_transcription
        self.model_tts = model_tts
        self.voice = voice
        self._client: Optional[openai.OpenAI] = None  # type: ignore[assignment]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    @property
    def client(self) -> "openai.OpenAI":  # type: ignore[name-defined]
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    # ---------------------------------------------------------------------
    # STTProvider API
    # ---------------------------------------------------------------------
    def transcribe(self, wav_path: str) -> str:  # noqa: D401
        with open(wav_path, "rb") as f:
            transcription = self.client.audio.transcriptions.create(
                model=self.model_transcription,
                file=f,
            )
        return transcription.text.strip()

    # ---------------------------------------------------------------------
    # TTSProvider API
    # ---------------------------------------------------------------------
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:  # noqa: D401
        response = self.client.audio.speech.create(
            model=self.model_tts,
            voice=self.voice,
            input=text,
        )
        audio_bytes = response.content  # type: ignore[attr-defined]

        # Decide output path
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=self.file_extension, prefix="voiceio_tts_")
        else:
            # Ensure directory exists and truncate if necessary
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return output_path 