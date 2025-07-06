from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class STTProvider(ABC):
    """Abstract base for speech-to-text implementations."""

    @abstractmethod
    def transcribe(self, wav_path: str) -> str:  # noqa: D401
        """Return the recognised text from the given *wav_path*."""


class TTSProvider(ABC):
    """Abstract base for text-to-speech implementations."""

    #: Default file extension (including leading dot) used by the provider.
    file_extension: str = ".mp3"

    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:  # noqa: D401
        """Create speech audio for *text* and write it to *output_path*.

        If *output_path* is *None*, the provider should create a temporary file.
        The method MUST return the path to the generated audio file.
        """ 