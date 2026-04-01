"""Voice transcription via Whisper API."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class VoiceTranscriber:
    """Transcribe audio to text using Whisper API."""

    def __init__(self, server_config: Any) -> None:
        self._model = server_config.whisper_model
        self._api_url = server_config.whisper_api_url

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "zh",
    ) -> str:
        """Transcribe audio bytes to text.

        Uses OpenAI Whisper API or compatible endpoint.
        """
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "httpx is required for voice transcription. "
                "Install with: pip install httpx"
            )

        url = self._api_url or "https://api.openai.com/v1/audio/transcriptions"

        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for Whisper API")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.webm", audio_bytes, "audio/webm")},
                data={
                    "model": self._model,
                    "language": language,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("text", "")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        language: str = "zh",
    ) -> AsyncIterator[str]:
        """Stream transcription (collects chunks then transcribes)."""
        chunks: list[bytes] = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        if chunks:
            full_audio = b"".join(chunks)
            text = await self.transcribe(full_audio, language=language)
            yield text
