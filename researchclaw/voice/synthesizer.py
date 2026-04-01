"""Text-to-speech synthesis."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VoiceSynthesizer:
    """Convert text to speech audio."""

    def __init__(self, server_config: Any) -> None:
        self._config = server_config

    async def synthesize(
        self,
        text: str,
        voice: str = "alloy",
        speed: float = 1.0,
    ) -> bytes:
        """Synthesize text to audio bytes using OpenAI TTS API."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required for TTS")

        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for TTS")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": voice,
                    "speed": speed,
                },
            )
            response.raise_for_status()
            return response.content
