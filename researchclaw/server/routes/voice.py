"""Voice upload / transcription API routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "zh",
) -> dict[str, Any]:
    """Transcribe uploaded audio using Whisper API."""
    try:
        from researchclaw.voice.transcriber import VoiceTranscriber
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Voice dependencies not installed. Run: pip install researchclaw[voice]",
        )

    from researchclaw.server.app import _app_state

    config = _app_state.get("config")
    if not config or not config.server.voice_enabled:
        raise HTTPException(status_code=403, detail="Voice is not enabled in config")

    audio_bytes = await file.read()
    transcriber = VoiceTranscriber(config.server)
    text = await transcriber.transcribe(audio_bytes, language=language)

    return {"text": text, "language": language}
