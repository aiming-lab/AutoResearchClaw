"""Voice command parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class VoiceCommand(str, Enum):
    """Recognized voice commands."""

    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    STATUS = "status"
    NONE = "none"  # Not a command, forward to chat


@dataclass
class ParsedVoiceInput:
    """Result of parsing voice input."""

    command: VoiceCommand
    text: str  # original or remaining text


# Command patterns (Chinese + English)
_COMMAND_PATTERNS: list[tuple[VoiceCommand, re.Pattern[str]]] = [
    (VoiceCommand.START, re.compile(r"^(?:start|run|开始|启动)", re.IGNORECASE)),
    (VoiceCommand.STOP, re.compile(r"^(?:stop|停止|结束|终止)", re.IGNORECASE)),
    (VoiceCommand.PAUSE, re.compile(r"^(?:pause|暂停|等一下)", re.IGNORECASE)),
    (VoiceCommand.RESUME, re.compile(r"^(?:resume|continue|继续|恢复)", re.IGNORECASE)),
    (VoiceCommand.STATUS, re.compile(r"^(?:status|progress|进度|到哪了|查看)", re.IGNORECASE)),
]


def parse_voice_input(text: str) -> ParsedVoiceInput:
    """Parse transcribed voice input into command + text."""
    stripped = text.strip()
    for cmd, pattern in _COMMAND_PATTERNS:
        if pattern.search(stripped):
            return ParsedVoiceInput(command=cmd, text=stripped)

    return ParsedVoiceInput(command=VoiceCommand.NONE, text=stripped)
