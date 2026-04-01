"""Intent classification for conversational research."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class Intent(str, Enum):
    """Research chat intents."""

    TOPIC_SELECTION = "topic_selection"
    START_PIPELINE = "start_pipeline"
    CHECK_STATUS = "check_status"
    MODIFY_CONFIG = "modify_config"
    DISCUSS_RESULTS = "discuss_results"
    EDIT_PAPER = "edit_paper"
    GENERAL_CHAT = "general_chat"
    HELP = "help"


# Keyword patterns for fast classification
_INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    (Intent.HELP, re.compile(
        r"(?:^\s*help\s*$|\bhow\s+to\b|\busage\b|帮助|怎么用)", re.IGNORECASE
    )),
    (Intent.START_PIPELINE, re.compile(
        r"(?:\b(?:start|run|begin|launch)\b|开始|启动|跑|运行)",
        re.IGNORECASE,
    )),
    (Intent.CHECK_STATUS, re.compile(
        r"(?:\b(?:status|progress|stage|current)\b|阶段|进度|到哪|第几|哪一步)", re.IGNORECASE
    )),
    (Intent.TOPIC_SELECTION, re.compile(
        r"(?:\b(?:topic|idea|direction)\b|research\s+direction|研究方向|选题|研究主题|想法)",
        re.IGNORECASE,
    )),
    (Intent.MODIFY_CONFIG, re.compile(
        r"(?:\b(?:config|setting|parameter|batch|epoch)\b|learning\s+rate|学习率|修改|设置)",
        re.IGNORECASE,
    )),
    (Intent.DISCUSS_RESULTS, re.compile(
        r"(?:\b(?:results?|metrics?|accuracy|loss|performance)\b|结果|指标|效果|怎么样)",
        re.IGNORECASE,
    )),
    (Intent.EDIT_PAPER, re.compile(
        r"(?:\b(?:paper|abstract|introduction|draft)\b|论文|摘要|改一下|写)",
        re.IGNORECASE,
    )),
]


def classify_intent(message: str) -> tuple[Intent, float]:
    """Classify user intent from message text.

    Returns (intent, confidence) where confidence is 0-1.
    Uses keyword matching for speed; can be replaced with LLM.
    """
    message_lower = message.strip().lower()

    if not message_lower:
        return Intent.GENERAL_CHAT, 0.0

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(message_lower):
            return intent, 0.8

    return Intent.GENERAL_CHAT, 0.5
