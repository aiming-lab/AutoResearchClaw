"""LLM integration — OpenAI-compatible and ACP agent clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from researchclaw.config import RCConfig
    from researchclaw.llm.acp_client import ACPClient
    from researchclaw.llm.client import LLMClient


def create_llm_client(config: RCConfig) -> LLMClient | ACPClient:
    """Factory: return the right LLM client based on ``config.llm.provider``.

    - ``"acp"`` → :class:`ACPClient` (spawns an ACP-compatible agent)
    - anything else → :class:`LLMClient` (OpenAI-compatible HTTP)
    """
    if config.llm.provider == "acp":
        from researchclaw.llm.acp_client import ACPClient as _ACP

        return _ACP.from_rc_config(config)

    from researchclaw.llm.client import LLMClient as _LLM

    return _LLM.from_rc_config(config)
