"""LLM integration — OpenAI-compatible and ACP agent clients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from researchclaw.config import RCConfig
    from researchclaw.llm.acp_client import ACPClient
    from researchclaw.llm.client import LLMClient

# Provider presets for common LLM services
PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
    },
    "kimi-anthropic": {
        "base_url": "https://api.kimi.com/coding/",
    },
    "novita": {
        "base_url": "https://api.novita.ai/openai",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
    },
    "openai-compatible": {
        "base_url": None,  # Use user-provided base_url
    },
}


def create_llm_client(config: RCConfig) -> LLMClient | ACPClient:
    """Factory: return the right LLM client based on ``config.llm.provider``.

    - ``"acp"`` → :class:`ACPClient` (spawns an ACP-compatible agent)
    - ``"anthropic"`` → :class:`LLMClient` with Anthropic Messages API adapter
    - ``"kimi-anthropic"`` → :class:`LLMClient` with Kimi Coding Anthropic adapter
    - ``"openrouter"`` → :class:`LLMClient` with OpenRouter base URL
    - ``"openai"`` → :class:`LLMClient` with OpenAI base URL
    - ``"deepseek"`` → :class:`LLMClient` with DeepSeek base URL
    - ``"novita"`` → :class:`LLMClient` with Novita AI base URL
    - ``"minimax"`` → :class:`LLMClient` with MiniMax base URL
    - ``"openai-compatible"`` (default) → :class:`LLMClient` with custom base_url

    OpenRouter is fully compatible with the OpenAI API format, making it
    a drop-in replacement with access to 200+ models from Anthropic, Google,
    Meta, Mistral, and more. See: https://openrouter.ai/models
    """
    if config.llm.provider == "acp":
        from researchclaw.llm.acp_client import ACPClient as _ACP
        return _ACP.from_rc_config(config)

    from researchclaw.llm.client import LLMClient as _LLM

    # Use from_rc_config to properly initialize adapters (e.g., Anthropic)
    return _LLM.from_rc_config(config)
