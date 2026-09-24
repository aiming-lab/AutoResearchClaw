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
    "atlascloud": {
        "base_url": "https://api.atlascloud.ai/v1",
    },
    "cheaperinference": {
        "base_url": "https://api.cheaperinference.com/v1",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "adapter": "anthropic",
    },
    "kimi-anthropic": {
        "base_url": "https://api.kimi.com/coding/",
        "adapter": "anthropic",
    },
    "novita": {
        "base_url": "https://api.novita.ai/openai",
    },
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
    },
    "minimax-global": {
        "base_url": "https://api.minimax.io/v1",
    },
    "minimax-anthropic": {
        "base_url": "https://api.minimax.io/anthropic",
        "adapter": "anthropic",
    },
    "minimax-anthropic-cn": {
        "base_url": "https://api.minimaxi.com/anthropic",
        "adapter": "anthropic",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
    },
    "openai-compatible": {
        "base_url": None,  # Use user-provided base_url
    },
}


def create_llm_client(config: RCConfig) -> LLMClient | ACPClient:
    """Factory: return the right LLM client based on ``config.llm.provider``.

    - ``"acp"`` → :class:`ACPClient` (spawns an ACP-compatible agent)
    - providers with an ``"anthropic"`` adapter → :class:`LLMClient` with
      Anthropic Messages API support
    - ``"openrouter"`` → :class:`LLMClient` with OpenRouter base URL
    - ``"openai"`` → :class:`LLMClient` with OpenAI base URL
    - ``"deepseek"`` → :class:`LLMClient` with DeepSeek base URL
    - ``"atlascloud"`` → :class:`LLMClient` with Atlas Cloud base URL
    - ``"cheaperinference"`` → :class:`LLMClient` with Cheaper Inference base URL
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


def build_reviewer_llm(config: RCConfig):
    """Build an independent reviewer/judge LLM client (P0-3) or None.

    Returns None when ``llm.reviewer_model`` is empty so callers fall back to
    the generator client (backward-compatible). ACP-provider runs do not get a
    separate reviewer (None) — reviewing reuses the generator agent.
    """
    if getattr(config.llm, "provider", "") == "acp":
        return None
    from researchclaw.llm.client import LLMClient as _LLM

    try:
        return _LLM.reviewer_from_rc_config(config)
    except Exception:  # noqa: BLE001 - never block the pipeline on reviewer setup
        return None


def build_panel_llms(config: RCConfig) -> list:
    """Build the multi-model debate panel (Stage 8/14/18) or [] when disabled.

    Opt-in via ``llm.debate_enabled``. The panel reuses existing models —
    ``primary_model`` + ``reviewer_model`` (if set) + ``fallback_models`` —
    deduplicated by model name, each cloned into its own single-model client
    (no fallback chain) so each debate role can be bound to a distinct model.
    Returns [] for ACP runs, when debate is disabled, or on any error.
    """
    import dataclasses

    if getattr(config.llm, "provider", "") == "acp":
        return []
    if not getattr(config.llm, "debate_enabled", False):
        return []
    from researchclaw.llm.client import LLMClient as _LLM

    try:
        base = _LLM.from_rc_config(config)
        names: list[str] = []
        seen: set[str] = set()
        for m in (
            config.llm.primary_model,
            getattr(config.llm, "reviewer_model", "") or "",
            *(config.llm.fallback_models or ()),
        ):
            m = (m or "").strip()
            if m and m not in seen:
                seen.add(m)
                names.append(m)
        clients = []
        for name in names:
            cfg = dataclasses.replace(
                base.config, primary_model=name, fallback_models=[]
            )
            client = _LLM(cfg)
            # Preserve the Anthropic/Kimi adapter from the base client (panel
            # members share the base endpoint).
            client._anthropic = base._anthropic
            clients.append(client)
        return clients
    except Exception:  # noqa: BLE001 - never block the pipeline on panel setup
        return []
