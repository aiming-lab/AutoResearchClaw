"""Multi-provider LLM client — stdlib only.

Supported providers:
  - OpenAI (default) — /chat/completions, Bearer auth
  - Anthropic — /messages, x-api-key auth, system as top-level param
  - OpenRouter — /chat/completions, Bearer auth + HTTP-Referer/X-Title

Features:
  - Model fallback chain (configurable per provider)
  - Auto-detect max_tokens vs max_completion_tokens per model (OpenAI)
  - Cloudflare User-Agent bypass
  - Exponential backoff retry with jitter
  - JSON mode support (OpenAI/OpenRouter only)
  - Streaming disabled (sync only)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens
_NEW_PARAM_MODELS = frozenset(
    {
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.4",
    }
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class LLMResponse:
    """Parsed response from the LLM API."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    base_url: str
    api_key: str
    provider: str = "openai"  # "openai" | "anthropic" | "openrouter"
    primary_model: str = "gpt-4o"
    fallback_models: list[str] = field(
        default_factory=lambda: ["gpt-4.1", "gpt-4o-mini"]
    )
    max_tokens: int = 4096
    temperature: float = 0.7
    max_retries: int = 3
    retry_base_delay: float = 2.0
    timeout_sec: int = 300
    user_agent: str = _DEFAULT_USER_AGENT


class LLMClient:
    """Multi-provider LLM client (OpenAI, Anthropic, OpenRouter)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._model_chain = [config.primary_model] + config.fallback_models

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> LLMClient:
        provider = getattr(rc_config.llm, "provider", "") or "openai"
        return cls(
            LLMConfig(
                base_url=rc_config.llm.base_url,
                api_key=str(
                    rc_config.llm.api_key
                    or os.environ.get(rc_config.llm.api_key_env, "")
                    or ""
                ),
                provider=provider,
                primary_model=rc_config.llm.primary_model,
                fallback_models=list(rc_config.llm.fallback_models or []),
            )
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with retry and fallback.

        Args:
            messages: List of {role, content} dicts.
            model: Override model (skips fallback chain).
            max_tokens: Override max token count.
            temperature: Override temperature.
            json_mode: Request JSON response format.
            system: Prepend a system message.

        Returns:
            LLMResponse with content and metadata.
        """
        if system:
            messages = [{"role": "system", "content": system}] + messages

        models = [model] if model else self._model_chain
        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        last_error: Exception | None = None

        for m in models:
            try:
                return self._call_with_retry(m, messages, max_tok, temp, json_mode)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Model %s failed: %s. Trying next.", m, exc)
                last_error = exc

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        ) from last_error

    def preflight(self) -> tuple[bool, str]:
        """Quick connectivity check - one minimal chat call.

        Returns (success, message).
        Distinguishes: 401 (bad key), 403 (model forbidden),
                       404 (bad endpoint), 429 (rate limited), timeout.
        """
        provider = self.config.provider

        # Anthropic requires max_tokens >= 1; reasoning models need more
        is_reasoning = any(
            self.config.primary_model.startswith(p) for p in _NEW_PARAM_MODELS
        )
        if provider == "anthropic":
            min_tokens = 16
        elif is_reasoning:
            min_tokens = 64
        else:
            min_tokens = 1

        try:
            _ = self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=min_tokens,
                temperature=0,
            )
            return True, (
                f"OK - {provider} model {self.config.primary_model} responding"
            )
        except urllib.error.HTTPError as e:
            status_map = {
                401: "Invalid API key",
                403: f"Model {self.config.primary_model} not allowed for this key",
                404: f"Endpoint not found: {self.config.base_url}",
                429: "Rate limited - try again in a moment",
            }
            msg = status_map.get(e.code, f"HTTP {e.code}")
            return False, msg
        except (urllib.error.URLError, OSError) as e:
            return False, f"Connection failed: {e}"
        except RuntimeError as e:
            return False, f"All models failed: {e}"

    def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Call with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                return self._raw_call(
                    model, messages, max_tokens, temperature, json_mode
                )
            except urllib.error.HTTPError as e:
                status = e.code
                body = ""
                try:
                    body = e.read().decode()[:500]
                except Exception:  # noqa: BLE001
                    pass

                # Non-retryable errors
                if status == 403 and "not allowed to use model" in body:
                    raise  # Model not available — let fallback handle
                if status == 400:
                    raise  # Bad request — fix the request, don't retry

                # Retryable: 429 (rate limit), 500, 502, 503, 504
                if status in (429, 500, 502, 503, 504):
                    delay = self.config.retry_base_delay * (2**attempt)
                    # Add jitter
                    import random

                    delay += random.uniform(0, delay * 0.3)
                    logger.info(
                        "Retry %d/%d for %s (HTTP %d). Waiting %.1fs.",
                        attempt + 1,
                        self.config.max_retries,
                        model,
                        status,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                raise  # Other HTTP errors
            except urllib.error.URLError:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2**attempt)
                    time.sleep(delay)
                    continue
                raise

        # Should not reach here, but just in case
        return self._raw_call(model, messages, max_tokens, temperature, json_mode)

    def _raw_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Route to the correct provider-specific call method."""
        provider = self.config.provider
        if provider == "anthropic":
            return self._call_anthropic(
                model, messages, max_tokens, temperature, json_mode
            )
        if provider == "openrouter":
            return self._call_openrouter(
                model, messages, max_tokens, temperature, json_mode
            )
        # Default: OpenAI-compatible (covers "openai", "openai-compatible", etc.)
        return self._call_openai(
            model, messages, max_tokens, temperature, json_mode
        )

    def _call_openai(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """OpenAI /chat/completions call."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Reasoning models need max_completion_tokens with higher minimum
        if any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS):
            reasoning_min = 32768
            body["max_completion_tokens"] = max(max_tokens, reasoning_min)
        else:
            body["max_tokens"] = max_tokens

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        payload = json.dumps(body).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            },
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            data = json.loads(resp.read())

        return self._parse_openai_response(data, model)

    def _call_anthropic(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Anthropic /messages call with x-api-key auth."""
        base = self.config.base_url.rstrip("/") or "https://api.anthropic.com/v1"

        # Extract system messages — Anthropic takes system as top-level param
        system_parts: list[str] = []
        non_system: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                non_system.append(msg)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": non_system,
        }

        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        # Anthropic thinking models (claude with extended thinking) should not
        # receive temperature — skip for safety on all Anthropic calls if temp==0
        # or if model looks like a thinking model. For now, always include unless 0.
        if temperature > 0:
            body["temperature"] = temperature

        if json_mode:
            logger.debug(
                "json_mode requested but Anthropic does not support "
                "response_format natively — passing messages as-is"
            )

        payload = json.dumps(body).encode("utf-8")
        url = f"{base}/messages"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
            },
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            data = json.loads(resp.read())

        # Anthropic response: {"content": [{"type": "text", "text": "..."}], ...}
        content_blocks = data.get("content", [])
        text = content_blocks[0]["text"] if content_blocks else ""

        usage = data.get("usage", {})

        return LLMResponse(
            content=text,
            model=data.get("model", model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", ""),
            truncated=(data.get("stop_reason", "") == "max_tokens"),
            raw=data,
        )

    def _call_openrouter(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """OpenRouter /chat/completions — OpenAI-compatible with extra headers."""
        base = self.config.base_url.rstrip("/") or "https://openrouter.ai/api/v1"

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        payload = json.dumps(body).encode("utf-8")
        url = f"{base}/chat/completions"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.config.user_agent,
                "HTTP-Referer": "https://github.com/ArielleTolome/AutoResearchClaw",
                "X-Title": "AutoResearchClaw",
            },
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
            data = json.loads(resp.read())

        return self._parse_openai_response(data, model)

    def _parse_openai_response(
        self, data: dict[str, Any], model: str
    ) -> LLMResponse:
        """Parse an OpenAI-compatible chat completion response."""
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"] or "",
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            truncated=(choice.get("finish_reason", "") == "length"),
            raw=data,
        )


def create_client_from_yaml(yaml_path: str | None = None) -> LLMClient:
    """Create an LLMClient from the ARC config file.

    Reads base_url and api_key from config.arc.yaml's llm section.
    """
    import yaml as _yaml

    if yaml_path is None:
        yaml_path = "config.yaml"

    with open(yaml_path, encoding="utf-8") as f:
        raw = _yaml.safe_load(f)

    llm_section = raw.get("llm", {})
    provider = llm_section.get("provider", "openai") or "openai"

    # Resolve default base_url per provider
    default_urls = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    default_key_envs = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    base_url = llm_section.get("base_url") or default_urls.get(
        provider, "https://api.openai.com/v1"
    )
    api_key_env = llm_section.get("api_key_env") or default_key_envs.get(
        provider, "OPENAI_API_KEY"
    )
    api_key = str(
        os.environ.get(api_key_env, llm_section.get("api_key", "")) or ""
    )

    return LLMClient(
        LLMConfig(
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            primary_model=llm_section.get("primary_model", "gpt-4o"),
            fallback_models=llm_section.get(
                "fallback_models", ["gpt-4.1", "gpt-4o-mini"]
            ),
        )
    )
