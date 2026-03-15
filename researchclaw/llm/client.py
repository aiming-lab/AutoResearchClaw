"""Multi-provider LLM client — stdlib only.

Supported providers:
  - OpenAI (default) — /chat/completions, Bearer auth
  - Anthropic — /messages, x-api-key auth, system as top-level param
  - OpenRouter — /chat/completions, Bearer auth + HTTP-Referer/X-Title
  - claude-cli — shells out to `claude --print` (Claude Max subscription, no API key needed)
  - anthropic-oauth — hits api.anthropic.com/v1/messages directly using the Claude CLI OAuth
                      token (sk-ant-oat01-…) with `anthropic-beta: oauth-2025-04-20`.
                      No Anthropic console key needed — uses your Claude Max subscription.
                      Token is auto-read from ~/.openclaw/agents/main/agent/auth-profiles.json
                      or ANTHROPIC_OAUTH_TOKEN env var.
  - gemini-cli — shells out to `gemini --output-format json` (Gemini CLI OAuth session,
                 free tier or Google One AI Premium, no API key needed).
                 Token is auto-refreshed by Gemini CLI. Default model: gemini-2.5-pro.
                 Auth is managed by `~/.config/gemini-cli/oauth_creds.json`.
  - codex-cli — shells out to `codex exec --json` (OpenAI Codex CLI, uses ChatGPT Pro/Plus
                subscription, no separate API key needed). Default model: gpt-5.3-codex-spark
                (or whatever is set in ~/.codex/config.toml). Supports all Codex CLI models
                including gpt-5.3-codex-spark, gpt-4.1, o3, etc.

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
import subprocess
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
    api_key_env: str = ""
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
        # Safely read optional fields that may not exist in older configs
        _get = lambda attr, default: getattr(rc_config.llm, attr, default) or default  # noqa: E731
        return cls(
            LLMConfig(
                base_url=_get("base_url", ""),
                api_key=str(
                    _get("api_key", "")
                    or os.environ.get(_get("api_key_env", ""), "")
                    or ""
                ),
                provider=provider,
                primary_model=rc_config.llm.primary_model,
                fallback_models=list(rc_config.llm.fallback_models or []),
                api_key_env=_get("api_key_env", ""),
                max_tokens=int(_get("max_tokens", 4096)),
                temperature=float(_get("temperature", 0.7)),
                timeout_sec=int(_get("timeout_sec", 300)),
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
        except Exception as e:  # noqa: BLE001
            return False, str(e)

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
            except RuntimeError as e:
                # CLI provider transient errors (empty response, timeout, etc.)
                # Non-retryable CLI errors: model not found, auth failure
                err_str = str(e).lower()
                if any(kw in err_str for kw in ("not found", "auth", "login", "install")):
                    raise
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2**attempt)
                    logger.info(
                        "CLI retry %d/%d for %s: %s. Waiting %.1fs.",
                        attempt + 1,
                        self.config.max_retries,
                        model,
                        str(e)[:100],
                        delay,
                    )
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
        if provider == "claude-cli":
            return self._call_claude_cli(model, messages, json_mode)
        if provider == "gemini-cli":
            return self._call_gemini_cli(model, messages, json_mode)
        if provider == "codex-cli":
            return self._call_codex_cli(model, messages, json_mode)
        if provider == "anthropic-oauth":
            return self._call_anthropic_oauth(
                model, messages, max_tokens, temperature, json_mode
            )
        # Default: OpenAI-compatible (covers "openai", "openai-compatible", etc.)
        return self._call_openai(
            model, messages, max_tokens, temperature, json_mode
        )

    # ------------------------------------------------------------------
    # anthropic-oauth: direct REST using Claude Max OAuth token
    # ------------------------------------------------------------------
    _OAUTH_BETAS = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
    _OAUTH_PROFILES_PATH = os.path.expanduser(
        "~/.openclaw/agents/main/agent/auth-profiles.json"
    )

    @classmethod
    def _resolve_oauth_token(cls, explicit_key: str) -> str:
        """Return the OAuth token to use, in priority order:
        1. explicit api_key / api_key_env value in config
        2. ANTHROPIC_OAUTH_TOKEN env var
        3. ~/.openclaw/agents/main/agent/auth-profiles.json (OpenClaw store)
        """
        if explicit_key and explicit_key.startswith("sk-ant-oat"):
            return explicit_key

        env_token = os.environ.get("ANTHROPIC_OAUTH_TOKEN", "").strip()
        if env_token:
            return env_token

        try:
            with open(cls._OAUTH_PROFILES_PATH) as f:
                data = json.load(f)
            profiles = data.get("profiles", {})
            for profile in profiles.values():
                token = profile.get("token", "")
                if isinstance(token, str) and token.startswith("sk-ant-oat"):
                    return token
        except (OSError, json.JSONDecodeError):
            pass

        raise RuntimeError(
            "No Claude OAuth token found. "
            "Run `openclaw models auth login --provider anthropic` or set "
            "ANTHROPIC_OAUTH_TOKEN env var."
        )

    def _call_anthropic_oauth(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Direct Anthropic API call using Claude Max OAuth token.

        Uses Authorization: Bearer <oat01-token> + anthropic-beta: oauth-2025-04-20.
        No Anthropic console API key needed.
        """
        explicit = self.config.api_key or os.environ.get(
            self.config.api_key_env or "", ""
        )
        token = self._resolve_oauth_token(explicit)
        base_url = (
            self.config.base_url.rstrip("/")
            if self.config.base_url
            else "https://api.anthropic.com"
        )

        # Split system messages
        system_parts: list[str] = []
        chat: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat.append(msg)

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature != 1.0:
            payload["temperature"] = temperature
        if json_mode:
            # Anthropic doesn't have a native json_mode; append instruction
            if payload["messages"] and payload["messages"][-1]["role"] == "user":
                payload["messages"][-1] = {
                    "role": "user",
                    "content": payload["messages"][-1]["content"]
                    + "\n\nRespond with valid JSON only.",
                }

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": self._OAUTH_BETAS,
                "User-Agent": "AutoResearchClaw/0.8 (anthropic-oauth)",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode()[:400]
            raise urllib.error.HTTPError(
                e.url, e.code, f"{e.reason}: {body_text}", e.headers, None
            ) from e

        content_blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        usage = data.get("usage", {})

        return LLMResponse(
            content=text,
            model=data.get("model", model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
            truncated=data.get("stop_reason") == "max_tokens",
            raw=data,
        )

    def _call_claude_cli(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
    ) -> LLMResponse:
        """Shell out to `claude --print` using Claude Max subscription.

        Concatenates message history into a single prompt for --print mode.
        No API key required — uses the locally authenticated Claude CLI session.
        """
        import shutil

        cli_path = shutil.which("claude") or "claude"

        # Separate system messages from the conversation
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append(msg)

        # Format conversation history as a single prompt block
        # claude --print is single-turn so we inline prior turns
        if len(conversation) == 1:
            prompt = conversation[0]["content"]
        else:
            lines: list[str] = []
            for msg in conversation:
                role_label = "Human" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role_label}: {msg['content']}")
            # Final turn must be a user message; append prompt cue
            prompt = "\n\n".join(lines)

        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        cmd = [cli_path, "--print", "--no-session-persistence"]

        if model and model not in ("default", "claude-cli"):
            cmd += ["--model", model]

        if system_parts:
            cmd += ["--system-prompt", "\n\n".join(system_parts)]

        # Run from /tmp to avoid reading workspace context files
        _cwd = tempfile.gettempdir()

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
                cwd=_cwd,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Install Claude Code: https://claude.ai/code"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI timed out after {self.config.timeout_sec}s"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()[:400]
            raise RuntimeError(f"claude CLI exited {result.returncode}: {stderr}")

        content = result.stdout.strip()
        if not content:
            raise RuntimeError("claude CLI returned empty response")

        # Rough token estimate (claude CLI doesn't expose usage)
        prompt_tokens = len(prompt.split()) * 4 // 3
        completion_tokens = len(content.split()) * 4 // 3

        return LLMResponse(
            content=content,
            model=model or "claude-cli",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            truncated=False,
            raw={"stdout": content, "stderr": result.stderr},
        )

    # ------------------------------------------------------------------
    # gemini-cli: subprocess using Gemini CLI OAuth session
    # ------------------------------------------------------------------
    def _call_gemini_cli(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
    ) -> LLMResponse:
        """Shell out to `gemini --output-format json` using Gemini CLI OAuth session.

        Uses `~/.config/gemini-cli/oauth_creds.json` (auto-refreshed by Gemini CLI).
        No API key required — uses the locally authenticated Gemini CLI session.
        Supports free tier and Google One AI Premium quota.
        """
        import shutil

        cli_path = shutil.which("gemini") or "gemini"

        # Separate system messages from conversation
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append(msg)

        # Build the prompt (gemini --prompt is single-turn via stdin or -p flag)
        if len(conversation) == 1:
            prompt = conversation[0]["content"]
        else:
            lines: list[str] = []
            for msg in conversation:
                role_label = "User" if msg["role"] == "user" else "Model"
                lines.append(f"{role_label}: {msg['content']}")
            prompt = "\n\n".join(lines)

        if system_parts:
            # Gemini CLI doesn't have a direct system prompt flag in non-interactive mode
            # so we prepend it to the prompt
            sys_block = "\n\n".join(system_parts)
            prompt = f"[System instructions: {sys_block}]\n\n{prompt}"

        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        # Resolve the effective model (gemini-2.5-pro is default)
        effective_model = model if model and model not in ("default", "gemini-cli") else "gemini-2.5-pro"

        cmd = [
            cli_path,
            "--output-format", "json",
            "--model", effective_model,
            "--prompt", prompt,
        ]

        # Run from /tmp to avoid reading workspace context files
        _cwd = tempfile.gettempdir()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
                cwd=_cwd,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "gemini CLI not found. Install: brew install gemini-cli or "
                "visit https://github.com/google-gemini/gemini-cli"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"gemini CLI timed out after {self.config.timeout_sec}s"
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()[:400]
            raise RuntimeError(f"gemini CLI exited {result.returncode}: {stderr}")

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("gemini CLI returned empty response")

        # Parse JSON output
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Fallback: treat raw output as the response
            return LLMResponse(
                content=stdout,
                model=effective_model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                finish_reason="stop",
                truncated=False,
                raw={"stdout": stdout},
            )

        content = data.get("response", "")
        stats = data.get("stats", {})
        model_stats = stats.get("models", {})
        # Aggregate token counts across all models used
        input_tokens = 0
        output_tokens = 0
        for ms in model_stats.values():
            tokens = ms.get("tokens", {})
            input_tokens += tokens.get("input", 0) or tokens.get("prompt", 0)
            output_tokens += tokens.get("candidates", 0)

        return LLMResponse(
            content=content,
            model=data.get("model", effective_model),
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            finish_reason="stop",
            truncated=False,
            raw=data,
        )

    # ------------------------------------------------------------------
    # codex-cli: subprocess using OpenAI Codex CLI (ChatGPT Pro/Plus subscription)
    # ------------------------------------------------------------------
    def _call_codex_cli(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
    ) -> LLMResponse:
        """Shell out to `codex exec --json` using ChatGPT Pro/Plus subscription.

        Uses ~/.codex/config.toml for auth. No separate OpenAI API key needed.
        The Codex CLI authenticates via the same OAuth session as ChatGPT.
        Supports: gpt-5.3-codex-spark, gpt-4.1, o3, o4-mini, and others available
        to your ChatGPT subscription tier.

        We use:
          codex exec --json -m <model> -o <tmpfile> -
        stdin receives the prompt; -o captures the final message cleanly;
        --json gives us JSONL with token usage from turn.completed.
        """
        import shutil
        import tempfile

        cli_path = shutil.which("codex") or "codex"

        # Separate system messages from conversation
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append(msg)

        # Build the prompt
        if len(conversation) == 1:
            prompt = conversation[0]["content"]
        else:
            lines: list[str] = []
            for msg in conversation:
                role_label = "Human" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role_label}: {msg['content']}")
            prompt = "\n\n".join(lines)

        if system_parts:
            sys_block = "\n\n".join(system_parts)
            prompt = f"[System instructions: {sys_block}]\n\n{prompt}"

        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        # Resolve effective model — fall back to Codex CLI default (from config.toml)
        effective_model = model if model and model not in ("default", "codex-cli") else None

        # Write output to a temp file so we can read the clean final message
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
            out_path = tf.name

        cmd = [cli_path, "exec", "--json", "--skip-git-repo-check", "-o", out_path]
        if effective_model:
            cmd += ["-c", f"model={effective_model}"]
        cmd += ["-"]  # read prompt from stdin

        import os as _os

        # Run codex from /tmp to avoid reading workspace context files
        # (AGENTS.md, SOUL.md, MEMORY.md) which add 50K+ tokens per call.
        _cwd = tempfile.gettempdir()

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec,
                cwd=_cwd,
            )
        except FileNotFoundError:
            _os.unlink(out_path) if _os.path.exists(out_path) else None
            raise RuntimeError(
                "codex CLI not found. Install: brew install codex or "
                "visit https://developers.openai.com/codex/cli"
            )
        except subprocess.TimeoutExpired:
            _os.unlink(out_path) if _os.path.exists(out_path) else None
            raise RuntimeError(
                f"codex CLI timed out after {self.config.timeout_sec}s"
            )

        # Parse JSONL for token usage and errors
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        error_msg: str | None = None
        final_text: str = ""

        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type", "")
            if etype == "turn.completed":
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cached_tokens = usage.get("cached_input_tokens", 0)
            elif etype == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    final_text = item.get("text", "")
            elif etype == "error":
                error_msg = event.get("message", "")
            elif etype == "turn.failed":
                err = event.get("error", {})
                error_msg = err.get("message", error_msg or "turn failed")

        if error_msg:
            _os.unlink(out_path) if _os.path.exists(out_path) else None
            raise RuntimeError(f"codex CLI error: {error_msg}")

        if not final_text:
            # Fallback: read the -o file written by codex exec -o (must read BEFORE cleanup)
            try:
                with open(out_path) as f:
                    final_text = f.read().strip()
            except OSError:
                pass

        # Cleanup temp file now that we've read it
        try:
            _os.unlink(out_path)
        except OSError:
            pass

        if not final_text:
            raise RuntimeError("codex CLI returned empty response")

        used_model = effective_model or "gpt-5.3-codex-spark"
        return LLMResponse(
            content=final_text,
            model=used_model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            finish_reason="stop",
            truncated=False,
            raw={"stdout": result.stdout, "cached_input_tokens": cached_tokens},
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
