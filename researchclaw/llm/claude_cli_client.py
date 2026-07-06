"""Claude Code CLI-backed LLM client."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from researchclaw.llm.client import LLMResponse


@dataclass
class ClaudeCliConfig:
    """Configuration for invoking ``claude -p``."""

    command: str = "claude"
    cwd: str = "."
    model: str = ""
    timeout_sec: int = 600
    max_budget_usd: str = ""


class ClaudeCliClient:
    """LLM client backed by the local Claude Code CLI."""

    def __init__(self, config: ClaudeCliConfig) -> None:
        self.config = config

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> ClaudeCliClient:
        acp = rc_config.llm.acp
        return cls(
            ClaudeCliConfig(
                command="claude",
                cwd=getattr(acp, "cwd", "."),
                model=getattr(rc_config.llm, "primary_model", ""),
                timeout_sec=int(getattr(acp, "timeout_sec", 600)),
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
        strip_thinking: bool = False,
    ) -> LLMResponse:
        """Send a prompt through Claude Code CLI.

        ``max_tokens`` and ``temperature`` are accepted for drop-in
        compatibility with :class:`LLMClient` but are not forwarded to the CLI.
        """
        _ = (max_tokens, temperature)
        prompt = self._messages_to_prompt(messages, system=system, json_mode=json_mode)
        content = self._run_claude(prompt, model=model)
        if strip_thinking:
            from researchclaw.utils.thinking_tags import strip_thinking_tags

            content = strip_thinking_tags(content)
        return LLMResponse(
            content=content,
            model=f"claude-cli:{model or self.config.model or 'default'}",
            finish_reason="stop",
        )

    def preflight(self) -> tuple[bool, str]:
        if not shutil.which(self.config.command):
            return False, f"Claude CLI not found: {self.config.command!r}"
        try:
            content = self._run_claude(
                "Reply with exactly: OK",
                timeout_sec=min(self.config.timeout_sec, 120),
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"Claude CLI preflight failed: {exc}"
        if content.strip() != "OK":
            return (
                False,
                f"Claude CLI returned unexpected preflight output: {content[:120]}",
            )
        return True, "OK - Claude CLI responding"

    def _run_claude(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout_sec: int | None = None,
    ) -> str:
        command = shutil.which(self.config.command) or self.config.command
        cmd = [command, "-p", "--output-format", "text"]
        effective_model = model or self.config.model
        if effective_model:
            cmd.extend(["--model", effective_model])
        if self.config.max_budget_usd:
            cmd.extend(["--max-budget-usd", self.config.max_budget_usd])

        result = subprocess.run(
            cmd,
            input=prompt,
            cwd=os.path.abspath(self.config.cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec or self.config.timeout_sec,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {stderr}")
        return (result.stdout or "").strip()

    @staticmethod
    def _messages_to_prompt(
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        parts: list[str] = []
        if system:
            parts.append(f"[System]\n{system}")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Previous Response]\n{content}")
            else:
                parts.append(content)
        if json_mode:
            parts.append(
                "Return only valid JSON. Do not wrap it in Markdown fences or add prose."
            )
        return "\n\n".join(parts)
