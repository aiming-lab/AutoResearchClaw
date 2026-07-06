"""Codex CLI-backed LLM client."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from researchclaw.llm.client import LLMResponse


@dataclass
class CodexCliConfig:
    """Configuration for invoking ``codex exec``."""

    command: str = "codex"
    cwd: str = "."
    model: str = ""
    timeout_sec: int = 600


class CodexCliClient:
    """LLM client backed by ``codex exec`` in read-only sandbox mode."""

    def __init__(self, config: CodexCliConfig) -> None:
        self.config = config

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> CodexCliClient:
        acp = rc_config.llm.acp
        return cls(
            CodexCliConfig(
                command="codex",
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
        """Send a prompt through Codex CLI.

        ``max_tokens`` and ``temperature`` are accepted for drop-in
        compatibility with :class:`LLMClient` but are not forwarded to the CLI.
        """
        _ = (max_tokens, temperature)
        prompt = self._messages_to_prompt(messages, system=system, json_mode=json_mode)
        content = self._run_codex(prompt, model=model)
        if strip_thinking:
            from researchclaw.utils.thinking_tags import strip_thinking_tags

            content = strip_thinking_tags(content)
        return LLMResponse(
            content=content,
            model=f"codex-cli:{model or self.config.model or 'default'}",
            finish_reason="stop",
        )

    def preflight(self) -> tuple[bool, str]:
        if not shutil.which(self.config.command):
            return False, f"Codex CLI not found: {self.config.command!r}"
        try:
            content = self._run_codex(
                "Reply with exactly: OK",
                timeout_sec=min(self.config.timeout_sec, 180),
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"Codex CLI preflight failed: {exc}"
        if content.strip() != "OK":
            return False, f"Codex CLI returned unexpected preflight output: {content[:120]}"
        return True, "OK - Codex CLI responding"

    def _run_codex(
        self,
        prompt: str,
        *,
        model: str | None = None,
        timeout_sec: int | None = None,
    ) -> str:
        command = shutil.which(self.config.command) or self.config.command
        with tempfile.NamedTemporaryFile(
            prefix="researchclaw_codex_", suffix=".txt", delete=False
        ) as tmp:
            output_path = Path(tmp.name)
        try:
            cmd = [
                command,
                "exec",
                "-C",
                os.path.abspath(self.config.cwd),
                "-s",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--output-last-message",
                str(output_path),
            ]
            effective_model = model or self.config.model
            if effective_model:
                cmd.extend(["--model", effective_model])
            cmd.append("-")

            result = subprocess.run(
                cmd,
                input=prompt,
                cwd=os.path.abspath(self.config.cwd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec or self.config.timeout_sec,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                raise RuntimeError(f"Codex CLI failed (exit {result.returncode}): {stderr}")
            return output_path.read_text(encoding="utf-8", errors="replace").strip()
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

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
