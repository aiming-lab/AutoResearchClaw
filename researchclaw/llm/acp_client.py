"""ACP (Agent Client Protocol) LLM client via acpx.

Uses acpx as the ACP bridge to communicate with any ACP-compatible agent
(Claude Code, Codex, Gemini CLI, etc.) via persistent named sessions.

Key advantage: a single persistent session maintains context across all
23 pipeline stages — the agent remembers everything.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

from researchclaw.llm.client import LLMResponse

logger = logging.getLogger(__name__)

# acpx output markers
_DONE_RE = re.compile(r"^\[done\]")
_CLIENT_RE = re.compile(r"^\[client\]")
_ACPX_RE = re.compile(r"^\[acpx\]")
_TOOL_RE = re.compile(r"^\[tool\]")


@dataclass
class ACPConfig:
    """Configuration for ACP agent connection."""

    agent: str = "claude"
    cwd: str = "."
    acpx_command: str = ""  # auto-detect if empty
    session_name: str = "researchclaw"
    timeout_sec: int = 1800  # per-prompt timeout


def _find_acpx() -> str | None:
    """Find the acpx binary — check PATH, then OpenClaw's plugin directory."""
    found = shutil.which("acpx")
    if found:
        return found
    # Check OpenClaw's bundled acpx plugin
    openclaw_acpx = os.path.expanduser(
        "~/.openclaw/extensions/acpx/node_modules/.bin/acpx"
    )
    if os.path.isfile(openclaw_acpx) and os.access(openclaw_acpx, os.X_OK):
        return openclaw_acpx
    return None


class ACPClient:
    """LLM client that uses acpx to communicate with ACP agents.

    Spawns persistent named sessions via acpx, reusing them across
    ``.chat()`` calls so the agent maintains context across the full
    23-stage pipeline.
    """

    # Track live instances for atexit cleanup (weak refs to avoid preventing GC)
    _live_instances: list[weakref.ref[ACPClient]] = []
    _atexit_registered: bool = False

    def __init__(self, acp_config: ACPConfig) -> None:
        self.config = acp_config
        self._acpx: str | None = acp_config.acpx_command or None
        self._session_ready = False
        # Prune dead weakrefs, then track this instance
        ACPClient._live_instances = [r for r in ACPClient._live_instances if r() is not None]
        ACPClient._live_instances.append(weakref.ref(self))
        if not ACPClient._atexit_registered:
            atexit.register(ACPClient._atexit_cleanup)
            ACPClient._atexit_registered = True

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> ACPClient:
        """Build from a ResearchClaw ``RCConfig``."""
        acp = rc_config.llm.acp
        return cls(ACPConfig(
            agent=acp.agent,
            cwd=acp.cwd,
            acpx_command=getattr(acp, "acpx_command", ""),
            session_name=getattr(acp, "session_name", "researchclaw"),
            timeout_sec=getattr(acp, "timeout_sec", 1800),
        ))

    # ------------------------------------------------------------------
    # Public interface (matches LLMClient)
    # ------------------------------------------------------------------

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
        """Send a prompt and return the agent's response.

        Parameters mirror ``LLMClient.chat()`` for drop-in compatibility.
        ``model``, ``max_tokens``, ``temperature``, and ``json_mode`` are
        accepted but not forwarded — the agent manages its own model and
        parameters.
        """
        prompt_text = self._messages_to_prompt(messages, system=system)
        content = self._send_prompt(prompt_text)
        if strip_thinking:
            from researchclaw.utils.thinking_tags import strip_thinking_tags
            content = strip_thinking_tags(content)
        return LLMResponse(
            content=content,
            model=f"acp:{self.config.agent}",
            finish_reason="stop",
        )

    def preflight(self) -> tuple[bool, str]:
        """Check that acpx and the agent are available."""
        acpx = self._resolve_acpx()
        if not acpx:
            return False, (
                "acpx not found. Install it: npm install -g acpx  "
                "or set llm.acp.acpx_command in config."
            )
        # Check the agent binary exists
        agent = self.config.agent
        if not shutil.which(agent):
            return False, f"ACP agent CLI not found: {agent!r} (not on PATH)"
        # Create the session
        try:
            self._ensure_session()
            return True, f"OK - ACP session ready ({agent} via acpx)"
        except Exception as exc:  # noqa: BLE001
            return False, f"ACP session init failed: {exc}"

    def close(self) -> None:
        """Close the acpx session."""
        if not self._session_ready:
            return
        acpx = self._resolve_acpx()
        if not acpx:
            return
        try:
            subprocess.run(
                [acpx, "--ttl", "0", "--cwd", self._abs_cwd(),
                 self.config.agent, "sessions", "close",
                 self.config.session_name],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15,
            )
        except Exception:  # noqa: BLE001
            pass
        self._session_ready = False

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def _atexit_cleanup(cls) -> None:
        """Close all live ACP sessions on interpreter shutdown."""
        for ref in cls._live_instances:
            inst = ref()
            if inst is not None:
                try:
                    inst.close()
                except Exception:  # noqa: BLE001
                    pass
        cls._live_instances.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_acpx(self) -> str | None:
        """Resolve the acpx binary path (cached)."""
        if self._acpx:
            return self._acpx
        self._acpx = _find_acpx()
        return self._acpx

    def _abs_cwd(self) -> str:
        return os.path.abspath(self.config.cwd)

    def _ensure_session(self) -> None:
        """Find or create the named acpx session."""
        if self._session_ready:
            return
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")

        # Use 'ensure' which finds existing or creates new
        result = subprocess.run(
            [acpx, "--ttl", "0", "--cwd", self._abs_cwd(),
             self.config.agent, "sessions", "ensure",
             "--name", self.config.session_name],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
        if result.returncode != 0:
            # Fall back to 'new'
            result = subprocess.run(
                [acpx, "--ttl", "0", "--cwd", self._abs_cwd(),
                 self.config.agent, "sessions", "new",
                 "--name", self.config.session_name],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create ACP session: {result.stderr.strip()}"
                )
        self._session_ready = True
        logger.info("ACP session '%s' ready (%s)", self.config.session_name, self.config.agent)

    # Linux MAX_ARG_STRLEN is 128 KB; Windows CreateProcess limit is ~32 KB
    # for the entire command line, not just the prompt payload. acpx adds
    # several fixed arguments plus quoting overhead, so leave generous headroom
    # on Windows and switch to temp-file transport earlier.
    _MAX_CLI_PROMPT_BYTES = 20_000 if sys.platform == "win32" else 100_000
    # On Windows, npm-installed CLIs usually resolve to ``.cmd`` launchers,
    # which are routed through ``cmd.exe`` and hit a much smaller practical
    # command-line limit (~8 KB). Use file transport much earlier there.
    _MAX_CMD_WRAPPER_PROMPT_BYTES = 6_000 if sys.platform == "win32" else 100_000

    # Localized error snippets for "command line too long" (may be in any OS language)
    _CMD_TOO_LONG_HINTS = (
        "too long",       # English Windows
        "trop long",      # French Windows
        "zu lang",        # German Windows
        "demasiado larg", # Spanish Windows
        "e2big",          # POSIX
    )

    # Error patterns that indicate a dead/stale session (retryable)
    _RECONNECT_ERRORS = (
        "agent needs reconnect",
        "session not found",
        "Query closed",
    )
    _MAX_RECONNECT_ATTEMPTS = 2

    @classmethod
    def _cli_prompt_limit(cls, acpx: str | None) -> int:
        """Return the safe inline-prompt size for the resolved ACP launcher."""
        limit = cls._MAX_CLI_PROMPT_BYTES
        if sys.platform == "win32" and acpx:
            lower = acpx.lower()
            if lower.endswith((".cmd", ".bat")):
                return min(limit, cls._MAX_CMD_WRAPPER_PROMPT_BYTES)
        return limit

    def _send_prompt(self, prompt: str) -> str:
        """Send a prompt via acpx and return the response text.

        For large prompts that would exceed the OS argument-length limit
        (``E2BIG``), the prompt is written to a temp file and the agent
        is asked to read it.

        If the session has died (common after long-running stages), retries
        up to ``_MAX_RECONNECT_ATTEMPTS`` times with automatic reconnection.
        """
        # Sanitize null bytes that may originate from web-scraped content
        # or OpenAlex API responses — subprocess.run() rejects \x00 because
        # the underlying C execve() treats it as a string terminator.
        prompt = prompt.replace("\x00", "")

        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")

        prompt_bytes = len(prompt.encode("utf-8"))
        prompt_limit = self._cli_prompt_limit(acpx)
        use_file = prompt_bytes > prompt_limit
        if use_file:
            logger.info(
                "Prompt too large for CLI arg (%d bytes > %d). Using temp file.",
                prompt_bytes,
                prompt_limit,
            )

        last_exc: RuntimeError | None = None
        for attempt in range(1 + self._MAX_RECONNECT_ATTEMPTS):
            self._ensure_session()
            try:
                if use_file:
                    return self._send_prompt_via_file(acpx, prompt)
                return self._send_prompt_cli(acpx, prompt)
            except OSError as os_exc:
                # OS-level failure (e.g., Windows CreateProcess arg limit).
                # Fall back to temp-file transport automatically.
                if not use_file:
                    logger.warning(
                        "CLI subprocess raised OSError, "
                        "falling back to temp file: %s",
                        os_exc,
                    )
                    use_file = True
                    return self._send_prompt_via_file(acpx, prompt)
                raise RuntimeError(
                    f"ACP prompt failed: {os_exc}"
                ) from os_exc
            except RuntimeError as exc:
                # Detect localized "command line too long" from subprocess stderr
                exc_lower = str(exc).lower()
                if not use_file and any(
                    h in exc_lower for h in self._CMD_TOO_LONG_HINTS
                ):
                    logger.warning(
                        "CLI prompt too long for OS, "
                        "falling back to temp file: %s",
                        exc,
                    )
                    use_file = True
                    return self._send_prompt_via_file(acpx, prompt)
                if not any(pat in str(exc) for pat in self._RECONNECT_ERRORS):
                    raise
                last_exc = exc
                if attempt < self._MAX_RECONNECT_ATTEMPTS:
                    logger.warning(
                        "ACP session died (%s), reconnecting (attempt %d/%d)...",
                        exc,
                        attempt + 1,
                        self._MAX_RECONNECT_ATTEMPTS,
                    )
                    self._force_reconnect()

        raise last_exc  # type: ignore[misc]

    def _force_reconnect(self) -> None:
        """Close the stale session and reset so _ensure_session creates a new one."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
        self._session_ready = False

    def _run_acp_with_heartbeat(
        self, cmd: list[str], *, label: str = "ACP prompt",
    ) -> subprocess.CompletedProcess[str]:
        """Run an ACP subprocess with periodic heartbeat logging.

        Instead of a silent blocking ``subprocess.run``, this uses ``Popen``
        with a background reader thread and logs a progress heartbeat every
        30 seconds so the user knows the agent is still working.
        """
        timeout = self.config.timeout_sec
        heartbeat_interval = 30  # seconds

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _reader(stream: Any, buf: list[str]) -> None:
            try:
                for line in stream:
                    buf.append(line)
            except Exception:  # noqa: BLE001
                pass

        t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

        start = time.monotonic()
        while True:
            try:
                proc.wait(timeout=heartbeat_interval)
                break  # process finished
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    proc.kill()
                    t_out.join(timeout=5)
                    t_err.join(timeout=5)
                    raise subprocess.TimeoutExpired(
                        cmd, timeout,
                        output="".join(stdout_chunks),
                        stderr="".join(stderr_chunks),
                    )
                logger.info(
                    "%s still running... %.0fs elapsed (timeout: %ds)",
                    label, elapsed, timeout,
                )

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    def _send_prompt_cli(self, acpx: str, prompt: str) -> str:
        """Send prompt as a CLI argument (original path)."""
        cmd = [
            acpx, "--approve-all", "--ttl", "0", "--cwd", self._abs_cwd(),
            self.config.agent, "-s", self.config.session_name, prompt,
        ]
        try:
            result = self._run_acp_with_heartbeat(cmd, label="ACP prompt (cli)")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ACP prompt timed out after {self.config.timeout_sec}s"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"ACP prompt failed (exit {result.returncode}): {stderr}")

        return self._extract_response(result.stdout)

    def _send_prompt_via_file(self, acpx: str, prompt: str) -> str:
        """Write prompt to a temp file, ask the agent to read and respond."""
        fd, prompt_path = tempfile.mkstemp(
            suffix=".md", prefix="rc_prompt_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(prompt)

            short_prompt = (
                f"Read the file at {prompt_path} in its entirety. "
                f"Follow ALL instructions contained in that file and "
                f"respond exactly as requested. Do NOT summarize, "
                f"just produce the requested output."
            )

            cmd = [
                acpx, "--approve-all", "--ttl", "0", "--cwd", self._abs_cwd(),
                self.config.agent, "-s", self.config.session_name,
                short_prompt,
            ]
            try:
                result = self._run_acp_with_heartbeat(cmd, label="ACP prompt (file)")
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"ACP prompt timed out after {self.config.timeout_sec}s"
                ) from exc

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                raise RuntimeError(
                    f"ACP prompt failed (exit {result.returncode}): {stderr}"
                )

            return self._extract_response(result.stdout)
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

    @staticmethod
    def _extract_response(raw_output: str | None) -> str:
        """Extract the agent's actual response from acpx output.

        Strips acpx metadata lines ([client], [acpx], [tool], [done])
        and their continuation lines (indented or sub-field lines like
        ``input:``, ``output:``, ``files:``, ``kind:``).
        """
        if not raw_output:
            return ""
        lines: list[str] = []
        in_tool_block = False
        for line in raw_output.splitlines():
            # Skip acpx control lines
            if _DONE_RE.match(line) or _CLIENT_RE.match(line) or _ACPX_RE.match(line):
                in_tool_block = False
                continue
            if _TOOL_RE.match(line):
                in_tool_block = True
                continue
            # Tool blocks have indented continuation lines
            if in_tool_block:
                if line.startswith("  ") or not line.strip():
                    continue
                # Non-indented, non-empty line = end of tool block
                in_tool_block = False
            # Skip empty lines at start
            if not lines and not line.strip():
                continue
            lines.append(line)

        # Trim trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)

    @staticmethod
    def _messages_to_prompt(
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
    ) -> str:
        """Flatten a chat-messages list into a single text prompt.

        Preserves role labels so the agent can distinguish context.
        """
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
        return "\n\n".join(parts)
