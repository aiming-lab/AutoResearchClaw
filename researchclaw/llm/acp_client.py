"""ACP (Agent Client Protocol) LLM client via acpx.

Uses acpx as the ACP bridge to communicate with any ACP-compatible agent
(Claude Code, Codex, Gemini CLI, etc.) via persistent named sessions.

Key advantage: a single persistent session maintains context across all
23 pipeline stages — the agent remembers everything.

On Windows, acpx subprocess communication suffers from encoding issues
(GBK codec errors) and process hangs.  When ``direct_mode`` is enabled
(the default on Windows), prompts are sent directly via the agent's
``--print`` flag, bypassing acpx entirely.  Large prompts are piped
through stdin to avoid the Windows CreateProcess ~32 KB argument limit.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import subprocess
import tempfile
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

# Agent CLIs known to support a ``--print`` (or equivalent) non-interactive
# mode that reads from stdin and writes the response to stdout.
_DIRECT_MODE_AGENTS: dict[str, list[str]] = {
    "claude": ["--print", "--output-format", "text"],
    # Future agents can be added here, e.g.:
    # "gemini": ["--print"],
    # "codex": ["--print"],
}

# Windows CreateProcess has a ~32 KB command-line limit.  On POSIX the
# limit is 128 KB (MAX_ARG_STRLEN).  We stay well under both.
_WIN_CLI_ARG_LIMIT = 20_000
_POSIX_CLI_ARG_LIMIT = 100_000


@dataclass
class ACPConfig:
    """Configuration for ACP agent connection."""

    agent: str = "claude"
    cwd: str = "."
    acpx_command: str = ""  # auto-detect if empty
    session_name: str = "researchclaw"
    timeout_sec: int = 1800  # per-prompt timeout
    direct_mode: bool | None = None  # None = auto-detect (True on Windows)


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


def _should_use_direct_mode(config: ACPConfig) -> bool:
    """Decide whether to use direct CLI mode instead of acpx.

    Returns ``True`` when:
    - ``direct_mode`` is explicitly ``True`` in config, OR
    - ``direct_mode`` is ``None`` (auto) AND we are on Windows, OR
    - ``direct_mode`` is ``None`` (auto) AND acpx is not found.
    """
    if config.direct_mode is True:
        return True
    if config.direct_mode is False:
        return False
    # Auto-detect: prefer direct mode on Windows (acpx has encoding bugs)
    if os.name == "nt":
        return True
    # Also fall back to direct mode if acpx is not installed
    if not _find_acpx():
        return True
    return False


class ACPClient:
    """LLM client that uses acpx to communicate with ACP agents.

    Spawns persistent named sessions via acpx, reusing them across
    ``.chat()`` calls so the agent maintains context across the full
    23-stage pipeline.

    On Windows (or when ``direct_mode`` is enabled), bypasses acpx and
    invokes the agent CLI directly via ``--print`` mode.  This avoids
    subprocess encoding issues (GBK codec errors on Windows) and process
    hangs that plague acpx on non-POSIX platforms.
    """

    # Track live instances for atexit cleanup (weak refs to avoid preventing GC)
    _live_instances: list[weakref.ref[ACPClient]] = []

    def __init__(self, acp_config: ACPConfig) -> None:
        self.config = acp_config
        self._acpx: str | None = acp_config.acpx_command or None
        self._session_ready = False
        self._use_direct = _should_use_direct_mode(acp_config)
        if self._use_direct:
            logger.info(
                "ACP direct mode enabled — bypassing acpx, using '%s --print'",
                acp_config.agent,
            )
        # Register for atexit cleanup to prevent zombie acpx processes
        ACPClient._live_instances.append(weakref.ref(self))
        atexit.register(ACPClient._atexit_cleanup)

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
            direct_mode=getattr(acp, "direct_mode", None),
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
        """Check that the agent is available (and acpx, if not in direct mode)."""
        agent = self.config.agent
        if not shutil.which(agent):
            return False, f"ACP agent CLI not found: {agent!r} (not on PATH)"

        if self._use_direct:
            if agent not in _DIRECT_MODE_AGENTS:
                return False, (
                    f"Direct mode not supported for agent {agent!r}. "
                    f"Supported agents: {', '.join(_DIRECT_MODE_AGENTS)}"
                )
            return True, (
                f"OK - ACP direct mode ready ({agent} --print)"
            )

        # acpx path
        acpx = self._resolve_acpx()
        if not acpx:
            return False, (
                "acpx not found. Install it: npm install -g acpx  "
                "or set llm.acp.acpx_command in config."
            )
        try:
            self._ensure_session()
            return True, f"OK - ACP session ready ({agent} via acpx)"
        except Exception as exc:  # noqa: BLE001
            return False, f"ACP session init failed: {exc}"

    def close(self) -> None:
        """Close the acpx session (no-op in direct mode)."""
        if self._use_direct or not self._session_ready:
            return
        acpx = self._resolve_acpx()
        if not acpx:
            return
        try:
            subprocess.run(
                [acpx, "--ttl", "0", "--cwd", self._abs_cwd(),
                 self.config.agent, "sessions", "close",
                 self.config.session_name],
                capture_output=True, timeout=15,
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
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            # Fall back to 'new'
            result = subprocess.run(
                [acpx, "--ttl", "0", "--cwd", self._abs_cwd(),
                 self.config.agent, "sessions", "new",
                 "--name", self.config.session_name],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to create ACP session: {result.stderr.strip()}"
                )
        self._session_ready = True
        logger.info("ACP session '%s' ready (%s)", self.config.session_name, self.config.agent)

    # Linux MAX_ARG_STRLEN is 128KB; stay well under to leave room for env
    _MAX_CLI_PROMPT_BYTES = 100_000

    # Error patterns that indicate a dead/stale session (retryable)
    _RECONNECT_ERRORS = (
        "agent needs reconnect",
        "session not found",
        "Query closed",
    )
    _MAX_RECONNECT_ATTEMPTS = 2

    def _send_prompt(self, prompt: str) -> str:
        """Route the prompt to direct mode or acpx depending on config."""
        if self._use_direct:
            agent_bin = shutil.which(self.config.agent)
            if not agent_bin:
                raise RuntimeError(
                    f"Agent CLI not found: {self.config.agent!r}"
                )
            return self._send_prompt_direct(agent_bin, prompt)

        return self._send_prompt_acpx(prompt)

    # ------------------------------------------------------------------
    # Direct mode (bypasses acpx — Windows-safe)
    # ------------------------------------------------------------------

    def _send_prompt_direct(self, agent_bin: str, prompt: str) -> str:
        """Send prompt directly via the agent's ``--print`` mode.

        Small prompts (< OS arg limit) are passed as a ``-p`` argument.
        Large prompts are piped through stdin to avoid the Windows
        ``CreateProcess`` ~32 KB command-line length limit.

        This bypasses acpx entirely, avoiding:
        - GBK encoding errors on Windows (subprocess stdout decoding)
        - Zombie acpx/node processes that hang indefinitely
        - Session reconnection failures
        """
        agent_name = self.config.agent
        extra_args = _DIRECT_MODE_AGENTS.get(agent_name, ["--print"])
        prompt_bytes = len(prompt.encode("utf-8"))
        arg_limit = _WIN_CLI_ARG_LIMIT if os.name == "nt" else _POSIX_CLI_ARG_LIMIT

        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

        if prompt_bytes > arg_limit:
            # Large prompt: pipe via stdin
            logger.debug(
                "Prompt too large for CLI arg (%d bytes > %d). Using stdin.",
                prompt_bytes, arg_limit,
            )
            try:
                result = subprocess.run(
                    [agent_bin, *extra_args],
                    input=prompt,
                    capture_output=True, text=True,
                    timeout=self.config.timeout_sec,
                    cwd=self._abs_cwd(),
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"{agent_name} --print timed out after "
                    f"{self.config.timeout_sec}s (stdin mode, "
                    f"{prompt_bytes} bytes)"
                ) from exc
        else:
            # Short prompt: pass as -p argument
            try:
                result = subprocess.run(
                    [agent_bin, *extra_args, "-p", prompt],
                    capture_output=True, text=True,
                    timeout=self.config.timeout_sec,
                    cwd=self._abs_cwd(),
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"{agent_name} --print timed out after "
                    f"{self.config.timeout_sec}s"
                ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"{agent_name} --print failed (exit {result.returncode}): "
                f"{stderr}"
            )

        return result.stdout.strip()

    # ------------------------------------------------------------------
    # acpx mode (original implementation)
    # ------------------------------------------------------------------

    def _send_prompt_acpx(self, prompt: str) -> str:
        """Send a prompt via acpx and return the response text.

        For large prompts that would exceed the OS argument-length limit
        (``E2BIG``), the prompt is written to a temp file and the agent
        is asked to read it.

        If the session has died (common after long-running stages), retries
        up to ``_MAX_RECONNECT_ATTEMPTS`` times with automatic reconnection.
        """
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")

        prompt_bytes = len(prompt.encode("utf-8"))
        use_file = prompt_bytes > self._MAX_CLI_PROMPT_BYTES
        if use_file:
            logger.info(
                "Prompt too large for CLI arg (%d bytes). Using temp file.",
                prompt_bytes,
            )

        last_exc: RuntimeError | None = None
        for attempt in range(1 + self._MAX_RECONNECT_ATTEMPTS):
            self._ensure_session()
            try:
                if use_file:
                    return self._send_prompt_via_file(acpx, prompt)
                return self._send_prompt_cli(acpx, prompt)
            except RuntimeError as exc:
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

    def _send_prompt_cli(self, acpx: str, prompt: str) -> str:
        """Send prompt as a CLI argument (original path)."""
        try:
            result = subprocess.run(
                [acpx, "--approve-all", "--ttl", "0", "--cwd", self._abs_cwd(),
                 self.config.agent, "-s", self.config.session_name,
                 prompt],
                capture_output=True, text=True,
                timeout=self.config.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ACP prompt timed out after {self.config.timeout_sec}s"
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"ACP prompt failed (exit {result.returncode}): {stderr}")

        return self._extract_response(result.stdout)

    def _send_prompt_via_file(self, acpx: str, prompt: str) -> str:
        """Write prompt to a temp file, ask the agent to read and respond."""
        tmp_dir = tempfile.gettempdir()
        fd, prompt_path = tempfile.mkstemp(
            suffix=".md", prefix="rc_prompt_", dir=tmp_dir
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

            try:
                result = subprocess.run(
                    [acpx, "--approve-all", "--ttl", "0", "--cwd", self._abs_cwd(),
                     self.config.agent, "-s", self.config.session_name,
                     short_prompt],
                    capture_output=True, text=True,
                    timeout=self.config.timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"ACP prompt timed out after {self.config.timeout_sec}s"
                ) from exc

            if result.returncode != 0:
                stderr = result.stderr.strip()
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
    def _extract_response(raw_output: str) -> str:
        """Extract the agent's actual response from acpx output.

        Strips acpx metadata lines ([client], [acpx], [tool], [done])
        and their continuation lines (indented or sub-field lines like
        ``input:``, ``output:``, ``files:``, ``kind:``).
        """
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
