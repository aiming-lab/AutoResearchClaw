"""ACP (Agent Client Protocol) LLM client via acpx.

Uses acpx as the ACP bridge to communicate with any ACP-compatible agent
(Claude Code, Codex, Gemini CLI, etc.) via persistent named sessions.

Key advantage: a single persistent session maintains context across all
23 pipeline stages — the agent remembers everything.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
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
    verbose: bool = False
    stateless_prompt: bool = False
    reconnect_retries: int = 2
    reconnect_backoff_sec: float = 2.0
    capture_status_on_failure: bool = False
    debug_log_path: str = ""
    archive_failed_prompt_files: bool = False


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
            verbose=getattr(acp, "verbose", False),
            stateless_prompt=getattr(acp, "stateless_prompt", False),
            reconnect_retries=getattr(acp, "reconnect_retries", 2),
            reconnect_backoff_sec=getattr(acp, "reconnect_backoff_sec", 2.0),
            capture_status_on_failure=getattr(acp, "capture_status_on_failure", False),
            debug_log_path=getattr(acp, "debug_log_path", ""),
            archive_failed_prompt_files=getattr(acp, "archive_failed_prompt_files", False),
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
        if self.config.stateless_prompt:
            return True, f"OK - ACP stateless prompt mode ready ({agent} via acpx)"
        # Create the session
        try:
            self._ensure_session()
            return True, f"OK - ACP session ready ({agent} via acpx)"
        except Exception as exc:  # noqa: BLE001
            return False, f"ACP session init failed: {exc}"

    def close(self) -> None:
        """Close the acpx session."""
        if self.config.stateless_prompt:
            self._session_ready = False
            return
        if not self._session_ready:
            return
        acpx = self._resolve_acpx()
        if not acpx:
            return
        try:
            subprocess.run(
                [
                    *self._acpx_base_command(acpx, approve_all=False),
                    "sessions",
                    "close",
                    self.config.session_name,
                ],
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

    def _acpx_base_command(self, acpx: str, *, approve_all: bool) -> list[str]:
        cmd = [acpx]
        if self.config.verbose:
            cmd.append("--verbose")
        if approve_all:
            cmd.append("--approve-all")
        cmd.extend(["--ttl", "0", "--cwd", self._abs_cwd(), self.config.agent])
        return cmd

    def _debug_log_path(self) -> Path | None:
        raw = str(getattr(self.config, "debug_log_path", "") or "").strip()
        if not raw:
            return None
        return Path(raw)

    @staticmethod
    def _debug_timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _append_debug_event(self, event: str, **payload: Any) -> None:
        record = {
            "ts": self._debug_timestamp(),
            "event": event,
            **payload,
        }
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
        logger.info("ACP_DEBUG %s", serialized)
        debug_path = self._debug_log_path()
        if debug_path is None:
            return
        try:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append ACP debug log %s: %s", debug_path, exc)

    def _archive_prompt_file(self, prompt_path: str, *, session_name: str) -> str:
        if not self.config.archive_failed_prompt_files:
            return ""
        source_path = Path(prompt_path)
        if not source_path.exists():
            return ""
        debug_path = self._debug_log_path()
        base_dir = debug_path.parent if debug_path is not None else Path(self._abs_cwd())
        archive_dir = base_dir / "acp_failed_prompts"
        archive_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{session_name}-{source_path.name}"
        target_path = archive_dir / target_name
        shutil.copy2(source_path, target_path)
        return str(target_path)

    def _status_command(self, acpx: str, session_name: str) -> list[str]:
        cmd = self._acpx_base_command(acpx, approve_all=False)
        cmd.extend(["status", "-s", session_name])
        return cmd

    def _capture_session_status(self, acpx: str, session_name: str) -> str:
        if not self.config.capture_status_on_failure:
            return ""
        try:
            result = subprocess.run(
                self._status_command(acpx, session_name),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            return f"<status lookup failed: {exc}>"

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        chunks = [f"exit={result.returncode}"]
        if stdout:
            chunks.append(f"stdout:\n{stdout}")
        if stderr:
            chunks.append(f"stderr:\n{stderr}")
        return "\n".join(chunks)

    def _record_failure_context(
        self,
        *,
        acpx: str,
        session_name: str,
        transport: str,
        prompt_bytes: int,
        prompt_limit: int,
        use_file: bool,
        error_text: str,
        prompt_path: str | None = None,
        returncode: int | None = None,
        timed_out: bool = False,
    ) -> None:
        archived_prompt_path = ""
        if prompt_path:
            try:
                archived_prompt_path = self._archive_prompt_file(
                    prompt_path,
                    session_name=session_name,
                )
            except Exception as exc:  # noqa: BLE001
                archived_prompt_path = ""
                logger.warning("Failed to archive ACP prompt file %s: %s", prompt_path, exc)

        status_text = self._capture_session_status(acpx, session_name)
        self._append_debug_event(
            "prompt_failure",
            session_name=session_name,
            transport=transport,
            prompt_bytes=prompt_bytes,
            prompt_limit=prompt_limit,
            use_file=use_file,
            prompt_path=prompt_path or "",
            archived_prompt_path=archived_prompt_path,
            returncode=returncode,
            timed_out=timed_out,
            error=error_text,
            session_status=status_text,
        )

    def _ensure_session(self) -> None:
        """Find or create the named acpx session."""
        if self._session_ready:
            return
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")
        self._create_or_ensure_session(acpx, self.config.session_name, ensure=True)
        self._session_ready = True
        logger.info("ACP session '%s' ready (%s)", self.config.session_name, self.config.agent)

    # Linux MAX_ARG_STRLEN is 128 KB; Windows CreateProcess limit is ~32 KB
    # for the entire command line, not just the prompt payload. acpx adds
    # several fixed arguments plus quoting overhead, so leave generous headroom
    # on Windows and switch to temp-file transport earlier.
    _MAX_CLI_PROMPT_BYTES = 6_000
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
        "query closed",
        "queue owner disconnected before prompt completion",
    )

    @classmethod
    def _cli_prompt_limit(cls, acpx: str | None) -> int:
        """Return the safe inline-prompt size for the resolved ACP launcher."""
        limit = cls._MAX_CLI_PROMPT_BYTES
        if sys.platform == "win32" and acpx:
            lower = acpx.lower()
            if lower.endswith((".cmd", ".bat")):
                return min(limit, cls._MAX_CMD_WRAPPER_PROMPT_BYTES)
        return limit

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str:
        """Strip NUL bytes before subprocess transport.

        ``subprocess.run()`` rejects arguments containing ``\\x00`` with
        ``ValueError: embedded null byte``. This can happen when upstream
        scraping or artifact text accidentally carries NULs into the prompt.
        """
        if "\x00" not in prompt:
            return prompt
        return prompt.replace("\x00", "")

    def _send_prompt(self, prompt: str) -> str:
        """Send a prompt via acpx and return the response text.

        For large prompts that would exceed the OS argument-length limit
        (``E2BIG``), the prompt is written to a temp file and the agent
        is asked to read it.

        If the session has died (common after long-running stages), retries
        up to the configured reconnect retry count with automatic reconnection.
        """
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")

        prompt = self._sanitize_prompt(prompt)

        prompt_bytes = len(prompt.encode("utf-8"))
        prompt_limit = self._cli_prompt_limit(acpx)
        use_file = prompt_bytes > prompt_limit
        if use_file:
            logger.info(
                "Prompt too large for CLI arg (%d bytes > %d). Using temp file.",
                prompt_bytes,
                prompt_limit,
            )

        if self.config.stateless_prompt:
            last_exc: RuntimeError | None = None
            for attempt in range(1 + self.config.reconnect_retries):
                session_name = self._new_ephemeral_session(acpx)
                self._append_debug_event(
                    "prompt_attempt",
                    session_name=session_name,
                    stateless=True,
                    attempt=attempt + 1,
                    max_attempts=1 + self.config.reconnect_retries,
                    prompt_bytes=prompt_bytes,
                    prompt_limit=prompt_limit,
                    use_file=use_file,
                )
                try:
                    if use_file:
                        return self._send_prompt_via_file(
                            acpx,
                            prompt,
                            session_name=session_name,
                            prompt_bytes=prompt_bytes,
                            prompt_limit=prompt_limit,
                        )
                    return self._send_prompt_cli(
                        acpx,
                        prompt,
                        session_name=session_name,
                        prompt_bytes=prompt_bytes,
                        prompt_limit=prompt_limit,
                    )
                except OSError as os_exc:
                    if not use_file:
                        logger.warning(
                            "Stateless ACP subprocess raised OSError, "
                            "falling back to temp file: %s",
                            os_exc,
                        )
                        use_file = True
                        return self._send_prompt_via_file(
                            acpx,
                            prompt,
                            session_name=session_name,
                            prompt_bytes=prompt_bytes,
                            prompt_limit=prompt_limit,
                        )
                    raise RuntimeError(
                        f"ACP prompt failed: {os_exc}"
                    ) from os_exc
                except RuntimeError as exc:
                    exc_lower = str(exc).lower()
                    if not use_file and any(
                        h in exc_lower for h in self._CMD_TOO_LONG_HINTS
                    ):
                        logger.warning(
                            "Stateless ACP prompt too long for OS, "
                            "falling back to temp file: %s",
                            exc,
                        )
                        use_file = True
                        return self._send_prompt_via_file(
                            acpx,
                            prompt,
                            session_name=session_name,
                        )
                    if not self._is_reconnect_error(exc):
                        raise
                    last_exc = exc
                    if attempt < self.config.reconnect_retries:
                        self._append_debug_event(
                            "prompt_retrying",
                            session_name=session_name,
                            stateless=True,
                            attempt=attempt + 1,
                            remaining_retries=self.config.reconnect_retries - attempt,
                            error=str(exc),
                        )
                        logger.warning(
                            "Stateless ACP session died (%s), retrying "
                            "with a fresh ephemeral session (attempt %d/%d)...",
                            exc,
                            attempt + 1,
                            self.config.reconnect_retries,
                        )
                        self._sleep_before_retry()
                        continue
                finally:
                    self._close_named_session(acpx, session_name)

            raise last_exc  # type: ignore[misc]

        last_exc: RuntimeError | None = None
        for attempt in range(1 + self.config.reconnect_retries):
            self._ensure_session()
            self._append_debug_event(
                "prompt_attempt",
                session_name=self.config.session_name,
                stateless=False,
                attempt=attempt + 1,
                max_attempts=1 + self.config.reconnect_retries,
                prompt_bytes=prompt_bytes,
                prompt_limit=prompt_limit,
                use_file=use_file,
            )
            try:
                if use_file:
                    return self._send_prompt_via_file(
                        acpx,
                        prompt,
                        prompt_bytes=prompt_bytes,
                        prompt_limit=prompt_limit,
                    )
                return self._send_prompt_cli(
                    acpx,
                    prompt,
                    prompt_bytes=prompt_bytes,
                    prompt_limit=prompt_limit,
                )
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
                    return self._send_prompt_via_file(
                        acpx,
                        prompt,
                        prompt_bytes=prompt_bytes,
                        prompt_limit=prompt_limit,
                    )
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
                if not self._is_reconnect_error(exc):
                    raise
                last_exc = exc
                if attempt < self.config.reconnect_retries:
                    self._append_debug_event(
                        "prompt_retrying",
                        session_name=self.config.session_name,
                        stateless=False,
                        attempt=attempt + 1,
                        remaining_retries=self.config.reconnect_retries - attempt,
                        error=str(exc),
                    )
                    logger.warning(
                        "ACP session died (%s), reconnecting (attempt %d/%d)...",
                        exc,
                        attempt + 1,
                        self.config.reconnect_retries,
                    )
                    self._force_reconnect()
                    self._sleep_before_retry()

        raise last_exc  # type: ignore[misc]

    def _force_reconnect(self) -> None:
        """Close the stale session and reset so _ensure_session creates a new one."""
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
        self._session_ready = False

    def _is_reconnect_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(pattern in text for pattern in self._RECONNECT_ERRORS)

    def _sleep_before_retry(self) -> None:
        delay = max(float(getattr(self.config, "reconnect_backoff_sec", 0.0) or 0.0), 0.0)
        if delay > 0:
            time.sleep(delay)

    def _send_prompt_cli(
        self,
        acpx: str,
        prompt: str,
        *,
        session_name: str | None = None,
        prompt_bytes: int,
        prompt_limit: int,
    ) -> str:
        """Send prompt as a CLI argument (original path)."""
        active_session = session_name or self.config.session_name
        try:
            result = subprocess.run(
                self._prompt_command(acpx, prompt, session_name=active_session),
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=self.config.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            self._record_failure_context(
                acpx=acpx,
                session_name=active_session,
                transport="cli",
                prompt_bytes=prompt_bytes,
                prompt_limit=prompt_limit,
                use_file=False,
                error_text=f"ACP prompt timed out after {self.config.timeout_sec}s",
                timed_out=True,
            )
            raise RuntimeError(
                f"ACP prompt timed out after {self.config.timeout_sec}s"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            self._record_failure_context(
                acpx=acpx,
                session_name=active_session,
                transport="cli",
                prompt_bytes=prompt_bytes,
                prompt_limit=prompt_limit,
                use_file=False,
                error_text=stderr,
                returncode=result.returncode,
            )
            raise RuntimeError(f"ACP prompt failed (exit {result.returncode}): {stderr}")

        response = self._extract_response(result.stdout)
        self._append_debug_event(
            "prompt_success",
            session_name=active_session,
            transport="cli",
            prompt_bytes=prompt_bytes,
            use_file=False,
            response_bytes=len(response.encode("utf-8")),
        )
        return response

    def _send_prompt_via_file(
        self,
        acpx: str,
        prompt: str,
        *,
        session_name: str | None = None,
        prompt_bytes: int,
        prompt_limit: int,
    ) -> str:
        """Write prompt to a temp file, ask the agent to read and respond."""
        fd, prompt_path = tempfile.mkstemp(
            suffix=".md", prefix="rc_prompt_",
        )
        active_session = session_name or self.config.session_name
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(prompt)
            self._append_debug_event(
                "prompt_file_written",
                session_name=active_session,
                transport="file",
                prompt_bytes=prompt_bytes,
                prompt_limit=prompt_limit,
                prompt_path=prompt_path,
            )

            short_prompt = (
                f"Read the file at {prompt_path} in its entirety. "
                f"Follow ALL instructions contained in that file and "
                f"respond exactly as requested. Do NOT summarize, "
                f"just produce the requested output."
            )

            try:
                result = subprocess.run(
                    self._prompt_command(acpx, short_prompt, session_name=active_session),
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=self.config.timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                self._record_failure_context(
                    acpx=acpx,
                    session_name=active_session,
                    transport="file",
                    prompt_bytes=prompt_bytes,
                    prompt_limit=prompt_limit,
                    use_file=True,
                    error_text=f"ACP prompt timed out after {self.config.timeout_sec}s",
                    prompt_path=prompt_path,
                    timed_out=True,
                )
                raise RuntimeError(
                    f"ACP prompt timed out after {self.config.timeout_sec}s"
                ) from exc

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                self._record_failure_context(
                    acpx=acpx,
                    session_name=active_session,
                    transport="file",
                    prompt_bytes=prompt_bytes,
                    prompt_limit=prompt_limit,
                    use_file=True,
                    error_text=stderr,
                    prompt_path=prompt_path,
                    returncode=result.returncode,
                )
                raise RuntimeError(
                    f"ACP prompt failed (exit {result.returncode}): {stderr}"
                )

            response = self._extract_response(result.stdout)
            self._append_debug_event(
                "prompt_success",
                session_name=active_session,
                transport="file",
                prompt_bytes=prompt_bytes,
                use_file=True,
                prompt_path=prompt_path,
                response_bytes=len(response.encode("utf-8")),
            )
            return response
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

    def _prompt_command(
        self,
        acpx: str,
        prompt: str,
        *,
        session_name: str | None = None,
    ) -> list[str]:
        """Build the acpx prompt command for session or stateless mode."""
        cmd = self._acpx_base_command(acpx, approve_all=True)
        cmd.append("prompt")
        active_session = session_name or self.config.session_name
        cmd.extend(["-s", active_session])
        cmd.append(prompt)
        return cmd

    def _create_or_ensure_session(
        self,
        acpx: str,
        session_name: str,
        *,
        ensure: bool,
    ) -> None:
        action = "ensure" if ensure else "new"
        result = subprocess.run(
            [
                *self._acpx_base_command(acpx, approve_all=False),
                "sessions",
                action,
                "--name",
                session_name,
            ],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30,
        )
        if result.returncode == 0:
            self._append_debug_event(
                "session_ready",
                session_name=session_name,
                ensure=ensure,
                stateless=self.config.stateless_prompt,
            )
            return
        if ensure:
            self._create_or_ensure_session(acpx, session_name, ensure=False)
            return
        raise RuntimeError(
            f"Failed to create ACP session: {(result.stderr or '').strip()}"
        )

    def _new_ephemeral_session(self, acpx: str) -> str:
        session_name = f"{self.config.session_name}-{uuid.uuid4().hex[:8]}"
        self._create_or_ensure_session(acpx, session_name, ensure=False)
        logger.info("ACP ephemeral session '%s' ready (%s)", session_name, self.config.agent)
        return session_name

    def _close_named_session(self, acpx: str, session_name: str) -> None:
        try:
            subprocess.run(
                [
                    *self._acpx_base_command(acpx, approve_all=False),
                    "sessions",
                    "close",
                    session_name,
                ],
                capture_output=True, timeout=15,
            )
            self._append_debug_event(
                "session_closed",
                session_name=session_name,
                stateless=self.config.stateless_prompt,
            )
        except Exception:  # noqa: BLE001
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
