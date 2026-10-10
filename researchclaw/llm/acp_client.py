"""ACP (Agent Client Protocol) LLM client via acpx.

Uses acpx as the ACP bridge to communicate with any ACP-compatible agent
(Claude Code, Codex, Gemini CLI, etc.) via persistent named sessions.

Persistent named sessions retain context between calls. Opt-in model routing
uses a separate fixed-model client and session for each configured role.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import sys
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
    max_turns: int = 1  # turns allowed per prompt before acpx aborts the call
    primary_model: str = ""  # non-empty enables strict, fixed-model sessions
    reasoning_effort: str = ""


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
        if bool(acp_config.primary_model) != bool(acp_config.reasoning_effort):
            raise ValueError("ACP model routing requires both primary_model and reasoning_effort")
        if acp_config.primary_model and any(
            not isinstance(value, str) or not value.strip()
            for value in (acp_config.primary_model, acp_config.reasoning_effort)
        ):
            raise ValueError("ACP model and reasoning effort must be non-empty strings")
        self.config = acp_config
        self._acpx: str | None = acp_config.acpx_command or None
        self._session_ready = False
        self._session_lock = threading.RLock()
        self.selection_error: str | None = None
        self.session_selection: dict[str, Any] = {
            "requested_model": acp_config.primary_model or None,
            "requested_reasoning_effort": acp_config.reasoning_effort or None,
            "selected_model": None,
            "selected_reasoning_effort": None,
            "model_advertised": False,
            "effort_advertised": False,
            "session_name": acp_config.session_name,
            "verified": False,
        }
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
        # Routing is opt-in: older agents continue to own their model choices.
        model_routing = getattr(acp, "model_routing", False) is True
        primary_model = getattr(rc_config.llm, "primary_model", "") if model_routing else ""
        reasoning_effort = getattr(acp, "reasoning_effort", "") if model_routing else ""
        if model_routing and (
            not isinstance(primary_model, str) or not primary_model.strip()
            or not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
        ):
            raise ValueError("ACP model routing requires a primary model and reasoning effort")
        return cls(ACPConfig(
            agent=acp.agent,
            cwd=acp.cwd,
            acpx_command=getattr(acp, "acpx_command", ""),
            session_name=getattr(acp, "session_name", "researchclaw"),
            timeout_sec=getattr(acp, "timeout_sec", 1800),
            max_turns=getattr(acp, "max_turns", 1),
            primary_model=primary_model,
            reasoning_effort=reasoning_effort,
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
        strip_thinking: bool = True,
    ) -> LLMResponse:
        """Send a prompt and return the agent's response.

        Parameters mirror ``LLMClient.chat()`` for drop-in compatibility.
        With model routing enabled, this instance is fixed to its configured
        model and reasoning effort. A different ``model`` is rejected; callers
        use a separate named-session client for another model. Otherwise the
        agent manages its model. ``max_tokens``, ``temperature``, and
        ``json_mode`` remain agent-managed.

        ``strip_thinking`` defaults to True: ACP agents (opencode, Claude
        Code) interleave ``[thinking]`` blocks and acpx metadata with the
        answer, and callers that use the response as paper text or code must
        not have to remember to ask for that to be removed. Pass False only
        when the reasoning trace itself is what you want.
        """
        with self._session_lock:
            if self.config.primary_model and model not in (None, self.config.primary_model):
                self.selection_error = (
                    f"ACP session {self.config.session_name!r} is fixed to "
                    f"{self.config.primary_model!r}; requested {model!r}"
                )
                self.session_selection["verified"] = False
                raise RuntimeError(self.selection_error)
            if self.config.primary_model:
                backend_instruction = (
                    "You are a text-generation backend for a research pipeline. "
                    "Respond with plain text only. Do not use tools, read or write "
                    "files, search, or run terminal commands."
                )
                system = f"{backend_instruction}\n\n{system}" if system else backend_instruction
            prompt_text = self._messages_to_prompt(messages, system=system)
            content = self._send_prompt(prompt_text)
            if strip_thinking:
                from researchclaw.utils.thinking_tags import strip_thinking_tags
                content = strip_thinking_tags(content)
            return LLMResponse(
                content=content,
                model=self.config.primary_model or f"acp:{self.config.agent}",
                finish_reason="stop",
                raw={"acp_selection": dict(self.session_selection)} if self.config.primary_model else {},
            )

    def prepare(self) -> dict[str, Any]:
        """Prepare the session and verify routing without a routed model prompt.

        Each call reapplies the fixed choices and reads the session metadata,
        including after reconnect. Selection failures are latched so pipeline
        callers can detect errors swallowed by a stage's exception handler.
        """
        with self._session_lock:
            try:
                self._ensure_session()
                if self.config.primary_model:
                    self._configure_and_verify_selection()
            except Exception as exc:
                if self.config.primary_model:
                    self.selection_error = str(exc)
                    self.session_selection.update(
                        verified=False, selected_model=None, selected_reasoning_effort=None,
                        model_advertised=False, effort_advertised=False,
                    )
                raise
            self.selection_error = None
            return dict(self.session_selection)

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
            self.prepare()
            return True, f"OK - ACP session ready ({agent} via acpx)"
        except Exception as exc:  # noqa: BLE001
            return False, f"ACP session init failed: {exc}"

    def close(self) -> None:
        """Close the acpx session."""
        with self._session_lock:
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
            self.session_selection["verified"] = False

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
        """Find or create the named acpx session.

        Legacy clients send a disposable warm-up to consume adapter greetings.
        Routed sessions perform configuration checks in ``prepare()`` and
        receive the text-backend instruction with each actual prompt instead.
        """
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

        # Routed Codex sessions need no greeting-consuming model request.
        # Mark the created session ready before verification, so a failed
        # selection can still be closed by normal cleanup.
        if self.config.primary_model:
            self._session_ready = True
            return

        # Warm-up: consume the agent's cold-start greeting and set
        # text-only mode so it does not use tools or pollute responses.
        _warmup = (
            "You are being used as a text-generation backend for a "
            "research pipeline. For ALL subsequent prompts in this "
            "session, you MUST respond with text output ONLY. "
            "Do NOT use any tools — no file reads, no file writes, "
            "no searches, no terminal commands. Generate your "
            "complete response as plain text. Confirm with: OK"
        )
        try:
            subprocess.run(
                [acpx, "--approve-all", "--max-turns", str(self.config.max_turns),
                 "--ttl", "0", "--cwd", self._abs_cwd(),
                 self.config.agent, "-s", self.config.session_name,
                 _warmup],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60,
            )
        except Exception:  # noqa: BLE001
            logger.debug("ACP warm-up prompt failed (non-fatal)")

        self._session_ready = True
        logger.info("ACP session '%s' ready (%s)", self.config.session_name, self.config.agent)

    def _selection_command(self, arguments: list[str], *, label: str) -> dict[str, Any]:
        acpx = self._resolve_acpx()
        if not acpx:
            raise RuntimeError("acpx not found")
        cmd = [acpx, "--format", "json", "--ttl", "0", "--cwd", self._abs_cwd(),
               self.config.agent, *arguments]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ACP {label} timed out") from exc
        if result.returncode != 0:
            raise RuntimeError(f"ACP {label} failed (exit {result.returncode}): {(result.stderr or '').strip()}")
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"ACP {label} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"ACP {label} returned invalid metadata")
        return payload

    @staticmethod
    def _advertised_values(options: Any) -> set[str]:
        """Read both flat and grouped ACP select-option values."""
        values: set[str] = set()
        if not isinstance(options, list):
            return values
        for option in options:
            if not isinstance(option, dict):
                continue
            value = option.get("value")
            if isinstance(value, str):
                values.add(value)
            values.update(ACPClient._advertised_values(option.get("options")))
        return values

    def _verify_config_options(self, options: Any, *, source: str) -> None:
        if not isinstance(options, list):
            raise RuntimeError(f"ACP {source} omitted config options")
        for config_id, requested in (
            ("model", self.config.primary_model),
            ("reasoning_effort", self.config.reasoning_effort),
        ):
            matches = [item for item in options if isinstance(item, dict) and item.get("id") == config_id]
            if len(matches) != 1:
                raise RuntimeError(f"ACP {source} did not advertise exactly one {config_id} option")
            option = matches[0]
            if requested not in self._advertised_values(option.get("options")):
                raise RuntimeError(f"ACP {source} does not advertise {config_id}={requested!r}")
            if option.get("currentValue") != requested:
                raise RuntimeError(
                    f"ACP {source} selected {config_id}={option.get('currentValue')!r}, "
                    f"requested {requested!r}"
                )

    def _configure_and_verify_selection(self) -> None:
        name = self.config.session_name
        # Model changes can alter the reasoning catalog, so effort follows model.
        self._selection_command(
            ["-s", name, "set", "model", self.config.primary_model], label="model selection",
        )
        accepted = self._selection_command(
            ["-s", name, "set", "reasoning_effort", self.config.reasoning_effort],
            label="reasoning effort selection",
        )
        # The CLI's echoed request is insufficient: check the adapter's complete
        # acknowledgement and then independently read the persisted session.
        self._verify_config_options(accepted.get("configOptions"), source="selection response")
        record = self._selection_command(["sessions", "show", name], label="selection readback")
        if record.get("name") != name or record.get("closed") is True:
            raise RuntimeError("ACP selection readback does not identify the active named session")
        for key in ("acpxRecordId", "acpxSessionId"):
            actual = record.get("acpSessionId" if key == "acpxSessionId" else key)
            if not isinstance(accepted.get(key), str) or not accepted[key] or actual != accepted[key]:
                raise RuntimeError(f"ACP selection readback has inconsistent {key}")
        state = record.get("acpx")
        self._verify_config_options(
            state.get("config_options") if isinstance(state, dict) else None,
            source="session readback",
        )
        self.session_selection.update(
            selected_model=self.config.primary_model,
            selected_reasoning_effort=self.config.reasoning_effort,
            model_advertised=True,
            effort_advertised=True,
            verified=True,
            acpx_record_id=record["acpxRecordId"],
            acp_session_id=record["acpSessionId"],
            agent_session_id=record.get("agentSessionId"),
        )

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

        # On Windows, .cmd/.bat wrappers route through cmd.exe which
        # silently truncates multi-line CLI arguments.  Always use stdin
        # pipe transport to avoid mangled prompts.
        prompt_bytes = len(prompt.encode("utf-8"))
        prompt_limit = self._cli_prompt_limit(acpx)
        use_file = prompt_bytes > prompt_limit or (
            sys.platform == "win32" and "\n" in prompt
        )
        if use_file:
            logger.info(
                "Using stdin-pipe prompt transport (%d bytes).",
                prompt_bytes,
            )

        last_exc: RuntimeError | None = None
        for attempt in range(1 + self._MAX_RECONNECT_ATTEMPTS):
            self.prepare()
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
        input_data: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run an ACP subprocess with periodic heartbeat logging.

        Instead of a silent blocking ``subprocess.run``, this uses ``Popen``
        with a background reader thread and logs a progress heartbeat every
        30 seconds so the user knows the agent is still working.

        When *input_data* is provided, it is written to the process's stdin
        (used for ``-f -`` stdin-pipe transport).
        """
        timeout = self.config.timeout_sec
        heartbeat_interval = 30  # seconds

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )

        # Write stdin data and close immediately so the process can read it.
        if input_data and proc.stdin:
            try:
                proc.stdin.write(input_data)
                proc.stdin.close()
            except OSError:
                pass

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
            acpx, "--approve-all", "--max-turns", str(self.config.max_turns),
            "--ttl", "0", "--cwd", self._abs_cwd(),
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
        """Send prompt via stdin pipe (``-f -``) to avoid CLI arg limits."""
        cmd = [
            acpx, "--approve-all", "--max-turns", str(self.config.max_turns),
            "--ttl", "0", "--cwd", self._abs_cwd(),
            self.config.agent, "-s", self.config.session_name,
            "-f", "-",
        ]
        try:
            result = self._run_acp_with_heartbeat(
                cmd, label="ACP prompt (stdin)", input_data=prompt,
            )
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
