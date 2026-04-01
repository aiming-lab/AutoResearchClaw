"""Agentic sandbox: launches a coding agent inside a Docker container.

The agent (e.g. Claude Code, Codex) gets full shell access and can run
arbitrary CLI commands, read/write files, and iteratively complete the
experiment.  This replaces the traditional code-generation + sandbox-execution
pipeline with a single agentic session.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from researchclaw.config import AgenticConfig
from researchclaw.experiment.sandbox import SandboxResult, parse_metrics

logger = logging.getLogger(__name__)

_CONTAINER_COUNTER = 0
_counter_lock = threading.Lock()


def _next_container_name() -> str:
    global _CONTAINER_COUNTER  # noqa: PLW0603
    with _counter_lock:
        _CONTAINER_COUNTER += 1
        return f"rc-agentic-{_CONTAINER_COUNTER}-{os.getpid()}"


@dataclass
class AgenticResult:
    """Result of an agentic experiment session."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    output_files: list[str] = field(default_factory=list)
    output_dirs: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    agent_log: str = ""
    steps_completed: int = 0


class AgenticSandbox:
    """Run a coding agent inside a Docker container with full shell access."""

    def __init__(
        self,
        config: AgenticConfig,
        workdir: Path,
        skills_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.skills_dir = skills_dir
        self._container_name: str | None = None

    # -- public API ----------------------------------------------------------

    def run_agent_session(
        self,
        prompt: str,
        workspace: Path,
        *,
        timeout_sec: int | None = None,
    ) -> AgenticResult:
        """Launch the agent inside Docker, send *prompt*, and collect results.

        1. ``docker run -d`` a long-lived container
        2. Install agent CLI (if ``agent_install_cmd`` is set)
        3. ``docker exec`` the agent with *prompt*
        4. Collect output files from ``/workspace``
        5. Stop + remove the container
        """
        timeout = timeout_sec or self.config.timeout_sec
        container = _next_container_name()
        self._container_name = container

        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        try:
            # 1. Start the container
            self._start_container(container, workspace)

            # 2. Install agent CLI
            if self.config.agent_install_cmd:
                self._docker_exec(
                    container,
                    self.config.agent_install_cmd,
                    timeout=min(300, timeout),
                )

            # 3. Run the agent
            agent_cmd = self._build_agent_command(prompt)
            proc = self._docker_exec(
                container,
                agent_cmd,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = proc.returncode

            # 4. Collect results
            output_files, output_dirs = self._collect_outputs(workspace)
            metrics = self._parse_result_metrics(workspace, stdout)
            agent_log = stdout
            steps = self._count_agent_steps(stdout)

            elapsed = time.monotonic() - start
            return AgenticResult(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                elapsed_sec=elapsed,
                output_files=output_files,
                output_dirs=output_dirs,
                metrics=metrics,
                agent_log=agent_log,
                steps_completed=steps,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            logger.warning(
                "Agentic session timed out after %ds (container %s)",
                timeout,
                container,
            )
            # Still try to collect partial results
            output_files, output_dirs = self._collect_outputs(workspace)
            metrics = self._parse_result_metrics(workspace, "")
            return AgenticResult(
                returncode=-1,
                stdout="",
                stderr=f"Agent session timed out after {timeout}s",
                elapsed_sec=elapsed,
                output_files=output_files,
                output_dirs=output_dirs,
                metrics=metrics,
                agent_log="",
                steps_completed=0,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.exception("Agentic session failed: %s", exc)
            return AgenticResult(
                returncode=-1,
                stdout="",
                stderr=str(exc),
                elapsed_sec=elapsed,
            )
        finally:
            self._cleanup_container(container)

    def to_sandbox_result(self, result: AgenticResult) -> SandboxResult:
        """Convert an AgenticResult to a SandboxResult for pipeline compat."""
        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_sec=result.elapsed_sec,
            metrics={k: v for k, v in result.metrics.items()},
            timed_out=(result.returncode == -1 and "timed out" in result.stderr),
        )

    # -- Docker helpers ------------------------------------------------------

    def _start_container(self, container: str, workspace: Path) -> None:
        """Start a long-lived Docker container with workspace mounted."""
        cmd = [
            "docker", "run", "-d",
            "--name", container,
            "-v", f"{workspace}:/workspace",
            "-w", "/workspace",
            f"--memory={self.config.memory_limit_mb}m",
        ]

        # Mount skills directory as read-only reference
        if self.config.mount_skills and self.skills_dir and self.skills_dir.is_dir():
            cmd.extend(["-v", f"{self.skills_dir}:/skills:ro"])

        # Network
        if self.config.network_policy == "none":
            cmd.extend(["--network", "none"])

        # GPU passthrough
        if self.config.gpu_enabled:
            cmd.extend(["--gpus", "all"])

        cmd.extend([self.config.image, "tail", "-f", "/dev/null"])

        logger.info("Starting agentic container: %s", container)
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _docker_exec(
        self,
        container: str,
        command: str,
        *,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the container."""
        cmd = ["docker", "exec", container, "bash", "-c", command]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _build_agent_command(self, prompt: str) -> str:
        """Build the shell command to invoke the agent CLI."""
        cli = self.config.agent_cli
        max_turns = self.config.max_turns

        import shlex as _shlex
        escaped = _shlex.quote(prompt)

        if cli == "claude":
            # Claude Code CLI
            return (
                f"{cli} -p {escaped} "
                f"--output-format json "
                f"--max-turns {max_turns} "
                f"--allowedTools 'Bash(*)' 'Read' 'Write' 'Edit' 'Glob' 'Grep'"
            )
        elif cli == "codex":
            # OpenAI Codex CLI
            return f"{cli} --quiet --approval-mode full-auto {escaped}"
        else:
            # Generic: pass prompt via -p flag
            return f"{cli} -p {escaped}"

    def _cleanup_container(self, container: str) -> None:
        """Stop and remove the container."""
        try:
            subprocess.run(
                ["docker", "stop", "-t", "5", container],
                capture_output=True,
                timeout=30,
                check=False,
            )
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                timeout=30,
                check=False,
            )
            logger.debug("Cleaned up container %s", container)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to cleanup container %s", container)

    # -- Result collection ---------------------------------------------------

    @staticmethod
    def _collect_outputs(workspace: Path) -> tuple[list[str], list[str]]:
        """Walk workspace and return lists of output files and directories."""
        output_files: list[str] = []
        output_dirs: list[str] = []
        if not workspace.exists():
            return output_files, output_dirs
        for item in sorted(workspace.rglob("*")):
            rel = str(item.relative_to(workspace))
            if item.is_dir():
                output_dirs.append(rel)
            elif item.is_file():
                output_files.append(rel)
        return output_files, output_dirs

    @staticmethod
    def _parse_result_metrics(
        workspace: Path, stdout: str
    ) -> dict[str, float]:
        """Parse metrics from results.json (preferred) or stdout."""
        metrics: dict[str, float] = {}

        # Try results.json first
        results_json = workspace / "results.json"
        if results_json.exists():
            try:
                data = json.loads(results_json.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Flatten metrics from various common formats
                    raw = data.get("metrics", data)
                    for k, v in raw.items():
                        try:
                            metrics[k] = float(v)
                        except (TypeError, ValueError):
                            pass
            except (json.JSONDecodeError, OSError):
                pass

        # Fall back to stdout metric parsing
        if not metrics and stdout:
            metrics = parse_metrics(stdout)

        return metrics

    @staticmethod
    def _count_agent_steps(stdout: str) -> int:
        """Estimate the number of agent turns from the output."""
        # For JSON-format Claude output, count tool-use entries
        try:
            data = json.loads(stdout)
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                # Claude Code JSON output has a "messages" or similar key
                messages = data.get("messages", data.get("turns", []))
                if isinstance(messages, list):
                    return len(messages)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: count lines that look like agent actions
        count = 0
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith(("$", ">>>", ">>", "claude>", "Agent:")):
                count += 1
        return count

    # -- Static checks -------------------------------------------------------

    @staticmethod
    def check_docker_available() -> bool:
        """Return True if Docker daemon is reachable."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
