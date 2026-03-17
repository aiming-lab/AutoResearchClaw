"""Renderer Agent — executes plotting scripts and verifies output.

Runs generated Python scripts in a subprocess (or Docker sandbox when
available), captures stdout/stderr, verifies output files exist with
correct format, and returns rendered image paths.

Security: When Docker is available, visualization code is executed inside
an isolated container (``--network none``) to prevent RCE from LLM-generated
code.  Falls back to a local subprocess when Docker is not available.

Architecture ref: Visual ChatGPT (Wu et al., 2023) — LLMs as controllers
calling deterministic render tools instead of generating pixels directly.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult

logger = logging.getLogger(__name__)

# Minimum acceptable file size (bytes) — filters out corrupt/empty PNGs
_MIN_FILE_SIZE = 1024  # 1 KB

# Docker image for sandboxed visualization rendering.
# The experiment image already has matplotlib, numpy, seaborn pre-installed.
_VIZ_DOCKER_IMAGE = "researchclaw/experiment:latest"


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        cp = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return cp.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class RendererAgent(BaseAgent):
    """Executes plotting scripts and verifies output files.

    Supports two execution modes:
      1. **Docker sandbox** (preferred): Runs scripts inside an isolated
         container with ``--network none`` to prevent RCE.
      2. **Local subprocess** (fallback): Direct execution when Docker
         is unavailable.

    The mode is auto-detected at instantiation time but can be forced via
    the ``use_docker`` parameter.
    """

    name = "figure_renderer"

    def __init__(
        self,
        llm: Any,
        *,
        timeout_sec: int = 30,
        python_path: str | None = None,
        use_docker: bool | None = None,
        docker_image: str | None = None,
    ) -> None:
        super().__init__(llm)
        self._timeout = timeout_sec
        self._python = python_path or sys.executable
        self._docker_image = docker_image or _VIZ_DOCKER_IMAGE

        # Auto-detect Docker availability if not explicitly set
        if use_docker is None:
            self._use_docker = _docker_available()
        else:
            self._use_docker = use_docker

        if self._use_docker:
            self.logger.info(
                "RendererAgent: Docker sandbox ENABLED (image=%s)",
                self._docker_image,
            )
        else:
            self.logger.info("RendererAgent: Docker sandbox disabled, using local subprocess")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Execute plotting scripts and verify outputs.

        Context keys:
            scripts (list[dict]): From CodeGen — each has 'figure_id',
                'script', 'output_filename'
            output_dir (str|Path): Directory for output charts and scripts
        """
        try:
            scripts = context.get("scripts", [])
            output_dir = Path(context.get("output_dir", "charts")).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            scripts_dir = output_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            results: list[dict[str, Any]] = []

            for script_info in scripts:
                figure_id = script_info.get("figure_id", "unknown")
                script_code = script_info.get("script", "")
                output_filename = script_info.get("output_filename", f"{figure_id}.png")

                result = self._render_one(
                    figure_id=figure_id,
                    script_code=script_code,
                    output_filename=output_filename,
                    output_dir=output_dir,
                    scripts_dir=scripts_dir,
                )
                result["title"] = script_info.get("title", "")
                result["caption"] = script_info.get("caption", "")
                result["section"] = script_info.get("section", "results")
                result["width"] = script_info.get("width", "single_column")
                results.append(result)

            success_count = sum(1 for r in results if r["success"])
            self.logger.info(
                "Rendered %d/%d figures successfully",
                success_count, len(scripts),
            )

            return self._make_result(
                success=success_count > 0,
                data={"rendered": results, "output_dir": str(output_dir)},
                error="" if success_count > 0 else "All renders failed",
            )
        except Exception as exc:
            self.logger.error("Renderer failed: %s", exc)
            return self._make_result(False, error=str(exc))

    # ------------------------------------------------------------------
    # Per-figure rendering
    # ------------------------------------------------------------------

    def _render_one(
        self,
        *,
        figure_id: str,
        script_code: str,
        output_filename: str,
        output_dir: Path,
        scripts_dir: Path,
    ) -> dict[str, Any]:
        """Render a single figure script."""
        result: dict[str, Any] = {
            "figure_id": figure_id,
            "success": False,
            "output_path": "",
            "script_path": "",
            "error": "",
        }

        if not script_code.strip():
            result["error"] = "Empty script"
            return result

        # Save script for reproducibility
        script_path = scripts_dir / f"{figure_id}.py"
        script_path.write_text(script_code, encoding="utf-8")
        result["script_path"] = str(script_path)

        # Choose execution backend
        if self._use_docker:
            proc_result = self._execute_in_docker(
                script_path=script_path,
                output_dir=output_dir,
                figure_id=figure_id,
            )
        else:
            proc_result = self._execute_local(
                script_path=script_path,
                output_dir=output_dir,
            )

        if proc_result["error"]:
            result["error"] = proc_result["error"]
            self.logger.warning(
                "Render failed for %s: %s", figure_id, result["error"][:200]
            )
            return result

        # Verify output file exists
        output_path = output_dir / output_filename
        if not output_path.exists():
            # Check if it was saved relative to script CWD
            alt_path = output_dir.parent / output_dir.name / output_filename
            if alt_path.exists():
                output_path = alt_path
            else:
                result["error"] = f"Output file not found: {output_path}"
                self.logger.warning("Output missing for %s", figure_id)
                return result

        # Verify file size
        file_size = output_path.stat().st_size
        if file_size < _MIN_FILE_SIZE:
            result["error"] = f"Output file too small ({file_size} bytes)"
            self.logger.warning(
                "Output too small for %s: %d bytes", figure_id, file_size
            )
            return result

        result["success"] = True
        result["output_path"] = str(output_path)
        result["file_size"] = file_size
        self.logger.info("Rendered %s: %s (%d bytes)", figure_id, output_path, file_size)
        return result

    # ------------------------------------------------------------------
    # Execution backends
    # ------------------------------------------------------------------

    def _execute_local(
        self,
        *,
        script_path: Path,
        output_dir: Path,
    ) -> dict[str, str]:
        """Execute script in a local subprocess (no sandbox)."""
        try:
            proc = subprocess.run(
                [self._python, str(script_path.resolve())],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(output_dir.resolve().parent),
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Script timed out after {self._timeout}s"}
        except FileNotFoundError:
            return {"error": f"Python executable not found: {self._python}"}

        if proc.returncode != 0:
            stderr = proc.stderr[:2000] if proc.stderr else "Unknown error"
            return {"error": f"Script failed (exit {proc.returncode}): {stderr}"}

        return {"error": ""}

    def _execute_in_docker(
        self,
        *,
        script_path: Path,
        output_dir: Path,
        figure_id: str,
    ) -> dict[str, str]:
        """Execute script inside an isolated Docker container.

        Security measures:
        - ``--network none``: No network access (prevents data exfiltration)
        - ``--read-only``: Root filesystem is read-only
        - ``--tmpfs /tmp``: Writable /tmp only in-memory
        - ``--memory 512m``: Hard memory limit
        - Volume mounts are restricted to the output directory
        - Script is bind-mounted read-only
        - Container is auto-removed after execution

        This prevents RCE from LLM-generated visualization code.
        """
        container_name = f"rc-viz-{figure_id}-{os.getpid()}"

        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            f"--memory=512m",
            "-v", f"{script_path.resolve()}:/workspace/script.py:ro",
            "-v", f"{output_dir.resolve()}:/workspace/output:rw",
            "-w", "/workspace",
            "--user", f"{os.getuid()}:{os.getgid()}",
            self._docker_image,
            "python3", "/workspace/script.py",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Kill the container on timeout
            try:
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            return {"error": f"Docker script timed out after {self._timeout}s"}
        except FileNotFoundError:
            return {"error": "Docker executable not found"}
        except Exception as exc:
            return {"error": f"Docker execution error: {exc}"}

        if proc.returncode != 0:
            stderr = proc.stderr[:2000] if proc.stderr else "Unknown error"
            return {"error": f"Docker script failed (exit {proc.returncode}): {stderr}"}

        return {"error": ""}
