"""Renderer Agent — executes plotting scripts and verifies output.

Runs generated Python scripts in a subprocess, captures stdout/stderr,
verifies output files exist with correct format, and returns rendered
image paths.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from researchclaw.agents.base import BaseAgent, AgentStepResult

logger = logging.getLogger(__name__)

# Minimum acceptable file size (bytes) — filters out corrupt/empty PNGs
_MIN_FILE_SIZE = 1024  # 1 KB


class RendererAgent(BaseAgent):
    """Executes plotting scripts and verifies output files."""

    name = "figure_renderer"

    def __init__(
        self,
        llm: Any,
        *,
        timeout_sec: int = 30,
        python_path: str | None = None,
    ) -> None:
        super().__init__(llm)
        self._timeout = timeout_sec
        self._python = python_path or sys.executable

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
            output_dir = Path(context.get("output_dir", "charts"))
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

        # Execute script
        try:
            proc = subprocess.run(
                [self._python, str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                cwd=str(output_dir.parent),
            )
        except subprocess.TimeoutExpired:
            result["error"] = f"Script timed out after {self._timeout}s"
            self.logger.warning("Render timeout for %s", figure_id)
            return result
        except FileNotFoundError:
            result["error"] = f"Python executable not found: {self._python}"
            return result

        if proc.returncode != 0:
            # Truncate stderr to reasonable length
            stderr = proc.stderr[:2000] if proc.stderr else "Unknown error"
            result["error"] = f"Script failed (exit {proc.returncode}): {stderr}"
            self.logger.warning(
                "Render failed for %s: %s", figure_id, stderr[:200]
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
