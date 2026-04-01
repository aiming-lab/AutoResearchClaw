"""Sandbox environment for safe experiment code execution."""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from researchclaw.config import SandboxConfig
from researchclaw.hardware import is_metric_name

logger = logging.getLogger(__name__)

# Matches both plain "metric: value" and "condition=xxx metric: value" formats
_FLOAT_RE = r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?"
_METRIC_PATTERN = re.compile(
    rf"^(?:\S+=\S+\s+)?(\w[\w.]*)\s*:\s*({_FLOAT_RE})\s*$"
)
# R17: Extract per-condition metrics with optional extra tags:
#   "condition=<name> [regime=<r>] [H=<h>] [seed=<s>] metric: value"
# Captures: (condition_name, extra_tags_string, metric_name, value)
_CONDITION_METRIC_PATTERN = re.compile(
    rf"^condition=(\S+)\s+((?:\S+=\S+\s+)*)(\w[\w.]*)\s*:\s*({_FLOAT_RE})\s*$"
)
# R16-1: Ratio format with optional extra tags
_CONDITION_RATIO_PATTERN = re.compile(
    r"^condition=(\S+)\s+((?:\S+=\S+\s+)*)(\w[\w.]*)\s*:\s*(\d+)/(\d+)\s*$"
)


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        stripped = line.strip()

        # R16-1: Try ratio format first: "condition=X [tags] metric: N/M"
        ratio_match = _CONDITION_RATIO_PATTERN.match(stripped)
        if ratio_match:
            cond_name, extra_tags, name, num, den = ratio_match.groups()
            if is_metric_name(name):
                try:
                    val = float(num) / float(den) if float(den) != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    continue
                # Build composite key from condition + extra tags
                tag_parts = [cond_name]
                for tag in extra_tags.strip().split():
                    if "=" in tag:
                        tag_parts.append(tag.split("=", 1)[1])
                composite_key = "/".join(tag_parts)
                metrics[f"{composite_key}/{name}"] = val
                metrics[f"{cond_name}/{name}"] = val
                metrics[name] = val
            continue

        # Try condition-prefixed format: "condition=X [tags] metric: value"
        cond_match = _CONDITION_METRIC_PATTERN.match(stripped)
        if cond_match:
            cond_name, extra_tags, name, value = cond_match.groups()
            if is_metric_name(name):
                try:
                    val = float(value)
                except ValueError:
                    continue
                if math.isnan(val) or math.isinf(val):
                    logger.warning("Skipping non-finite metric %s=%s", name, value)
                    continue
                # Build composite key from condition + extra tags
                tag_parts = [cond_name]
                for tag in extra_tags.strip().split():
                    if "=" in tag:
                        tag_parts.append(tag.split("=", 1)[1])
                composite_key = "/".join(tag_parts)
                metrics[f"{composite_key}/{name}"] = val
                metrics[f"{cond_name}/{name}"] = val
                metrics[name] = val
            continue

        # Plain format: "metric: value"
        match = _METRIC_PATTERN.match(stripped)
        if match is None:
            continue
        name, value = match.groups()
        if not is_metric_name(name):
            continue
        try:
            val = float(value)
        except ValueError:
            continue
        # R5-3: Skip NaN/Inf values — they indicate divergence
        if math.isnan(val) or math.isinf(val):
            logger.warning("Skipping non-finite metric %s=%s", name, value)
            continue
        metrics[name] = val
    return metrics


def extract_paired_comparisons(stdout: str) -> list[dict[str, object]]:
    """R18-1: Extract PAIRED statistical comparison lines from stdout.

    Matches: PAIRED: <method> vs <baseline> [regime=<r>] mean_diff=<v> ...
    Returns a list of dicts with method, baseline, regime, and stats.
    """
    results: list[dict[str, object]] = []
    pattern = re.compile(
        r"^PAIRED:\s+(\S+)\s+vs\s+(\S+)\s*(.*?)mean_diff=([+-]?\d+\.?\d*)"
        r".*?std_diff=([+-]?\d+\.?\d*)"
        r".*?t_stat=([+-]?\d+\.?\d*)"
        r".*?p_value=([+-]?\d+\.?\d*)"
    )
    for line in stdout.splitlines():
        m = pattern.match(line.strip())
        if m:
            method, baseline, tags, mean_diff, std_diff, t_stat, p_value = m.groups()
            entry: dict[str, object] = {
                "method": method,
                "baseline": baseline,
                "mean_diff": float(mean_diff),
                "std_diff": float(std_diff),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
            # Extract regime if present
            regime_m = re.search(r"regime=(\S+)", tags)
            if regime_m:
                entry["regime"] = regime_m.group(1)
            # Extract CI if present
            ci_m = re.search(r"ci95=\(([^,]+),([^)]+)\)", line)
            if ci_m:
                entry["ci95_low"] = float(ci_m.group(1))
                entry["ci95_high"] = float(ci_m.group(2))
            results.append(entry)
    return results


def detect_nan_divergence(stdout: str, stderr: str) -> str | None:
    """Check stdout/stderr for NaN/Inf/divergence indicators.

    Returns a description of the issue if detected, None otherwise.
    """
    issues: list[str] = []
    combined = (stdout or "") + "\n" + (stderr or "")
    lower = combined.lower()

    # Check for NaN indicators
    if "nan" in lower:
        for pattern in ("loss: nan", "nan loss", "math domain error", "loss is nan"):
            if pattern in lower:
                issues.append(f"NaN detected: '{pattern}' found in output")
                break
        else:
            # Generic NaN mention — could be a false positive but worth flagging
            if re.search(r"\bnan\b", lower):
                issues.append("Possible NaN detected in output")

    # Check for Inf indicators
    if "inf" in lower:
        if re.search(r"\binf\b", lower) and "info" not in lower.split("inf")[0][-4:]:
            issues.append("Possible Inf value detected in output")

    # Check for divergence (loss > 100 is a common fast-fail threshold)
    for line in stdout.splitlines():
        match = _METRIC_PATTERN.match(line.strip())
        if match:
            name, value = match.groups()
            try:
                val = float(value)
                if math.isnan(val) or math.isinf(val):
                    issues.append(f"Non-finite metric: {name}={value}")
                elif "loss" in name.lower() and val > 100:
                    issues.append(f"Diverging loss: {name}={val} (>100)")
            except ValueError:
                pass

    return "; ".join(issues) if issues else None


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    metrics: dict[str, object]
    timed_out: bool = False


class SandboxProtocol(Protocol):
    """Structural type for sandbox backends (ExperimentSandbox, DockerSandbox)."""

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult: ...

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> SandboxResult: ...


class ExperimentSandbox:
    def __init__(self, config: SandboxConfig, workdir: Path) -> None:
        self.config: SandboxConfig = config
        self.workdir: Path = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter: int = 0

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        script_path = self._next_script_path()
        self._write_script(script_path, code)

        start = time.monotonic()
        command = self._build_command(script_path)
        logger.debug("Running sandbox command: %s", command)

        result: SandboxResult
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                cwd=self.workdir,
                env=env,
                check=False,
            )
            result = self._result_from_completed(
                completed, elapsed_sec=time.monotonic() - start
            )
        except subprocess.TimeoutExpired as exc:
            result = self._result_from_timeout(
                exc, timeout_sec=timeout_sec, elapsed_sec=time.monotonic() - start
            )
        except Exception as exc:  # noqa: BLE001
            result = self._result_from_exception(
                exc, elapsed_sec=time.monotonic() - start
            )

        if self._should_cleanup(result):
            self._cleanup_script(script_path)

        return result

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> SandboxResult:
        """Run a multi-file experiment project in the sandbox.

        Copies all ``.py`` files from *project_dir* into the sandbox work
        directory and executes *entry_point*.
        """
        import shutil

        sandbox_project = self.workdir / "_project"
        if sandbox_project.exists():
            shutil.rmtree(sandbox_project)
        sandbox_project.mkdir(parents=True, exist_ok=True)

        # R5-4: Inject immutable experiment harness before copying project files
        self._inject_harness(sandbox_project)

        # Copy all project files (will NOT overwrite harness — harness name is unique)
        for src_file in project_dir.iterdir():
            if src_file.is_file():
                dest = sandbox_project / src_file.name
                # Do not allow project to overwrite the harness
                if dest.name == "experiment_harness.py":
                    logger.warning("Project contains experiment_harness.py — skipping (immutable)")
                    continue
                dest.write_bytes(src_file.read_bytes())

        entry = sandbox_project / entry_point
        if not entry.exists():
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"Entry point {entry_point} not found in project",
                elapsed_sec=0.0,
                metrics={},
            )

        start = time.monotonic()
        command = self._build_command(entry)
        logger.debug("Running project sandbox command: %s (cwd=%s)", command, sandbox_project)

        result: SandboxResult
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                cwd=sandbox_project,
                env=env,
                check=False,
            )
            result = self._result_from_completed(
                completed, elapsed_sec=time.monotonic() - start
            )
        except subprocess.TimeoutExpired as exc:
            result = self._result_from_timeout(
                exc, timeout_sec=timeout_sec, elapsed_sec=time.monotonic() - start
            )
        except Exception as exc:  # noqa: BLE001
            result = self._result_from_exception(
                exc, elapsed_sec=time.monotonic() - start
            )

        return result

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        """Copy the immutable experiment harness into the target directory."""
        harness_src = Path(__file__).parent / "harness_template.py"
        if harness_src.exists():
            dest = target_dir / "experiment_harness.py"
            dest.write_text(harness_src.read_text(encoding="utf-8"), encoding="utf-8")
            logger.debug("Injected experiment harness into %s", target_dir)
        else:
            logger.warning("Harness template not found at %s", harness_src)

    def _next_script_path(self) -> Path:
        self._run_counter += 1
        return self.workdir / f"_experiment_{self._run_counter}.py"

    @staticmethod
    def _write_script(script_path: Path, code: str) -> None:
        _ = script_path.write_text(code, encoding="utf-8")

    def _build_command(self, script_path: Path) -> list[str]:
        # Convert relative python_path to absolute WITHOUT resolving symlinks.
        # Using .resolve() would follow venv symlinks to the system Python binary,
        # which loses the venv context (site-packages like numpy become unavailable).
        python = self.config.python_path
        python_path = Path(python)
        if not python_path.is_absolute():
            python_path = Path.cwd() / python_path
        # -u: unbuffered stdout/stderr so subprocess.run captures all output
        return [str(python_path), "-u", str(script_path)]

    @staticmethod
    def _result_from_completed(
        completed: subprocess.CompletedProcess[str], *, elapsed_sec: float
    ) -> SandboxResult:
        metrics = parse_metrics(completed.stdout)
        return SandboxResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_sec=elapsed_sec,
            metrics={key: value for key, value in metrics.items()},
        )

    @staticmethod
    def _result_from_timeout(
        exc: subprocess.TimeoutExpired,
        *,
        timeout_sec: int,
        elapsed_sec: float,
    ) -> SandboxResult:
        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        metrics = parse_metrics(stdout)
        logger.warning("Sandbox execution timed out after %ss", timeout_sec)
        return SandboxResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed_sec,
            metrics={key: value for key, value in metrics.items()},
            timed_out=True,
        )

    @staticmethod
    def _result_from_exception(exc: Exception, *, elapsed_sec: float) -> SandboxResult:
        logger.exception("Sandbox execution failed: %s", exc)
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=str(exc),
            elapsed_sec=elapsed_sec,
            metrics={},
        )

    @staticmethod
    def _should_cleanup(result: SandboxResult) -> bool:
        return result.returncode == 0 and not result.timed_out

    @staticmethod
    def _cleanup_script(script_path: Path) -> None:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete temporary file: %s", script_path)
