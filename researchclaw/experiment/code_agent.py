"""Pluggable code-generation backends for experiment Stages 10 & 13.

Supports three providers:
  - ``llm``         — existing LLM chat API (backward-compatible default)
  - ``claude_code`` — Claude Code CLI (``claude -p``)
  - ``codex``       — OpenAI Codex CLI (``codex exec``)

Usage::

    from researchclaw.experiment.code_agent import create_code_agent

    agent = create_code_agent(config, llm=llm_client, prompts=pm)
    result = agent.generate(exp_plan=plan, topic=topic, ...)
    if result.ok:
        files = result.files  # dict[str, str]
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from researchclaw.config import RCConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeAgentResult:
    """Output from a code agent invocation."""

    files: dict[str, str]  # filename -> code content
    provider_name: str  # "llm", "claude_code", "codex"
    elapsed_sec: float
    raw_output: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.files)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class CodeAgentProvider(Protocol):
    """Protocol for code generation backends."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        """Generate experiment code from scratch (Stage 10)."""
        ...

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        """Refine existing experiment code based on run results (Stage 13)."""
        ...

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        """Fix validation or runtime issues in code."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _collect_py_files(workdir: Path) -> dict[str, str]:
    """Read all .py files from a directory (flat, no subdirs)."""
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith("_codex_") or pyfile.name.startswith("_agent_"):
            continue
        files[pyfile.name] = pyfile.read_text(encoding="utf-8")
    return files


def _seed_workdir(workdir: Path, files: dict[str, str]) -> None:
    """Pre-populate workdir with files for refinement/repair."""
    workdir.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (workdir / fname).write_text(content, encoding="utf-8")


def format_feedback_for_agent(
    sandbox_result: Any,
    metric_key: str,
    metric_direction: str,
    best_metric: float | None,
) -> str:
    """Format sandbox run results as structured feedback for CLI agents."""
    parts = ["## Previous Run Results"]
    parts.append(f"Return code: {sandbox_result.returncode}")
    parts.append(f"Elapsed: {sandbox_result.elapsed_sec:.1f}s")
    parts.append(f"Timed out: {sandbox_result.timed_out}")
    if sandbox_result.metrics:
        parts.append("Metrics:")
        for k, v in sandbox_result.metrics.items():
            parts.append(f"  {k}: {v}")
    if sandbox_result.stderr:
        parts.append(f"Stderr (last 1000 chars):\n{sandbox_result.stderr[-1000:]}")
    parts.append(f"\nTarget: {metric_direction} '{metric_key}'")
    if best_metric is not None:
        parts.append(f"Best so far: {best_metric}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LlmCodeAgent — wraps existing LLM chat API (backward-compatible)
# ---------------------------------------------------------------------------

class LlmCodeAgent:
    """Code agent backed by the existing OpenAI-compatible LLM chat API.

    This implementation extracts the LLM call + response parsing logic that was
    previously inline in ``_execute_code_generation`` and
    ``_execute_iterative_refine``, preserving exact behavior.
    """

    def __init__(
        self,
        llm: Any,
        prompts: Any,
        config: RCConfig,
    ) -> None:
        self._llm = llm
        self._pm = prompts
        self._config = config

    @property
    def name(self) -> str:
        return "llm"

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_multi_file_blocks,
        )

        start = time.monotonic()
        sp = self._pm.for_stage(
            "code_generation",
            topic=topic,
            metric=metric_key,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
        )
        # Higher max_tokens for reasoning models
        _code_max_tokens = sp.max_tokens or 8192
        if any(
            self._config.llm.primary_model.startswith(p)
            for p in ("gpt-5", "o3", "o4")
        ):
            _code_max_tokens = max(_code_max_tokens, 16384)

        try:
            resp = _chat_with_prompt(
                self._llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=_code_max_tokens,
            )
            files = _extract_multi_file_blocks(resp.content)
            # Retry on empty response with higher token limit
            if not files and not resp.content.strip():
                logger.warning(
                    "LlmCodeAgent: empty response (len=%d, finish=%s). "
                    "Retrying with 32768 tokens.",
                    len(resp.content),
                    resp.finish_reason,
                )
                resp = _chat_with_prompt(
                    self._llm,
                    sp.system,
                    sp.user,
                    json_mode=sp.json_mode,
                    max_tokens=32768,
                )
                files = _extract_multi_file_blocks(resp.content)

            elapsed = time.monotonic() - start
            if not files:
                logger.warning(
                    "LlmCodeAgent: no files extracted (resp len=%d)",
                    len(resp.content),
                )
            return CodeAgentResult(
                files=files,
                provider_name="llm",
                elapsed_sec=elapsed,
                raw_output=resp.content[:2000],
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("LlmCodeAgent.generate failed: %s", exc)
            return CodeAgentResult(
                files={},
                provider_name="llm",
                elapsed_sec=elapsed,
                error=str(exc),
            )

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_code_block,
            _extract_multi_file_blocks,
        )

        start = time.monotonic()

        def _files_to_context(project_files: dict[str, str]) -> str:
            parts = []
            for fname, code in sorted(project_files.items()):
                parts.append(f"```filename:{fname}\n{code}\n```")
            return "\n\n".join(parts)

        try:
            ip = self._pm.sub_prompt(
                "iterative_improve",
                metric_key=metric_key,
                metric_direction=metric_direction,
                files_context=_files_to_context(current_files),
                run_summaries=chr(10).join(run_summaries[:20]),
                condition_coverage_hint="",
                topic=topic,
            )
            user_prompt = ip.user + extra_hints

            response = _chat_with_prompt(
                self._llm,
                ip.system,
                user_prompt,
                max_tokens=ip.max_tokens or 8192,
            )
            extracted_files = _extract_multi_file_blocks(response.content)
            if not extracted_files:
                single_code = _extract_code_block(response.content)
                if single_code.strip():
                    extracted_files = {"main.py": single_code}

            elapsed = time.monotonic() - start
            return CodeAgentResult(
                files=extracted_files,
                provider_name="llm",
                elapsed_sec=elapsed,
                raw_output=response.content[:2000],
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("LlmCodeAgent.refine failed: %s", exc)
            return CodeAgentResult(
                files={},
                provider_name="llm",
                elapsed_sec=elapsed,
                error=str(exc),
            )

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        from researchclaw.pipeline.executor import (
            _chat_with_prompt,
            _extract_code_block,
            _extract_multi_file_blocks,
        )

        start = time.monotonic()
        all_files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        try:
            rp = self._pm.sub_prompt(
                "code_repair",
                fname="main.py",
                issues_text=issues,
                all_files_ctx=all_files_ctx,
            )
            resp = _chat_with_prompt(self._llm, rp.system, rp.user)
            # Try multi-file extraction first, then single-block
            repaired = _extract_multi_file_blocks(resp.content)
            if not repaired:
                code = _extract_code_block(resp.content)
                if code.strip():
                    repaired = {"main.py": code}

            elapsed = time.monotonic() - start
            return CodeAgentResult(
                files=repaired,
                provider_name="llm",
                elapsed_sec=elapsed,
                raw_output=resp.content[:2000],
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return CodeAgentResult(
                files={},
                provider_name="llm",
                elapsed_sec=elapsed,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# CLI agent base — shared subprocess logic for Claude Code / Codex
# ---------------------------------------------------------------------------

class _CliAgentBase:
    """Shared infrastructure for CLI-based coding agents."""

    _provider_name: str = ""

    def __init__(
        self,
        binary_path: str,
        model: str = "",
        max_budget_usd: float = 5.0,
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
    ) -> None:
        self._binary = binary_path
        self._model = model
        self._max_budget_usd = max_budget_usd
        self._default_timeout = timeout_sec
        self._extra_args = extra_args or []

    @property
    def name(self) -> str:
        return self._provider_name

    def _run_subprocess(
        self,
        cmd: list[str],
        workdir: Path,
        timeout_sec: int,
    ) -> tuple[int, str, str, float, bool]:
        """Run command as subprocess with process-group cleanup on timeout.

        Returns (returncode, stdout, stderr, elapsed_sec, timed_out).
        """
        workdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        timed_out = False
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env={**os.environ},
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)

        elapsed = time.monotonic() - start
        return (
            proc.returncode or -1,
            _to_text(stdout_bytes),
            _to_text(stderr_bytes),
            elapsed,
            timed_out,
        )

    def _build_result(
        self,
        workdir: Path,
        returncode: int,
        stdout: str,
        stderr: str,
        elapsed: float,
        timed_out: bool,
    ) -> CodeAgentResult:
        """Collect .py files from workdir and build result."""
        files = _collect_py_files(workdir)
        error = None
        if timed_out:
            error = f"Timed out after {elapsed:.0f}s"
        elif returncode != 0 and not files:
            error = f"Exited {returncode}: {stderr[:500]}"
        return CodeAgentResult(
            files=files,
            provider_name=self._provider_name,
            elapsed_sec=elapsed,
            raw_output=stdout[:3000],
            error=error,
        )

    @staticmethod
    def _generate_prompt(
        topic: str,
        exp_plan: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
    ) -> str:
        return (
            "You are generating experiment code for a research paper.\n\n"
            f"TOPIC: {topic}\n\n"
            f"EXPERIMENT PLAN:\n{exp_plan}\n\n"
            f"PRIMARY METRIC: {metric_key}\n"
            f"{pkg_hint}\n{compute_budget}\n{extra_guidance}\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a multi-file Python project in the current directory.\n"
            "2. The entry point MUST be main.py.\n"
            "3. main.py must print metrics as 'name: value' lines to stdout.\n"
            f"4. Use condition labels: 'condition=<name> {metric_key}: <value>'\n"
            "5. FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket, "
            "network calls, external data files.\n"
            "6. Use deterministic seeds (numpy.random.seed or random.seed).\n"
            "7. Write ALL files to the current working directory.\n"
            "8. Do NOT create subdirectories.\n"
        )

    @staticmethod
    def _refine_prompt(
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
    ) -> str:
        files_listing = "\n".join(
            f"  - {fname} ({len(code)} chars)" for fname, code in current_files.items()
        )
        summaries_text = "\n".join(run_summaries[:10]) if run_summaries else "(no prior runs)"
        return (
            "You are improving experiment code for a research paper.\n\n"
            f"TOPIC: {topic}\n"
            f"TARGET: {metric_direction} '{metric_key}'\n\n"
            f"EXISTING FILES in current directory:\n{files_listing}\n\n"
            "Read the existing files, then improve them based on these run results:\n\n"
            f"PRIOR RUN SUMMARIES:\n{summaries_text}\n\n"
            f"{extra_hints}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read existing code, understand the experiment structure.\n"
            "2. Modify files to improve the metric.\n"
            "3. Keep the entry point as main.py.\n"
            "4. Write modified files to the current directory.\n"
            "5. FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket.\n"
        )

    @staticmethod
    def _repair_prompt(
        files: dict[str, str],
        issues: str,
    ) -> str:
        files_listing = "\n".join(
            f"  - {fname} ({len(code)} chars)" for fname, code in files.items()
        )
        return (
            "The experiment code has validation or runtime issues.\n\n"
            f"ISSUES:\n{issues}\n\n"
            f"FILES in current directory:\n{files_listing}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the existing files in the current directory.\n"
            "2. Fix ALL reported issues.\n"
            "3. Write the corrected files back.\n"
            "4. FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket.\n"
        )


# ---------------------------------------------------------------------------
# ClaudeCodeAgent
# ---------------------------------------------------------------------------

class ClaudeCodeAgent(_CliAgentBase):
    """Code agent backed by Claude Code CLI (``claude -p``)."""

    _provider_name = "claude_code"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--allowed-tools", "Bash Edit Write Read",
            "--add-dir", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._max_budget_usd:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        cmd.extend(self._extra_args)
        return cmd

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        prompt = self._generate_prompt(
            topic, exp_plan, metric_key, pkg_hint, compute_budget, extra_guidance,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._default_timeout,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        _seed_workdir(workdir, current_files)
        prompt = self._refine_prompt(
            current_files, run_summaries, metric_key, metric_direction,
            topic, extra_hints,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._default_timeout,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        _seed_workdir(workdir, files)
        prompt = self._repair_prompt(files, issues)
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._default_timeout,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)


# ---------------------------------------------------------------------------
# CodexAgent
# ---------------------------------------------------------------------------

class CodexAgent(_CliAgentBase):
    """Code agent backed by OpenAI Codex CLI (``codex exec``)."""

    _provider_name = "codex"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "exec", prompt,
            "--sandbox", "workspace-write",
            "--json",
            "-C", str(workdir),
        ]
        if self._model:
            cmd += ["-m", self._model]
        cmd.extend(self._extra_args)
        return cmd

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        prompt = self._generate_prompt(
            topic, exp_plan, metric_key, pkg_hint, compute_budget, extra_guidance,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._default_timeout,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        _seed_workdir(workdir, current_files)
        prompt = self._refine_prompt(
            current_files, run_summaries, metric_key, metric_direction,
            topic, extra_hints,
        )
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._default_timeout,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        _seed_workdir(workdir, files)
        prompt = self._repair_prompt(files, issues)
        cmd = self._build_cmd(prompt, workdir)
        rc, stdout, stderr, elapsed, to = self._run_subprocess(
            cmd, workdir, timeout_sec or self._default_timeout,
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, to)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_code_agent(
    config: RCConfig,
    llm: Any | None = None,
    prompts: Any | None = None,
) -> CodeAgentProvider:
    """Create the appropriate code agent based on config.experiment.cli_agent."""
    agent_cfg = config.experiment.cli_agent
    provider = agent_cfg.provider

    if provider == "llm":
        if llm is None:
            raise RuntimeError("LLM code agent requires an LLM client")
        from researchclaw.prompts import PromptManager

        return LlmCodeAgent(llm, prompts or PromptManager(), config)  # type: ignore[return-value]

    if provider == "claude_code":
        binary = agent_cfg.binary_path or shutil.which("claude")
        if not binary:
            raise RuntimeError(
                "Claude Code binary not found. "
                "Install it or set experiment.code_agent.binary_path."
            )
        return ClaudeCodeAgent(  # type: ignore[return-value]
            binary_path=binary,
            model=agent_cfg.model or "sonnet",
            max_budget_usd=agent_cfg.max_budget_usd,
            timeout_sec=agent_cfg.timeout_sec,
            extra_args=list(agent_cfg.extra_args),
        )

    if provider == "codex":
        binary = agent_cfg.binary_path or shutil.which("codex")
        if not binary:
            raise RuntimeError(
                "Codex binary not found. "
                "Install it or set experiment.code_agent.binary_path."
            )
        return CodexAgent(  # type: ignore[return-value]
            binary_path=binary,
            model=agent_cfg.model or "",
            max_budget_usd=agent_cfg.max_budget_usd,
            timeout_sec=agent_cfg.timeout_sec,
            extra_args=list(agent_cfg.extra_args),
        )

    raise ValueError(f"Unknown code agent provider: {provider}")