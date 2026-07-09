"""OpenCode 'Beast Mode' bridge — routes complex code generation to OpenCode CLI.

OpenCode (https://github.com/anomalyco/opencode) is an external AI coding agent
invoked via ``opencode run --format json "prompt"``.  This module provides:

1. **ComplexityScore / score_complexity()** — analyses an experiment plan to
   decide whether beast mode is warranted.
2. **OpenCodeBridge** — manages workspace creation, OpenCode invocation, file
   collection, and cleanup.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

# Keywords that indicate multi-component architectures
_COMPONENT_KEYWORDS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "discriminator",
    "generator",
    "critic",
    "actor",
    "teacher",
    "student",
    "backbone",
    "head",
    "neck",
    "classifier",
    "embedder",
    "attention",
    "transformer",
    "tokenizer",
    "vae",
    "autoencoder",
)

# Indicators that multi-file generation is needed
_FILE_HINT_KEYWORDS: tuple[str, ...] = (
    "model.py",
    "trainer.py",
    "dataset.py",
    "utils.py",
    "config.py",
    "multiple files",
    "modular",
    "separate module",
    "multi-file",
)

# Domain-complexity keywords
_DOMAIN_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "multi-modal",
    "multimodal",
    "distributed",
    "gan",
    "diffusion",
    "nerf",
    "mixture of experts",
    "moe",
    "meta-learning",
    "meta learning",
    "maml",
    "neural ode",
    "neural sde",
    "physics-informed",
    "pinn",
    "graph neural",
    "gnn",
    "reinforcement learning",
    "multi-agent",
    "world model",
    "vision-language",
    "text-to-image",
    "image-to-text",
)

# Patterns suggesting deep dependency chains
_DEPENDENCY_KEYWORDS: tuple[str, ...] = (
    "custom layer",
    "custom loss",
    "wrapper",
    "registry",
    "hook",
    "callback",
    "scheduler",
    "custom optimizer",
    "custom dataset",
    "custom sampler",
    "custom transform",
)


@dataclass
class ComplexityScore:
    """Result of complexity analysis on an experiment plan."""

    score: float  # 0.0-1.0
    signals: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""  # "beast_mode" | "code_agent" | "legacy"
    reason: str = ""


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_complexity(
    exp_plan: str,
    topic: str = "",
    *,
    historical_failures: int = 0,
    threshold: float = 0.6,
) -> ComplexityScore:
    """Score the complexity of an experiment to determine if beast mode is warranted.

    Returns a ComplexityScore with score in [0.0, 1.0].
    """
    if not exp_plan and not topic:
        return ComplexityScore(
            score=0.0,
            signals={},
            recommendation="legacy",
            reason="Empty plan",
        )

    combined = f"{topic}\n{exp_plan}"

    # Signal 1: Component count (weight 0.25)
    comp_hits = _count_keyword_hits(combined, _COMPONENT_KEYWORDS)
    component_score = min(comp_hits / 5.0, 1.0)

    # Signal 2: File count hint (weight 0.20)
    file_hits = _count_keyword_hits(combined, _FILE_HINT_KEYWORDS)
    file_score = min(file_hits / 3.0, 1.0)

    # Signal 3: Domain complexity (weight 0.20)
    domain_hits = _count_keyword_hits(combined, _DOMAIN_COMPLEX_KEYWORDS)
    domain_score = min(domain_hits / 3.0, 1.0)

    # Signal 4: Condition count (weight 0.15)
    # Look for numbered conditions, ablation mentions, variant mentions
    condition_pattern = re.compile(
        r"(?:condition|ablation|variant|experiment)\s*[\-_:]?\s*\d+",
        re.IGNORECASE,
    )
    condition_matches = len(condition_pattern.findall(combined))
    # Also count bullet points in conditions/ablations sections
    condition_matches += combined.lower().count("baseline")
    condition_score = min(condition_matches / 8.0, 1.0)

    # Signal 5: Historical failures (weight 0.10)
    failure_score = min(historical_failures / 3.0, 1.0)

    # Signal 6: Dependency depth (weight 0.10)
    dep_hits = _count_keyword_hits(combined, _DEPENDENCY_KEYWORDS)
    dep_score = min(dep_hits / 3.0, 1.0)

    # Weighted sum
    weighted = (
        0.25 * component_score
        + 0.20 * file_score
        + 0.20 * domain_score
        + 0.15 * condition_score
        + 0.10 * failure_score
        + 0.10 * dep_score
    )
    final_score = min(max(weighted, 0.0), 1.0)

    signals = {
        "component_count": round(component_score, 3),
        "file_count_hint": round(file_score, 3),
        "domain_complexity": round(domain_score, 3),
        "condition_count": round(condition_score, 3),
        "historical_failure": round(failure_score, 3),
        "dependency_depth": round(dep_score, 3),
    }

    if final_score >= threshold:
        recommendation = "beast_mode"
        reason = (
            f"Complexity {final_score:.2f} >= threshold {threshold:.2f}: "
            f"top signals: "
            + ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(signals.items(), key=lambda x: -x[1])[:3]
            )
        )
    else:
        recommendation = "code_agent"
        reason = f"Complexity {final_score:.2f} < threshold {threshold:.2f}"

    return ComplexityScore(
        score=round(final_score, 4),
        signals=signals,
        recommendation=recommendation,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# OpenCode bridge
# ---------------------------------------------------------------------------

@dataclass
class OpenCodeResult:
    """Result from an OpenCode invocation."""

    success: bool
    files: dict[str, str] = field(default_factory=dict)
    opencode_log: str = ""
    elapsed_sec: float = 0.0
    error: str = ""


_MEGA_PROMPT_TEMPLATE = """\
You are implementing a complete, runnable ML/science experiment.

Read the files in the current workspace:
- EXPERIMENT_PLAN.yaml — the full experiment design
- GUIDANCE.md — topic, metric, environment constraints, domain-specific guidance

Your task:
1. Design the file structure (main.py is the required entry point).
2. Implement ALL files with complete, runnable code. No placeholders or TODOs.
3. main.py must be the entry point and print the primary metric as:
   {metric}: <value>
4. Include numerical stability guards (gradient clipping, NaN detection, etc.).
5. Use multi-seed evaluation (seeds 0, 1, 2) and report mean ± std.
6. Each ablation/condition MUST be genuinely different — not copy-paste with a renamed variable.
7. Implement a time guard: stop gracefully at 80% of the time budget ({time_budget_sec} seconds).
8. Write requirements.txt listing any extra pip packages needed.
9. If the experiment needs dataset downloads, write a setup.py that handles them.

IMPORTANT CONSTRAINTS:
- The code will run in an isolated Docker container with PyTorch, torchvision, and common ML packages pre-installed.
- Do NOT use argparse or CLI arguments — hardcode all configuration.
- All output must go to stdout (print statements).
- Keep the experiment feasible within {time_budget_sec} seconds total.
"""


class OpenCodeBridge:
    """Manages OpenCode CLI invocations for beast mode code generation."""

    def __init__(
        self,
        *,
        model: str = "",
        llm_base_url: str = "",
        api_key_env: str = "",
        llm_provider: str = "openai-compatible",
        timeout_sec: int = 600,
        max_retries: int = 1,
        workspace_cleanup: bool = True,
    ) -> None:
        self._model = model
        self._llm_base_url = llm_base_url
        self._api_key_env = api_key_env
        self._llm_provider = llm_provider
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._workspace_cleanup = workspace_cleanup
        self._stage_dir: Path | None = None

    # -- availability check ---------------------------------------------------

    @staticmethod
    def check_available() -> bool:
        """Return True if the ``opencode`` CLI is installed and callable."""
        opencode_cmd = shutil.which("opencode")
        if not opencode_cmd:
            return False
            
        try:
            result = subprocess.run(
                [opencode_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:  # noqa: BLE001
            return False

    # -- workspace preparation ------------------------------------------------

    def _prepare_workspace(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        extra_guidance: str,
        time_budget_sec: int,
    ) -> Path:
        """Create a temporary workspace directory with context files.

        BUG-OB-03: The workspace MUST live outside the run-output tree.
        Placing it under ``stage_dir`` (i.e. ``runs/.../stage-10/``) nested it
        inside the main project's git repo, whose ``.gitignore`` excludes
        ``runs/``.  OpenCode then resolved its *project root* to the OUTER
        repo and its file-discovery (which honours the outer ``.gitignore``)
        could not see the workspace's own ``EXPERIMENT_PLAN.yaml`` /
        ``GUIDANCE.md`` — the model reported "files not found", wrote nothing,
        and the run failed with exit-0/zero-files.  Non-ASCII characters in
        the project path (e.g. ``工作区``) compounded the root-detection
        mismatch.  Creating the workspace in an isolated ASCII temp directory
        makes OpenCode treat it as a standalone project.  Debug artifacts are
        mirrored back into ``stage_dir`` by :meth:`_mirror_workspace`.
        """
        self._stage_dir = stage_dir
        ws_name = f"opencode_beast_{int(time.time())}_{time.monotonic_ns() % 100000}"
        base = Path(tempfile.mkdtemp(prefix="rc_beast_"))
        ws = base / ws_name
        ws.mkdir(parents=True, exist_ok=True)

        # Write experiment plan
        (ws / "EXPERIMENT_PLAN.yaml").write_text(
            exp_plan or "# No experiment plan provided\n",
            encoding="utf-8",
        )

        # Write guidance document
        guidance_parts = [
            f"# Experiment Guidance\n",
            f"## Topic\n{topic}\n",
            f"## Primary Metric\n{metric}\n",
            f"## Time Budget\n{time_budget_sec} seconds\n",
        ]
        if pkg_hint:
            guidance_parts.append(f"## Environment\n{pkg_hint}\n")
        if extra_guidance:
            guidance_parts.append(f"## Additional Guidance\n{extra_guidance}\n")
        (ws / "GUIDANCE.md").write_text(
            "\n".join(guidance_parts), encoding="utf-8",
        )

        # Write opencode.json config
        opencode_cfg = self._build_opencode_config()
        (ws / "opencode.json").write_text(
            json.dumps(opencode_cfg, indent=2), encoding="utf-8",
        )

        # OpenCode requires a git repository — initialise one with
        # a single commit so that ``opencode run`` doesn't hang.
        # BUG-OB-01/OB-02: Check return codes and catch TimeoutExpired.
        try:
            r = subprocess.run(
                ["git", "init"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git init failed: {r.stderr}")
            r = subprocess.run(
                ["git", "add", "-A"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git add failed: {r.stderr}")
            r = subprocess.run(
                ["git", "-c", "user.email=beast@researchclaw",
                 "-c", "user.name=BeastMode",
                 "commit", "-m", "init workspace"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git commit failed: {r.stderr}")
        except subprocess.TimeoutExpired as exc:
            raise OSError(f"git workspace init timed out: {exc}") from exc

        return ws

    def _finalize_workspace(self, workspace: Path, stage_dir: Path) -> None:
        """Mirror artifacts into ``stage_dir`` (unless cleanup) and remove temp.

        The live workspace lives in an out-of-tree temp directory created by
        :func:`tempfile.mkdtemp`; that scratch dir is always removed.  When
        ``workspace_cleanup`` is disabled, a debug copy is left under
        ``stage_dir`` first.
        """
        if not self._workspace_cleanup:
            self._mirror_workspace(workspace, stage_dir)
        # Remove the mkdtemp() base (workspace.parent) so no temp dirs leak.
        temp_base = workspace.parent
        try:
            if temp_base.exists() and temp_base.name.startswith("rc_beast_"):
                shutil.rmtree(temp_base, ignore_errors=True)
            elif workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)
        except OSError as exc:
            logger.warning("Beast mode: failed to remove temp workspace: %s", exc)

    @staticmethod
    def _mirror_workspace(workspace: Path, stage_dir: Path) -> Path | None:
        """Copy the isolated temp workspace into ``stage_dir`` for debugging.

        Preserves the historical artifact contract
        (``stage-10/opencode_beast_*/opencode.json`` and generated files) now
        that the live workspace runs from an out-of-tree temp directory.
        The ``.git`` scaffold is skipped to keep the mirror small.
        """
        try:
            dest = stage_dir / workspace.name
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(
                workspace,
                dest,
                ignore=shutil.ignore_patterns(".git", "__pycache__"),
            )
            return dest
        except OSError as exc:
            logger.warning("Beast mode: failed to mirror workspace: %s", exc)
            return None

    def _is_azure(self) -> bool:
        """Detect Azure OpenAI from base URL or provider string."""
        return (
            "azure" in (self._llm_base_url or "").lower()
            or "azure" in (self._llm_provider or "").lower()
        )

    def _uses_builtin_openai_provider(self) -> bool:
        """Return True for endpoints compatible with OpenCode's built-in OpenAI provider."""
        base = (self._llm_base_url or "").lower()
        provider = (self._llm_provider or "").lower()
        return (
            self._is_azure()
            or provider == "openai"
            or "api.openai.com" in base
        )

    def _opencode_provider_id(self) -> str:
        """Provider id used in opencode model strings."""
        if self._uses_builtin_openai_provider():
            return "openai"
        base = (self._llm_base_url or "").lower()
        if "deepseek" in base or "deepseek" in (self._llm_provider or "").lower():
            return "deepseek"
        return "openai_compatible"

    def _opencode_base_url(self) -> str:
        """Return the base URL shape expected by the selected OpenCode provider."""
        base = (self._llm_base_url or "").rstrip("/")
        if not base:
            return base
        if self._uses_builtin_openai_provider():
            return base
        # @ai-sdk/openai-compatible expects the OpenAI-compatible /v1 base.
        if not base.endswith("/v1"):
            return f"{base}/v1"
        return base

    def _build_opencode_config(self) -> dict[str, Any]:
        """Build the opencode.json configuration.

        Standard OpenAI/Azure use OpenCode's built-in ``openai`` provider.
        Generic OpenAI-compatible endpoints use a custom provider backed by
        ``@ai-sdk/openai-compatible`` so OpenCode calls ``/v1/chat/completions``.
        """
        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
        }

        if self._llm_base_url:
            provider_id = self._opencode_provider_id()
            if self._model:
                cfg["model"] = (
                    self._model if "/" in self._model
                    else f"{provider_id}/{self._model}"
                )
            provider_cfg: dict[str, Any] = {
                "options": {
                    "baseURL": self._opencode_base_url(),
                    "apiKey": f"{{env:{self._api_key_env}}}"
                    if self._api_key_env
                    else "",
                },
                "models": {},
            }
            if not self._uses_builtin_openai_provider():
                provider_cfg["npm"] = "@ai-sdk/openai-compatible"
                provider_cfg["name"] = provider_id.replace("_", " ").title()
            cfg["provider"] = {provider_id: provider_cfg}
            # Register the model so OpenCode knows it exists
            if self._model:
                model_name = self._model.split("/")[-1]
                cfg["provider"][provider_id]["models"] = {
                    model_name: {
                        "name": model_name,
                        # Explicitly mark tool-calling support: custom models
                        # unknown to the models.dev catalog may otherwise be
                        # treated as text-only, producing sessions that exit 0
                        # without editing any files.
                        "tool_call": True,
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"],
                        },
                    }
                }
        elif self._model:
            cfg["model"] = (
                self._model if "/" in self._model
                else f"{self._opencode_provider_id()}/{self._model}"
            )

        return cfg

    # -- model resolution -------------------------------------------------------

    def _resolve_opencode_model(self) -> str:
        """Resolve the model identifier for OpenCode CLI's ``-m`` flag.

        Resolution order:
        1. If model already contains "/" (e.g. "anthropic/claude-sonnet-4-6") → use as-is
        2. Otherwise → "{provider}/{model}", where provider is "openai" for
           OpenAI/Azure and a custom OpenAI-compatible provider for endpoints
           such as DeepSeek.
        """
        if not self._model:
            return "anthropic/claude-sonnet-4-6"
        if "/" in self._model:
            return self._model
        return f"{self._opencode_provider_id()}/{self._model}"

    # -- invocation ------------------------------------------------------------

    def _invoke_opencode(
        self,
        workspace: Path,
        prompt: str,
    ) -> tuple[bool, str, float]:
        """Run ``opencode run`` in the workspace. Returns (success, log, elapsed)."""
        env = os.environ.copy()
        # Pass API key via environment if configured.  The configured
        # api_key_env (e.g. DEEPSEEK_API_KEY) is inherited via os.environ
        # and resolved by the "{env:...}" reference in opencode.json.
        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env, "")
            if api_key and self._uses_builtin_openai_provider():
                # Only the built-in "openai" provider (OpenAI/Azure) reads
                # OPENAI_API_KEY.  Do NOT leak other providers' keys into
                # OPENAI_API_KEY — it would enable OpenCode's built-in
                # openai provider with a foreign key.
                env["OPENAI_API_KEY"] = api_key

        # Use -m flag to specify model (more reliable than opencode.json)
        resolved_model = self._resolve_opencode_model()
        opencode_cmd = shutil.which("opencode") or "opencode"
        # BUG-OB-04: pin the project directory explicitly with --dir.  Passing
        # cwd= to subprocess is NOT sufficient — on some platforms OpenCode
        # resolved its project root to the directory it was *launched from*
        # (the repo root where ``python -m researchclaw`` runs), not the
        # workspace.  It then read/wrote files in the repo instead of the
        # workspace, so no main.py was ever collected (exit-0/zero-files) and
        # the model even tried to reuse a pre-existing repo experiment.
        # ``--dir`` forces OpenCode to treat the workspace as the project.
        cmd = [
            opencode_cmd, "run",
            "-m", resolved_model,
            "--dir", str(workspace),
            "--format", "json",
            prompt,
        ]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout_sec,
                env=env,
            )
            elapsed = time.monotonic() - t0
            log = result.stdout + "\n" + result.stderr
            return result.returncode == 0, log, elapsed
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            log = f"TIMEOUT after {elapsed:.1f}s"
            if exc.stdout:
                log += f"\nstdout: {exc.stdout[:2000] if isinstance(exc.stdout, str) else exc.stdout.decode(errors='replace')[:2000]}"
            return False, log, elapsed
        except FileNotFoundError:
            return False, "opencode CLI not found", 0.0
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            return False, f"Unexpected error: {exc}", elapsed

    # -- file collection -------------------------------------------------------

    @staticmethod
    def _collect_files(workspace: Path) -> dict[str, str]:
        """Collect generated Python files, requirements.txt, and setup.py.

        File names are flattened to basenames (e.g. ``src/main.py`` → ``main.py``)
        because the downstream executor expects a flat file dict.  If two files
        share the same basename, the one closer to the workspace root wins.
        """
        files: dict[str, str] = {}
        # Sort by depth (fewer parts first) so root-level files take priority
        py_files = sorted(
            workspace.rglob("*.py"),
            key=lambda p: len(p.relative_to(workspace).parts),
        )
        for py_file in py_files:
            rel = py_file.relative_to(workspace)
            parts = rel.parts
            if any(p.startswith("__pycache__") or p.startswith(".") for p in parts):
                continue
            # Flatten to basename — executor expects flat structure
            basename = rel.name
            if basename not in files:
                try:
                    files[basename] = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Beast mode: failed to read %s: %s", py_file, exc)

        # Also collect requirements.txt and setup.py at root
        for extra in ("requirements.txt", "setup.py"):
            p = workspace / extra
            if p.exists() and extra not in files:
                files[extra] = p.read_text(encoding="utf-8", errors="replace")

        return files

    # -- entry-point validation ------------------------------------------------

    @staticmethod
    def _has_main_guard(source: str) -> bool:
        """Return True if *source* contains ``if __name__ == "__main__":``."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
                    if test.left.id == "__name__" and len(test.comparators) == 1:
                        comp = test.comparators[0]
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            return True
        return False

    @staticmethod
    def _ensure_main_entry_point(files: dict[str, str]) -> dict[str, str]:
        """Ensure ``main.py`` has an ``if __name__ == "__main__"`` guard.

        Beast Mode often generates multi-file projects where ``main.py`` is a
        library module and the real entry point lives in another file (e.g.
        ``run_experiment.py``).  Since the Docker sandbox always executes
        ``python3 main.py``, a library-only ``main.py`` exits immediately with
        no output.

        Strategy:
        1. If ``main.py`` already has the guard → return unchanged.
        2. Find the first other ``.py`` file that **does** have the guard.
        3. Swap: rename that file to ``main.py`` and the old ``main.py`` to a
           helper module (its original basename, or ``_lib.py``).
        4. If no file has a guard, append a minimal stub to ``main.py`` that
           calls the most likely entry function (``main()``, ``run()``, etc.).
        """
        main_code = files.get("main.py", "")
        if not main_code:
            return files

        if OpenCodeBridge._has_main_guard(main_code):
            return files

        # -- Strategy 2/3: find another file with the guard and swap -----------
        for fname, code in files.items():
            if fname == "main.py" or not fname.endswith(".py"):
                continue
            if OpenCodeBridge._has_main_guard(code):
                logger.info(
                    "Beast mode: main.py lacks __main__ guard; swapping "
                    "entry point with %s",
                    fname,
                )
                new_files = dict(files)
                # Rename original main.py → helper module
                helper_name = fname  # reuse the other file's name for old main
                new_files[helper_name] = main_code
                new_files["main.py"] = code
                return new_files

        # -- Strategy 4: inject a minimal entry point into main.py -------------
        # Look for common entry functions defined in main.py
        entry_func: str | None = None
        try:
            tree = ast.parse(main_code)
            candidates = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ("main", "run", "run_experiment", "train",
                               "run_experiments", "experiment", "run_all")
            ]
            if candidates:
                entry_func = candidates[0]
        except SyntaxError:
            pass

        if entry_func:
            logger.info(
                "Beast mode: main.py lacks __main__ guard; injecting call "
                "to %s()",
                entry_func,
            )
            new_files = dict(files)
            new_files["main.py"] = (
                main_code.rstrip()
                + "\n\n\nif __name__ == \"__main__\":\n"
                + f"    {entry_func}()\n"
            )
            return new_files

        logger.warning(
            "Beast mode: main.py lacks __main__ guard and no known entry "
            "function found — experiment may exit without producing output",
        )
        return files

    # -- main entry point ------------------------------------------------------

    def generate(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str = "",
        extra_guidance: str = "",
        time_budget_sec: int = 300,
    ) -> OpenCodeResult:
        """Run OpenCode to generate experiment code.

        Returns an OpenCodeResult with success status and generated files.
        """
        # Check availability first
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="OpenCode CLI not installed or not callable",
            )

        workspace: Path | None = None
        last_error = ""

        for attempt in range(1 + self._max_retries):
            # Prepare workspace
            try:
                workspace = self._prepare_workspace(
                    stage_dir=stage_dir,
                    topic=topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint,
                    extra_guidance=extra_guidance,
                    time_budget_sec=time_budget_sec,
                )
            except OSError as exc:
                last_error = f"Failed to prepare workspace: {exc}"
                logger.warning("Beast mode: %s", last_error)
                continue

            # Build the mega-prompt (use replace instead of .format() to
            # avoid KeyError when metric contains curly braces like "F{1}")
            prompt = _MEGA_PROMPT_TEMPLATE.replace(
                "{metric}", metric
            ).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )

            logger.info(
                "Beast mode: invoking OpenCode (attempt %d/%d, timeout=%ds)",
                attempt + 1,
                1 + self._max_retries,
                self._timeout_sec,
            )

            success, log, elapsed = self._invoke_opencode(workspace, prompt)

            if success:
                files = self._collect_files(workspace)
                if "main.py" not in files:
                    logger.warning(
                        "Beast mode: OpenCode succeeded but no main.py found "
                        "(files: %s)", list(files.keys()),
                    )
                    last_error = "No main.py in OpenCode output"
                    # Persist the transcript — without it, "exit 0 but no
                    # files" failures are undiagnosable (BUG: this branch
                    # previously skipped the log write, leaving a stale
                    # opencode_log.txt from an earlier attempt).
                    try:
                        (stage_dir / "opencode_log.txt").write_text(
                            log or "", encoding="utf-8",
                        )
                    except OSError as _wexc:
                        logger.warning(
                            "Beast mode: failed to write no-main.py log: %s",
                            _wexc,
                        )
                    # Mirror artifacts for debugging, then drop temp workspace
                    self._finalize_workspace(workspace, stage_dir)
                    continue

                # BUG-R52-01: Ensure main.py has an entry point
                files = self._ensure_main_entry_point(files)

                # Write log
                try:
                    (stage_dir / "opencode_log.txt").write_text(
                        log or "", encoding="utf-8",
                    )
                except OSError as _wexc:
                    logger.warning("Beast mode: failed to write log: %s", _wexc)

                # Mirror artifacts for debugging, then drop temp workspace
                self._finalize_workspace(workspace, stage_dir)

                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = log
            try:
                (stage_dir / "opencode_log.txt").write_text(
                    log or "", encoding="utf-8",
                )
            except OSError as _wexc:
                logger.warning("Beast mode: failed to write failure log: %s", _wexc)
            logger.warning(
                "Beast mode: OpenCode attempt %d failed (%.1fs): %s",
                attempt + 1,
                elapsed,
                log[:500],
            )
            # Mirror artifacts for debugging, then drop temp workspace
            if workspace:
                self._finalize_workspace(workspace, stage_dir)

        # All attempts failed
        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            error=f"OpenCode failed after {1 + self._max_retries} attempt(s)",
        )


# ---------------------------------------------------------------------------
# Helper: count historical failures
# ---------------------------------------------------------------------------

def count_historical_failures(run_dir: Path, stage_name: str = "stage-10") -> int:
    """Count past Stage 10 failures from stage directories and logs.

    Each stage directory is counted at most once, even if multiple failure
    indicators are present.
    """
    failures = 0
    for d in run_dir.glob(f"{stage_name}*"):
        failed = False
        # Check for beast_mode_log.json
        bm_log = d / "beast_mode_log.json"
        if bm_log.exists():
            try:
                data = json.loads(bm_log.read_text(encoding="utf-8"))
                if not data.get("success", True):
                    failed = True
            except Exception:  # noqa: BLE001
                pass
        # Check for stage health failures
        if not failed:
            health = d / "stage_health.json"
            if health.exists():
                try:
                    data = json.loads(health.read_text(encoding="utf-8"))
                    if data.get("status") == "FAILED":
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        # Check for validation report with FAILED status
        if not failed:
            vr = d / "validation_report.md"
            if vr.exists():
                try:
                    content = vr.read_text(encoding="utf-8")
                    if "BLOCKED" in content or "FAILED" in content:
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        if failed:
            failures += 1
    return failures
