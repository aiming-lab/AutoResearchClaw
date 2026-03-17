"""Advanced multi-phase code generation agent.

Phases
------
1. **Blueprint Planning** — produce deep implementation blueprint with
   per-file pseudocode, tensor shapes, and generation ordering.
2. **Sequential File Generation** — generate files one-by-one following
   the dependency order from the blueprint, with CodeMem summaries.
   Falls back to single-shot generation if blueprint parsing fails.
3. **Execution-in-the-Loop** — run in sandbox, feed errors back for repair.
4. **Solution Tree Search** — explore multiple candidate implementations,
   evaluate via sandbox, select the best (optional, higher cost).
5. **Multi-Agent Review** — coder-reviewer dialog for quality assurance.

Integration
-----------
``CodeAgent`` is instantiated inside ``_execute_code_generation`` in
``executor.py`` when ``config.experiment.code_agent.enabled`` is True.
It receives the same inputs (topic, exp_plan, metric, pkg_hint) and
returns ``CodeAgentResult`` with the generated files.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for the advanced code generation agent.

    All phases are independently toggleable.  The default profile enables
    Phases 1 (blueprint), 2 (sequential generation + exec-fix), and 5
    (review), which gives a large quality boost at moderate extra cost.
    Phase 4 (tree search) is opt-in because it multiplies both LLM and
    sandbox usage.
    """

    enabled: bool = True

    # Phase 1: Blueprint planning (deep implementation blueprint)
    architecture_planning: bool = True

    # Phase 2: Sequential file generation (generate files one-by-one
    # following dependency order from blueprint, with CodeMem summaries)
    sequential_generation: bool = True

    # Phase 2.5: Hard validation gates (AST-based)
    hard_validation: bool = True
    hard_validation_max_repairs: int = 2

    # Phase 3: Execution-in-the-loop
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60

    # Phase 4: Solution tree search (off by default)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2
    tree_search_eval_timeout_sec: int = 120

    # Phase 5: Multi-agent review dialog
    review_max_rounds: int = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SolutionNode:
    """One candidate solution in the search tree."""

    node_id: str
    files: dict[str, str]
    parent_id: str | None = None
    depth: int = 0
    # Evaluation
    runs_ok: bool = False
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    generation_method: str = "initial"


@dataclass
class CodeAgentResult:
    """Final output from the code agent."""

    files: dict[str, str]
    architecture_spec: str = ""
    validation_log: list[str] = field(default_factory=list)
    total_llm_calls: int = 0
    total_sandbox_runs: int = 0
    best_score: float = 0.0
    tree_nodes_explored: int = 0
    review_rounds: int = 0


# ---------------------------------------------------------------------------
# Sandbox protocol (structural typing — no import dependency)
# ---------------------------------------------------------------------------


class _SandboxResult(Protocol):  # pragma: no cover
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    metrics: dict[str, object]
    timed_out: bool


class _SandboxLike(Protocol):  # pragma: no cover
    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# CodeAgent
# ---------------------------------------------------------------------------


class CodeAgent:
    """Multi-phase code generation agent.

    Parameters
    ----------
    llm : LLMClient
        The language model client to use for code generation.
    prompts : PromptManager
        Manages prompt templates.
    config : CodeAgentConfig
        Agent configuration (toggles, limits, timeouts).
    stage_dir : Path
        Working directory for this stage (e.g. ``run_dir/stage-10``).
    sandbox_factory : callable, optional
        ``(ExperimentConfig, Path) -> SandboxLike``.  Required for
        Phases 2 and 3.
    experiment_config : ExperimentConfig, optional
        Passed to ``sandbox_factory`` when creating sandboxes.
    """

    def __init__(
        self,
        llm: Any,
        prompts: Any,
        config: CodeAgentConfig,
        stage_dir: Path,
        sandbox_factory: Any | None = None,
        experiment_config: Any | None = None,
    ) -> None:
        self._llm = llm
        self._pm = prompts
        self._cfg = config
        self._stage_dir = stage_dir
        self._sandbox_factory = sandbox_factory
        self._exp_config = experiment_config
        self._calls = 0
        self._runs = 0
        self._log: list[str] = []
        self._sandbox: _SandboxLike | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        max_tokens: int = 8192,
    ) -> CodeAgentResult:
        """Execute all enabled phases and return generated files."""
        t0 = time.time()
        self._log_event("CodeAgent.generate() started")

        # Phase 1: Blueprint planning
        arch_spec = ""
        blueprint = None
        if self._cfg.architecture_planning:
            arch_spec, blueprint = self._phase1_blueprint(
                topic, exp_plan, metric,
            )

        # Phase 2: Code generation
        nodes_explored = 0
        if self._cfg.tree_search_enabled and self._sandbox_factory:
            best, nodes_explored = self._phase3_tree_search(
                topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
            )
        elif (
            self._cfg.sequential_generation
            and blueprint is not None
            and self._is_valid_blueprint(blueprint)
        ):
            # Sequential file generation following blueprint
            files = self._phase2_sequential_generate(
                topic, exp_plan, metric, pkg_hint, arch_spec, blueprint,
            )
            # Hard validation gates (E-03)
            if self._cfg.hard_validation:
                files = self._hard_validate_and_repair(
                    files, topic, exp_plan, metric, pkg_hint, arch_spec,
                )
            # Exec-fix loop
            files = self._exec_fix_loop(files)
            best = SolutionNode(
                node_id="sequential", files=files, runs_ok=True, score=1.0,
            )
        else:
            # Fallback: single-shot generation
            if self._cfg.sequential_generation and blueprint is None:
                self._log_event(
                    "  Sequential generation requested but blueprint "
                    "invalid — falling back to single-shot"
                )
            files = self._phase2_generate_and_fix(
                topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
            )
            # Hard validation gates (E-03) for single-shot too
            if self._cfg.hard_validation:
                files = self._hard_validate_and_repair(
                    files, topic, exp_plan, metric, pkg_hint, arch_spec,
                )
            best = SolutionNode(
                node_id="single", files=files, runs_ok=True, score=1.0,
            )

        # Phase 5: Review dialog
        review_rounds = 0
        if self._cfg.review_max_rounds > 0:
            best.files, review_rounds = self._phase4_review(
                best.files, topic, exp_plan, metric,
            )

        elapsed = time.time() - t0
        self._log_event(
            f"CodeAgent.generate() done in {elapsed:.1f}s — "
            f"{self._calls} LLM calls, {self._runs} sandbox runs"
        )

        return CodeAgentResult(
            files=best.files,
            architecture_spec=arch_spec,
            validation_log=list(self._log),
            total_llm_calls=self._calls,
            total_sandbox_runs=self._runs,
            best_score=best.score,
            tree_nodes_explored=nodes_explored,
            review_rounds=review_rounds,
        )

    # ── Phase 1: Blueprint Planning ──────────────────────────────────────

    def _phase1_blueprint(
        self, topic: str, exp_plan: str, metric: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Generate a deep implementation blueprint.

        Returns (raw_yaml_str, parsed_blueprint_dict_or_None).
        """
        self._log_event("Phase 1: Blueprint planning")

        sp = self._pm.sub_prompt(
            "architecture_planning",
            topic=topic,
            exp_plan=exp_plan,
            metric=metric,
        )
        resp = self._chat(sp.system, sp.user, max_tokens=8192)

        # Extract YAML block from response
        arch_spec = resp.content
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", arch_spec, re.DOTALL)
        if yaml_match:
            arch_spec = yaml_match.group(1).strip()

        self._log_event(f"  Blueprint spec: {len(arch_spec)} chars")

        # Parse YAML into structured blueprint
        blueprint = self._parse_blueprint(arch_spec)
        if blueprint:
            n_files = len(blueprint.get("files", []))
            self._log_event(f"  Parsed blueprint: {n_files} files")
        else:
            self._log_event("  WARNING: Could not parse blueprint YAML")

        return arch_spec, blueprint

    def _parse_blueprint(self, yaml_text: str) -> dict[str, Any] | None:
        """Parse blueprint YAML into a structured dict."""
        try:
            import yaml
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict) and "files" in data:
                return data
        except Exception as exc:
            self._log_event(f"  Blueprint YAML parse error: {exc}")
        return None

    @staticmethod
    def _is_valid_blueprint(blueprint: dict[str, Any]) -> bool:
        """Check if a blueprint has the minimum required structure."""
        files = blueprint.get("files", [])
        if not files or not isinstance(files, list):
            return False
        # Need at least 2 files with generation_order
        has_order = sum(
            1 for f in files
            if isinstance(f, dict) and "generation_order" in f
        )
        return has_order >= 2

    # ── Phase 2a: Sequential File Generation ─────────────────────────────

    def _phase2_sequential_generate(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        blueprint: dict[str, Any],
    ) -> dict[str, str]:
        """Generate files one-by-one following blueprint dependency order."""
        self._log_event("Phase 2: Sequential generation (blueprint-guided)")

        generated_files: dict[str, str] = {}
        code_memory: dict[str, dict[str, Any]] = {}  # CodeMem summaries

        # Sort files by generation_order
        file_specs = blueprint.get("files", [])
        file_specs = [f for f in file_specs if isinstance(f, dict)]

        # Ensure generation_order exists; default to list position
        for i, fs in enumerate(file_specs):
            if "generation_order" not in fs:
                fs["generation_order"] = i + 1

        file_specs.sort(key=lambda f: f.get("generation_order", 99))

        for file_spec in file_specs:
            file_name = file_spec.get("name", "")
            if not file_name:
                continue

            self._log_event(
                f"  Generating {file_name} "
                f"(order={file_spec.get('generation_order')})"
            )

            # Build dependency context
            deps = file_spec.get("dependencies", [])
            dep_summaries = ""
            dep_code = ""

            for dep in deps:
                if isinstance(dep, str):
                    if dep in code_memory:
                        dep_summaries += (
                            f"\n### {dep} (summary)\n"
                            + json.dumps(code_memory[dep], indent=2)
                            + "\n"
                        )
                    if dep in generated_files:
                        dep_code += (
                            f"\n### {dep}\n```python\n"
                            + generated_files[dep]
                            + "\n```\n"
                        )

            if not dep_summaries:
                dep_summaries = "(no dependencies yet)"
            if not dep_code:
                dep_code = "(no dependencies yet)"

            # Generate this file via LLM
            file_spec_str = json.dumps(file_spec, indent=2, default=str)
            sp = self._pm.sub_prompt(
                "generate_single_file",
                file_name=file_name,
                file_spec=file_spec_str,
                blueprint=arch_spec,
                dependency_summaries=dep_summaries,
                dependency_code=dep_code,
                topic=topic,
                exp_plan=exp_plan[:4000],  # Truncate to avoid token overflow
                pkg_hint=pkg_hint,
            )
            resp = self._chat(sp.system, sp.user, max_tokens=8192)

            # Extract code from response
            code = self._extract_single_file_code(resp.content, file_name)
            if not code:
                self._log_event(f"  WARNING: Empty code for {file_name}")
                continue

            generated_files[file_name] = code

            # Build CodeMem summary via AST
            code_memory[file_name] = self._build_code_summary(
                file_name, code,
            )

            self._log_event(
                f"  {file_name}: {len(code.split(chr(10)))} lines, "
                f"{len(code_memory[file_name].get('classes', []))} classes"
            )

        # Verify we have main.py
        if "main.py" not in generated_files:
            self._log_event("  WARNING: No main.py generated, promoting first file")
            if generated_files:
                first_key = next(iter(generated_files))
                generated_files["main.py"] = generated_files.pop(first_key)

        self._log_event(
            f"  Sequential generation complete: {len(generated_files)} files"
        )
        return generated_files

    @staticmethod
    def _extract_single_file_code(content: str, expected_name: str) -> str:
        """Extract Python code from LLM response for a single file."""
        # Try to extract from ```python``` block
        m = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Try ```filename:xxx.py block
        m = re.search(
            rf"```(?:filename:)?{re.escape(expected_name)}\s*\n(.*?)```",
            content, re.DOTALL,
        )
        if m:
            return m.group(1).strip()

        # If content looks like raw Python (starts with import/from/# or def)
        stripped = content.strip()
        if stripped and (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("#")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith('"""')
        ):
            return stripped

        return ""

    @staticmethod
    def _build_code_summary(
        filename: str, code: str,
    ) -> dict[str, Any]:
        """Build a CodeMem-style compressed summary via AST analysis."""
        summary: dict[str, Any] = {
            "filename": filename,
            "classes": [],
            "functions": [],
            "imports": [],
        }

        try:
            tree = ast.parse(code)
        except SyntaxError:
            summary["parse_error"] = True
            return summary

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for n in node.body:
                    if isinstance(n, ast.FunctionDef):
                        args = [a.arg for a in n.args.args if a.arg != "self"]
                        methods.append({
                            "name": n.name,
                            "args": args,
                        })
                summary["classes"].append({
                    "name": node.name,
                    "bases": [ast.unparse(b) for b in node.bases],
                    "methods": methods,
                })
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                args = [a.arg for a in node.args.args]
                summary["functions"].append({
                    "name": node.name,
                    "args": args,
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    summary["imports"].append(ast.unparse(node))
                except Exception:
                    pass

        return summary

    # ── Phase 2.5: Hard Validation Gates (E-03) ─────────────────────────

    def _hard_validate_and_repair(
        self,
        files: dict[str, str],
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
    ) -> dict[str, str]:
        """Run AST-based hard validation and repair critical issues.

        Critical issues trigger targeted file regeneration.  Non-critical
        issues are logged as warnings only.
        """
        self._log_event("Phase 2.5: Hard validation gates")

        for attempt in range(self._cfg.hard_validation_max_repairs + 1):
            critical, warnings = self._hard_validate(files)

            # Log warnings
            for w in warnings:
                self._log_event(f"  WARNING: {w}")

            if not critical:
                self._log_event(
                    f"  Hard validation passed "
                    f"({len(warnings)} warning(s), attempt {attempt})"
                )
                return files

            self._log_event(
                f"  Hard validation found {len(critical)} CRITICAL issue(s) "
                f"(attempt {attempt}/{self._cfg.hard_validation_max_repairs})"
            )
            for c in critical:
                self._log_event(f"  CRITICAL: {c}")

            if attempt >= self._cfg.hard_validation_max_repairs:
                self._log_event(
                    "  Max repair attempts reached — proceeding with warnings"
                )
                return files

            # Targeted repair: ask LLM to fix specific critical issues
            files = self._repair_critical_issues(
                files, critical, topic, exp_plan, metric, arch_spec,
            )

        return files

    def _hard_validate(
        self, files: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Run AST-based checks and classify as CRITICAL or WARNING.

        Returns (critical_issues, warning_issues).
        """
        critical: list[str] = []
        warnings: list[str] = []

        from researchclaw.experiment.validator import (
            check_class_quality,
            check_code_complexity,
            check_api_correctness,
            check_variable_scoping,
            validate_syntax,
        )

        # 1. Syntax check — always critical
        for fname, code in files.items():
            if not fname.endswith(".py"):
                continue
            syn = validate_syntax(code)
            if not syn.ok:
                for issue in syn.errors:
                    critical.append(
                        f"[{fname}] Syntax error: {issue.message} "
                        f"(line {issue.line})"
                    )

        # 2. Class quality — some are critical
        class_warns = check_class_quality(files)
        for w in class_warns:
            if "identical AST to parent" in w:
                critical.append(w)
            elif "NOT a real ablation" in w:
                critical.append(w)
            elif "creates nn.Module" in w and "inside forward()" in w:
                critical.append(w)
            elif "empty or trivial subclass" in w:
                # Critical: ablation classes must have real implementations
                critical.append(w)
            else:
                warnings.append(w)

        # 3. Code complexity — hardcoded metrics are critical
        for fname, code in files.items():
            if not fname.endswith(".py"):
                continue
            complexity_warns = check_code_complexity(code)
            for w in complexity_warns:
                if "hardcoded metric" in w.lower():
                    critical.append(f"[{fname}] {w}")
                elif "trivial computation" in w.lower():
                    critical.append(f"[{fname}] {w}")
                else:
                    warnings.append(f"[{fname}] {w}")

        # 4. API correctness — NameError-causing issues are critical
        for fname, code in files.items():
            if not fname.endswith(".py"):
                continue
            api_warns = check_api_correctness(code, fname)
            for w in api_warns:
                if "NameError" in w or "Import-usage mismatch" in w:
                    critical.append(w)
                elif "does not exist" in w:
                    critical.append(w)
                else:
                    warnings.append(w)

        # 5. Variable scoping — UnboundLocalError is critical
        for fname, code in files.items():
            if not fname.endswith(".py"):
                continue
            scope_warns = check_variable_scoping(code, fname)
            for w in scope_warns:
                if "UnboundLocalError" in w:
                    critical.append(w)
                else:
                    warnings.append(w)

        # 6. Cross-file import consistency — check local imports resolve
        known_modules = {
            fname.replace(".py", "")
            for fname in files
            if fname.endswith(".py")
        }
        for fname, code in files.items():
            if not fname.endswith(".py"):
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod_top = node.module.split(".")[0]
                    if (
                        mod_top in known_modules
                        and mod_top not in known_modules
                    ):
                        pass  # impossible, skip
                    # Check if importing from a local module that exists
                    if mod_top in known_modules:
                        # Verify imported names exist in target file
                        target_file = f"{mod_top}.py"
                        if target_file in files and node.names:
                            target_code = files[target_file]
                            try:
                                target_tree = ast.parse(target_code)
                            except SyntaxError:
                                continue
                            exported = set()
                            for tnode in ast.walk(target_tree):
                                if isinstance(tnode, ast.ClassDef):
                                    exported.add(tnode.name)
                                elif isinstance(tnode, ast.FunctionDef):
                                    exported.add(tnode.name)
                                elif isinstance(tnode, ast.Assign):
                                    for t in tnode.targets:
                                        if isinstance(t, ast.Name):
                                            exported.add(t.id)
                            for alias in node.names:
                                name = alias.name
                                if name != "*" and name not in exported:
                                    critical.append(
                                        f"[{fname}] ImportError: "
                                        f"'{name}' not defined in "
                                        f"'{target_file}' — will crash"
                                    )

        return critical, warnings

    def _repair_critical_issues(
        self,
        files: dict[str, str],
        critical_issues: list[str],
        topic: str,
        exp_plan: str,
        metric: str,
        arch_spec: str,
    ) -> dict[str, str]:
        """Ask LLM to fix critical validation issues."""
        self._log_event("  Targeted repair for critical issues")

        # Identify which files need repair
        affected_files: set[str] = set()
        for issue in critical_issues:
            # Extract filename from issue string: [filename.py] ...
            m = re.match(r"\[([^\]]+\.py)\]", issue)
            if m:
                affected_files.add(m.group(1))
            else:
                # If no filename found, assume all files affected
                affected_files.update(
                    f for f in files if f.endswith(".py")
                )

        if not affected_files:
            affected_files.update(f for f in files if f.endswith(".py"))

        files_ctx = self._format_files(files)
        issues_text = "\n".join(f"- {issue}" for issue in critical_issues)

        prompt = (
            "Your generated code has CRITICAL issues that will cause "
            "runtime failures or produce invalid results. Fix ALL of them.\n\n"
            "## Critical Issues Found\n"
            f"{issues_text}\n\n"
            "## Architecture Blueprint\n"
            f"{arch_spec[:4000]}\n\n"
            "## Current Code\n"
            f"{files_ctx}\n\n"
            "## Rules\n"
            "1. Fix every critical issue listed above\n"
            "2. Ablation/variant classes MUST have different implementations "
            "from their parent — change the forward() or core method\n"
            "3. Never hardcode metric values — compute them from actual data\n"
            "4. nn.Module layers must be created in __init__(), not forward()\n"
            "5. All cross-file imports must reference names that actually exist\n"
            "6. Output ALL files in ```filename:xxx.py``` format\n"
        )

        sys_prompt = self._pm.system("code_generation")
        resp = self._chat(sys_prompt, prompt, max_tokens=16384)

        fixed = self._extract_files(resp.content)
        if fixed:
            merged = dict(files)
            merged.update(fixed)
            self._log_event(
                f"  Repair updated {len(fixed)} file(s): "
                f"{', '.join(sorted(fixed))}"
            )
            return merged

        self._log_event("  WARNING: Repair produced no extractable files")
        return files

    # ── Phase 2b: Single-Shot Generate + Exec-Fix (legacy) ───────────────

    def _phase2_generate_and_fix(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        max_tokens: int,
    ) -> dict[str, str]:
        """Generate code in single shot, then iteratively fix via sandbox."""
        self._log_event("Phase 2: Single-shot generate + exec-fix")

        # Initial generation (uses the existing code_generation prompt)
        files = self._generate_code(
            topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
        )
        if not files:
            self._log_event("  WARNING: empty generation, returning fallback")
            return files

        return self._exec_fix_loop(files)

    def _exec_fix_loop(self, files: dict[str, str]) -> dict[str, str]:
        """Run exec-fix loop if sandbox is available."""
        if not self._sandbox_factory or self._cfg.exec_fix_max_iterations <= 0:
            return files

        for i in range(self._cfg.exec_fix_max_iterations):
            result = self._run_in_sandbox(files)
            if result.returncode == 0:
                self._log_event(f"  Exec-fix iter {i}: code runs OK")
                break

            self._log_event(
                f"  Exec-fix iter {i}: crashed (rc={result.returncode}), "
                f"stderr={len(result.stderr)} chars"
            )
            files = self._fix_runtime_error(files, result)

        return files

    def _generate_code(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        max_tokens: int,
    ) -> dict[str, str]:
        """Single code generation call with architecture spec injected."""
        # Inject architecture specification into the pkg_hint slot
        hint = pkg_hint
        if arch_spec:
            hint = (
                f"{pkg_hint}\n\n"
                "## ARCHITECTURE SPECIFICATION (follow this file and class structure)\n"
                f"{arch_spec}\n"
            )

        sp = self._pm.for_stage(
            "code_generation",
            topic=topic,
            metric=metric,
            pkg_hint=hint,
            exp_plan=exp_plan,
        )
        resp = self._chat(sp.system, sp.user, max_tokens=max_tokens)

        files = self._extract_files(resp.content)
        if not files and resp.content.strip():
            # Retry with higher token budget
            self._log_event("  Empty extraction, retrying with 32768 tokens")
            resp = self._chat(sp.system, sp.user, max_tokens=32768)
            files = self._extract_files(resp.content)

        return files

    def _fix_runtime_error(
        self, files: dict[str, str], result: Any,
    ) -> dict[str, str]:
        """Fix a runtime error using targeted or full-file repair.

        E-05: Parse the error traceback to identify the failing file and
        line, then send only the affected file with a focused context
        window.  Falls back to full-file repair if parsing fails.
        """
        stderr_tail = (result.stderr or "")[-3000:]
        stdout_tail = "\n".join(
            (result.stdout or "").split("\n")[-50:]
        )

        # Try targeted repair first (E-05)
        error_loc = self._parse_error_location(stderr_tail, files)
        if error_loc:
            fname, lineno, error_msg = error_loc
            self._log_event(
                f"  Targeted repair: {fname}:{lineno} — {error_msg[:80]}"
            )
            fixed = self._targeted_file_repair(
                files, fname, lineno, error_msg, stderr_tail,
            )
            if fixed:
                return fixed

        # Fallback: full-file repair
        files_ctx = self._format_files(files)
        sp = self._pm.sub_prompt(
            "code_exec_fix",
            stderr=stderr_tail or "(empty)",
            stdout_tail=stdout_tail or "(empty)",
            returncode=str(result.returncode),
            files_context=files_ctx,
        )
        resp = self._chat(sp.system, sp.user, max_tokens=16384)

        fixed = self._extract_files(resp.content)
        if fixed:
            merged = dict(files)
            merged.update(fixed)
            return merged
        return files

    @staticmethod
    def _parse_error_location(
        stderr: str, files: dict[str, str],
    ) -> tuple[str, int, str] | None:
        """Parse Python traceback to find failing file and line.

        Returns (filename, line_number, error_message) or None.
        """
        known_files = set(files.keys())
        # Parse traceback lines: File "xxx.py", line NNN
        tb_pattern = re.compile(
            r'File "(?:[^"]*[/\\])?([^"]+\.py)", line (\d+)'
        )
        matches = list(tb_pattern.finditer(stderr))
        if not matches:
            return None

        # Find the last match that references one of our files
        for m in reversed(matches):
            fname = m.group(1)
            lineno = int(m.group(2))
            if fname in known_files:
                # Extract error message (last line of stderr)
                lines = stderr.strip().split("\n")
                error_msg = lines[-1] if lines else "Unknown error"
                return fname, lineno, error_msg

        return None

    def _targeted_file_repair(
        self,
        files: dict[str, str],
        target_file: str,
        error_line: int,
        error_msg: str,
        full_stderr: str,
    ) -> dict[str, str] | None:
        """Repair a single file with focused context around the error."""
        if target_file not in files:
            return None

        code = files[target_file]
        code_lines = code.split("\n")
        total_lines = len(code_lines)

        # Extract context window: ±30 lines around error
        window = 30
        start = max(0, error_line - window - 1)
        end = min(total_lines, error_line + window)
        context_lines = code_lines[start:end]

        # Number the lines for the LLM
        numbered = "\n".join(
            f"{start + i + 1:4d} | {line}"
            for i, line in enumerate(context_lines)
        )

        # Build compact dependency context (summaries only)
        dep_summaries = ""
        for fname, fcode in files.items():
            if fname != target_file and fname.endswith(".py"):
                summary = self._build_code_summary(fname, fcode)
                dep_summaries += (
                    f"\n### {fname}: "
                    f"{len(summary.get('classes', []))} classes, "
                    f"{len(summary.get('functions', []))} functions\n"
                )
                for cls in summary.get("classes", []):
                    methods = ", ".join(
                        m["name"] for m in cls.get("methods", [])
                    )
                    dep_summaries += (
                        f"  class {cls['name']}"
                        f"({', '.join(cls.get('bases', []))})"
                        f": [{methods}]\n"
                    )

        prompt = (
            f"Fix the runtime error in `{target_file}` at line {error_line}.\n\n"
            f"## Error\n```\n{error_msg}\n```\n\n"
            f"## Full Traceback (last 1500 chars)\n"
            f"```\n{full_stderr[-1500:]}\n```\n\n"
            f"## {target_file} (lines {start + 1}-{end})\n"
            f"```python\n{numbered}\n```\n\n"
            f"## Other Files in Project\n{dep_summaries}\n\n"
            f"## Full File ({target_file}, {total_lines} lines)\n"
            f"```python\n{code}\n```\n\n"
            "Output the COMPLETE fixed `{target_file}` in "
            "```filename:{target_file}``` format. Fix the root cause, "
            "not just the symptom."
        ).format(target_file=target_file)

        sys_prompt = (
            "You are a debugging expert. Fix the specific runtime error "
            "shown. Preserve experiment design and scientific methodology. "
            "Output the COMPLETE fixed file."
        )
        resp = self._chat(sys_prompt, prompt, max_tokens=16384)

        fixed = self._extract_files(resp.content)
        if not fixed:
            # Try extracting as single file
            code_match = re.search(
                r"```(?:python|filename:\S+)\s*\n(.*?)```",
                resp.content, re.DOTALL,
            )
            if code_match:
                fixed = {target_file: code_match.group(1).strip()}

        if fixed and target_file in fixed:
            merged = dict(files)
            merged.update(fixed)
            self._log_event(
                f"  Targeted repair applied to {target_file} "
                f"({len(fixed[target_file].split(chr(10)))} lines)"
            )
            return merged

        return None

    # ── Phase 3: Solution Tree Search ─────────────────────────────────────

    def _phase3_tree_search(
        self,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        arch_spec: str,
        max_tokens: int,
    ) -> tuple[SolutionNode, int]:
        """Explore multiple candidate solutions via tree search."""
        self._log_event("Phase 3: Solution tree search")
        all_nodes: list[SolutionNode] = []

        # Generate initial candidates
        n_cand = self._cfg.tree_search_candidates
        for k in range(n_cand):
            self._log_event(f"  Generating candidate {k + 1}/{n_cand}")
            files = self._generate_code(
                topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
            )
            node = SolutionNode(
                node_id=f"gen-{k}",
                files=files,
                depth=0,
                generation_method="initial",
            )
            all_nodes.append(node)

        # Iterative evaluate-fix-branch loop
        for depth in range(self._cfg.tree_search_max_depth):
            # Evaluate unevaluated nodes
            for node in all_nodes:
                if node.returncode == -1:
                    self._evaluate_node(node, metric)

            # Sort by score
            all_nodes.sort(key=lambda n: n.score, reverse=True)

            self._log_event(
                f"  Depth {depth}: {len(all_nodes)} nodes, "
                f"best={all_nodes[0].node_id} score={all_nodes[0].score:.2f}"
            )

            # If best runs OK, we're done
            if all_nodes[0].runs_ok:
                break

            # Generate fix variants for top-2 crashing candidates
            new_nodes: list[SolutionNode] = []
            for node in all_nodes[:2]:
                if not node.runs_ok:
                    fixed_files = self._fix_runtime_error(
                        node.files,
                        _SimpleResult(
                            returncode=node.returncode,
                            stdout=node.stdout,
                            stderr=node.stderr,
                        ),
                    )
                    new_node = SolutionNode(
                        node_id=f"{node.node_id}-fix{depth}",
                        files=fixed_files,
                        parent_id=node.node_id,
                        depth=depth + 1,
                        generation_method="fix",
                    )
                    new_nodes.append(new_node)

            all_nodes.extend(new_nodes)

        # Final evaluation of any remaining unevaluated nodes
        for node in all_nodes:
            if node.returncode == -1:
                self._evaluate_node(node, metric)

        all_nodes.sort(key=lambda n: n.score, reverse=True)
        best = all_nodes[0]
        self._log_event(
            f"  Tree search complete: best={best.node_id} "
            f"score={best.score:.2f}, explored {len(all_nodes)} nodes"
        )

        return best, len(all_nodes)

    def _evaluate_node(self, node: SolutionNode, metric_key: str) -> None:
        """Run a node's code in sandbox and update its score."""
        if not node.files:
            node.score = 0.0
            return

        result = self._run_in_sandbox(
            node.files,
            timeout_sec=self._cfg.tree_search_eval_timeout_sec,
        )
        node.returncode = result.returncode
        node.stdout = result.stdout
        node.stderr = result.stderr
        node.runs_ok = result.returncode == 0
        node.metrics = dict(result.metrics) if result.metrics else {}
        node.score = self._score_node(node, metric_key)

    @staticmethod
    def _score_node(node: SolutionNode, metric_key: str) -> float:
        """Score a solution node based on execution results."""
        score = 0.0
        if node.runs_ok:
            score += 1.0
        if node.stdout and len(node.stdout) > 100:
            score += 0.3  # produces meaningful output
        if node.metrics:
            score += 0.5
            if metric_key in node.metrics:
                score += 0.5
        if node.stderr and "Error" in node.stderr:
            score -= 0.2
        return max(score, 0.0)

    # ── Phase 5: Multi-Agent Review Dialog ────────────────────────────────

    def _phase4_review(
        self,
        files: dict[str, str],
        topic: str,
        exp_plan: str,
        metric: str,
    ) -> tuple[dict[str, str], int]:
        """Reviewer agent examines code; coder fixes critical issues."""
        self._log_event("Phase 4: Review dialog")

        rounds = 0
        for r in range(self._cfg.review_max_rounds):
            rounds += 1
            files_ctx = self._format_files(files)

            sp = self._pm.sub_prompt(
                "code_reviewer",
                topic=topic,
                exp_plan=exp_plan,
                metric=metric,
                files_context=files_ctx,
            )
            resp = self._chat(sp.system, sp.user, max_tokens=4096)

            review = self._parse_json(resp.content)
            if not isinstance(review, dict) or not review:
                self._log_event(
                    f"  Review round {r + 1}: could not parse JSON, skipping"
                )
                break

            verdict = review.get("verdict", "APPROVE")
            score = review.get("score", 10)
            critical = review.get("critical_issues", [])

            self._log_event(
                f"  Review round {r + 1}: verdict={verdict}, score={score}, "
                f"critical_issues={len(critical)}"
            )

            if verdict == "APPROVE" or not critical:
                break

            # Fix critical issues using the code_generation system prompt
            fix_prompt = (
                "A code reviewer found these critical issues in your experiment code.\n"
                "Fix ALL of them while preserving the experiment design.\n\n"
                "## Critical Issues\n"
                + "\n".join(f"- {issue}" for issue in critical)
                + f"\n\n## Current Code\n{files_ctx}\n\n"
                "Output ALL files in ```filename:xxx.py``` format, "
                "including unchanged files."
            )
            sys_prompt = self._pm.system("code_generation")
            fix_resp = self._chat(sys_prompt, fix_prompt, max_tokens=16384)

            fixed = self._extract_files(fix_resp.content)
            if fixed:
                files = dict(files)
                files.update(fixed)

        return files, rounds

    # ── Helpers ────────────────────────────────────────────────────────────

    def _chat(self, system: str, user: str, max_tokens: int = 8192) -> Any:
        """Make an LLM call and track count."""
        self._calls += 1
        messages = [{"role": "user", "content": user}]
        return self._llm.chat(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
        )

    def _get_or_create_sandbox(self) -> _SandboxLike:
        """Lazily create a single sandbox instance for all validation runs."""
        if self._sandbox is None:
            sandbox_dir = self._stage_dir / "agent_sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            self._sandbox = self._sandbox_factory(
                self._exp_config, sandbox_dir,
            )
        return self._sandbox

    def _run_in_sandbox(
        self,
        files: dict[str, str],
        timeout_sec: int | None = None,
    ) -> Any:
        """Write files to a temp directory and run in sandbox."""
        if not self._sandbox_factory:
            raise RuntimeError("No sandbox factory configured")

        self._runs += 1
        timeout = timeout_sec or self._cfg.exec_fix_timeout_sec

        # Write files to a numbered attempt directory
        run_dir = self._stage_dir / "agent_runs" / f"attempt_{self._runs:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in files.items():
            fpath = run_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")

        # Run using the sandbox
        sandbox = self._get_or_create_sandbox()
        try:
            result = sandbox.run_project(run_dir, timeout_sec=timeout)
        except Exception as exc:
            self._log_event(f"  Sandbox run failed: {exc}")
            result = _SimpleResult(
                returncode=1,
                stdout="",
                stderr=f"Sandbox exception: {exc}",
            )

        return result

    def _extract_files(self, content: str) -> dict[str, str]:
        """Extract multi-file code blocks from LLM output."""
        # Local import to avoid circular dependency with executor.py
        from researchclaw.pipeline.executor import _extract_multi_file_blocks

        return _extract_multi_file_blocks(content)

    @staticmethod
    def _format_files(files: dict[str, str]) -> str:
        """Format files for inclusion in a prompt."""
        parts = []
        for fname in sorted(files):
            parts.append(f"```filename:{fname}\n{files[fname]}\n```")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Best-effort JSON extraction from LLM response.

        BUG-17: Always returns ``dict | None`` — never a bare string or list,
        which would cause ``.get()`` crashes in callers.
        """
        def _as_dict(val: Any) -> dict[str, Any] | None:
            return val if isinstance(val, dict) else None

        # Direct parse
        try:
            return _as_dict(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            pass
        # ```json``` fenced block
        m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if m:
            try:
                return _as_dict(json.loads(m.group(1)))
            except (json.JSONDecodeError, ValueError):
                pass
        # First {...} object
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if m:
            try:
                return _as_dict(json.loads(m.group(0)))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _log_event(self, msg: str) -> None:
        """Log to both Python logger and the internal validation log."""
        logger.info("[CodeAgent] %s", msg)
        self._log.append(msg)


# ---------------------------------------------------------------------------
# Lightweight result stand-in for error plumbing
# ---------------------------------------------------------------------------


@dataclass
class _SimpleResult:
    """Minimal sandbox result for internal error plumbing."""

    returncode: int = 1
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0
    metrics: dict[str, object] = field(default_factory=dict)
    timed_out: bool = False
