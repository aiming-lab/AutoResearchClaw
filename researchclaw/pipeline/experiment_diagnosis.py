"""Experiment Diagnosis Agent — analyzes WHY experiments failed.

Parses experiment artifacts (stdout, stderr, metrics, experiment plan)
to produce a structured failure diagnosis with root cause classification
and concrete repair instructions.

Used by the experiment repair loop (``experiment_repair.py``) to generate
targeted fixes via OpenCode.
"""

from __future__ import annotations

import enum
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class DeficiencyType(enum.Enum):
    """Classification of experiment failure modes."""

    NO_CONDITIONS_COMPLETED = "no_conditions"
    TOO_FEW_CONDITIONS = "few_conditions"
    MISSING_BASELINE = "no_baseline"
    MISSING_PROPOSED = "no_proposed"
    INSUFFICIENT_SEEDS = "few_seeds"
    TIME_GUARD_DOMINANT = "time_guard"
    SYNTHETIC_DATA_FALLBACK = "synthetic_data"
    CODE_CRASH = "code_crash"
    MISSING_DEPENDENCY = "missing_dep"
    HYPERPARAMETER_ISSUE = "bad_hyperparams"
    IDENTICAL_CONDITIONS = "identical_conditions"
    PERMISSION_ERROR = "permission_error"
    DATASET_UNAVAILABLE = "dataset_unavailable"
    GPU_OOM = "gpu_oom"


@dataclass
class Deficiency:
    """A single identified deficiency in the experiment."""

    type: DeficiencyType
    severity: str  # "critical" | "major" | "minor"
    description: str
    affected_conditions: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    error_message: str = ""  # The actual error text from logs


@dataclass
class ExperimentDiagnosis:
    """Structured diagnosis of an experiment run."""

    deficiencies: list[Deficiency] = field(default_factory=list)
    repairable: bool = True
    reason: str = ""  # Why not repairable (if applicable)
    summary: str = ""
    conditions_completed: list[str] = field(default_factory=list)
    conditions_failed: list[str] = field(default_factory=list)
    total_planned: int = 0
    completion_rate: float = 0.0

    def has_critical(self) -> bool:
        return any(d.severity == "critical" for d in self.deficiencies)

    def to_repair_prompt(self) -> str:
        """Generate a structured repair prompt for OpenCode."""
        lines = ["## EXPERIMENT DIAGNOSIS\n"]
        lines.append(f"Completion rate: {self.completion_rate:.0%} "
                     f"({len(self.conditions_completed)}/{self.total_planned} conditions)\n")

        if self.conditions_completed:
            lines.append(f"Completed: {', '.join(self.conditions_completed)}")
        if self.conditions_failed:
            lines.append(f"Failed: {', '.join(self.conditions_failed)}\n")

        lines.append("## DEFICIENCIES (ordered by severity)\n")
        for d in sorted(self.deficiencies, key=lambda x: {"critical": 0, "major": 1, "minor": 2}.get(x.severity, 3)):
            lines.append(f"### [{d.severity.upper()}] {d.type.value}")
            lines.append(f"**Problem**: {d.description}")
            if d.error_message:
                lines.append(f"**Error**: ```{d.error_message[:500]}```")
            if d.affected_conditions:
                lines.append(f"**Affected conditions**: {', '.join(d.affected_conditions)}")
            lines.append(f"**Fix**: {d.suggested_fix}\n")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "deficiencies": [
                {
                    "type": d.type.value,
                    "severity": d.severity,
                    "description": d.description,
                    "affected_conditions": d.affected_conditions,
                    "suggested_fix": d.suggested_fix,
                    "error_message": d.error_message,
                }
                for d in self.deficiencies
            ],
            "repairable": self.repairable,
            "reason": self.reason,
            "summary": self.summary,
            "conditions_completed": self.conditions_completed,
            "conditions_failed": self.conditions_failed,
            "total_planned": self.total_planned,
            "completion_rate": self.completion_rate,
        }


# ---------------------------------------------------------------------------
# Experiment Quality Assessment
# ---------------------------------------------------------------------------


class PaperMode(enum.Enum):
    """Paper writing mode based on experiment quality."""

    FULL_PAPER = "full_paper"
    PRELIMINARY_STUDY = "preliminary_study"
    NEGATIVE_RESULT = "negative_result"
    TECHNICAL_REPORT = "technical_report"


@dataclass
class ExperimentQualityAssessment:
    """Deterministic assessment of experiment readiness for paper writing."""

    sufficient: bool
    mode: PaperMode
    deficiencies: list[Deficiency] = field(default_factory=list)
    repair_possible: bool = True
    diagnosis: ExperimentDiagnosis | None = None


def assess_experiment_quality(
    experiment_summary: dict,
    refinement_log: dict | None = None,
    experiment_plan: dict | None = None,
    *,
    min_conditions: int = 3,
    min_seeds: int = 2,
) -> ExperimentQualityAssessment:
    """Deterministic quality assessment of experiment data.

    Parameters
    ----------
    experiment_summary:
        Parsed ``experiment_summary.json``.
    refinement_log:
        Parsed ``refinement_log.json``.
    experiment_plan:
        Parsed experiment plan (conditions list).
    min_conditions:
        Minimum conditions required for ``full_paper`` mode.
    min_seeds:
        Minimum seeds per condition for ``full_paper`` mode.
    """
    # Run full diagnosis
    stdout = _extract_stdout(experiment_summary, refinement_log)
    stderr = _extract_stderr(experiment_summary, refinement_log)
    diagnosis = diagnose_experiment(
        experiment_summary=experiment_summary,
        refinement_log=refinement_log,
        stdout=stdout,
        stderr=stderr,
        experiment_plan=experiment_plan,
    )

    # Determine paper mode
    mode = _select_paper_mode(experiment_summary, diagnosis, min_conditions, min_seeds)
    sufficient = mode == PaperMode.FULL_PAPER
    repair_possible = not diagnosis.has_critical() or diagnosis.repairable

    return ExperimentQualityAssessment(
        sufficient=sufficient,
        mode=mode,
        deficiencies=diagnosis.deficiencies,
        repair_possible=repair_possible,
        diagnosis=diagnosis,
    )


def _select_paper_mode(
    experiment_summary: dict,
    diagnosis: ExperimentDiagnosis,
    min_conditions: int,
    min_seeds: int,
) -> PaperMode:
    """Select paper mode based on experiment quality."""
    # Check for synthetic data
    if any(d.type == DeficiencyType.SYNTHETIC_DATA_FALLBACK for d in diagnosis.deficiencies):
        return PaperMode.TECHNICAL_REPORT

    # Check for no conditions
    if not diagnosis.conditions_completed:
        return PaperMode.TECHNICAL_REPORT

    # Check if only 1 condition
    if len(diagnosis.conditions_completed) <= 1:
        return PaperMode.PRELIMINARY_STUDY

    # Check for sufficient conditions and seeds
    cond_summaries = experiment_summary.get("condition_summaries", {})
    conditions_with_enough_seeds = 0
    for cond_name in diagnosis.conditions_completed:
        cond_data = cond_summaries.get(cond_name, {})
        n_seeds = cond_data.get("n_seeds", 1)
        # Heuristic: count per-seed keys in best_run metrics
        metrics = experiment_summary.get("best_run", {}).get("metrics", {})
        seed_keys = [k for k in metrics if k.startswith(f"{cond_name}/") and re.match(r".*/\d+/", k)]
        # BUG-R6-05: Guard against re.match returning None
        actual_seeds = len(set(
            m.group(1) for k in seed_keys if (m := re.match(r".*/(\d+)/", k)) is not None
        )) if seed_keys else 1
        if actual_seeds >= min_seeds:
            conditions_with_enough_seeds += 1

    if len(diagnosis.conditions_completed) >= min_conditions and conditions_with_enough_seeds >= min_conditions:
        # Check for negative result
        best_run = experiment_summary.get("best_run", {})
        metrics = best_run.get("metrics", {})
        # Simple heuristic: if primary_metric is very low, might be negative result
        # This is refined by checking baseline vs proposed in the full pipeline
        return PaperMode.FULL_PAPER

    if len(diagnosis.conditions_completed) >= 2:
        return PaperMode.PRELIMINARY_STUDY

    return PaperMode.TECHNICAL_REPORT


# ---------------------------------------------------------------------------
# Core diagnosis logic
# ---------------------------------------------------------------------------


def diagnose_experiment(
    experiment_summary: dict,
    refinement_log: dict | None = None,
    stdout: str = "",
    stderr: str = "",
    experiment_plan: dict | None = None,
    *,
    prior_diagnoses: list[dict] | None = None,
) -> ExperimentDiagnosis:
    """Analyze experiment failures and produce structured diagnosis.

    Parameters
    ----------
    experiment_summary:
        Parsed ``experiment_summary.json``.
    refinement_log:
        Parsed ``refinement_log.json``.
    stdout:
        Combined stdout from experiment execution.
    stderr:
        Combined stderr from experiment execution.
    experiment_plan:
        The designed experiment plan (planned conditions).
    prior_diagnoses:
        Previous diagnosis results (to avoid recommending same fix twice).
    """
    diag = ExperimentDiagnosis()

    # Determine planned vs completed conditions
    planned_conditions = _get_planned_conditions(experiment_plan, experiment_summary)
    completed_conditions = _get_completed_conditions(experiment_summary)
    diag.total_planned = len(planned_conditions)
    diag.conditions_completed = sorted(completed_conditions)
    diag.conditions_failed = sorted(set(planned_conditions) - completed_conditions)
    diag.completion_rate = len(completed_conditions) / max(len(planned_conditions), 1)

    combined_output = stdout + "\n" + stderr

    # --- Pattern-based checks ---

    # 1. Missing dependencies
    _check_missing_deps(diag, combined_output)

    # 2. Permission errors
    _check_permission_errors(diag, combined_output)

    # 3. GPU OOM
    _check_gpu_oom(diag, combined_output)

    # 4. Time guard dominance
    _check_time_guard(diag, combined_output, planned_conditions, completed_conditions)

    # 5. Synthetic data fallback
    _check_synthetic_data(diag, combined_output)

    # 6. Dataset unavailability
    _check_dataset_issues(diag, combined_output)

    # 7. Code crashes
    _check_code_crashes(diag, stderr, combined_output)

    # 8. Hyperparameter issues
    _check_hyperparams(diag, combined_output, experiment_summary)

    # 9. Identical conditions
    _check_identical_conditions(diag, experiment_summary)

    # 10. Insufficient seeds
    _check_insufficient_seeds(diag, experiment_summary)

    # 11. Near-random accuracy (BUG-204)
    _check_near_random_accuracy(diag, experiment_summary)

    # 12. No conditions at all
    if not completed_conditions:
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.NO_CONDITIONS_COMPLETED,
            severity="critical",
            description="No experimental conditions completed successfully.",
            suggested_fix="Fix the root cause errors above, then re-run.",
        ))

    # Determine repairability
    _assess_repairability(diag, prior_diagnoses)

    # Build summary
    diag.summary = (
        f"{len(diag.deficiencies)} deficiency(ies) found. "
        f"{len(completed_conditions)}/{len(planned_conditions)} conditions completed. "
        f"Repairable: {diag.repairable}."
    )

    return diag


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_missing_deps(diag: ExperimentDiagnosis, output: str) -> None:
    """Detect missing Python package errors."""
    pattern = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
    for m in pattern.finditer(output):
        module = m.group(1)
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.MISSING_DEPENDENCY,
            severity="critical",
            description=f"Missing Python package: {module}",
            error_message=m.group(0),
            suggested_fix=f"Add '{module}' to requirements.txt and re-run.",
        ))

    # Also check for missing system libraries (Box2D, etc.)
    if "box2d" in output.lower() or "Box2D" in output:
        if "not available" in output.lower() or "not installed" in output.lower():
            diag.deficiencies.append(Deficiency(
                type=DeficiencyType.MISSING_DEPENDENCY,
                severity="critical",
                description="Box2D library not available — LunarLander environments will fail.",
                suggested_fix="Add 'box2d-py' and 'gymnasium[box2d]' to requirements.txt.",
            ))


def _check_permission_errors(diag: ExperimentDiagnosis, output: str) -> None:
    """Detect file/network permission errors."""
    patterns = [
        (r"PermissionError.*?(?:huggingface|hf|model|download)", "HuggingFace model download blocked"),
        (r"PermissionError", "File permission error"),
        (r"403.*?Forbidden.*?(?:huggingface|hf)", "HuggingFace API access denied"),
    ]
    for pat, desc in patterns:
        if re.search(pat, output, re.IGNORECASE):
            diag.deficiencies.append(Deficiency(
                type=DeficiencyType.PERMISSION_ERROR,
                severity="critical",
                description=desc,
                error_message=_extract_context(output, pat),
                suggested_fix=(
                    "Pre-cache the model in setup.py, or switch to a smaller "
                    "model (e.g., distilgpt2 instead of gpt2). Ensure HF_TOKEN "
                    "is set if using gated models."
                ),
            ))
            break  # One permission error is enough


def _check_gpu_oom(diag: ExperimentDiagnosis, output: str) -> None:
    """Detect GPU out-of-memory errors."""
    if re.search(r"CUDA out of memory|RuntimeError.*?OOM|torch\.cuda\.OutOfMemoryError", output):
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.GPU_OOM,
            severity="major",
            description="GPU out of memory during training.",
            error_message=_extract_context(output, r"CUDA out of memory"),
            suggested_fix=(
                "Reduce batch size by 50%. If still OOM, reduce model size "
                "or use gradient checkpointing."
            ),
        ))


def _check_time_guard(
    diag: ExperimentDiagnosis,
    output: str,
    planned: set[str],
    completed: set[str],
) -> None:
    """Detect time guard killing too many conditions."""
    # Count TIME_GUARD mentions
    time_guard_hits = len(re.findall(r"TIME_GUARD|time.guard|time guard", output, re.IGNORECASE))
    skipped_conditions = planned - completed
    skipped_pct = len(skipped_conditions) / max(len(planned), 1)

    if skipped_pct > 0.5 and len(skipped_conditions) > 1:
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.TIME_GUARD_DOMINANT,
            severity="major",
            description=(
                f"Time guard killed {len(skipped_conditions)}/{len(planned)} conditions "
                f"({skipped_pct:.0%}). Too many conditions for the time budget."
            ),
            affected_conditions=sorted(skipped_conditions),
            suggested_fix=(
                f"Reduce from {len(planned)} to {min(5, len(planned))} conditions "
                f"(keep baseline + proposed + 1 ablation). "
                f"Reduce epochs by 50%. Reduce seeds from 3 to 2."
            ),
        ))


def _check_synthetic_data(diag: ExperimentDiagnosis, output: str) -> None:
    """Detect synthetic/fake data fallback."""
    patterns = [
        r"using synthetic data",
        r"synthetic.*?fallback",
        r"random.*?tokens",
        r"WARNING.*?load failed.*?using",
    ]
    for pat in patterns:
        if re.search(pat, output, re.IGNORECASE):
            diag.deficiencies.append(Deficiency(
                type=DeficiencyType.SYNTHETIC_DATA_FALLBACK,
                severity="critical",
                description="Experiment fell back to synthetic/random data instead of real dataset.",
                error_message=_extract_context(output, pat),
                suggested_fix=(
                    "Fix dataset loading. Use a pre-cached dataset "
                    "(CIFAR-10/MNIST are available at /opt/datasets). "
                    "Ensure download happens in setup.py, not main.py."
                ),
            ))
            break


def _check_dataset_issues(diag: ExperimentDiagnosis, output: str) -> None:
    """Detect dataset loading failures."""
    patterns = [
        (r"FileNotFoundError.*?(?:dataset|data|csv|json)", "Dataset file not found"),
        (r"No such file.*?(?:dataset|data|train|test)", "Dataset path does not exist"),
        # BUG-203: HuggingFace DatasetNotFoundError (e.g. cifar10_corrupted)
        (r"DatasetNotFoundError.*?doesn't exist", "HuggingFace dataset not found on Hub"),
    ]
    for pat, desc in patterns:
        if re.search(pat, output, re.IGNORECASE):
            diag.deficiencies.append(Deficiency(
                type=DeficiencyType.DATASET_UNAVAILABLE,
                severity="critical",
                description=desc,
                error_message=_extract_context(output, pat),
                suggested_fix=(
                    "The dataset does not exist on HuggingFace Hub. "
                    "Use ONLY pre-cached datasets: CIFAR-10, CIFAR-100, MNIST, "
                    "FashionMNIST, STL-10 (available at /opt/datasets). "
                    "Remove the failing download from setup.py and use "
                    "torchvision.datasets with root='/opt/datasets' instead."
                ),
            ))


def _check_code_crashes(diag: ExperimentDiagnosis, stderr: str, output: str) -> None:
    """Detect Python runtime crashes."""
    # Look for tracebacks — use MULTILINE, not DOTALL, so each traceback
    # is matched independently (DOTALL would eat all tracebacks into one).
    tb_pattern = re.compile(
        r"(?:Error|Exception):\s*(.+)$",
        re.MULTILINE,
    )
    seen_errors: set[str] = set()
    for m in tb_pattern.finditer(output):
        error_msg = m.group(1).strip()[:200]
        # Skip if already handled by more specific checks
        if "ModuleNotFoundError" in error_msg:
            continue
        if "PermissionError" in error_msg:
            continue
        if "CUDA out of memory" in error_msg:
            continue
        if "DatasetNotFoundError" in error_msg:
            continue
        if error_msg in seen_errors:
            continue
        seen_errors.add(error_msg)
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.CODE_CRASH,
            severity="major",
            description=f"Runtime error: {error_msg}",
            error_message=m.group(0)[:500],
            suggested_fix="Fix the code error. See traceback for details.",
        ))


def _check_hyperparams(diag: ExperimentDiagnosis, output: str, summary: dict) -> None:
    """Detect hyperparameter issues (diverging loss, NaN gradients)."""
    # NaN in training — use word boundary to avoid matching "Shannan" etc.
    if re.search(r"loss.*?\bnan\b|\bnan\b.*?loss|gradient.*?\bnan\b", output, re.IGNORECASE):
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.HYPERPARAMETER_ISSUE,
            severity="major",
            description="NaN detected in loss or gradients — likely learning rate too high.",
            suggested_fix="Reduce learning rate by 10×. Add gradient clipping (max_norm=1.0).",
        ))

    # Diverging loss — parse each value individually so one malformed value
    # doesn't silence the entire check (EDGE-1 fix)
    loss_values = re.findall(r"loss[=:]\s*([\d.]+)", output, re.IGNORECASE)
    if loss_values:
        losses: list[float] = []
        for v in loss_values[-10:]:
            try:
                losses.append(float(v))
            except (ValueError, TypeError):
                continue
        if losses and any(l > 100 for l in losses):
            diag.deficiencies.append(Deficiency(
                type=DeficiencyType.HYPERPARAMETER_ISSUE,
                severity="major",
                description=f"Loss diverging (max={max(losses):.1f}). Training is unstable.",
                suggested_fix="Reduce learning rate. Add gradient clipping. Check data normalization.",
            ))


def _check_near_random_accuracy(diag: ExperimentDiagnosis, summary: dict) -> None:
    """BUG-204: Detect when all conditions produce near-random accuracy.

    If the metric name suggests accuracy/top-1 and the best value is below
    15%, the model likely isn't learning (wrong LR, broken forward pass, etc.).
    """
    ms = summary.get("metrics_summary", {})
    if not ms:
        return

    # Find accuracy-like metrics
    _ACC_KEYS = {"accuracy", "acc", "top1", "top1_accuracy", "val_acc", "test_acc"}
    best_acc: float | None = None
    acc_key: str = ""
    for key, val in ms.items():
        key_lower = key.lower().split("/")[-1]  # strip condition prefix
        if key_lower in _ACC_KEYS or "accuracy" in key_lower or "top1" in key_lower:
            v = val.get("max", val) if isinstance(val, dict) else val
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if best_acc is None or fv > best_acc:
                best_acc = fv
                acc_key = key

    if best_acc is not None and 0 < best_acc < 15.0:
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.HYPERPARAMETER_ISSUE,
            severity="critical",
            description=(
                f"Best accuracy is {best_acc:.1f}% ({acc_key}), near random chance. "
                f"The model is not learning."
            ),
            suggested_fix=(
                "Check: (1) Learning rate too high/low — try 0.001 for Adam, 0.1 for SGD. "
                "(2) Data preprocessing — normalize to [0,1] or ImageNet stats. "
                "(3) Forward pass — ensure loss backward reaches all parameters. "
                "(4) KD — ensure teacher is loaded with correct pretrained weights."
            ),
        ))


def _check_identical_conditions(diag: ExperimentDiagnosis, summary: dict) -> None:
    """Detect ablation conditions producing identical results."""
    warnings = summary.get("ablation_warnings", [])
    if warnings:
        affected = []
        for w in warnings:
            m = re.search(r"Conditions '([^']+)' and '([^']+)'", w)
            if m:
                affected.extend([m.group(1), m.group(2)])
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.IDENTICAL_CONDITIONS,
            severity="major",
            description=(
                f"{len(warnings)} ablation pair(s) produce identical outputs. "
                "The differentiating parameter is likely not wired into the code."
            ),
            affected_conditions=sorted(set(affected)),
            suggested_fix=(
                "Check that each ablation condition actually modifies the model/training. "
                "The condition parameter must affect the forward pass, not just be logged."
            ),
        ))


def _check_insufficient_seeds(diag: ExperimentDiagnosis, summary: dict) -> None:
    """Check if completed conditions have too few seeds."""
    metrics = summary.get("best_run", {}).get("metrics", {})
    seed_pattern = re.compile(r"^(.+)/(\d+)/(.+)$")
    cond_seeds: dict[str, set[int]] = {}
    for key in metrics:
        m = seed_pattern.match(key)
        if m:
            cond_name, seed_str = m.group(1), m.group(2)
            cond_seeds.setdefault(cond_name, set()).add(int(seed_str))

    single_seed_conds = [c for c, seeds in cond_seeds.items() if len(seeds) < 2]
    if single_seed_conds:
        diag.deficiencies.append(Deficiency(
            type=DeficiencyType.INSUFFICIENT_SEEDS,
            severity="minor",
            description=f"{len(single_seed_conds)} condition(s) have only 1 seed (no variance estimate).",
            affected_conditions=single_seed_conds,
            suggested_fix="Increase seeds to at least 2 per condition, or reduce epoch count to fit time budget.",
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_planned_conditions(plan: dict | None, summary: dict) -> set[str]:
    """Extract planned condition names from experiment plan or summary."""
    if plan:
        conditions = plan.get("conditions", [])
        if isinstance(conditions, list):
            return {c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in conditions}
    # Fallback: use condition_summaries from summary
    return set(summary.get("condition_summaries", {}).keys())


def _get_completed_conditions(summary: dict) -> set[str]:
    """Extract conditions that actually produced metrics."""
    completed = set()
    cond_summaries = summary.get("condition_summaries", {})
    for cond_name, data in cond_summaries.items():
        metrics = data.get("metrics", {})
        if metrics and any(
            isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values()
        ):
            completed.add(cond_name)
    return completed


def _extract_stdout(summary: dict, ref_log: dict | None) -> str:
    """Extract combined stdout from experiment artifacts."""
    parts: list[str] = []
    # From best_run
    stdout = summary.get("best_run", {}).get("stdout", "")
    if stdout:
        parts.append(stdout)
    # From refinement log iterations
    if ref_log:
        for it in ref_log.get("iterations", []):
            for key in ("sandbox", "sandbox_after_fix"):
                sb = it.get(key, {})
                if isinstance(sb, dict):
                    out = sb.get("stdout", "")
                    if out:
                        parts.append(out)
    return "\n".join(parts)


def _extract_stderr(summary: dict, ref_log: dict | None) -> str:
    """Extract combined stderr from experiment artifacts."""
    parts: list[str] = []
    stderr = summary.get("best_run", {}).get("stderr", "")
    if stderr:
        parts.append(stderr)
    if ref_log:
        for it in ref_log.get("iterations", []):
            for key in ("sandbox", "sandbox_after_fix"):
                sb = it.get(key, {})
                if isinstance(sb, dict):
                    err = sb.get("stderr", "")
                    if err:
                        parts.append(err)
    return "\n".join(parts)


def _extract_context(text: str, pattern: str, context_chars: int = 200) -> str:
    """Extract surrounding context for an error pattern match."""
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return ""
    start = max(0, m.start() - context_chars // 2)
    end = min(len(text), m.end() + context_chars // 2)
    return text[start:end].strip()


def _assess_repairability(diag: ExperimentDiagnosis, prior: list[dict] | None) -> None:
    """Determine if the experiment can be repaired."""
    if not diag.deficiencies:
        diag.repairable = True
        return

    # Count how many times we've tried to fix the same issues
    if prior:
        prior_types = set()
        for pd in prior:
            for d in pd.get("deficiencies", []):
                prior_types.add(d.get("type", ""))
        current_types = {d.type.value for d in diag.deficiencies}
        repeated = current_types & prior_types
        if len(repeated) >= 3:
            diag.repairable = False
            diag.reason = f"Same deficiencies recur after {len(prior)} repair cycles: {repeated}"
            return

    # All types are potentially repairable
    diag.repairable = True
