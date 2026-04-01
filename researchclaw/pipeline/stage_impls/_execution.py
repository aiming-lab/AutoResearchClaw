"""Stages 11-13: Resource planning, experiment execution, and iterative refinement."""

from __future__ import annotations

import json
import logging
import math
import re
import time as _time
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.validator import (
    CodeValidation,
    format_issues_for_llm,
    validate_code,
)
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _chat_with_prompt,
    _detect_runtime_issues,
    _ensure_sandbox_deps,
    _extract_code_block,
    _extract_multi_file_blocks,
    _get_evolution_overlay,
    _load_hardware_profile,
    _parse_metrics_from_stdout,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _utcnow_iso,
    _write_stage_meta,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_resource_planning(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    schedule: dict[str, Any] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "resource_planning")
        sp = _pm.for_stage("resource_planning", evolution_overlay=_overlay, exp_plan=exp_plan)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        parsed = _safe_json_loads(resp.content, {})
        if isinstance(parsed, dict):
            schedule = parsed
    if schedule is None:
        schedule = {
            "tasks": [
                {
                    "id": "baseline",
                    "name": "Run baseline",
                    "depends_on": [],
                    "gpu_count": 1,
                    "estimated_minutes": 20,
                    "priority": "high",
                },
                {
                    "id": "proposed",
                    "name": "Run proposed method",
                    "depends_on": ["baseline"],
                    "gpu_count": 1,
                    "estimated_minutes": 30,
                    "priority": "high",
                },
            ],
            "total_gpu_budget": 1,
            "generated": _utcnow_iso(),
        }
    schedule.setdefault("generated", _utcnow_iso())
    (stage_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    return StageResult(
        stage=Stage.RESOURCE_PLANNING,
        status=StageStatus.DONE,
        artifacts=("schedule.json",),
        evidence_refs=("stage-11/schedule.json",),
    )


def _execute_experiment_run(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.experiment.factory import create_sandbox
    from researchclaw.experiment.runner import ExperimentRunner

    schedule_text = _read_prior_artifact(run_dir, "schedule.json") or "{}"
    # Try multi-file experiment directory first, fall back to single file
    exp_dir_path = _read_prior_artifact(run_dir, "experiment/")
    code_text = ""
    if exp_dir_path and Path(exp_dir_path).is_dir():
        main_path = Path(exp_dir_path) / "main.py"
        if main_path.exists():
            try:
                code_text = main_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                code_text = ""
    if not code_text:
        code_text = _read_prior_artifact(run_dir, "experiment.py") or ""

    runs_dir = stage_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    mode = config.experiment.mode
    if mode in ("sandbox", "docker"):
        # P7: Auto-install missing dependencies before subprocess sandbox
        if mode == "sandbox":
            _all_code = code_text
            if exp_dir_path and Path(exp_dir_path).is_dir():
                for _pyf in Path(exp_dir_path).glob("*.py"):
                    try:
                        _all_code += "\n" + _pyf.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
            _ensure_sandbox_deps(_all_code, config.experiment.sandbox.python_path)

        sandbox = create_sandbox(config.experiment, runs_dir / "sandbox")
        # Use run_project for multi-file, run for single-file
        if exp_dir_path and Path(exp_dir_path).is_dir():
            result = sandbox.run_project(
                Path(exp_dir_path), timeout_sec=config.experiment.time_budget_sec
            )
        else:
            result = sandbox.run(
                code_text, timeout_sec=config.experiment.time_budget_sec
            )
        # Try to read structured results.json from sandbox working dir
        structured_results: dict[str, Any] | None = None
        sandbox_project = runs_dir / "sandbox" / "_project"
        results_json_path = sandbox_project / "results.json"
        if results_json_path.exists():
            try:
                structured_results = json.loads(
                    results_json_path.read_text(encoding="utf-8")
                )
                # Copy results.json to runs dir for easy access
                (runs_dir / "results.json").write_text(
                    results_json_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, OSError):
                structured_results = None

        # If sandbox metrics are empty, try to parse from stdout
        effective_metrics = result.metrics
        if not effective_metrics and result.stdout:
            effective_metrics = _parse_metrics_from_stdout(result.stdout)

        # Determine run status: completed / partial (timed out with data) / failed
        # R6-2: Detect stdout failure signals even when exit code is 0
        _stdout_has_failure = bool(
            result.stdout
            and not effective_metrics
            and any(
                sig in result.stdout
                for sig in ("FAIL:", "NaN/divergence", "Traceback (most recent")
            )
        )
        if result.returncode == 0 and not result.timed_out and not _stdout_has_failure:
            run_status = "completed"
        elif result.timed_out and effective_metrics:
            run_status = "partial"
            logger.warning(
                "Experiment timed out but captured %d partial metrics",
                len(effective_metrics),
            )
        else:
            run_status = "failed"
            if _stdout_has_failure:
                logger.warning(
                    "Experiment exited cleanly but stdout contains failure signals"
                )

        # P1: Warn if experiment completed suspiciously fast (trivially easy benchmark)
        if run_status == "completed" and result.elapsed_sec and result.elapsed_sec < 5.0:
            logger.warning(
                "Stage 12: Experiment completed in %.2fs — benchmark may be trivially easy. "
                "Consider increasing task difficulty.",
                result.elapsed_sec,
            )

        run_payload: dict[str, Any] = {
            "run_id": "run-1",
            "task_id": "sandbox-main",
            "status": run_status,
            "metrics": effective_metrics,
            "elapsed_sec": result.elapsed_sec,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "completed_at": _utcnow_iso(),
        }
        if structured_results is not None:
            run_payload["structured_results"] = structured_results
        # Auto-generate results.json from parsed metrics if sandbox didn't produce one
        if structured_results is None and effective_metrics:
            auto_results = {"source": "stdout_parsed", "metrics": effective_metrics}
            (runs_dir / "results.json").write_text(
                json.dumps(auto_results, indent=2), encoding="utf-8"
            )
            logger.info("Stage 12: Auto-generated results.json from stdout metrics (%d keys)", len(effective_metrics))
        (runs_dir / "run-1.json").write_text(
            json.dumps(run_payload, indent=2), encoding="utf-8"
        )

        # R11-6: Time budget adequacy check
        if result.timed_out or (result.elapsed_sec and result.elapsed_sec > config.experiment.time_budget_sec * 0.9):
            # Parse stdout to estimate how many conditions/seeds completed
            _stdout = result.stdout or ""
            _completed_conditions = set()
            _completed_seeds = 0
            for _line in _stdout.splitlines():
                if "condition=" in _line and "seed=" in _line:
                    _completed_seeds += 1
                    _cond_match = re.match(r".*condition=(\S+)", _line)
                    if _cond_match:
                        _completed_conditions.add(_cond_match.group(1))
            _time_budget_warning = {
                "timed_out": result.timed_out,
                "elapsed_sec": result.elapsed_sec,
                "budget_sec": config.experiment.time_budget_sec,
                "conditions_completed": sorted(_completed_conditions),
                "total_seed_runs": _completed_seeds,
                "warning": (
                    f"Experiment used {result.elapsed_sec:.0f}s of "
                    f"{config.experiment.time_budget_sec}s budget. "
                    f"Only {len(_completed_conditions)} conditions completed "
                    f"({_completed_seeds} seed-runs). Consider increasing "
                    f"time_budget_sec for more complete results."
                ),
            }
            logger.warning(
                "Stage 12: %s", _time_budget_warning["warning"]
            )
            (stage_dir / "time_budget_warning.json").write_text(
                json.dumps(_time_budget_warning, indent=2), encoding="utf-8"
            )

        # FIX-8: Validate seed count from structured results
        if structured_results and isinstance(structured_results, dict):
            _sr_conditions = structured_results.get("conditions", structured_results.get("per_condition", {}))
            if isinstance(_sr_conditions, dict):
                for _cname, _cdata in _sr_conditions.items():
                    if isinstance(_cdata, dict):
                        _seeds_run = _cdata.get("seeds_run", _cdata.get("n_seeds", 0))
                        if isinstance(_seeds_run, (int, float)) and 0 < _seeds_run < 3:
                            logger.warning(
                                "Stage 12: Condition '%s' ran only %d seed(s) — "
                                "minimum 3 required for statistical validity",
                                _cname, int(_seeds_run),
                            )

    elif mode == "simulated":
        schedule = _safe_json_loads(schedule_text, {})
        tasks = schedule.get("tasks", []) if isinstance(schedule, dict) else []
        if not isinstance(tasks, list):
            tasks = []
        for idx, task in enumerate(tasks or [{"id": "task-1", "name": "simulated"}]):
            task_id = (
                str(task.get("id", f"task-{idx + 1}"))
                if isinstance(task, dict)
                else f"task-{idx + 1}"
            )
            payload = {
                "run_id": f"run-{idx + 1}",
                "task_id": task_id,
                "status": "simulated",
                "key_metrics": {
                    config.experiment.metric_key: round(0.3 + idx * 0.03, 4),
                    "secondary_metric": round(0.6 - idx * 0.04, 4),
                },
                "notes": "Simulated run result",
                "completed_at": _utcnow_iso(),
            }
            run_id = str(payload["run_id"])
            (runs_dir / f"{_safe_filename(run_id)}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    else:
        runner = ExperimentRunner(config.experiment, runs_dir / "workspace")
        history = runner.run_loop(code_text, run_id=f"exp-{run_dir.name}", llm=llm)
        runner.save_history(stage_dir / "experiment_history.json")
        for item in history.results:
            payload = {
                "run_id": f"run-{item.iteration}",
                "task_id": item.run_id,
                "status": "completed" if item.error is None else "failed",
                "metrics": item.metrics,
                "primary_metric": item.primary_metric,
                "improved": item.improved,
                "kept": item.kept,
                "elapsed_sec": item.elapsed_sec,
                "error": item.error,
                "completed_at": _utcnow_iso(),
            }
            run_id = str(payload["run_id"])
            (runs_dir / f"{_safe_filename(run_id)}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    return StageResult(
        stage=Stage.EXPERIMENT_RUN,
        status=StageStatus.DONE,
        artifacts=("runs/",),
        evidence_refs=("stage-12/runs/",),
    )


def _execute_iterative_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.experiment.factory import create_sandbox
    from researchclaw.experiment.validator import format_issues_for_llm, validate_code

    def _to_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            f = float(value)
            # BUG-EX-01: NaN/Inf block all future improvement detection
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    # R10-Fix3: Skip iterative refinement in simulated mode (no real execution)
    if config.experiment.mode == "simulated":
        logger.info(
            "Stage 13: Skipping iterative refinement in simulated mode "
            "(no real code execution available)"
        )
        import shutil

        final_dir = stage_dir / "experiment_final"
        # Copy latest experiment code as final (directory or single file)
        copied = False
        for stage_num in (12, 10):
            src_dir = run_dir / f"stage-{stage_num:02d}" / "experiment"
            if src_dir.is_dir():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(src_dir, final_dir)
                copied = True
                break
            # Also check for single experiment.py
            src_file = run_dir / f"stage-{stage_num:02d}" / "experiment.py"
            if src_file.is_file():
                (stage_dir / "experiment_final.py").write_text(
                    src_file.read_text(encoding="utf-8"), encoding="utf-8"
                )
                copied = True
                break

        log: dict[str, Any] = {
            "generated": _utcnow_iso(),
            "mode": "simulated",
            "skipped": True,
            "skip_reason": "Iterative refinement not meaningful in simulated mode",
            "metric_key": config.experiment.metric_key,
        }
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=("refinement_log.json",),
            evidence_refs=(),
        )

    metric_key = config.experiment.metric_key
    metric_direction = config.experiment.metric_direction

    # P9: Detect metric direction mismatch between config and experiment code.
    # The code-gen stage instructs experiments to print a line like:
    #   METRIC_DEF: primary_metric | direction=higher | desc=...
    # Log a warning if mismatch is detected, but trust the config value
    # (BUG-06 fix: no longer auto-override, since Stage 9 and 12 now
    # explicitly enforce config.metric_direction in prompts).
    _runs_dir_detect = _read_prior_artifact(run_dir, "runs/")
    if _runs_dir_detect and Path(_runs_dir_detect).is_dir():
        import re as _re_detect

        for _rf in sorted(Path(_runs_dir_detect).glob("*.json"))[:5]:
            try:
                _rp = _safe_json_loads(_rf.read_text(encoding="utf-8"), {})
                _stdout = _rp.get("stdout", "") if isinstance(_rp, dict) else ""
                _match = _re_detect.search(
                    r"METRIC_DEF:.*direction\s*=\s*(higher|lower)", _stdout
                )
                if _match:
                    _detected = _match.group(1)
                    _detected_dir = "maximize" if _detected == "higher" else "minimize"
                    if _detected_dir != metric_direction:
                        logger.warning(
                            "P9: Metric direction mismatch — config says '%s' but "
                            "experiment code declares 'direction=%s'. "
                            "Keeping config value '%s'. Code will be "
                            "corrected in next refinement cycle.",
                            metric_direction,
                            _detected,
                            metric_direction,
                        )
                    break
            except OSError:
                pass

    maximize = metric_direction == "maximize"

    def _is_better(candidate: float | None, current: float | None) -> bool:
        if candidate is None:
            return False
        if current is None:
            return True
        return candidate > current if maximize else candidate < current

    def _find_metric(metrics: dict[str, object], key: str) -> float | None:
        """R13-4: Find metric value with fuzzy key matching.

        Tries exact match first, then looks for aggregate keys that contain
        the metric name (e.g. 'primary_metric_mean' when key='primary_metric').
        """
        # Exact match
        val = _to_float(metrics.get(key))
        if val is not None:
            return val
        # Try aggregate/mean keys containing the metric name
        # Prefer keys ending with the metric name or containing '_mean'
        candidates: list[tuple[str, float]] = []
        for mk, mv in metrics.items():
            fv = _to_float(mv)
            if fv is None:
                continue
            if mk == key or mk.endswith(f"/{key}"):
                return fv  # Exact match via condition prefix
            if key in mk and ("mean" in mk or "avg" in mk):
                candidates.append((mk, fv))
            elif mk.endswith(f"_{key}") or mk.endswith(f"/{key}_mean"):
                candidates.append((mk, fv))
        if candidates:
            # Take the aggregate mean if available, otherwise first match
            for ck, cv in candidates:
                if "mean" in ck:
                    return cv
            return candidates[0][1]
        # Last resort: if there's an "overall" or root-level aggregate
        for mk, mv in metrics.items():
            fv = _to_float(mv)
            if fv is not None and key in mk and "/" not in mk and "seed" not in mk:
                return fv
        return None

    requested_iterations = int(getattr(config.experiment, "max_iterations", 10) or 10)
    max_iterations = max(1, min(requested_iterations, 10))

    # BUG-57: Wall-clock time cap for the entire refinement stage.
    # Default: 3× the per-iteration time budget (e.g., 2400s → 7200s = 2h).
    import time as _time_bug57
    _refine_start_time = _time_bug57.monotonic()
    _per_iter_budget = int(getattr(config.experiment, "time_budget_sec", 2400) or 2400)
    _max_refine_wall_sec = int(
        getattr(config.experiment, "max_refine_duration_sec", 0) or 0
    ) or int(_per_iter_budget * 1.5)

    # --- Collect baseline metrics from prior runs ---
    runs_dir_path: Path | None = None
    runs_dir_text = _read_prior_artifact(run_dir, "runs/")
    if runs_dir_text:
        runs_dir_path = Path(runs_dir_text)

    run_summaries: list[str] = []
    baseline_metric: float | None = None
    if runs_dir_path is not None:
        for run_file in sorted(runs_dir_path.glob("*.json"))[:40]:
            payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if not isinstance(payload, dict):
                continue
            # R5-5: Truncate stdout/stderr for context efficiency
            summary = dict(payload)
            if "stdout" in summary and isinstance(summary["stdout"], str):
                lines = summary["stdout"].splitlines()
                if len(lines) > 30:
                    summary["stdout"] = (
                        f"[...truncated {len(lines) - 30} lines...]\n"
                        + "\n".join(lines[-30:])
                    )
                if len(summary["stdout"]) > 2000:
                    summary["stdout"] = summary["stdout"][-2000:]
            if "stderr" in summary and isinstance(summary["stderr"], str):
                lines = summary["stderr"].splitlines()
                if len(lines) > 50:
                    summary["stderr"] = "\n".join(lines[-50:])
                if len(summary["stderr"]) > 2000:
                    summary["stderr"] = summary["stderr"][-2000:]
            run_summaries.append(json.dumps(summary, ensure_ascii=False))
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                metrics = (
                    payload.get("key_metrics")
                    if isinstance(payload.get("key_metrics"), dict)
                    else {}
                )
            metric_val = (
                _find_metric(metrics, metric_key)
                if isinstance(metrics, dict)
                else None
            )
            if metric_val is None:
                metric_val = _to_float(payload.get("primary_metric"))
            if _is_better(metric_val, baseline_metric):
                baseline_metric = metric_val

    # --- Read experiment project (multi-file or single-file) ---
    # BUG-58: When PIVOT rolls back to Stage 13, prefer the best refined code
    # from a previous cycle (stage-13_vX/experiment_final/) over the original
    # unrefined code (stage-12/experiment/ or stage-10/experiment/).
    # Enhanced: try ALL versioned directories (latest first) with fallback chain.
    exp_dir_text: str | None = None
    _prev_refine_dirs = sorted(
        run_dir.glob("stage-13_v*/experiment_final"),
        key=lambda p: p.parent.name,
        reverse=True,  # latest version first
    )
    # BUG-58 fix: Find the best version across ALL cycles (not just latest)
    _best_prev_metric: float | None = None
    _best_prev_dir: Path | None = None
    for _prd in _prev_refine_dirs:
        if not _prd.is_dir():
            continue
        _prd_log = _prd.parent / "refinement_log.json"
        if _prd_log.is_file():
            _prd_data = _safe_json_loads(
                _prd_log.read_text(encoding="utf-8"), {}
            )
            _prd_metric = _prd_data.get("best_metric") if isinstance(_prd_data, dict) else None
            if isinstance(_prd_metric, (int, float)) and _is_better(_prd_metric, _best_prev_metric):
                _best_prev_metric = _prd_metric
                _best_prev_dir = _prd
        elif _best_prev_dir is None:
            # No log but directory exists — use as fallback
            _best_prev_dir = _prd
    if _best_prev_dir is not None:
        exp_dir_text = str(_best_prev_dir)
        logger.info(
            "BUG-58: Recovered best refined code from PIVOT cycle: %s (metric=%s)",
            _best_prev_dir.parent.name,
            f"{_best_prev_metric:.4f}" if _best_prev_metric is not None else "N/A",
        )
    if not exp_dir_text:
        exp_dir_text = _read_prior_artifact(run_dir, "experiment/")
    best_files: dict[str, str] = {}
    if exp_dir_text and Path(exp_dir_text).is_dir():
        # BUG-EX-02: Load ALL text files (not just .py) — requirements.txt,
        # setup.py, config files are needed for Docker sandbox phases.
        for src_file in sorted(Path(exp_dir_text).iterdir()):
            if src_file.is_file() and src_file.suffix in (
                ".py", ".txt", ".yaml", ".yml", ".json", ".cfg", ".ini", ".sh",
            ):
                try:
                    best_files[src_file.name] = src_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # skip binary files
    if not best_files:
        # Backward compat: single experiment.py
        original_code = _read_prior_artifact(run_dir, "experiment.py") or ""
        if original_code:
            best_files = {"main.py": original_code}

    # --- Detect if prior experiment timed out ---
    prior_timed_out = False
    prior_time_budget = config.experiment.time_budget_sec
    if runs_dir_path is not None:
        for run_file in sorted(runs_dir_path.glob("*.json"))[:5]:
            try:
                payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
                if isinstance(payload, dict) and payload.get("timed_out"):
                    prior_timed_out = True
                    break
            except OSError:
                pass

    best_metric = baseline_metric
    best_version = "experiment/"
    # BUG-58: Recover best_metric from best previous PIVOT cycle
    if _best_prev_metric is not None and _is_better(_best_prev_metric, best_metric):
        best_metric = _best_prev_metric
        logger.info(
            "BUG-58: Recovered best_metric=%.4f from previous PIVOT",
            best_metric,
        )
    no_improve_streak = 0
    consecutive_no_metrics = 0

    log: dict[str, Any] = {
        "generated": _utcnow_iso(),
        "mode": config.experiment.mode,
        "metric_key": metric_key,
        "metric_direction": metric_direction,
        "max_iterations_requested": requested_iterations,
        "max_iterations_executed": max_iterations,
        "baseline_metric": baseline_metric,
        "project_files": list(best_files.keys()),
        "iterations": [],
        "converged": False,
        "stop_reason": "max_iterations_reached",
    }

    # --- Helper: write files to a directory ---
    def _write_project(target_dir: Path, project_files: dict[str, str]) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in project_files.items():
            (target_dir / fname).write_text(code, encoding="utf-8")

    # --- Helper: format all files for LLM context ---
    def _files_to_context(project_files: dict[str, str]) -> str:
        parts = []
        for fname, code in sorted(project_files.items()):
            parts.append(f"```filename:{fname}\n{code}\n```")
        return "\n\n".join(parts)

    def _write_refinement_log() -> None:
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )

    def _pause_refinement(
        *,
        reason: str,
        stop_reason: str,
        iteration: int | None = None,
    ) -> StageResult:
        log.update(
            {
                "paused": True,
                "converged": False,
                "stop_reason": stop_reason,
                "pause_reason": reason,
                "best_metric": best_metric,
                "best_version": best_version,
                "iterations_completed": len(log["iterations"]),
            }
        )
        if iteration is not None:
            log["pause_iteration"] = iteration
        _write_refinement_log()
        artifacts = ("refinement_log.json",)
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.PAUSED,
            artifacts=artifacts,
            error=reason,
            decision="resume",
            evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
        )

    if llm is None:
        logger.info("Stage 13: LLM unavailable, saving original experiment as final")
        final_dir = stage_dir / "experiment_final"
        _write_project(final_dir, best_files)
        # Backward compat
        if "main.py" in best_files:
            (stage_dir / "experiment_final.py").write_text(
                best_files["main.py"], encoding="utf-8"
            )
        log.update(
            {
                "converged": True,
                "stop_reason": "llm_unavailable",
                "best_metric": best_metric,
                "best_version": "experiment_final/",
                "iterations": [
                    {
                        "iteration": 0,
                        "version_dir": "experiment_final/",
                        "source": "fallback_original",
                        "metric": best_metric,
                    }
                ],
            }
        )
        _write_refinement_log()
        artifacts = ("refinement_log.json", "experiment_final/")
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=artifacts,
            evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
        )

    _pm = prompts or PromptManager()
    timeout_refine_attempts = 0

    # R7-3: Read experiment plan to detect condition coverage gaps
    _exp_plan_text = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    _condition_coverage_hint = ""
    if _exp_plan_text and run_summaries:
        # Check if stdout contains condition labels
        _all_stdout = " ".join(run_summaries)
        _has_condition_labels = "condition=" in _all_stdout
        if not _has_condition_labels and _exp_plan_text.strip():
            _condition_coverage_hint = (
                "\nCONDITION COVERAGE GAP DETECTED:\n"
                "The experiment plan specifies multiple conditions/treatments, "
                "but the output contains NO condition labels (no 'condition=...' in stdout).\n"
                "You MUST:\n"
                "1. Run ALL conditions/treatments from the experiment plan independently\n"
                "2. Label each metric output: `condition=<name> {metric_key}: <value>`\n"
                "3. Print a SUMMARY line comparing all conditions after completion\n"
                "This is the MOST IMPORTANT improvement — a single unlabeled metric stream "
                "cannot support any comparative conclusions.\n\n"
            )
            logger.info(
                "Stage 13: condition coverage gap detected, injecting multi-condition hint"
            )

    # P1: Track metrics history for saturation detection
    _metrics_history: list[float | None] = [baseline_metric]

    for iteration in range(1, max_iterations + 1):
        # BUG-57: Check wall-clock time before starting a new iteration
        _elapsed = _time_bug57.monotonic() - _refine_start_time
        if _elapsed > _max_refine_wall_sec:
            logger.warning(
                "Stage 13: Wall-clock time cap reached (%.0fs > %ds). "
                "Stopping refinement after %d iterations.",
                _elapsed, _max_refine_wall_sec, iteration - 1,
            )
            log["stop_reason"] = "wall_clock_time_cap"
            break
        logger.info("Stage 13: refinement iteration %d/%d (%.0fs elapsed, cap %ds)",
                    iteration, max_iterations, _elapsed, _max_refine_wall_sec)

        # P1: Detect metric saturation and inject difficulty upgrade hint
        _saturation_hint = ""
        _valid_metrics = [m for m in _metrics_history if m is not None]
        if len(_valid_metrics) >= 2:
            _last_two = _valid_metrics[-2:]
            _saturated = False
            # Use relative change rate instead of hard-coded thresholds
            _change_rate = abs(_last_two[-1] - _last_two[-2]) / max(abs(_last_two[-2]), 1e-8)
            if metric_direction == "minimize":
                _saturated = all(m <= 0.001 for m in _last_two) or (
                    _change_rate < 0.001 and _last_two[-1] < 0.01
                )
            else:
                _saturated = all(m >= 0.999 for m in _last_two) or (
                    _change_rate < 0.001 and _last_two[-1] > 0.99
                )
            if _saturated:
                _saturation_hint = (
                    "\n\nWARNING — BENCHMARK SATURATION DETECTED:\n"
                    "All methods achieve near-perfect scores, making the task too easy "
                    "to discriminate between methods.\n"
                    "YOU MUST increase benchmark difficulty in this iteration:\n"
                    "1. Increase the number of actions/decisions from 8 to at least 20\n"
                    "2. Increase the horizon from 12-18 to at least 50-100 steps\n"
                    "3. Increase noise level to at least 0.3-0.5\n"
                    "4. Add partial observability (agent cannot see full state)\n"
                    "5. Add delayed rewards (reward only at episode end)\n"
                    "6. Ensure random search achieves < 50% success rate\n"
                    "Without this change, the experiment produces meaningless results.\n"
                )
                logger.warning("Stage 13: metric saturation detected, injecting difficulty upgrade hint")

        files_context = _files_to_context(best_files)
        # BUG-10 fix: anchor refinement to original experiment plan
        _exp_plan_anchor = ""
        if _exp_plan_text.strip():
            _exp_plan_anchor = (
                "Original experiment plan (exp_plan.yaml):\n"
                "```yaml\n" + _exp_plan_text[:4000] + "\n```\n"
                "You MUST preserve ALL condition names from this plan.\n\n"
            )
        ip = _pm.sub_prompt(
            "iterative_improve",
            metric_key=metric_key,
            metric_direction=metric_direction,
            files_context=files_context,
            run_summaries=chr(10).join(run_summaries[:20]),
            condition_coverage_hint=_condition_coverage_hint,
            topic=config.research.topic,
            exp_plan_anchor=_exp_plan_anchor,
        )

        # --- Timeout-aware prompt injection ---
        user_prompt = ip.user + _saturation_hint
        if prior_timed_out and baseline_metric is None:
            timeout_refine_attempts += 1
            timeout_hint = (
                f"\n\nCRITICAL: The experiment TIMED OUT after {prior_time_budget}s "
                f"with NO results. You MUST drastically reduce the experiment scale:\n"
                f"- Reduce total runs to ≤50\n"
                f"- Reduce steps per run to ≤2000\n"
                f"- Remove conditions that are not essential\n"
                f"- Add time.time() checks to stop gracefully before timeout\n"
                f"- Print intermediate metrics frequently so partial data is captured\n"
                f"- Time budget is {prior_time_budget}s — design for ≤{int(prior_time_budget * 0.7)}s\n"
            )
            user_prompt = user_prompt + timeout_hint
            logger.warning(
                "Stage 13: injecting timeout-aware prompt (attempt %d)",
                timeout_refine_attempts,
            )

        try:
            response = _chat_with_prompt(
                llm,
                ip.system,
                user_prompt,
                max_tokens=ip.max_tokens or 8192,
            )
        except RuntimeError as exc:
            if "ACP prompt timed out after" in str(exc):
                logger.warning(
                    "Stage 13: ACP prompt timed out during iteration %d; pausing for resume",
                    iteration,
                )
                return _pause_refinement(
                    reason=str(exc),
                    stop_reason="acp_prompt_timeout",
                    iteration=iteration,
                )
            raise
        extracted_files = _extract_multi_file_blocks(response.content)
        # If LLM returns only single block, treat as main.py update
        if not extracted_files:
            single_code = _extract_code_block(response.content)
            if single_code.strip():
                extracted_files = {"main.py": single_code}
        # R8-2: Merge with best_files to preserve supporting modules
        # (e.g., graphs.py, game.py) that the LLM didn't rewrite
        candidate_files = dict(best_files)
        if extracted_files:
            candidate_files.update(extracted_files)
        # If LLM returned nothing at all, candidate_files == best_files (unchanged)

        # BUG-R6-02: Preserve entry point when LLM strips main() function.
        # The LLM often returns only class/function improvements without the
        # main() entry point, causing the script to exit with no output.
        _new_main = candidate_files.get("main.py", "")
        _old_main = best_files.get("main.py", "")
        if (
            _new_main
            and _old_main
            and "if __name__" not in _new_main
            and "if __name__" in _old_main
        ):
            # Extract the entry-point block from original main.py
            _ep_idx = _old_main.rfind("\ndef main(")
            if _ep_idx == -1:
                _ep_idx = _old_main.rfind("\nif __name__")
            if _ep_idx != -1:
                _entry_block = _old_main[_ep_idx:]
                candidate_files["main.py"] = _new_main.rstrip() + "\n\n" + _entry_block
                logger.info(
                    "Stage 13 iter %d: restored entry point stripped by LLM "
                    "(%d chars appended from original main.py)",
                    iteration,
                    len(_entry_block),
                )

        # Validate main.py
        main_code = candidate_files.get("main.py", "")
        validation = validate_code(main_code)
        issue_text = ""
        repaired = False

        if not validation.ok:
            issue_text = format_issues_for_llm(validation)
            logger.info(
                "Stage 13 iteration %d validation failed: %s",
                iteration,
                validation.summary(),
            )
            irp = _pm.sub_prompt(
                "iterative_repair",
                issue_text=issue_text,
                all_files_ctx=_files_to_context(candidate_files),
            )
            try:
                repair_response = _chat_with_prompt(llm, irp.system, irp.user)
            except RuntimeError as exc:
                if "ACP prompt timed out after" in str(exc):
                    logger.warning(
                        "Stage 13: ACP repair prompt timed out during iteration %d; pausing for resume",
                        iteration,
                    )
                    return _pause_refinement(
                        reason=str(exc),
                        stop_reason="acp_prompt_timeout",
                        iteration=iteration,
                    )
                raise
            candidate_files["main.py"] = _extract_code_block(repair_response.content)
            validation = validate_code(candidate_files["main.py"])
            repaired = True

        # Save version directory
        version_dir = stage_dir / f"experiment_v{iteration}"
        _write_project(version_dir, candidate_files)

        iter_record: dict[str, Any] = {
            "iteration": iteration,
            "version_dir": f"experiment_v{iteration}/",
            "files": list(candidate_files.keys()),
            "validation_ok": validation.ok,
            "validation_summary": validation.summary(),
            "repaired": repaired,
            "metric": None,
            "improved": False,
        }
        if issue_text:
            iter_record["validation_issues"] = issue_text

        metric_val = None  # R6-3: initialize before conditional block
        if validation.ok and config.experiment.mode in ("sandbox", "docker"):
            # P7: Ensure deps for refined code (subprocess sandbox only)
            if config.experiment.mode == "sandbox":
                _refine_code = "\n".join(candidate_files.values())
                _ensure_sandbox_deps(_refine_code, config.experiment.sandbox.python_path)

            sandbox = create_sandbox(
                config.experiment,
                stage_dir / f"refine_sandbox_v{iteration}",
            )
            rerun = sandbox.run_project(
                version_dir,
                timeout_sec=config.experiment.time_budget_sec,
            )
            metric_val = _find_metric(rerun.metrics, metric_key)
            # R19-1: Store stdout (capped) so PAIRED lines survive for Stage 14
            _stdout_cap = rerun.stdout[:50000] if rerun.stdout else ""
            iter_record["sandbox"] = {
                "returncode": rerun.returncode,
                "metrics": rerun.metrics,
                "elapsed_sec": rerun.elapsed_sec,
                "timed_out": rerun.timed_out,
                "stderr": rerun.stderr[:2000] if rerun.stderr else "",
                "stdout": _stdout_cap,
            }
            iter_record["metric"] = metric_val

            # BUG-110: Parse ABLATION_CHECK lines from stdout
            if rerun.stdout:
                import re as _re_ablation
                _ablation_checks = _re_ablation.findall(
                    r"ABLATION_CHECK:\s*(\S+)\s+vs\s+(\S+)\s+outputs_differ=(True|False)",
                    rerun.stdout,
                )
                if _ablation_checks:
                    _identical_pairs = [
                        (c1, c2) for c1, c2, diff in _ablation_checks if diff == "False"
                    ]
                    iter_record["ablation_checks"] = [
                        {"cond1": c1, "cond2": c2, "differ": diff == "True"}
                        for c1, c2, diff in _ablation_checks
                    ]
                    if _identical_pairs:
                        _pairs_str = ", ".join(f"{c1} vs {c2}" for c1, c2 in _identical_pairs)
                        logger.warning(
                            "BUG-110: Identical ablation outputs detected: %s. "
                            "Ablation conditions may not be wired correctly.",
                            _pairs_str,
                        )
                        iter_record["ablation_identical"] = True

            # --- Track timeout in refine sandbox ---
            if rerun.timed_out:
                prior_timed_out = True
                timeout_refine_attempts += 1
                logger.warning(
                    "Stage 13 iteration %d: sandbox timed out after %.1fs",
                    iteration,
                    rerun.elapsed_sec,
                )
                # If still no metrics after timeout, use partial stdout metrics
                if not rerun.metrics and rerun.stdout:
                    from researchclaw.experiment.sandbox import parse_metrics as _parse_sb_metrics
                    partial = _parse_sb_metrics(rerun.stdout)
                    if partial:
                        iter_record["sandbox"]["metrics"] = partial
                        metric_val = _find_metric(partial, metric_key)
                        iter_record["metric"] = metric_val
                        logger.info(
                            "Stage 13 iteration %d: recovered %d partial metrics from timeout stdout",
                            iteration,
                            len(partial),
                        )

            # --- Detect runtime issues (NaN/Inf, stderr warnings) ---
            runtime_issues = _detect_runtime_issues(rerun)
            if runtime_issues:
                iter_record["runtime_issues"] = runtime_issues
                logger.info(
                    "Stage 13 iteration %d: runtime issues detected: %s",
                    iteration,
                    runtime_issues[:200],
                )
                # Attempt LLM repair with runtime context
                rrp = _pm.sub_prompt(
                    "iterative_repair",
                    issue_text=runtime_issues,
                    all_files_ctx=_files_to_context(candidate_files),
                )
                try:
                    repair_resp = _chat_with_prompt(llm, rrp.system, rrp.user)
                except RuntimeError as exc:
                    if "ACP prompt timed out after" in str(exc):
                        logger.warning(
                            "Stage 13: ACP runtime-repair prompt timed out during iteration %d; pausing for resume",
                            iteration,
                        )
                        return _pause_refinement(
                            reason=str(exc),
                            stop_reason="acp_prompt_timeout",
                            iteration=iteration,
                        )
                    raise
                repaired_files = _extract_multi_file_blocks(repair_resp.content)
                if not repaired_files:
                    single = _extract_code_block(repair_resp.content)
                    if single.strip():
                        repaired_files = dict(candidate_files)
                        repaired_files["main.py"] = single
                if repaired_files:
                    # BUG-106 fix: merge instead of replace to preserve
                    # supporting modules (trainers.py, utils.py, etc.)
                    merged = dict(candidate_files)
                    merged.update(repaired_files)
                    candidate_files = merged
                    _write_project(version_dir, candidate_files)
                    # Re-run after runtime fix
                    sandbox2 = create_sandbox(
                        config.experiment,
                        stage_dir / f"refine_sandbox_v{iteration}_fix",
                    )
                    rerun2 = sandbox2.run_project(
                        version_dir,
                        timeout_sec=config.experiment.time_budget_sec,
                    )
                    metric_val = _find_metric(rerun2.metrics, metric_key)
                    iter_record["sandbox_after_fix"] = {
                        "returncode": rerun2.returncode,
                        "metrics": rerun2.metrics,
                        "elapsed_sec": rerun2.elapsed_sec,
                        "timed_out": rerun2.timed_out,
                    }
                    iter_record["metric"] = metric_val
                    iter_record["runtime_repaired"] = True

            if metric_val is not None:
                consecutive_no_metrics = 0
                # R6-1: Only count toward no_improve_streak when we have real metrics
                if _is_better(metric_val, best_metric):
                    best_metric = metric_val
                    best_files = dict(candidate_files)
                    best_version = f"experiment_v{iteration}/"
                    iter_record["improved"] = True
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            else:
                consecutive_no_metrics += 1
        elif validation.ok and best_version == "experiment/":
            best_files = dict(candidate_files)
            best_version = f"experiment_v{iteration}/"

        # P1: Track metric for saturation detection
        _metrics_history.append(metric_val)

        log["iterations"].append(iter_record)

        if consecutive_no_metrics >= 3:
            log["stop_reason"] = "consecutive_no_metrics"
            logger.warning("Stage 13: Aborting after %d consecutive iterations without metrics", consecutive_no_metrics)
            break

        if no_improve_streak >= 2:
            log["converged"] = True
            log["stop_reason"] = "no_improvement_for_2_iterations"
            logger.info(
                "Stage 13 converged after %d iterations (no improvement streak=%d)",
                iteration,
                no_improve_streak,
            )
            break

    # Write final experiment directory
    final_dir = stage_dir / "experiment_final"
    _write_project(final_dir, best_files)
    # Backward compat: also write experiment_final.py (copy of main.py)
    if "main.py" in best_files:
        (stage_dir / "experiment_final.py").write_text(
            best_files["main.py"], encoding="utf-8"
        )

    log["best_metric"] = best_metric
    log["best_version"] = best_version
    log["final_version"] = "experiment_final/"
    # BUG-110: Aggregate ablation check results across iterations
    _all_ablation_identical = any(
        iter_rec.get("ablation_identical", False)
        for iter_rec in log.get("iterations", [])
        if isinstance(iter_rec, dict)
    )
    if _all_ablation_identical:
        log["ablation_identical_warning"] = True
    _write_refinement_log()

    artifacts = ["refinement_log.json", "experiment_final/"]
    artifacts.extend(
        entry["version_dir"]
        for entry in log["iterations"]
        if isinstance(entry, dict) and isinstance(entry.get("version_dir"), str)
    )
    return StageResult(
        stage=Stage.ITERATIVE_REFINE,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
    )
