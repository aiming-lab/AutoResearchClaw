"""Stages 14-15: Result analysis and research decision."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain, _is_ml_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_context_preamble,
    _chat_with_prompt,
    _collect_experiment_results,
    _collect_json_context,
    _get_evolution_overlay,
    _multi_perspective_generate,
    _read_prior_artifact,
    _safe_json_loads,
    _synthesize_perspectives,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_result_analysis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    # --- Collect experiment data ---
    exp_data = _collect_experiment_results(
        run_dir,
        metric_key=config.experiment.metric_key,
        metric_direction=config.experiment.metric_direction,
    )
    runs_dir = _read_prior_artifact(run_dir, "runs/") or ""
    context = ""
    if runs_dir:
        context = _collect_json_context(Path(runs_dir), max_files=30)

    # --- R13-1: Merge Stage 13 (ITERATIVE_REFINE) results if available ---
    # Stage 13 stores richer per-condition metrics in refinement_log.json
    # that _collect_experiment_results() misses (it only scans runs/ dirs).
    _refine_log_text = _read_prior_artifact(run_dir, "refinement_log.json")
    if _refine_log_text:
        try:
            _refine_data = json.loads(_refine_log_text)
            _best_iter = None
            _best_ver = _refine_data.get("best_version", "")

            def _get_best_sandbox(it: dict) -> dict:
                """BUG-181: Metrics may be in sandbox or sandbox_after_fix."""
                sbx = it.get("sandbox", {})
                if sbx.get("metrics"):
                    return sbx
                sbx_fix = it.get("sandbox_after_fix", {})
                if sbx_fix.get("metrics"):
                    return sbx_fix
                return sbx

            for _it in _refine_data.get("iterations", []):
                _sbx = _get_best_sandbox(_it)
                _it_metrics = _sbx.get("metrics", {})
                if _it.get("version_dir", "") == _best_ver and _it_metrics:
                    _best_iter = _it
                    break
            # If no version match, take the first iteration with metrics
            if _best_iter is None:
                for _it in _refine_data.get("iterations", []):
                    _sbx = _get_best_sandbox(_it)
                    if _sbx.get("metrics"):
                        _best_iter = _it
                        break
            if _best_iter is not None:
                _sbx = _get_best_sandbox(_best_iter)
                _refine_metrics = _sbx.get("metrics", {})
                # BUG-165 fix: Prefer Stage 13 refinement data when it is
                # actually better.  The old `or True` unconditionally
                # replaced existing data, causing catastrophic regressions
                # (BUG-205: v1=78.93% destroyed by v3=8.65%).
                _refine_is_better = not exp_data["metrics_summary"]
                if not _refine_is_better and _refine_metrics:
                    # Compare primary_metric values to decide
                    _mkey = config.experiment.metric_key or "primary_metric"
                    _mdir = config.experiment.metric_direction or "maximize"
                    _existing_pm: float | None = None
                    _refine_pm: float | None = None
                    # BUG-214: Use exact match first, then substring fallback
                    # to avoid "accuracy" matching "balanced_accuracy".
                    _ms_items = list((exp_data.get("metrics_summary") or {}).items())
                    for _k, _v in _ms_items:
                        if _k == _mkey:
                            try:
                                _existing_pm = float(_v["mean"] if isinstance(_v, dict) else _v)
                            except (TypeError, ValueError, KeyError):
                                pass
                            break
                    else:
                        for _k, _v in _ms_items:
                            if _mkey in _k:
                                try:
                                    _existing_pm = float(_v["mean"] if isinstance(_v, dict) else _v)
                                except (TypeError, ValueError, KeyError):
                                    pass
                                break
                    _refine_items = list(_refine_metrics.items())
                    for _k, _v in _refine_items:
                        if _k == _mkey:
                            try:
                                _refine_pm = float(_v)
                            except (TypeError, ValueError):
                                pass
                            break
                    else:
                        for _k, _v in _refine_items:
                            if _mkey in _k:
                                try:
                                    _refine_pm = float(_v)
                                except (TypeError, ValueError):
                                    pass
                                break
                    if _existing_pm is None:
                        _refine_is_better = True  # no existing data
                    elif _refine_pm is not None:
                        if _mdir == "maximize":
                            _refine_is_better = _refine_pm > _existing_pm
                        else:
                            _refine_is_better = _refine_pm < _existing_pm
                    logger.info(
                        "Stage 14: Refine metric comparison: existing=%s, refine=%s, "
                        "direction=%s → refine_is_better=%s",
                        _existing_pm, _refine_pm, _mdir, _refine_is_better,
                    )
                if _refine_metrics and _refine_is_better:
                    # Refinement has richer data — rebuild metrics_summary from it
                    _new_summary: dict[str, dict[str, float | None]] = {}
                    for _mk, _mv in _refine_metrics.items():
                        try:
                            _fv = float(_mv)
                            _new_summary[_mk] = {
                                "min": round(_fv, 6),
                                "max": round(_fv, 6),
                                "mean": round(_fv, 6),
                                "count": 1,
                            }
                        except (ValueError, TypeError):
                            pass
                    if _new_summary:
                        exp_data["metrics_summary"] = _new_summary
                        # Also update best_run with refinement data
                        exp_data["best_run"] = {
                            "run_id": "iterative-refine-best",
                            "task_id": "sandbox-main",
                            "status": "completed",
                            "metrics": {
                                k: v for k, v in _refine_metrics.items()
                            },
                            "elapsed_sec": _sbx.get("elapsed_sec", 0),
                            "stdout": "",  # omit for brevity
                            "stderr": _sbx.get("stderr", ""),
                            "timed_out": _sbx.get("timed_out", False),
                        }
                        # Rebuild latex table
                        _ltx = [
                            r"\begin{table}[h]", r"\centering",
                            r"\caption{Experiment Results (Best Refinement Iteration)}",
                            r"\begin{tabular}{lrrrr}", r"\hline",
                            r"Metric & Min & Max & Mean & N \\", r"\hline",
                        ]
                        for _col in sorted(_new_summary.keys()):
                            _s = _new_summary[_col]
                            _ltx.append(
                                f"{_col} & {_s['min']:.4f} & {_s['max']:.4f} "
                                f"& {_s['mean']:.4f} & {_s['count']} \\\\"
                            )
                        _ltx.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
                        exp_data["latex_table"] = "\n".join(_ltx)
                        # Count unique conditions (keys without 'seed' and not ending in _mean/_std)
                        _conditions = {
                            k for k in _refine_metrics
                            if "seed" not in k and not k.endswith("_std")
                        }
                        exp_data["runs"] = [exp_data["best_run"]]
                        # Store condition count for accurate reporting
                        exp_data["best_run"]["condition_count"] = len(_conditions)
                        if not context:
                            context = json.dumps(
                                {"refinement_best_metrics": _refine_metrics},
                                indent=2, default=str,
                            )
                        _bm_val = _refine_data.get("best_metric")
                        logger.info(
                            "R13-1: Merged %d metrics from refinement_log (best_metric=%.4f)",
                            len(_refine_metrics),
                            float(_bm_val) if isinstance(_bm_val, (int, float)) else 0.0,
                        )
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("R13-1: Failed to parse refinement_log.json, using Stage 12 data")

    # --- R19-2: Extract PAIRED comparisons from refinement stdout ---
    from researchclaw.experiment.sandbox import extract_paired_comparisons as _extract_paired

    _all_paired: list[dict[str, object]] = []
    # First: from _collect_experiment_results (Stage 12 runs/)
    if exp_data.get("paired_comparisons"):
        _all_paired.extend(exp_data["paired_comparisons"])
    # Second: from refinement_log iterations (Stage 13)
    if _refine_log_text:
        try:
            _rl = json.loads(_refine_log_text)
            for _it in _rl.get("iterations", []):
                for _sbx_key in ("sandbox", "sandbox_after_fix"):
                    _sbx_stdout = (_it.get(_sbx_key) or {}).get("stdout", "")
                    if _sbx_stdout:
                        _all_paired.extend(_extract_paired(_sbx_stdout))
        except (json.JSONDecodeError, OSError):
            pass

    # --- R19-3: Build structured condition_summaries from metrics ---
    _condition_summaries: dict[str, dict[str, Any]] = {}
    _ms = exp_data.get("metrics_summary", {})
    _best_metrics = {}
    if exp_data.get("best_run") and isinstance(exp_data["best_run"], dict):
        _best_metrics = exp_data["best_run"].get("metrics", {})

    # Group metrics by condition prefix (e.g., "ppo/primary_metric" → condition "ppo")
    for _mk, _mv in _best_metrics.items():
        parts = _mk.split("/")
        if len(parts) >= 2:
            cond = parts[0]
            metric_name = parts[-1]
            if cond not in _condition_summaries:
                _condition_summaries[cond] = {"metrics": {}}
            try:
                _condition_summaries[cond]["metrics"][metric_name] = float(_mv)
            except (ValueError, TypeError):
                pass

    # BUG-09 fix: If no condition summaries were built (metrics don't use
    # condition/metric format), try to extract from metrics_summary or
    # structured_results so FigureAgent has data to work with.
    if not _condition_summaries and _ms:
        # Try to parse condition data from metrics_summary keys
        for _mk, _mv in _ms.items():
            parts = _mk.split("/")
            if len(parts) >= 2:
                cond = parts[0]
                metric_name = parts[-1]
                if cond not in _condition_summaries:
                    _condition_summaries[cond] = {"metrics": {}}
                try:
                    # BUG-182: metrics_summary values are dicts {min,max,mean,count},
                    # not plain floats. Extract the mean value.
                    if isinstance(_mv, dict):
                        _val = float(_mv["mean"]) if "mean" in _mv else None
                    else:
                        _val = float(_mv)
                    if _val is not None:
                        _condition_summaries[cond]["metrics"][metric_name] = _val
                except (ValueError, TypeError, KeyError):
                    pass
    if not _condition_summaries:
        # Last resort: build from structured_results condition keys
        _sr = exp_data.get("structured_results", {})
        if isinstance(_sr, dict):
            for _sk, _sv in _sr.items():
                if isinstance(_sv, dict) and _sk not in ("metadata", "config"):
                    _condition_summaries[_sk] = {"metrics": {}}
                    for _smk, _smv in _sv.items():
                        try:
                            _condition_summaries[_sk]["metrics"][_smk] = float(_smv)
                        except (ValueError, TypeError):
                            pass

    # R33: Build per-seed data structure (needed for CIs and paired tests below)
    _seed_data: dict[str, dict[int, float]] = {}  # {condition: {seed: value}}
    for _mk, _mv in _best_metrics.items():
        parts = _mk.split("/")
        # Pattern: condition/regime/seed_id/primary_metric
        if len(parts) >= 4 and parts[-1] == config.experiment.metric_key:
            cond = parts[0]
            try:
                seed_id = int(parts[2])
                val = float(_mv)
                _seed_data.setdefault(cond, {})[seed_id] = val
            except (ValueError, TypeError):
                pass

    # Enrich condition summaries with seed counts, success rates, and CIs
    for _ck, _cv in _condition_summaries.items():
        # Look for success_rate in metrics
        sr_key = f"{_ck}/success_rate"
        if sr_key in _best_metrics:
            try:
                _cv["success_rate"] = float(_best_metrics[sr_key])
            except (ValueError, TypeError):
                pass
        # Count seed-level entries to estimate n_seeds
        _seed_count = 0
        for _mk in _best_metrics:
            if _mk.startswith(f"{_ck}/") and "seed" in _mk.lower():
                _seed_count += 1
        if _seed_count > 0:
            _cv["n_seed_metrics"] = _seed_count

        # R33: Compute mean ± std and bootstrap 95% CI from per-seed data
        if _ck in _seed_data and len(_seed_data[_ck]) >= 3:
            _vals = list(_seed_data[_ck].values())
            import statistics as _stats_mod
            _mean = _stats_mod.mean(_vals)
            _std = _stats_mod.stdev(_vals)
            _cv["metrics"][f"{config.experiment.metric_key}_mean"] = round(_mean, 6)
            _cv["metrics"][f"{config.experiment.metric_key}_std"] = round(_std, 6)
            _cv["n_seeds"] = len(_vals)
            # Bootstrap 95% CI (use local RNG to avoid corrupting global state)
            import random as _rng_mod
            _rng_local = _rng_mod.Random(42)
            _boot_means = []
            for _ in range(1000):
                _sample = [_rng_local.choice(_vals) for _ in range(len(_vals))]
                _boot_means.append(_stats_mod.mean(_sample))
            _boot_means.sort()
            _ci_low = round(_boot_means[int(0.025 * len(_boot_means))], 6)
            _ci_high = round(_boot_means[int(0.975 * len(_boot_means))], 6)
            # IMP-16: Sanity check — CI must contain the mean
            if _ci_low > _mean or _ci_high < _mean:
                logger.warning(
                    "Bootstrap CI [%.4f, %.4f] does not contain mean %.4f "
                    "for condition %s — replacing CI with mean ± 1.96*SE",
                    _ci_low, _ci_high, _mean, _ck,
                )
                _se = _std / (len(_vals) ** 0.5)
                _ci_low = round(_mean - 1.96 * _se, 6)
                _ci_high = round(_mean + 1.96 * _se, 6)
            _cv["ci95_low"] = _ci_low
            _cv["ci95_high"] = _ci_high

    # Count totals
    _total_conditions = len(_condition_summaries) if _condition_summaries else None
    _total_metrics = len(_best_metrics) if _best_metrics else None

    # --- R33: Pipeline-level paired computation as fallback ---
    # If the experiment code's PAIRED lines are sparse or suspicious (e.g.,
    # all identical t-stats), compute fresh paired tests from per-seed data.
    # (_seed_data was built above before condition summary enrichment)
    if len(_seed_data) >= 2:
        # Find common seeds across conditions
        _all_seeds_sets = [set(v.keys()) for v in _seed_data.values()]
        _common_seeds = set.intersection(*_all_seeds_sets) if _all_seeds_sets else set()

        if len(_common_seeds) >= 3:
            _cond_names_sorted = sorted(_seed_data.keys())
            _pipeline_paired: list[dict[str, object]] = []
            # Compare each condition against the first baseline (alphabetically)
            _baseline_cond = _cond_names_sorted[0]
            for _other_cond in _cond_names_sorted[1:]:
                _diffs = []
                for _sid in sorted(_common_seeds):
                    _diffs.append(
                        _seed_data[_other_cond][_sid] - _seed_data[_baseline_cond][_sid]
                    )
                if _diffs:
                    import statistics
                    _n = len(_diffs)
                    _mean_d = statistics.mean(_diffs)
                    _std_d = statistics.stdev(_diffs) if _n > 1 else 0.0
                    _t = (_mean_d / (_std_d / (_n ** 0.5))) if _std_d > 0 else 0.0
                    _df = _n - 1
                    # Two-tailed p-value using t-distribution
                    import math
                    try:
                        from scipy.stats import t as _t_dist
                        _p = float(2 * _t_dist.sf(abs(_t), _df))
                    except ImportError:
                        _p = 2 * (1 - 0.5 * (1 + math.erf(abs(_t) / (2 ** 0.5))))
                        if _df < 30:
                            _p = min(1.0, _p * (1 + 2.5 / max(_df, 1)))
                    _pipeline_paired.append({
                        "method": _other_cond,
                        "baseline": _baseline_cond,
                        "mean_diff": round(_mean_d, 6),
                        "std_diff": round(_std_d, 6),
                        "t_stat": round(_t, 4),
                        "p_value": round(_p, 6),
                        "n_seeds": _n,
                        "source": "pipeline_computed",
                    })

            # Use pipeline-computed if experiment code's are suspicious
            _exp_t_stats = {round(p.get("t_stat", 0), 4) for p in _all_paired}
            _all_identical = len(_exp_t_stats) <= 1 and len(_all_paired) > 1
            if _pipeline_paired and (_all_identical or len(_all_paired) < len(_pipeline_paired)):
                logger.info(
                    "R33: Using %d pipeline-computed paired tests (experiment code had %d, identical=%s)",
                    len(_pipeline_paired), len(_all_paired), _all_identical,
                )
                _all_paired = _pipeline_paired

    # --- P8: Detect identical conditions (broken ablations) ---
    _ablation_warnings: list[str] = []
    if _condition_summaries and len(_condition_summaries) >= 2:
        _cond_names = sorted(_condition_summaries.keys())
        for _i in range(len(_cond_names)):
            for _j in range(_i + 1, len(_cond_names)):
                _c1, _c2 = _cond_names[_i], _cond_names[_j]
                _s1_raw = _condition_summaries[_c1]
                _s2_raw = _condition_summaries[_c2]
                # BUG-133 fix: compare inner metrics dicts, not top-level keys
                _s1_m = _s1_raw.get("metrics", {}) if isinstance(_s1_raw, dict) else {}
                _s2_m = _s2_raw.get("metrics", {}) if isinstance(_s2_raw, dict) else {}
                if not isinstance(_s1_m, dict):
                    _s1_m = {}
                if not isinstance(_s2_m, dict):
                    _s2_m = {}
                _shared_keys = set(_s1_m.keys()) & set(_s2_m.keys())
                if not _shared_keys:
                    continue
                _all_equal = True
                for _sk in _shared_keys:
                    _v1 = _s1_m[_sk]
                    _v2 = _s2_m[_sk]
                    if _v1 != _v2:
                        _all_equal = False
                        break
                if _all_equal and _shared_keys:
                    _warn = (
                        f"ABLATION FAILURE: Conditions '{_c1}' and '{_c2}' produce "
                        f"identical outputs across all {len(_shared_keys)} metrics. "
                        f"The ablation is invalid — the differentiating parameter "
                        f"is likely not used in the code."
                    )
                    _ablation_warnings.append(_warn)
                    logger.warning("P8: %s", _warn)
                elif _shared_keys:
                    # R5-BUG-03: Also flag near-identical conditions (< 1% relative diff)
                    _near_identical = True
                    for _sk in _shared_keys:
                        _v1 = _s1_m[_sk]
                        _v2 = _s2_m[_sk]
                        try:
                            _v1f, _v2f = float(_v1), float(_v2)
                            _denom = max(abs(_v1f), abs(_v2f), 1e-12)
                            if abs(_v1f - _v2f) / _denom > 0.01:
                                _near_identical = False
                                break
                        except (TypeError, ValueError):
                            _near_identical = False
                            break
                    if _near_identical:
                        _warn = (
                            f"ABLATION WARNING: Conditions '{_c1}' and '{_c2}' produce "
                            f"near-identical outputs (<1% relative difference) across "
                            f"all {len(_shared_keys)} metrics. The ablation may be trivial."
                        )
                        _ablation_warnings.append(_warn)
                        logger.warning("P8: %s", _warn)

    # --- Improvement B: Validate seed counts ---
    _seed_insufficiency_warnings: list[str] = []
    for _sc_name, _sc_seeds in _seed_data.items():
        _n_seeds = len(_sc_seeds)
        if 0 < _n_seeds < 3:
            _warn = (
                f"SEED_INSUFFICIENCY: Condition '{_sc_name}' has only "
                f"{_n_seeds} seed(s) (minimum 3 required for statistical validity)"
            )
            _seed_insufficiency_warnings.append(_warn)
            logger.warning("B: %s", _warn)

    # --- Write structured experiment summary ---
    summary_payload = {
        "metrics_summary": exp_data["metrics_summary"],
        "total_runs": len(exp_data["runs"]),
        "best_run": exp_data["best_run"],
        "latex_table": exp_data["latex_table"],
        "generated": _utcnow_iso(),
    }
    if _seed_insufficiency_warnings:
        summary_payload["seed_insufficiency_warnings"] = _seed_insufficiency_warnings
    # R13-1: Detect zero-variance across conditions (all conditions identical primary metric)
    if _condition_summaries and len(_condition_summaries) >= 2:
        _primary_vals = []
        for _cs in _condition_summaries.values():
            if isinstance(_cs, dict):
                # Try 'metrics' dict first (actual structure), then 'primary_metric' fallback
                _metrics = _cs.get("metrics", {})
                if isinstance(_metrics, dict) and _metrics:
                    _pv_candidate = next(iter(_metrics.values()), None)
                    if isinstance(_pv_candidate, dict):
                        _pv_candidate = _pv_candidate.get("mean")
                    if isinstance(_pv_candidate, (int, float)):
                        _primary_vals.append(_pv_candidate)
                        continue
                _pm = _cs.get("primary_metric", {})
                _pv = _pm.get("mean") if isinstance(_pm, dict) else _pm
                if isinstance(_pv, (int, float)):
                    _primary_vals.append(_pv)
        if len(_primary_vals) >= 2 and len(set(_primary_vals)) == 1:
            _zv_warn = (
                f"ZERO VARIANCE: All {len(_primary_vals)} conditions have "
                f"identical primary_metric ({_primary_vals[0]}). "
                f"Experiment condition wiring is likely broken."
            )
            _ablation_warnings.append(_zv_warn)
            logger.warning("R13-1: %s", _zv_warn)

    if _ablation_warnings:
        summary_payload["ablation_warnings"] = _ablation_warnings
    if _all_paired:
        summary_payload["paired_comparisons"] = _all_paired
    if _condition_summaries:
        summary_payload["condition_summaries"] = _condition_summaries
        summary_payload["condition_metrics"] = _condition_summaries  # alias for quality gate
        summary_payload["total_conditions"] = _total_conditions
    if _total_metrics:
        summary_payload["total_metric_keys"] = _total_metrics
    (stage_dir / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )
    if exp_data["latex_table"]:
        (stage_dir / "results_table.tex").write_text(
            exp_data["latex_table"], encoding="utf-8"
        )

    # --- Build data-augmented prompt ---
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    data_context = ""
    if exp_data["metrics_summary"]:
        lines = ["\n## Quantitative Results"]
        for mk, mv in exp_data["metrics_summary"].items():
            if isinstance(mv, dict):
                lines.append(
                    f"- {mk}: mean={mv.get('mean', '?')}, min={mv.get('min', '?')}, "
                    f"max={mv.get('max', '?')}, n={mv.get('count', '?')}"
                )
        data_context = "\n".join(lines)

    # Append structured results if available
    if exp_data.get("structured_results"):
        structured_text = json.dumps(
            exp_data["structured_results"], indent=2, default=str
        )
        # Truncate to avoid blowing up context
        if len(structured_text) > 6000:
            structured_text = structured_text[:6000] + "\n... (truncated)"
        data_context += (
            f"\n\n## Structured Experiment Results (from results.json)\n"
            f"```json\n{structured_text}\n```"
        )

    # P8: Inject ablation warnings into data context
    if _ablation_warnings:
        data_context += "\n\nCRITICAL ABLATION WARNINGS:\n"
        for _aw in _ablation_warnings:
            data_context += f"- {_aw}\n"
        data_context += (
            "\nYou MUST address these in your analysis. Identical conditions "
            "mean the ablation design is broken and the comparison is meaningless.\n"
        )

    if llm is not None:
        _pm = prompts or PromptManager()
        from researchclaw.prompts import DEBATE_ROLES_ANALYSIS  # noqa: PLC0415

        # --- Multi-perspective debate ---
        perspectives_dir = stage_dir / "perspectives"
        variables = {
            "preamble": preamble,
            "data_context": data_context,
            "context": context,
        }
        perspectives = _multi_perspective_generate(
            llm, DEBATE_ROLES_ANALYSIS, variables, perspectives_dir
        )
        # --- Synthesize into unified analysis ---
        analysis = _synthesize_perspectives(
            llm, perspectives, "analysis_synthesize", _pm
        )
    else:
        # Template with real data if available
        ms = exp_data["metrics_summary"]
        metrics_block = ""
        if ms:
            for mk, mv in ms.items():
                if isinstance(mv, dict):
                    metrics_block += (
                        f"- **{mk}**: mean={mv.get('mean')}, "
                        f"min={mv.get('min')}, max={mv.get('max')}, n={mv.get('count')}\n"
                    )
        else:
            metrics_block = f"- Primary metric key: `{config.experiment.metric_key}`\n- No quantitative data yet.\n"

        analysis = f"""# Result Analysis

## Metrics Summary
{metrics_block}
## Comparative Findings
- Proposed approach results from {len(exp_data["runs"])} run(s) collected.

## Statistical Checks
- Recommend confidence interval and seed-wise variance reporting.

## Limitations
- Limited runs and synthetic constraints.

## Conclusion
- Proceed to decision stage with moderate confidence.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "analysis.md").write_text(analysis, encoding="utf-8")

    artifacts = ["analysis.md", "experiment_summary.json"]
    if (stage_dir / "results_table.tex").exists():
        artifacts.append("results_table.tex")

    # IMP-6 + FA: Generate charts early (Stage 14) so paper draft can reference them
    # Try FigureAgent first (multi-agent intelligent charts), fall back to visualize.py
    _figure_plan_saved = False
    if config.experiment.figure_agent.enabled and llm is not None:
        try:
            from researchclaw.agents.figure_agent import FigureOrchestrator
            from researchclaw.agents.figure_agent.orchestrator import FigureAgentConfig as _FACfg

            _fa_cfg = _FACfg(
                enabled=True,
                min_figures=config.experiment.figure_agent.min_figures,
                max_figures=config.experiment.figure_agent.max_figures,
                max_iterations=config.experiment.figure_agent.max_iterations,
                render_timeout_sec=config.experiment.figure_agent.render_timeout_sec,
                use_docker=config.experiment.figure_agent.use_docker,
                docker_image=config.experiment.figure_agent.docker_image,
                output_format=config.experiment.figure_agent.output_format,
                gemini_api_key=config.experiment.figure_agent.gemini_api_key,
                gemini_model=config.experiment.figure_agent.gemini_model,
                nano_banana_enabled=config.experiment.figure_agent.nano_banana_enabled,
                strict_mode=config.experiment.figure_agent.strict_mode,
                dpi=config.experiment.figure_agent.dpi,
            )
            _fa = FigureOrchestrator(llm, _fa_cfg, stage_dir=stage_dir)

            # Build conditions list from condition_summaries
            _fa_conditions = list(_condition_summaries.keys()) if _condition_summaries else []

            # BUG-09 fix: pass best_run metrics as fallback data if
            # structured_results is empty, so Planner has some data to chart
            _fa_exp_results = exp_data.get("structured_results", {})
            if not _fa_exp_results and _best_metrics:
                _fa_exp_results = {"best_run_metrics": _best_metrics}

            # Read paper draft for Decision Agent analysis
            _paper_draft = (
                _read_prior_artifact(run_dir, "paper_draft.md")
                or _read_prior_artifact(run_dir, "outline.md")
                or ""
            )

            _fa_plan = _fa.orchestrate({
                "experiment_results": _fa_exp_results,
                "condition_summaries": _condition_summaries,
                "metrics_summary": exp_data.get("metrics_summary", {}),
                "metric_key": config.experiment.metric_key,
                "conditions": _fa_conditions,
                "topic": _read_prior_artifact(run_dir, "topic.md") or config.research.topic,
                "hypothesis": _read_prior_artifact(run_dir, "hypotheses.md") or "",
                "paper_draft": _paper_draft,
                "output_dir": str(stage_dir / "charts"),
            })

            if _fa_plan.figure_count > 0:
                # Save figure plan for Stage 17 to read
                (stage_dir / "figure_plan.json").write_text(
                    json.dumps(_fa_plan.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
                _figure_plan_saved = True
                for _cf_name in _fa_plan.get_chart_files():
                    artifacts.append(f"charts/{_cf_name}")
                logger.info(
                    "Stage 14: FigureAgent generated %d charts (%d passed review, %.1fs)",
                    _fa_plan.figure_count,
                    _fa_plan.passed_count,
                    _fa_plan.elapsed_sec,
                )
            else:
                logger.warning("Stage 14: FigureAgent produced no charts, falling back")
        except Exception as _fa_exc:
            logger.warning("Stage 14: FigureAgent failed (%s), falling back to visualize.py", _fa_exc)

    # Fallback: legacy visualize.py chart generation
    if not _figure_plan_saved:
        try:
            from researchclaw.experiment.visualize import (
                generate_all_charts as _gen_charts_early,
            )

            _charts_dir = stage_dir / "charts"
            _early_charts = _gen_charts_early(
                run_dir,
                _charts_dir,
                metric_key=config.experiment.metric_key,
            )
            if _early_charts:
                for _cp in _early_charts:
                    artifacts.append(f"charts/{_cp.name}")
                logger.info(
                    "Stage 14: Generated %d early charts (legacy) for paper embedding",
                    len(_early_charts),
                )
        except Exception as _chart_exc:
            logger.warning("Stage 14: Early chart generation failed: %s", _chart_exc)

    return StageResult(
        stage=Stage.RESULT_ANALYSIS,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-14/{a}" for a in artifacts),
    )


def _parse_decision(text: str) -> str:
    """Extract PROCEED/PIVOT/REFINE from decision text.

    Looks for the first standalone keyword on its own line after a
    ``## Decision`` heading.  Falls back to a keyword scan of the first
    few lines after the heading, but only matches the keyword itself
    (not mentions inside explanatory prose like "PIVOT is not warranted").
    Returns lowercase ``"proceed"`` / ``"pivot"`` / ``"refine"``.
    Defaults to ``"proceed"`` if nothing matches.
    """
    import re as _re

    text_upper = text.upper()
    # Look in the first occurrence after "## Decision" heading
    decision_section = ""
    for keyword in ("## DECISION", "## Decision", "## decision"):
        if keyword.upper() in text_upper:
            idx = text_upper.index(keyword.upper())
            decision_section = text[idx : idx + 200]
            break
    search_text = decision_section or text[:500]

    # First try: look for a line that is just the keyword (possibly with
    # whitespace / markdown bold / trailing punctuation).
    for line in search_text.splitlines():
        stripped = line.strip().strip("*").strip("#").strip()
        if stripped.upper() in ("PROCEED", "PIVOT", "REFINE"):
            return stripped.lower()

    # Fallback: regex for standalone word boundaries so that
    # "PIVOT is not warranted" does NOT match as a decision.
    for kw in ("PIVOT", "REFINE", "PROCEED"):
        # Only match if the keyword appears as the FIRST keyword-class token
        # on its own (not embedded in a sentence saying "not PIVOT").
        pattern = _re.compile(
            r"(?:^|##\s*Decision\s*\n\s*)" + kw, _re.IGNORECASE | _re.MULTILINE
        )
        if pattern.search(search_text):
            return kw.lower()

    # Last resort: position-based — prefer whichever keyword appears LAST
    # (the final conclusion after deliberation is more reliable than early mentions)
    # BUG-DA8-08: Old code always returned "refine" when both keywords present
    search_upper = search_text.upper()
    last_refine = search_upper.rfind("REFINE")
    last_pivot = search_upper.rfind("PIVOT")
    if last_refine >= 0 and (last_pivot < 0 or last_refine > last_pivot):
        return "refine"
    if last_pivot >= 0 and (last_refine < 0 or last_pivot > last_refine):
        return "pivot"
    return "proceed"


def _execute_research_decision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""

    # P6: Detect degenerate REFINE cycles — inject warning if metrics stagnate
    _degenerate_hint = ""
    _refine_log = _read_prior_artifact(run_dir, "refinement_log.json")
    if _refine_log:
        try:
            _rl = json.loads(_refine_log)
            _iters = _rl.get("iterations", [])
            _metrics = [it.get("metric") for it in _iters if isinstance(it, dict)]
            _valid = [m for m in _metrics if m is not None]
            _all_saturated = _valid and all(m <= 0.001 or m >= 0.999 for m in _valid)
            _all_identical = len(set(_valid)) <= 1 and len(_valid) >= 2
            if _all_saturated or _all_identical:
                _degenerate_hint = (
                    "\n\nSYSTEM WARNING — DEGENERATE REFINE CYCLE DETECTED:\n"
                    f"Metrics across {len(_valid)} iterations: {_valid}\n"
                    "All iterations produce identical/saturated results. Further REFINE "
                    "cycles CANNOT fix this — the underlying benchmark design is too "
                    "easy/hard. You SHOULD choose PROCEED with a quality caveat rather "
                    "than REFINE again.\n"
                )
                logger.warning("P6: Degenerate refine cycle detected, injecting PROCEED hint")
        except (json.JSONDecodeError, OSError):
            pass

    # Phase 2: Inject experiment diagnosis into decision prompt
    _diagnosis_hint = ""
    _diag_path = run_dir / "experiment_diagnosis.json"
    if _diag_path.exists():
        try:
            _diag_data = json.loads(_diag_path.read_text(encoding="utf-8"))
            _qa = _diag_data.get("quality_assessment", {})
            _mode = _qa.get("mode", "unknown")
            _sufficient = _qa.get("sufficient", False)
            _deficiency_types = _qa.get("deficiency_types", [])
            if not _sufficient:
                _diagnosis_hint = (
                    "\n\n## EXPERIMENT DIAGNOSIS (from automated analysis)\n"
                    f"Quality mode: {_mode}\n"
                    f"Sufficient for full paper: NO\n"
                    f"Issues found: {', '.join(_deficiency_types)}\n\n"
                    "IMPORTANT: The experiment has significant issues. "
                    "If REFINE is chosen, a structured repair prompt is available "
                    "at repair_prompt.txt with specific fixes for identified issues.\n"
                    "If the same issues persist after 2+ REFINE cycles, choose PROCEED "
                    "with appropriate quality caveats.\n"
                )
                logger.info(
                    "Stage 15: Injected experiment diagnosis — mode=%s, issues=%s",
                    _mode, _deficiency_types,
                )
        except (json.JSONDecodeError, OSError):
            pass

    # Improvement C: Check ablation quality — if >50% trivial, push REFINE
    _ablation_refine_hint = ""
    # BUG-DA8-16: Prefer experiment_summary_best.json (promoted best) over
    # alphabetically-last stage-14* (which could be a stale versioned dir)
    _exp_sum_path = run_dir / "experiment_summary_best.json"
    if not _exp_sum_path.is_file():
        _exp_sum_path = None
        for _s14 in sorted(run_dir.glob("stage-14*/experiment_summary.json"), reverse=True):
            _exp_sum_path = _s14
            break
    if _exp_sum_path and _exp_sum_path.is_file():
        try:
            from researchclaw.pipeline.stage_impls._paper_writing import _check_ablation_effectiveness
            _abl_exp = json.loads(_exp_sum_path.read_text(encoding="utf-8"))
            _abl_warnings = _check_ablation_effectiveness(_abl_exp, threshold=0.02)
            if _abl_warnings:
                _trivial_count = sum(1 for w in _abl_warnings if "ineffective" in w.lower() or "trivial" in w.lower())
                _total_abl = max(1, len(_abl_warnings))
                if _trivial_count / _total_abl > 0.5:
                    _ablation_refine_hint = (
                        "\n\n## ABLATION QUALITY ASSESSMENT (CRITICAL)\n"
                        f"STRONG RECOMMENDATION: Choose REFINE.\n"
                        f"{_trivial_count}/{_total_abl} ablations show <2% difference from baseline "
                        f"(trivially similar). This means the ablation design is broken.\n"
                        "Warnings:\n" + "\n".join(f"- {w}" for w in _abl_warnings) + "\n"
                    )
                    logger.warning("C: %d/%d ablations trivial → recommending REFINE", _trivial_count, _total_abl)
        except Exception:  # noqa: BLE001
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "research_decision")
        sp = _pm.for_stage("research_decision", evolution_overlay=_overlay, analysis=analysis)
        _user = sp.user + _degenerate_hint + _diagnosis_hint + _ablation_refine_hint
        resp = _chat_with_prompt(llm, sp.system, _user)
        decision_md = resp.content
    else:
        decision_md = f"""# Research Decision

## Decision
PROCEED

## Justification
Current evidence suggests measurable progress with actionable limitations.

## Next Actions
- Build detailed paper outline
- Expand ablation and uncertainty analysis in writing

Generated: {_utcnow_iso()}
"""
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")

    # --- Extract structured decision ---
    decision = _parse_decision(decision_md)

    # T3.1: Validate decision quality — check for minimum experiment rigor
    _quality_warnings: list[str] = []
    _dec_lower = decision_md.lower()
    if "baseline" not in _dec_lower and "control" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention baselines")
    if "seed" not in _dec_lower and "replicat" not in _dec_lower and "run" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention multi-seed/replicate runs")
    if "metric" not in _dec_lower and "accuracy" not in _dec_lower and "loss" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention evaluation metrics")
    if _quality_warnings:
        logger.warning("T3.1: Decision quality warnings: %s", _quality_warnings)

    decision_payload = {
        "decision": decision,
        "raw_text_excerpt": decision_md[:500],
        "quality_warnings": _quality_warnings,
        "generated": _utcnow_iso(),
    }
    (stage_dir / "decision_structured.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8"
    )
    logger.info("Research decision: %s", decision)

    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md", "decision_structured.json"),
        evidence_refs=("stage-15/decision.md",),
        decision=decision,
    )
