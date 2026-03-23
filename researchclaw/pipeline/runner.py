from __future__ import annotations

import json
import importlib
import logging
import os
import shutil
import tempfile
import time as _time
from pathlib import Path

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.evolution import EvolutionStore, extract_lessons
from researchclaw.knowledge.base import write_stage_to_kb
from researchclaw.pipeline.executor import StageResult, execute_stage
from researchclaw.pipeline.stages import (
    DECISION_ROLLBACK,
    MAX_DECISION_PIVOTS,
    NONCRITICAL_STAGES,
    STAGE_SEQUENCE,
    Stage,
    StageStatus,
)


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _should_start(stage: Stage, from_stage: Stage, started: bool) -> bool:
    if started:
        return True
    return stage == from_stage


def _build_pipeline_summary(
    *,
    run_id: str,
    results: list[StageResult],
    from_stage: Stage,
    run_dir: Path | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "run_id": run_id,
        "stages_executed": len(results),
        "stages_done": sum(1 for item in results if item.status == StageStatus.DONE),
        "stages_blocked": sum(
            1 for item in results if item.status == StageStatus.BLOCKED_APPROVAL
        ),
        "stages_failed": sum(
            1 for item in results if item.status == StageStatus.FAILED
        ),
        "degraded": any(r.decision == "degraded" for r in results),
        "from_stage": int(from_stage),
        "final_stage": int(results[-1].stage) if results else int(from_stage),
        "final_status": results[-1].status.value if results else "no_stages",
        "generated": _utcnow_iso(),
        "content_metrics": _collect_content_metrics(run_dir),
    }
    return summary


def _write_pipeline_summary(run_dir: Path, summary: dict[str, object]) -> None:
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _write_checkpoint(run_dir: Path, stage: Stage, run_id: str) -> None:
    """Write checkpoint atomically via temp file + rename to prevent corruption."""
    checkpoint = {
        "last_completed_stage": int(stage),
        "last_completed_name": stage.name,
        "run_id": run_id,
        "timestamp": _utcnow_iso(),
    }
    target = run_dir / "checkpoint.json"
    fd, tmp_path = tempfile.mkstemp(dir=run_dir, suffix=".tmp", prefix="checkpoint_")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(checkpoint, indent=2))
        Path(tmp_path).replace(target)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _write_heartbeat(run_dir: Path, stage: Stage, run_id: str) -> None:
    """Write heartbeat file for sentinel watchdog monitoring."""
    import os

    heartbeat = {
        "pid": os.getpid(),
        "last_stage": int(stage),
        "last_stage_name": stage.name,
        "run_id": run_id,
        "timestamp": _utcnow_iso(),
    }
    (run_dir / "heartbeat.json").write_text(
        json.dumps(heartbeat, indent=2), encoding="utf-8"
    )


def read_checkpoint(run_dir: Path) -> Stage | None:
    """Read checkpoint and return the NEXT stage to execute, or None if no checkpoint."""
    cp_path = run_dir / "checkpoint.json"
    if not cp_path.exists():
        return None
    try:
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        last_num = data.get("last_completed_stage")
        if last_num is None:
            return None
        for i, stage in enumerate(STAGE_SEQUENCE):
            if int(stage) == last_num:
                if i + 1 < len(STAGE_SEQUENCE):
                    return STAGE_SEQUENCE[i + 1]
                return None
        return None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def resume_from_checkpoint(
    run_dir: Path, default_stage: Stage = Stage.TOPIC_INIT
) -> Stage:
    """Resolve the stage to resume from using checkpoint metadata."""
    next_stage = read_checkpoint(run_dir)
    return next_stage if next_stage is not None else default_stage


def _collect_content_metrics(run_dir: Path | None) -> dict[str, object]:
    """Collect content authenticity metrics from stage outputs."""
    metrics: dict[str, object] = {
        "template_ratio": None,
        "citation_verify_score": None,
        "total_citations": None,
        "verified_citations": None,
        "degraded_sources": [],
    }
    if run_dir is None:
        return metrics

    draft_path = run_dir / "stage-17" / "paper_draft.md"
    if draft_path.exists():
        try:
            quality_module = importlib.import_module("researchclaw.quality")
            compute_template_ratio = quality_module.compute_template_ratio
            text = draft_path.read_text(encoding="utf-8")
            metrics["template_ratio"] = round(compute_template_ratio(text), 4)
        except (
            AttributeError,
            ModuleNotFoundError,
            UnicodeDecodeError,
            OSError,
            ValueError,
            TypeError,
        ):
            pass

    verify_path = run_dir / "stage-23" / "verification_report.json"
    if verify_path.exists():
        try:
            vdata = json.loads(verify_path.read_text(encoding="utf-8"))
            if isinstance(vdata, dict):
                summary = vdata.get("summary", vdata)
                total = summary.get("total", 0) if isinstance(summary, dict) else None
                verified = summary.get("verified", 0) if isinstance(summary, dict) else None
                if isinstance(total, int | float) and isinstance(verified, int | float):
                    total_num = int(total)
                    verified_num = int(verified)
                    metrics["total_citations"] = total_num
                    metrics["verified_citations"] = verified_num
                    if total_num > 0:
                        metrics["citation_verify_score"] = round(
                            verified_num / total_num, 4
                        )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    return metrics


logger = logging.getLogger(__name__)


def _run_experiment_diagnosis(run_dir: Path, config: RCConfig, run_id: str) -> None:
    """Run experiment diagnosis after Stage 14 and save reports.

    Produces:
    - ``run_dir/experiment_diagnosis.json`` — structured diagnosis + quality assessment
    - ``run_dir/repair_prompt.txt`` — repair instructions (if quality is insufficient)
    """
    try:
        from researchclaw.pipeline.experiment_diagnosis import (
            diagnose_experiment,
            assess_experiment_quality,
        )

        # Find the most recent stage-14 experiment_summary.json
        summary_path = None
        for candidate in sorted(run_dir.glob("stage-14*/experiment_summary.json")):
            summary_path = candidate
        if not summary_path or not summary_path.exists():
            return

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # Collect stdout/stderr from experiment runs
        stdout, stderr = "", ""
        runs_dir = summary_path.parent / "runs"
        if runs_dir.is_dir():
            for run_file in sorted(runs_dir.glob("*.json"))[:5]:
                try:
                    run_data = json.loads(run_file.read_text(encoding="utf-8"))
                    if isinstance(run_data, dict):
                        stdout += run_data.get("stdout", "") + "\n"
                        stderr += run_data.get("stderr", "") + "\n"
                except (json.JSONDecodeError, OSError):
                    continue

        # Load experiment plan from stage-09
        plan = None
        for candidate in sorted(run_dir.glob("stage-09*/exp_plan.yaml")):
            try:
                import yaml as _yaml_diag
                plan = _yaml_diag.safe_load(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
        if plan is None:
            for candidate in sorted(run_dir.glob("stage-09*/experiment_design.json")):
                try:
                    plan = json.loads(candidate.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

        # Load refinement log if available
        ref_log = None
        for candidate in sorted(run_dir.glob("stage-13*/refinement_log.json")):
            try:
                ref_log = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        # Run diagnosis
        diag = diagnose_experiment(
            experiment_summary=summary,
            experiment_plan=plan,
            refinement_log=ref_log,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
        )

        # Run quality assessment
        qa = assess_experiment_quality(summary, ref_log)

        # Save diagnosis report
        diag_report = {
            "diagnosis": diag.to_dict(),
            "quality_assessment": {
                "mode": qa.mode.value,
                "sufficient": qa.sufficient,
                "repair_possible": qa.repair_possible,
                "deficiency_types": [d.type.value for d in qa.deficiencies],
            },
            "repair_needed": not qa.sufficient,
            "generated": _utcnow_iso(),
        }
        (run_dir / "experiment_diagnosis.json").write_text(
            json.dumps(diag_report, indent=2), encoding="utf-8"
        )

        if not qa.sufficient:
            # Generate repair prompt for the REFINE loop
            from researchclaw.pipeline.experiment_repair import build_repair_prompt

            code: dict[str, str] = {}
            # Try refined code first, then stage-10 experiment dir, then raw stage-10
            for _glob_pat in (
                "stage-13*/experiment_final/*.py",
                "stage-10*/experiment/*.py",
                "stage-10*/*.py",
            ):
                for candidate in sorted(run_dir.glob(_glob_pat)):
                    try:
                        code[candidate.name] = candidate.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
                if code:
                    break

            repair_prompt = build_repair_prompt(
                diag, code, time_budget_sec=config.experiment.time_budget_sec
            )
            (run_dir / "repair_prompt.txt").write_text(
                repair_prompt, encoding="utf-8"
            )
            logger.info(
                "[%s] Experiment diagnosis: mode=%s, deficiencies=%d — repair prompt saved",
                run_id, qa.mode.value, len(diag.deficiencies),
            )
            print(
                f"[{run_id}] Experiment diagnosis: {qa.mode.value} "
                f"({len(diag.deficiencies)} issues found, repair needed)"
            )
        else:
            logger.info(
                "[%s] Experiment diagnosis: mode=%s, sufficient=True — quality OK",
                run_id, qa.mode.value,
            )
            print(f"[{run_id}] Experiment diagnosis: {qa.mode.value} — quality OK")

    except Exception as exc:
        logger.warning("Experiment diagnosis failed: %s", exc)


def _run_experiment_repair(run_dir: Path, config: RCConfig, run_id: str) -> None:
    """Execute the experiment repair loop when diagnosis finds quality issues.

    Calls the repair loop from ``experiment_repair.py`` which:
    1. Loads experiment code and diagnosis
    2. Gets fixes from LLM or OpenCode
    3. Re-runs experiment in sandbox
    4. Re-assesses quality
    5. Repeats up to max_cycles
    """
    try:
        from researchclaw.pipeline.experiment_repair import run_repair_loop

        repair_result = run_repair_loop(
            run_dir=run_dir,
            config=config,
            run_id=run_id,
        )

        # Save repair result
        (run_dir / "experiment_repair_result.json").write_text(
            json.dumps(repair_result.to_dict(), indent=2), encoding="utf-8"
        )

        # BUG-186: Promote best experiment summary to stage-14/ so
        # downstream stages (sanitizer, paper_verifier) see it.
        # BUG-198: Only promote if the repair summary is RICHER than
        # the existing stage-14 summary.  The repair loop can produce
        # empty summaries (metrics: {}, 0 conditions) which would
        # overwrite enriched data from the analysis stage.
        if repair_result.best_experiment_summary:
            from researchclaw.pipeline.experiment_repair import (
                _summary_quality_score,
            )

            best_path = run_dir / "stage-14" / "experiment_summary.json"
            existing_score = 0.0
            if best_path.exists():
                try:
                    existing = json.loads(
                        best_path.read_text(encoding="utf-8")
                    )
                    existing_score = _summary_quality_score(existing)
                except (json.JSONDecodeError, OSError):
                    pass

            repair_score = _summary_quality_score(
                repair_result.best_experiment_summary
            )

            if repair_score > existing_score:
                best_path.write_text(
                    json.dumps(
                        repair_result.best_experiment_summary, indent=2
                    ),
                    encoding="utf-8",
                )
                logger.info(
                    "[%s] Promoted repair results to stage-14 "
                    "(score %.1f > %.1f, success=%s)",
                    run_id, repair_score, existing_score,
                    repair_result.success,
                )
            else:
                logger.info(
                    "[%s] Kept existing stage-14 summary (score %.1f >= "
                    "repair score %.1f)",
                    run_id, existing_score, repair_score,
                )

        if repair_result.success:
            # Re-run diagnosis with updated results
            _run_experiment_diagnosis(run_dir, config, run_id)
        else:
            logger.info(
                "[%s] Repair loop completed without reaching full_paper quality "
                "(best mode: %s, %d cycles)",
                run_id, repair_result.final_mode.value, repair_result.total_cycles,
            )

    except Exception as exc:
        logger.warning("[%s] Experiment repair failed: %s", run_id, exc)
        print(f"[{run_id}] Experiment repair failed: {exc}")


def execute_pipeline(
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    from_stage: Stage = Stage.TOPIC_INIT,
    auto_approve_gates: bool = False,
    stop_on_gate: bool = False,
    skip_noncritical: bool = False,
    kb_root: Path | None = None,
) -> list[StageResult]:
    """Execute pipeline stages sequentially from `from_stage` and write summary."""

    results: list[StageResult] = []
    started = False
    total_stages = len(STAGE_SEQUENCE)

    for stage in STAGE_SEQUENCE:
        started = _should_start(stage, from_stage, started)
        if not started:
            continue

        stage_num = int(stage)
        prefix = f"[{run_id}] Stage {stage_num:02d}/{total_stages}"
        print(f"{prefix} {stage.name} — running...")

        # BUG-218: Ensure the best stage-14 experiment data is promoted
        # BEFORE paper writing begins.  Without this, the recursive REFINE
        # path writes the paper using the latest (potentially worse)
        # iteration's data, because the post-recursion promotion at line
        # ~547 runs only after the recursive call—i.e. after the paper
        # has already been written.
        if stage == Stage.PAPER_OUTLINE:
            _promote_best_stage14(run_dir, config)

        t0 = _time.monotonic()

        result = execute_stage(
            stage,
            run_dir=run_dir,
            run_id=run_id,
            config=config,
            adapters=adapters,
            auto_approve_gates=auto_approve_gates,
        )
        elapsed = _time.monotonic() - t0
        if result.status == StageStatus.DONE:
            arts = ", ".join(result.artifacts) if result.artifacts else "none"
            if result.decision == "degraded":
                print(
                    f"{prefix} {stage.name} — DEGRADED ({elapsed:.1f}s) "
                    f"— continuing with sanitization → {arts}"
                )
            else:
                print(f"{prefix} {stage.name} — done ({elapsed:.1f}s) → {arts}")
        elif result.status == StageStatus.FAILED:
            err = result.error or "unknown error"
            print(f"{prefix} {stage.name} — FAILED ({elapsed:.1f}s) — {err}")
        elif result.status == StageStatus.BLOCKED_APPROVAL:
            print(f"{prefix} {stage.name} — blocked (awaiting approval)")
        results.append(result)

        if kb_root is not None and result.status == StageStatus.DONE:
            try:
                stage_dir = run_dir / f"stage-{int(stage):02d}"
                write_stage_to_kb(
                    kb_root,
                    stage_id=int(stage),
                    stage_name=stage.name.lower(),
                    run_id=run_id,
                    artifacts=list(result.artifacts),
                    stage_dir=stage_dir,
                    backend=config.knowledge_base.backend,
                    topic=config.research.topic,
                )
            except Exception:  # noqa: BLE001
                pass

        if result.status == StageStatus.DONE:
            _write_checkpoint(run_dir, stage, run_id)

        # --- Experiment diagnosis + repair after Stage 14 (result_analysis) ---
        if (
            stage == Stage.RESULT_ANALYSIS
            and result.status == StageStatus.DONE
            and config.experiment.repair.enabled
        ):
            _run_experiment_diagnosis(run_dir, config, run_id)

            # Check if repair loop should run
            _diag_path = run_dir / "experiment_diagnosis.json"
            if _diag_path.exists():
                try:
                    _diag_data = json.loads(_diag_path.read_text(encoding="utf-8"))
                    if _diag_data.get("repair_needed"):
                        _run_experiment_repair(run_dir, config, run_id)
                except (json.JSONDecodeError, OSError):
                    pass

        # --- Heartbeat for sentinel watchdog ---
        _write_heartbeat(run_dir, stage, run_id)

        # --- PIVOT/REFINE decision handling ---
        if (
            stage == Stage.RESEARCH_DECISION
            and result.status == StageStatus.DONE
            and result.decision in DECISION_ROLLBACK
        ):
            pivot_count = _read_pivot_count(run_dir)
            # R6-4: Skip REFINE if experiment metrics are empty for consecutive cycles
            if pivot_count > 0 and _consecutive_empty_metrics(run_dir, pivot_count):
                logger.warning(
                    "Consecutive REFINE cycles produced empty metrics — forcing PROCEED"
                )
                print(
                    f"[{run_id}] Consecutive empty metrics across REFINE cycles — forcing PROCEED"
                )
                # BUG-211: Promote best stage-14 before proceeding with
                # empty data — an earlier iteration may have real metrics.
                _promote_best_stage14(run_dir, config)
            elif pivot_count < MAX_DECISION_PIVOTS:
                rollback_target = DECISION_ROLLBACK[result.decision]
                _record_decision_history(
                    run_dir, result.decision, rollback_target, pivot_count + 1
                )
                logger.info(
                    "Decision %s: rolling back to %s (attempt %d/%d)",
                    result.decision.upper(),
                    rollback_target.name,
                    pivot_count + 1,
                    MAX_DECISION_PIVOTS,
                )
                print(
                    f"[{run_id}] Decision: {result.decision.upper()} → "
                    f"rollback to {rollback_target.name} "
                    f"(attempt {pivot_count + 1}/{MAX_DECISION_PIVOTS})"
                )
                # Version existing stage directories before overwriting
                _version_rollback_stages(
                    run_dir, rollback_target, pivot_count + 1
                )
                # Recurse from rollback target
                pivot_results = execute_pipeline(
                    run_dir=run_dir,
                    run_id=run_id,
                    config=config,
                    adapters=adapters,
                    from_stage=rollback_target,
                    auto_approve_gates=auto_approve_gates,
                    stop_on_gate=stop_on_gate,
                    skip_noncritical=skip_noncritical,
                    kb_root=kb_root,
                )
                results.extend(pivot_results)
                # BUG-211: Promote best stage-14 after REFINE completes so
                # downstream stages use the best data, not just the latest.
                _promote_best_stage14(run_dir, config)
                break  # Exit current loop; recursive call handles the rest
            else:
                # Quality gate: check if experiment results are actually usable
                _quality_ok, _quality_msg = _check_experiment_quality(
                    run_dir, pivot_count
                )
                if not _quality_ok:
                    logger.warning(
                        "Max pivot attempts (%d) reached — forcing PROCEED "
                        "with quality warning: %s",
                        MAX_DECISION_PIVOTS,
                        _quality_msg,
                    )
                    print(
                        f"[{run_id}] QUALITY WARNING: {_quality_msg}"
                    )
                    # Write quality warning to run directory
                    _qw_path = run_dir / "quality_warning.txt"
                    _qw_path.write_text(
                        f"Max pivots ({MAX_DECISION_PIVOTS}) reached.\n"
                        f"Quality gate failed: {_quality_msg}\n"
                        f"Paper will be written but may have significant issues.\n",
                        encoding="utf-8",
                    )
                else:
                    logger.warning(
                        "Max pivot attempts (%d) reached — forcing PROCEED",
                        MAX_DECISION_PIVOTS,
                    )
                print(
                    f"[{run_id}] Max pivot attempts reached — forcing PROCEED"
                )

                # BUG-205: After forced PROCEED, promote the BEST stage-14
                # experiment summary across all REFINE iterations.
                _promote_best_stage14(run_dir, config)

        if result.status == StageStatus.FAILED:
            if skip_noncritical and stage in NONCRITICAL_STAGES:
                logger.warning("Noncritical stage %s failed - skipping", stage.name)
            else:
                break
        if result.status == StageStatus.BLOCKED_APPROVAL and stop_on_gate:
            break

    summary = _build_pipeline_summary(
        run_id=run_id,
        results=results,
        from_stage=from_stage,
        run_dir=run_dir,
    )
    _write_pipeline_summary(run_dir, summary)

    # --- Evolution: extract and store lessons ---
    lessons: list[object] = []
    try:
        lessons = extract_lessons(results, run_id=run_id, run_dir=run_dir)
        if lessons:
            store = EvolutionStore(run_dir / "evolution")
            store.append_many(lessons)
            logger.info("Extracted %d lessons from pipeline run", len(lessons))
    except Exception:  # noqa: BLE001
        logger.warning("Evolution lesson extraction failed (non-blocking)")

    # --- MetaClaw bridge: convert high-severity lessons to skills ---
    try:
        _metaclaw_post_pipeline(config, results, lessons, run_id, run_dir)
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw post-pipeline hook failed (non-blocking)")

    # --- Package deliverables into a single folder ---
    try:
        deliverables_dir = _package_deliverables(run_dir, run_id, config)
        if deliverables_dir is not None:
            print(f"[{run_id}] Deliverables packaged → {deliverables_dir}")
    except Exception:  # noqa: BLE001
        logger.warning("Deliverables packaging failed (non-blocking)")

    return results


def _package_deliverables(
    run_dir: Path,
    run_id: str,
    config: RCConfig,
) -> Path | None:
    """Collect all final user-facing deliverables into a single ``deliverables/`` folder.

    Returns the deliverables directory path, or None if nothing was packaged.

    Packaged artifacts (best-available version selected automatically):
    - paper_final.md          — Final paper (Markdown)
    - paper.tex               — Conference-ready LaTeX
    - references.bib          — BibTeX bibliography
    - code/                   — Experiment code package
    - verification_report.json — Citation verification report (if available)
    """
    dest = run_dir / "deliverables"
    dest.mkdir(parents=True, exist_ok=True)

    packaged: list[str] = []

    # --- 1. Final paper (Markdown) ---
    # Prefer verified version (stage 23) over base version (stage 22)
    paper_md = None
    for candidate in [
        run_dir / "stage-23" / "paper_final_verified.md",
        run_dir / "stage-22" / "paper_final.md",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            paper_md = candidate
            break
    if paper_md is not None:
        shutil.copy2(paper_md, dest / "paper_final.md")
        packaged.append("paper_final.md")

    # --- 2. LaTeX paper ---
    # BUG-183: Stage 22's paper.tex has been sanitized (fabricated numbers
    # replaced with ---).  Regenerating from Markdown would undo this because
    # the Markdown was never sanitized.  Prefer Stage-22 paper.tex when a
    # sanitization report exists.  Only regenerate from verified Markdown if
    # no sanitization was performed (i.e., the run was clean).
    tex_regenerated = False
    _sanitization_report = run_dir / "stage-22" / "sanitization_report.json"
    _was_sanitized = _sanitization_report.exists()
    verified_md = run_dir / "stage-23" / "paper_final_verified.md"
    if (
        not _was_sanitized
        and paper_md is not None
        and paper_md == verified_md
        and verified_md.exists()
        and verified_md.stat().st_size > 0
    ):
        try:
            from researchclaw.templates import get_template, markdown_to_latex
            from researchclaw.pipeline.executor import _extract_paper_title

            tpl = get_template(config.export.target_conference)
            v_text = verified_md.read_text(encoding="utf-8")
            tex_content = markdown_to_latex(
                v_text,
                tpl,
                title=_extract_paper_title(v_text),
                authors=config.export.authors,
                bib_file=config.export.bib_file,
            )
            # IMP-17: Quality check — ensure regenerated LaTeX has
            # proper structure (abstract, multiple sections)
            _has_abstract = (
                "\\begin{abstract}" in tex_content
                and tex_content.split("\\begin{abstract}")[1]
                .split("\\end{abstract}")[0]
                .strip()
            )
            _section_count = tex_content.count("\\section{")
            if _has_abstract and _section_count >= 3:
                (dest / "paper.tex").write_text(tex_content, encoding="utf-8")
                packaged.append("paper.tex")
                tex_regenerated = True
                logger.info(
                    "Deliverables: regenerated paper.tex from verified markdown"
                )
            else:
                logger.warning(
                    "Regenerated paper.tex has poor structure "
                    "(abstract=%s, sections=%d) — using Stage 22 version",
                    bool(_has_abstract),
                    _section_count,
                )
        except Exception:  # noqa: BLE001
            logger.debug("paper.tex regeneration from verified md failed")
    elif _was_sanitized:
        logger.info(
            "Deliverables: using Stage 22 paper.tex (sanitized) — "
            "skipping markdown regeneration to preserve sanitization"
        )

    if not tex_regenerated:
        tex_src = run_dir / "stage-22" / "paper.tex"
        if tex_src.exists() and tex_src.stat().st_size > 0:
            shutil.copy2(tex_src, dest / "paper.tex")
            packaged.append("paper.tex")

    # --- 3. References (BibTeX) ---
    # Prefer verified bib (stage 23) over base bib (stage 22)
    bib_src = None
    for candidate in [
        run_dir / "stage-23" / "references_verified.bib",
        run_dir / "stage-22" / "references.bib",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            bib_src = candidate
            break
    if bib_src is not None:
        shutil.copy2(bib_src, dest / "references.bib")
        packaged.append("references.bib")

    # --- 4. Experiment code package ---
    code_src = run_dir / "stage-22" / "code"
    if code_src.is_dir():
        code_dest = dest / "code"
        if code_dest.exists():
            shutil.rmtree(code_dest)
        shutil.copytree(code_src, code_dest)
        packaged.append("code/")

    # --- 5. Verification report (optional) ---
    verify_src = run_dir / "stage-23" / "verification_report.json"
    if verify_src.exists() and verify_src.stat().st_size > 0:
        shutil.copy2(verify_src, dest / "verification_report.json")
        packaged.append("verification_report.json")

    # --- 5b. Sanitization report (degraded mode) ---
    san_src = run_dir / "stage-22" / "sanitization_report.json"
    if san_src.exists() and san_src.stat().st_size > 0:
        shutil.copy2(san_src, dest / "sanitization_report.json")
        packaged.append("sanitization_report.json")

    # --- 6. Charts (optional) ---
    charts_src = run_dir / "stage-22" / "charts"
    if charts_src.is_dir() and any(charts_src.iterdir()):
        charts_dest = dest / "charts"
        if charts_dest.exists():
            shutil.rmtree(charts_dest)
        shutil.copytree(charts_src, charts_dest)
        packaged.append("charts/")

    # --- 7. Conference style files (.sty, .bst) ---
    try:
        from researchclaw.templates import get_template

        tpl = get_template(config.export.target_conference)
        style_files = tpl.get_style_files()
        for sf in style_files:
            shutil.copy2(sf, dest / sf.name)
            packaged.append(sf.name)
        if style_files:
            logger.info(
                "Deliverables: bundled %d style files for %s",
                len(style_files),
                tpl.display_name,
            )
    except Exception:  # noqa: BLE001
        logger.debug("Style file bundling skipped (template lookup failed)")

    # --- 8. Verify & repair cite key coverage (IMP-12 + IMP-14) ---
    tex_path = dest / "paper.tex"
    bib_path = dest / "references.bib"
    if tex_path.exists() and bib_path.exists():
        try:
            tex_text = tex_path.read_text(encoding="utf-8")
            bib_text = bib_path.read_text(encoding="utf-8")
            import re as _re

            # IMP-15: Deduplicate .bib entries
            _seen_bib_keys: set[str] = set()
            _deduped_entries: list[str] = []
            for _bm in _re.finditer(
                r"(@\w+\{([^,]+),.*?\n\})", bib_text, _re.DOTALL
            ):
                _bkey = _bm.group(2).strip()
                if _bkey not in _seen_bib_keys:
                    _seen_bib_keys.add(_bkey)
                    _deduped_entries.append(_bm.group(1))
            if len(_deduped_entries) < len(
                list(_re.finditer(r"@\w+\{", bib_text))
            ):
                bib_text = "\n\n".join(_deduped_entries) + "\n"
                bib_path.write_text(bib_text, encoding="utf-8")
                logger.info(
                    "Deliverables: deduplicated .bib → %d entries",
                    len(_deduped_entries),
                )

            # Collect all cite keys from \cite{key1, key2}
            all_cite_keys: set[str] = set()
            for cm in _re.finditer(r"\\cite\{([^}]+)\}", tex_text):
                all_cite_keys.update(k.strip() for k in cm.group(1).split(","))
            bib_keys = set(_re.findall(r"@\w+\{([^,]+),", bib_text))
            missing = all_cite_keys - bib_keys

            # IMP-14: Strip orphaned \cite{key} from paper.tex
            if missing:
                logger.warning(
                    "Deliverables: stripping %d orphaned cite keys from "
                    "paper.tex: %s",
                    len(missing),
                    sorted(missing)[:10],
                )

                def _filter_cite(m: _re.Match[str]) -> str:
                    keys = [k.strip() for k in m.group(1).split(",")]
                    kept = [k for k in keys if k not in missing]
                    if not kept:
                        return ""
                    return "\\cite{" + ", ".join(kept) + "}"

                tex_text = _re.sub(r"\\cite\{([^}]+)\}", _filter_cite, tex_text)
                # Clean up whitespace artifacts: double spaces, space before period
                tex_text = _re.sub(r"  +", " ", tex_text)
                tex_text = _re.sub(r" ([.,;:)])", r"\1", tex_text)
                tex_path.write_text(tex_text, encoding="utf-8")
                logger.info(
                    "Deliverables: paper.tex repaired — all remaining cite "
                    "keys verified"
                )
            else:
                logger.info(
                    "Deliverables: all %d cite keys verified in references.bib",
                    len(all_cite_keys),
                )
        except Exception:  # noqa: BLE001
            logger.debug("Cite key verification/repair skipped")

    # --- 9. IMP-18: Compile LaTeX to verify paper.tex ---
    if tex_path.exists() and bib_path.exists():
        try:
            from researchclaw.templates.compiler import compile_latex

            compile_result = compile_latex(tex_path, max_attempts=3, timeout=120)
            if compile_result.success:
                logger.info("IMP-18: paper.tex compiles successfully")
                # Keep the generated PDF
                pdf_path = dest / tex_path.stem
                pdf_file = dest / (tex_path.stem + ".pdf")
                if pdf_file.exists():
                    packaged.append(f"{tex_path.stem}.pdf")
            else:
                logger.warning(
                    "IMP-18: paper.tex compilation failed after %d attempts: %s",
                    compile_result.attempts,
                    compile_result.errors[:3],
                )
            if compile_result.fixes_applied:
                logger.info(
                    "IMP-18: Applied %d auto-fixes: %s",
                    len(compile_result.fixes_applied),
                    compile_result.fixes_applied,
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-18: LaTeX compilation skipped (non-blocking)")

    if not packaged:
        # Nothing to package — remove empty dir
        dest.rmdir()
        return None

    # --- Write manifest ---
    manifest = {
        "run_id": run_id,
        "target_conference": config.export.target_conference,
        "files": packaged,
        "generated": _utcnow_iso(),
        "notes": {
            "paper_final.md": "Final paper in Markdown format",
            "paper.tex": f"Conference-ready LaTeX ({config.export.target_conference})",
            "references.bib": "BibTeX bibliography (verified citations only)",
            "code/": "Experiment source code with requirements.txt",
            "verification_report.json": "Citation integrity & relevance verification",
            "charts/": "Result visualizations",
        },
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info(
        "Deliverables packaged: %s (%d items)",
        dest,
        len(packaged),
    )
    return dest


def _version_rollback_stages(
    run_dir: Path, rollback_target: Stage, attempt: int
) -> None:
    """Rename stage directories that will be overwritten by a PIVOT/REFINE.

    For example, if rolling back to Stage 8 (attempt 2), renames:
      stage-08/ → stage-08_v1/
      stage-09/ → stage-09_v1/
      ... up to stage-15/
    """
    import shutil

    rollback_num = int(rollback_target)
    # Stages from rollback target up to RESEARCH_DECISION (15) will be rerun
    decision_num = int(Stage.RESEARCH_DECISION)

    for stage_num in range(rollback_num, decision_num + 1):
        stage_dir = run_dir / f"stage-{stage_num:02d}"
        if stage_dir.exists():
            version_dir = run_dir / f"stage-{stage_num:02d}_v{attempt}"
            if version_dir.exists():
                shutil.rmtree(version_dir)
            stage_dir.rename(version_dir)
            logger.debug(
                "Versioned %s → %s", stage_dir.name, version_dir.name
            )


def _consecutive_empty_metrics(run_dir: Path, pivot_count: int) -> bool:
    """R6-4: Check if the current and previous REFINE cycles both produced empty metrics."""
    # Check the most recent experiment_summary.json (stage-14) and its versioned predecessor.
    # BUG-215: When stage-14/ doesn't exist (renamed to stage-14_v{N} without
    # promotion), fall back to the latest versioned directory as "current".
    current = run_dir / "stage-14" / "experiment_summary.json"
    if not current.exists():
        # Try the latest versioned directory
        for _v in range(pivot_count + 1, 0, -1):
            alt = run_dir / f"stage-14_v{_v}" / "experiment_summary.json"
            if alt.exists():
                current = alt
                break
    prev = run_dir / f"stage-14_v{pivot_count}" / "experiment_summary.json"
    for path in (current, prev):
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Check all possible metric locations
            has_metrics = False
            ms = data.get("metrics_summary", {})
            if isinstance(ms, dict) and ms:
                has_metrics = True
            br = data.get("best_run", {})
            if isinstance(br, dict) and br.get("metrics"):
                has_metrics = True
            if has_metrics:
                return False  # At least one cycle had real metrics
        except (json.JSONDecodeError, OSError, AttributeError):
            return False
    return True  # Both cycles had empty metrics


def _promote_best_stage14(run_dir: Path, config: RCConfig) -> None:
    """BUG-205: After forced PROCEED, promote the best stage-14 experiment.

    Scans all ``stage-14*`` directories, scores them by primary metric,
    and copies the best experiment_summary.json into ``stage-14/`` if the
    current ``stage-14/`` is not already the best.
    """
    import shutil

    metric_key = config.experiment.metric_key or "primary_metric"
    metric_dir = config.experiment.metric_direction or "maximize"

    candidates: list[tuple[float, Path]] = []
    for d in sorted(run_dir.glob("stage-14*")):
        summary_path = d / "experiment_summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ms = data.get("metrics_summary", {})
        pm_val: float | None = None
        # BUG-DA8-03: Exact match first, then substring fallback
        # (avoids "accuracy" matching "balanced_accuracy")
        if metric_key in ms:
            _v = ms[metric_key]
            try:
                pm_val = float(_v["mean"] if isinstance(_v, dict) else _v)
            except (TypeError, ValueError, KeyError):
                pass
        if pm_val is None:
            for k, v in ms.items():
                if metric_key in k:
                    try:
                        pm_val = float(v["mean"] if isinstance(v, dict) else v)
                    except (TypeError, ValueError, KeyError):
                        pass
                    break
        if pm_val is not None:
            candidates.append((pm_val, d))

    if not candidates:
        return  # nothing to promote

    current_dir = run_dir / "stage-14"

    # Sort: best first
    candidates.sort(key=lambda x: x[0], reverse=(metric_dir == "maximize"))

    # BUG-226: Detect degenerate near-zero metrics (broken normalization or
    # collapsed training).  When minimising, a value >1000x smaller than the
    # second-best almost certainly comes from a degenerate iteration.
    if metric_dir == "minimize" and len(candidates) > 1:
        _bv, _bd = candidates[0]
        _sv = candidates[1][0]
        if 0 < _bv < _sv * 1e-3:
            logger.warning(
                "BUG-226: Degenerate best value %.6g is >1000× smaller than "
                "second-best %.6g — skipping degenerate iteration %s",
                _bv, _sv, _bd.name,
            )
            candidates.pop(0)

    best_val, best_dir = candidates[0]

    # BUG-223: Always write canonical best summary at run root BEFORE any
    # early return, so downstream consumers (Stage 17, Stage 20, Stage 22,
    # VerifiedRegistry) always find experiment_summary_best.json.
    _best_src = best_dir / "experiment_summary.json"
    if _best_src.exists():
        shutil.copy2(_best_src, run_dir / "experiment_summary_best.json")
        logger.info(
            "BUG-223: Wrote experiment_summary_best.json from %s (%.4f)",
            best_dir.name, best_val,
        )
        # BUG-225: Also copy analysis.md from the best iteration so Stage 17
        # doesn't read stale analysis from a degenerate non-versioned stage-14.
        _best_analysis = best_dir / "analysis.md"
        if _best_analysis.exists():
            shutil.copy2(_best_analysis, run_dir / "analysis_best.md")

    if best_dir == current_dir:
        logger.info("BUG-205: stage-14/ already has the best result (%.4f)", best_val)
        return

    # Promote: copy best summary into stage-14/
    current_summary = current_dir / "experiment_summary.json"
    best_summary = best_dir / "experiment_summary.json"
    # BUG-213: Also promote when stage-14/ is missing or empty
    if best_summary.exists():
        current_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "BUG-205: Promoting %s (%.4f) over stage-14/",
            best_dir.name, best_val,
        )
        shutil.copy2(best_summary, current_summary)
        # Also copy charts, analysis, and figure plans if they exist
        for fname in [
            "analysis.md",
            "results_table.tex",
            "figure_plan.json",           # BUG-213: must travel with metrics
            "figure_plan_final.json",     # BUG-213: ditto
        ]:
            src = best_dir / fname
            if src.exists():
                shutil.copy2(src, current_dir / fname)
        # Copy charts directory
        best_charts = best_dir / "charts"
        current_charts = current_dir / "charts"
        if best_charts.is_dir():
            if current_charts.is_dir():
                shutil.rmtree(current_charts)
            shutil.copytree(best_charts, current_charts)


def _check_experiment_quality(
    run_dir: Path, pivot_count: int
) -> tuple[bool, str]:
    """Quality gate before forced PROCEED.

    Returns (ok, message). ok=False means experiment results have critical
    quality issues and the forced-PROCEED paper will likely be poor.
    """
    # BUG-DA8-18: Check experiment_summary_best.json first (repair-promoted)
    summary_path = run_dir / "experiment_summary_best.json"
    if not summary_path.exists():
        summary_path = run_dir / "stage-14" / "experiment_summary.json"
    if not summary_path.exists():
        for v in range(pivot_count, 0, -1):
            alt = run_dir / f"stage-14_v{v}" / "experiment_summary.json"
            if alt.exists():
                summary_path = alt
                break

    if not summary_path.exists():
        return False, "No experiment_summary.json found — no metrics produced"

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, "experiment_summary.json is malformed"

    # Check 1: Are all metrics zero?
    ms = data.get("metrics_summary", {})
    if isinstance(ms, dict):
        values: list[float] = []
        for k, v in ms.items():
            if isinstance(v, (int, float)):
                values.append(float(v))
            # BUG-212: metrics_summary values are often dicts {min,max,mean,count}
            elif isinstance(v, dict) and "mean" in v:
                _mv = v["mean"]
                if isinstance(_mv, (int, float)):
                    values.append(float(_mv))
        if values and all(v == 0.0 for v in values):
            return False, "All experiment metrics are zero — experiments likely failed"

    # Check 2: Zero variance across conditions (R13-1)
    # Look for ablation_warnings or condition comparison data
    ablation_warnings = data.get("ablation_warnings", [])
    # BUG-212: Key is "condition_summaries", not "conditions"
    conditions = data.get(
        "condition_summaries", data.get("condition_metrics", {})
    )
    if isinstance(conditions, dict) and len(conditions) >= 2:
        primary_values: list[float] = []
        for cond_name, cond_data in conditions.items():
            if isinstance(cond_data, dict):
                # BUG-212: Primary metric lives inside cond_data["metrics"]
                _metrics = cond_data.get("metrics", cond_data)
                pm = _metrics.get(
                    "primary_metric",
                    _metrics.get("primary_metric_mean"),
                )
                if isinstance(pm, (int, float)):
                    primary_values.append(float(pm))
        if len(primary_values) >= 2 and len(set(primary_values)) == 1:
            return False, (
                f"All {len(primary_values)} conditions have identical primary_metric "
                f"({primary_values[0]}) — condition implementations are likely broken"
            )

    # Check 3: Too many ablation warnings
    if isinstance(ablation_warnings, list) and len(ablation_warnings) >= 3:
        return False, (
            f"{len(ablation_warnings)} ablation warnings — most conditions "
            f"produce identical results"
        )

    # Check 4: Analysis quality score (if available)
    quality = data.get("analysis_quality", data.get("quality_score"))
    if isinstance(quality, (int, float)) and quality < 3.0:
        return False, f"Analysis quality score {quality}/10 — below minimum threshold"

    return True, "Quality checks passed"


def _read_pivot_count(run_dir: Path) -> int:
    """Read how many PIVOT/REFINE decisions have been made so far."""
    history_path = run_dir / "decision_history.json"
    if not history_path.exists():
        return 0
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
    except (json.JSONDecodeError, OSError):
        pass
    return 0


def _record_decision_history(
    run_dir: Path, decision: str, rollback_target: Stage, attempt: int
) -> None:
    """Append a decision event to the history log."""
    history_path = run_dir / "decision_history.json"
    history: list[dict[str, object]] = []
    if history_path.exists():
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                history = data
        except (json.JSONDecodeError, OSError):
            pass
    history.append({
        "decision": decision,
        "rollback_target": rollback_target.name,
        "rollback_stage_num": int(rollback_target),
        "attempt": attempt,
        "timestamp": _utcnow_iso(),
    })
    history_path.write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


logger = logging.getLogger(__name__)


def _read_quality_score(run_dir: Path) -> float | None:
    """Extract quality score from the most recent quality_report.json."""
    report_path = run_dir / "stage-20" / "quality_report.json"
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Try common keys: score_1_to_10, score, quality_score
            for key in ("score_1_to_10", "score", "quality_score", "overall_score"):
                if key in data:
                    return float(data[key])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def _write_iteration_context(
    run_dir: Path, iteration: int, reviews: str, quality_score: float | None
) -> None:
    """Write iteration feedback file so next round can read it."""
    ctx = {
        "iteration": iteration,
        "quality_score": quality_score,
        "reviews_excerpt": reviews[:3000] if reviews else "",
        "generated": _utcnow_iso(),
    }
    (run_dir / "iteration_context.json").write_text(
        json.dumps(ctx, indent=2), encoding="utf-8"
    )


def execute_iterative_pipeline(
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    auto_approve_gates: bool = False,
    kb_root: Path | None = None,
    max_iterations: int = 3,
    quality_threshold: float = 7.0,
    convergence_rounds: int = 2,
) -> dict[str, object]:
    """Run the full pipeline with iterative quality improvement.

    After the first full pass (stages 1-22), if the quality gate score is below
    *quality_threshold*, re-run stages 16-22 (paper writing + finalization) with
    review feedback injected.  Stop when:
      - Score >= quality_threshold, OR
      - Score hasn't improved for *convergence_rounds* consecutive iterations, OR
      - *max_iterations* reached.

    Returns a summary dict with iteration history.
    """
    iteration_scores: list[float | None] = []
    all_results: list[list[StageResult]] = []

    # --- First full pass ---
    logger.info("Iteration 1/%d: running full pipeline (stages 1-22)", max_iterations)
    results = execute_pipeline(
        run_dir=run_dir,
        run_id=f"{run_id}-iter1",
        config=config,
        adapters=adapters,
        auto_approve_gates=auto_approve_gates,
        kb_root=kb_root,
    )
    all_results.append(results)
    score = _read_quality_score(run_dir)
    iteration_scores.append(score)
    logger.info("Iteration 1 score: %s", score)

    # --- Iterative improvement ---
    for iteration in range(2, max_iterations + 1):
        # Check if we've met quality threshold
        if score is not None and score >= quality_threshold:
            logger.info(
                "Quality threshold %.1f met (score=%.1f). Stopping.",
                quality_threshold,
                score,
            )
            break

        # Check convergence (score hasn't improved)
        if len(iteration_scores) >= convergence_rounds:
            recent = iteration_scores[-convergence_rounds:]
            if all(s is not None for s in recent):
                recent_scores = [float(s) for s in recent if s is not None]
                if max(recent_scores) - min(recent_scores) < 0.5:
                    logger.info(
                        "Convergence detected: scores %s unchanged for %d rounds. Stopping.",
                        recent,
                        convergence_rounds,
                    )
                    break

        # Write iteration context with feedback from reviews
        reviews_text = ""
        reviews_path = run_dir / "stage-18" / "reviews.md"
        if reviews_path.exists():
            reviews_text = reviews_path.read_text(encoding="utf-8")
        _write_iteration_context(run_dir, iteration, reviews_text, score)

        # Re-run from PAPER_OUTLINE (stage 16) through EXPORT_PUBLISH (stage 22)
        logger.info(
            "Iteration %d/%d: re-running stages 16-22 with feedback",
            iteration,
            max_iterations,
        )
        results = execute_pipeline(
            run_dir=run_dir,
            run_id=f"{run_id}-iter{iteration}",
            config=config,
            adapters=adapters,
            from_stage=Stage.PAPER_OUTLINE,
            auto_approve_gates=auto_approve_gates,
            kb_root=kb_root,
        )
        all_results.append(results)
        score = _read_quality_score(run_dir)
        iteration_scores.append(score)
        logger.info("Iteration %d score: %s", iteration, score)

    # --- Build iterative summary ---
    converged = False
    if len(iteration_scores) >= convergence_rounds:
        recent_window = iteration_scores[-convergence_rounds:]
        if all(s is not None for s in recent_window):
            recent_scores = [float(s) for s in recent_window if s is not None]
            converged = max(recent_scores) - min(recent_scores) < 0.5

    summary: dict[str, object] = {
        "run_id": run_id,
        "total_iterations": len(iteration_scores),
        "iteration_scores": iteration_scores,
        "quality_threshold": quality_threshold,
        "converged": converged,
        "final_score": iteration_scores[-1] if iteration_scores else None,
        "met_threshold": score is not None and score >= quality_threshold,
        "stages_per_iteration": [len(r) for r in all_results],
        "generated": _utcnow_iso(),
    }
    (run_dir / "iteration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # --- Package deliverables into a single folder ---
    try:
        deliverables_dir = _package_deliverables(run_dir, run_id, config)
        if deliverables_dir is not None:
            print(f"[{run_id}] Deliverables packaged → {deliverables_dir}")
    except Exception:  # noqa: BLE001
        logger.warning("Deliverables packaging failed (non-blocking)")

    return summary


def _metaclaw_post_pipeline(
    config: RCConfig,
    results: list[StageResult],
    lessons: list[object],
    run_id: str,
    run_dir: Path,
) -> None:
    """MetaClaw bridge: post-pipeline hook.

    1. Convert high-severity lessons into MetaClaw skills.
    2. Record skill effectiveness feedback.
    3. Signal session end to MetaClaw proxy.
    """
    bridge = getattr(config, "metaclaw_bridge", None)
    if not bridge or not getattr(bridge, "enabled", False):
        return

    from researchclaw.llm.client import LLMClient

    # 1. Lesson-to-skill conversion
    l2s = getattr(bridge, "lesson_to_skill", None)
    if l2s and getattr(l2s, "enabled", False) and lessons:
        try:
            from researchclaw.metaclaw_bridge.lesson_to_skill import (
                convert_lessons_to_skills,
            )

            min_sev = getattr(l2s, "min_severity", "warning")
            llm = LLMClient.from_rc_config(config)
            new_skills = convert_lessons_to_skills(
                lessons,
                llm,
                getattr(bridge, "skills_dir", "~/.metaclaw/skills"),
                min_severity=min_sev,
                max_skills=getattr(l2s, "max_skills_per_run", 3),
            )
            if new_skills:
                logger.info(
                    "MetaClaw: generated %d new skills from lessons: %s",
                    len(new_skills),
                    new_skills,
                )
        except Exception:  # noqa: BLE001
            logger.warning("MetaClaw lesson-to-skill conversion failed", exc_info=True)

    # 2. Skill effectiveness feedback
    try:
        from researchclaw.metaclaw_bridge.skill_feedback import (
            SkillFeedbackStore,
            record_stage_skills,
        )
        from researchclaw.metaclaw_bridge.stage_skill_map import get_stage_config

        feedback_store = SkillFeedbackStore(run_dir / "evolution" / "skill_effectiveness.jsonl")
        for result in results:
            stage_num = int(getattr(result, "stage", 0))
            stage_name = {
                1: "topic_init", 2: "problem_decompose", 3: "search_strategy",
                4: "literature_collect", 5: "literature_screen", 6: "knowledge_extract",
                7: "synthesis", 8: "hypothesis_gen", 9: "experiment_design",
                10: "code_generation", 11: "resource_planning", 12: "experiment_run",
                13: "iterative_refine", 14: "result_analysis", 15: "research_decision",
                16: "paper_outline", 17: "paper_draft", 18: "peer_review",
                19: "paper_revision", 20: "quality_gate", 21: "knowledge_archive",
                22: "export_publish", 23: "citation_verify",
            }.get(stage_num, "")
            if not stage_name:
                continue

            stage_config = get_stage_config(stage_name)
            active_skills = stage_config.get("skills", [])
            status = str(getattr(result, "status", ""))
            success = "done" in status.lower()

            if active_skills:
                record_stage_skills(
                    feedback_store,
                    stage_name,
                    run_id,
                    success,
                    active_skills,
                )
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw skill feedback recording failed")

    # 3. Signal session end (fire-and-forget)
    try:
        from researchclaw.metaclaw_bridge.session import MetaClawSession
        import json as _json
        import urllib.request as _urllib_req

        session = MetaClawSession(run_id)
        end_headers = session.end()
        # Send a minimal request to signal session end
        proxy_url = getattr(bridge, "proxy_url", "http://localhost:30000")
        url = f"{proxy_url.rstrip('/')}/v1/chat/completions"
        body = _json.dumps({
            "model": "session-end",
            "messages": [{"role": "user", "content": "session complete"}],
            "max_tokens": 1,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(end_headers)
        req = _urllib_req.Request(url, data=body, headers=headers)
        try:
            _urllib_req.urlopen(req, timeout=5)
        except Exception:  # noqa: BLE001
            pass  # Best-effort signal
    except Exception:  # noqa: BLE001
        pass
