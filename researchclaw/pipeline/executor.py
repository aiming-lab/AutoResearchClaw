from __future__ import annotations

import json
import logging
import math
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.hardware import HardwareProfile, detect_hardware, ensure_torch_available, is_metric_name
from researchclaw.llm import create_llm_client
from researchclaw.llm.client import LLMClient
from researchclaw.prompts import PromptManager
from researchclaw.pipeline.stages import (
    NEXT_STAGE,
    Stage,
    StageStatus,
    TransitionEvent,
    TransitionOutcome,
    advance,
    gate_required,
)
from researchclaw.pipeline.contracts import CONTRACTS, StageContract
from researchclaw.experiment.validator import (
    CodeValidation,
    format_issues_for_llm,
    validate_code,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain detection (extracted to _domain.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline._domain import (  # noqa: E402
    _detect_domain,
    _is_ml_domain,
)


# ---------------------------------------------------------------------------
# Shared helpers (extracted to _helpers.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline._helpers import (  # noqa: E402
    StageResult,
    _METACLAW_SKILLS_DIR,
    _SANDBOX_SAFE_PACKAGES,
    _STOP_WORDS,
    _build_context_preamble,
    _build_fallback_queries,
    _chat_with_prompt,
    _collect_experiment_results,
    _collect_json_context,
    _default_hypotheses,
    _default_paper_outline,
    _default_quality_report,
    _detect_runtime_issues,
    _ensure_sandbox_deps,
    _extract_code_block,
    _extract_multi_file_blocks,
    _extract_paper_title,
    _extract_topic_keywords,
    _extract_yaml_block,
    _find_prior_file,
    _generate_framework_diagram_prompt,
    _generate_neurips_checklist,
    _get_evolution_overlay,
    _load_hardware_profile,
    _multi_perspective_generate,
    _parse_jsonl_rows,
    _parse_metrics_from_stdout,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _synthesize_perspectives,
    _topic_constraint_block,
    _utcnow_iso,
    _write_jsonl,
    _write_stage_meta,
    reconcile_figure_refs,
)

# ---------------------------------------------------------------------------
# Stages 1-2 (extracted to stage_impls/_topic.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._topic import (  # noqa: E402
    _execute_topic_init,
    _execute_problem_decompose,
)

# ---------------------------------------------------------------------------
# Stages 3-6 (extracted to stage_impls/_literature.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._literature import (  # noqa: E402
    _execute_search_strategy,
    _execute_literature_collect,
    _execute_literature_screen,
    _execute_knowledge_extract,
    _expand_search_queries,
)

# ---------------------------------------------------------------------------
# Stages 7-8 (extracted to stage_impls/_synthesis.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._synthesis import (  # noqa: E402
    _execute_synthesis,
    _execute_hypothesis_gen,
)

# ---------------------------------------------------------------------------
# Stage 9 (extracted to stage_impls/_experiment_design.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._experiment_design import (  # noqa: E402
    _execute_experiment_design,
)

# ---------------------------------------------------------------------------
# Stage 10 (extracted to stage_impls/_code_generation.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._code_generation import (  # noqa: E402
    _execute_code_generation,
)

# ---------------------------------------------------------------------------
# Stages 11-13 (extracted to stage_impls/_execution.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._execution import (  # noqa: E402
    _execute_resource_planning,
    _execute_experiment_run,
    _execute_iterative_refine,
)

# ---------------------------------------------------------------------------
# Stages 14-15 (extracted to stage_impls/_analysis.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._analysis import (  # noqa: E402
    _execute_result_analysis,
    _parse_decision,
    _execute_research_decision,
)

# ---------------------------------------------------------------------------
# Stages 16-17 (extracted to stage_impls/_paper_writing.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._paper_writing import (  # noqa: E402
    _execute_paper_outline,
    _execute_paper_draft,
    _collect_raw_experiment_metrics,
    _write_paper_sections,
    _validate_draft_quality,
    _review_compiled_pdf,
    _check_ablation_effectiveness,
    _detect_result_contradictions,
    _BULLET_LENIENT_SECTIONS,
    _BALANCE_SECTIONS,
)

# ---------------------------------------------------------------------------
# Stages 18-23 (extracted to stage_impls/_review_publish.py)
# ---------------------------------------------------------------------------
from researchclaw.pipeline.stage_impls._review_publish import (  # noqa: E402
    _execute_peer_review,
    _execute_paper_revision,
    _execute_quality_gate,
    _execute_knowledge_archive,
    _execute_export_publish,
    _execute_citation_verify,
    _sanitize_fabricated_data,
    _collect_experiment_evidence,
    _check_citation_relevance,
    _remove_bibtex_entries,
    _remove_citations_from_text,
)


_STAGE_EXECUTORS: dict[Stage, Callable[..., StageResult]] = {
    Stage.TOPIC_INIT: _execute_topic_init,
    Stage.PROBLEM_DECOMPOSE: _execute_problem_decompose,
    Stage.SEARCH_STRATEGY: _execute_search_strategy,
    Stage.LITERATURE_COLLECT: _execute_literature_collect,
    Stage.LITERATURE_SCREEN: _execute_literature_screen,
    Stage.KNOWLEDGE_EXTRACT: _execute_knowledge_extract,
    Stage.SYNTHESIS: _execute_synthesis,
    Stage.HYPOTHESIS_GEN: _execute_hypothesis_gen,
    Stage.EXPERIMENT_DESIGN: _execute_experiment_design,
    Stage.CODE_GENERATION: _execute_code_generation,
    Stage.RESOURCE_PLANNING: _execute_resource_planning,
    Stage.EXPERIMENT_RUN: _execute_experiment_run,
    Stage.ITERATIVE_REFINE: _execute_iterative_refine,
    Stage.RESULT_ANALYSIS: _execute_result_analysis,
    Stage.RESEARCH_DECISION: _execute_research_decision,
    Stage.PAPER_OUTLINE: _execute_paper_outline,
    Stage.PAPER_DRAFT: _execute_paper_draft,
    Stage.PEER_REVIEW: _execute_peer_review,
    Stage.PAPER_REVISION: _execute_paper_revision,
    Stage.QUALITY_GATE: _execute_quality_gate,
    Stage.KNOWLEDGE_ARCHIVE: _execute_knowledge_archive,
    Stage.EXPORT_PUBLISH: _execute_export_publish,
    Stage.CITATION_VERIFY: _execute_citation_verify,
}


def execute_stage(
    stage: Stage,
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    auto_approve_gates: bool = False,
) -> StageResult:
    """Execute one pipeline stage, validate outputs, and apply gate logic."""

    stage_dir = run_dir / f"stage-{int(stage):02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _t_health_start = _time.monotonic()
    contract: StageContract = CONTRACTS[stage]

    if contract.input_files:
        for input_file in contract.input_files:
            found = _read_prior_artifact(run_dir, input_file)
            if found is None:
                result = StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    artifacts=(),
                    error=f"Missing input: {input_file} (required by {stage.name})",
                    decision="retry",
                )
                _write_stage_meta(stage_dir, stage, run_id, result)
                return result

    bridge = config.openclaw_bridge
    if bridge.use_message and config.notifications.on_stage_start:
        adapters.message.notify(
            config.notifications.channel,
            f"stage-{int(stage):02d}-start",
            f"Starting {stage.name}",
        )
    if bridge.use_memory:
        adapters.memory.append("stages", f"{run_id}:{int(stage)}:running")

    llm = None
    try:
        if config.llm.provider == "acp":
            llm = create_llm_client(config)
        else:
            candidate = LLMClient.from_rc_config(config)
            if candidate.config.base_url and candidate.config.api_key:
                llm = candidate
    except Exception as _llm_exc:  # noqa: BLE001
        logger.warning("LLM client creation failed: %s", _llm_exc)
        llm = None

    try:
        _ = advance(stage, StageStatus.PENDING, TransitionEvent.START)
        executor = _STAGE_EXECUTORS[stage]
        prompts = PromptManager(config.prompts.custom_file or None)  # type: ignore[attr-defined]
        try:
            result = executor(
                stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
            )
        except TypeError as exc:
            if "unexpected keyword argument 'prompts'" not in str(exc):
                raise
            result = executor(stage_dir, run_dir, config, adapters, llm=llm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage %s failed", stage.name)
        result = StageResult(
            stage=stage,
            status=StageStatus.FAILED,
            artifacts=(),
            error=str(exc),
            decision="retry",
        )

    if result.status == StageStatus.DONE:
        for output_file in contract.output_files:
            if output_file.endswith("/"):
                path = stage_dir / output_file.rstrip("/")
                if not path.is_dir() or not any(path.iterdir()):
                    result = StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        artifacts=result.artifacts,
                        error=f"Missing output directory: {output_file}",
                        decision="retry",
                        evidence_refs=result.evidence_refs,
                    )
                    break
            else:
                path = stage_dir / output_file
                if not path.exists() or path.stat().st_size == 0:
                    result = StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        artifacts=result.artifacts,
                        error=f"Missing or empty output: {output_file}",
                        decision="retry",
                        evidence_refs=result.evidence_refs,
                    )
                    break

    # --- MetaClaw PRM quality gate evaluation ---
    try:
        mc_bridge = getattr(config, "metaclaw_bridge", None)
        if (
            mc_bridge
            and getattr(mc_bridge, "enabled", False)
            and result.status == StageStatus.DONE
        ):
            mc_prm = getattr(mc_bridge, "prm", None)
            if mc_prm and getattr(mc_prm, "enabled", False):
                prm_stages = getattr(mc_prm, "gate_stages", (5, 9, 15, 20))
                if int(stage) in prm_stages:
                    from researchclaw.metaclaw_bridge.prm_gate import ResearchPRMGate

                    prm_gate = ResearchPRMGate.from_bridge_config(mc_prm)
                    if prm_gate is not None:
                        # Read stage output for PRM evaluation
                        output_text = ""
                        for art in result.artifacts:
                            art_path = stage_dir / art
                            if art_path.exists() and art_path.is_file():
                                try:
                                    output_text += art_path.read_text(encoding="utf-8")[:4000]
                                except (UnicodeDecodeError, OSError):
                                    pass
                        if output_text:
                            prm_score = prm_gate.evaluate_stage(int(stage), output_text)
                            logger.info(
                                "MetaClaw PRM score for stage %d: %.1f",
                                int(stage),
                                prm_score,
                            )
                            # Write PRM score to stage health
                            import json as _prm_json

                            prm_report = {
                                "stage": int(stage),
                                "prm_score": prm_score,
                                "model": prm_gate.model,
                                "votes": prm_gate.votes,
                            }
                            (stage_dir / "prm_score.json").write_text(
                                _prm_json.dumps(prm_report, indent=2),
                                encoding="utf-8",
                            )
                            # If PRM score is -1 (fail), mark stage as failed
                            if prm_score == -1.0:
                                logger.warning(
                                    "MetaClaw PRM rejected stage %d output",
                                    int(stage),
                                )
                                result = StageResult(
                                    stage=result.stage,
                                    status=StageStatus.FAILED,
                                    artifacts=result.artifacts,
                                    error="PRM quality gate: output below quality threshold",
                                    decision="retry",
                                    evidence_refs=result.evidence_refs,
                                )
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw PRM evaluation failed (non-blocking)")

    if gate_required(stage, config.security.hitl_required_stages):
        if auto_approve_gates:
            if bridge.use_memory:
                adapters.memory.append("gates", f"{run_id}:{int(stage)}:auto-approved")
        else:
            result = StageResult(
                stage=result.stage,
                status=StageStatus.BLOCKED_APPROVAL,
                artifacts=result.artifacts,
                error=result.error,
                decision="block",
                evidence_refs=result.evidence_refs,
            )
            if bridge.use_message and config.notifications.on_gate_required:
                adapters.message.notify(
                    config.notifications.channel,
                    f"gate-{int(stage):02d}",
                    f"Approval required for {stage.name}",
                )

    if bridge.use_memory:
        adapters.memory.append("stages", f"{run_id}:{int(stage)}:{result.status.value}")

    _write_stage_meta(stage_dir, stage, run_id, result)

    _t_health_end = _time.monotonic()
    stage_health = {
        "stage_id": f"{int(stage):02d}-{stage.name.lower()}",
        "run_id": run_id,
        "duration_sec": round(_t_health_end - _t_health_start, 2),
        "status": result.status.value,
        "artifacts_count": len(result.artifacts),
        "error": result.error,
        "timestamp": _utcnow_iso(),
    }
    try:
        (stage_dir / "stage_health.json").write_text(
            json.dumps(stage_health, indent=2), encoding="utf-8"
        )
    except OSError:
        pass

    return result
