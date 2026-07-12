"""Stages 7-8: Synthesis and hypothesis generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.literature.evidence_cards import (
    EvidenceCardContractError,
    load_validated_cards,
)
from researchclaw.pipeline._helpers import (
    StageResult,
    _default_hypotheses,
    _get_evolution_overlay,
    _multi_perspective_generate,
    _parse_jsonl_rows,
    _read_prior_artifact,
    _synthesize_perspectives,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_synthesis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    synthesis_path = stage_dir / "synthesis.md"
    try:
        synthesis_path.unlink(missing_ok=True)
        cards = load_validated_cards(run_dir, config)
        successful_card_ids = [
            str(card["card_id"])
            for card in cards
            if card["extraction_status"] == "success"
        ]
        if not successful_card_ids:
            raise EvidenceCardContractError("no successful evidence cards")
        snippets = [
            (run_dir / "stage-06" / "cards" / f"{card_id}.md").read_text(
                encoding="utf-8"
            )
            for card_id in successful_card_ids[:24]
        ]
        cards_context = "\n\n".join(snippets)
    except (EvidenceCardContractError, OSError, UnicodeDecodeError) as exc:
        return StageResult(
            stage=Stage.SYNTHESIS,
            status=StageStatus.FAILED,
            artifacts=(),
            error=f"Stage 6 evidence cards are invalid: {exc}",
            decision="retry",
        )

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "synthesis")
        sp = _pm.for_stage(
            "synthesis",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            cards_context=cards_context,
            domain_context="",
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
            max_tokens=sp.max_tokens or 8192,
        )
        synthesis_md = resp.content
    else:
        synthesis_md = f"""# Synthesis

## Cluster Overview
- Cluster A: Representation methods
- Cluster B: Training strategies
- Cluster C: Evaluation robustness

## Gap 1
Limited consistency across benchmark protocols.

## Gap 2
Under-reported failure behavior under distribution shift.

## Prioritized Opportunities
1. Unified experimental protocol
2. Robustness-aware evaluation suite

## Generated
{_utcnow_iso()}
"""
    synthesis_path.write_text(synthesis_md, encoding="utf-8")
    return StageResult(
        stage=Stage.SYNTHESIS,
        status=StageStatus.DONE,
        artifacts=("synthesis.md",),
        evidence_refs=("stage-07/synthesis.md",),
    )


def _execute_hypothesis_gen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    synthesis = _read_prior_artifact(run_dir, "synthesis.md") or ""

    if llm is not None:
        _pm = prompts or PromptManager()
        # Debate roles come from the PromptManager's active bank
        # (ML bank -> innovator/pragmatist/contrarian,
        #  HEP bank -> theorist/phenomenologist/experimentalist).
        _active_roles = _pm.debate_roles_hypothesis()

        # --- Multi-perspective debate ---
        perspectives_dir = stage_dir / "perspectives"
        variables = {"topic": config.research.topic, "synthesis": synthesis}
        perspectives = _multi_perspective_generate(
            llm, _active_roles, variables, perspectives_dir
        )
        # BUG-S2: If all debate perspectives failed, fall back to defaults
        # instead of sending empty context to the LLM (pure hallucination).
        if not perspectives:
            logger.warning("All debate perspectives failed; using default hypotheses")
            hypotheses_md = _default_hypotheses(config.research.topic)
        else:
            # --- Synthesize into final hypotheses ---
            hypotheses_md = _synthesize_perspectives(
                llm, perspectives, "hypothesis_synthesize", _pm
            )
    else:
        hypotheses_md = _default_hypotheses(config.research.topic)
    # --- HITL: Read human guidance if available ---
    guidance_file = stage_dir / "hitl_guidance.md"
    if guidance_file.exists():
        try:
            guidance = guidance_file.read_text(encoding="utf-8").strip()
            if guidance and llm is not None:
                logger.info("Applying HITL guidance to hypotheses")
                resp = llm.chat(
                    [{"role": "user", "content": (
                        f"Refine the following hypotheses based on this human guidance.\n\n"
                        f"## Current Hypotheses\n{hypotheses_md}\n\n"
                        f"## Human Guidance\n{guidance}\n\n"
                        f"Produce improved hypotheses that incorporate the guidance."
                    )}],
                    max_tokens=4096,
                )
                hypotheses_md = resp.content
        except Exception:
            logger.debug("HITL guidance application failed (non-blocking)")

    # --- HITL: Idea Workshop data persistence ---
    try:
        from researchclaw.hitl.workshops.idea import IdeaWorkshop

        workshop = IdeaWorkshop(run_dir, llm_client=llm)
        workshop.candidates = [
            type("IC", (), {"title": "Generated Hypothesis", "description": hypotheses_md[:500],
                            "to_dict": lambda self: {"title": self.title, "description": self.description},
                            "human_approved": False, "baselines": [], "keywords": [],
                            "novelty_notes": "", "feasibility_notes": "", "impact_notes": "",
                            "score": 0.0})()
        ]
        workshop.save()
    except Exception:
        pass

    (stage_dir / "hypotheses.md").write_text(hypotheses_md, encoding="utf-8")

    # --- Novelty check (non-blocking) ---
    novelty_artifacts: tuple[str, ...] = ()
    try:
        from researchclaw.literature.novelty import check_novelty  # noqa: PLC0415

        candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""
        papers_seen = _parse_jsonl_rows(candidates_text) if candidates_text else []
        novelty_report = check_novelty(
            topic=config.research.topic,
            hypotheses_text=hypotheses_md,
            papers_already_seen=papers_seen,
            s2_api_key=getattr(config.llm, "s2_api_key", ""),
        )
        (stage_dir / "novelty_report.json").write_text(
            json.dumps(novelty_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        novelty_artifacts = ("novelty_report.json",)
        logger.info(
            "Novelty check: score=%.3f  assessment=%s  recommendation=%s",
            novelty_report["novelty_score"],
            novelty_report["assessment"],
            novelty_report["recommendation"],
        )
    except Exception:  # noqa: BLE001
        logger.warning("Novelty check failed (non-blocking)", exc_info=True)

    return StageResult(
        stage=Stage.HYPOTHESIS_GEN,
        status=StageStatus.DONE,
        artifacts=("hypotheses.md",) + novelty_artifacts,
        evidence_refs=("stage-08/hypotheses.md",),
    )
