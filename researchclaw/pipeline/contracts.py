"""Stage I/O contracts for the 23-stage ResearchClaw pipeline.

Each StageContract declares:
  - input_files: artifacts this stage reads (produced by prior stages)
  - output_files: artifacts this stage must produce
  - dod: Definition of Done — human-readable acceptance criterion
  - error_code: unique error identifier for diagnostics
  - max_retries: how many times the stage may be retried on failure
"""

from __future__ import annotations

from dataclasses import dataclass

from researchclaw.pipeline.stages import Stage


@dataclass(frozen=True)
class StageContract:
    stage: Stage
    input_files: tuple[str, ...]
    output_files: tuple[str, ...]
    dod: str
    error_code: str
    max_retries: int = 1


CONTRACTS: dict[Stage, StageContract] = {
    # Phase A: Research Scoping
    Stage.TOPIC_INIT: StageContract(
        stage=Stage.TOPIC_INIT,
        input_files=(),
        output_files=("goal.md", "hardware_profile.json"),
        dod="SMART goal statement with topic, scope, and constraints",
        error_code="E01_INVALID_GOAL",
        max_retries=0,
    ),
    Stage.PROBLEM_DECOMPOSE: StageContract(
        stage=Stage.PROBLEM_DECOMPOSE,
        input_files=("goal.md",),
        output_files=("problem_tree.md",),
        dod=">=3 prioritized sub-questions identified",
        error_code="E02_DECOMP_FAIL",
    ),
    # Phase B: Literature Discovery
    Stage.SEARCH_STRATEGY: StageContract(
        stage=Stage.SEARCH_STRATEGY,
        input_files=("problem_tree.md",),
        output_files=("search_plan.yaml", "sources.json", "queries.json"),
        dod=">=2 search strategies defined with verified data sources",
        error_code="E03_STRATEGY_BAD",
    ),
    Stage.LITERATURE_COLLECT: StageContract(
        stage=Stage.LITERATURE_COLLECT,
        input_files=("search_plan.yaml",),
        output_files=("candidates.jsonl",),
        dod=">=N candidate papers collected from specified sources",
        error_code="E04_COLLECT_EMPTY",
        max_retries=2,
    ),
    Stage.LITERATURE_SCREEN: StageContract(
        stage=Stage.LITERATURE_SCREEN,
        input_files=("candidates.jsonl",),
        output_files=("shortlist.jsonl",),
        dod="Relevance + quality dual screening completed and approved",
        error_code="E05_GATE_REJECT",
        max_retries=0,
    ),
    Stage.KNOWLEDGE_EXTRACT: StageContract(
        stage=Stage.KNOWLEDGE_EXTRACT,
        input_files=("shortlist.jsonl",),
        output_files=("cards/",),
        dod="Structured knowledge card per shortlisted paper",
        error_code="E06_EXTRACT_FAIL",
    ),
    # Phase C: Knowledge Synthesis
    Stage.SYNTHESIS: StageContract(
        stage=Stage.SYNTHESIS,
        input_files=("cards/",),
        output_files=("synthesis.md",),
        dod="Topic clusters + >=2 research gaps identified",
        error_code="E07_SYNTHESIS_WEAK",
    ),
    Stage.HYPOTHESIS_GEN: StageContract(
        stage=Stage.HYPOTHESIS_GEN,
        input_files=("synthesis.md",),
        output_files=("hypotheses.md",),
        dod=">=2 falsifiable research hypotheses",
        error_code="E08_HYP_INVALID",
    ),
    # Phase D: Experiment Design
    Stage.EXPERIMENT_DESIGN: StageContract(
        stage=Stage.EXPERIMENT_DESIGN,
        input_files=("hypotheses.md",),
        output_files=("exp_plan.yaml",),
        dod="Experiment plan with baselines, ablations, metrics approved",
        error_code="E09_GATE_REJECT",
        max_retries=0,
    ),
    Stage.CODE_GENERATION: StageContract(
        stage=Stage.CODE_GENERATION,
        input_files=("exp_plan.yaml",),
        output_files=("experiment/", "experiment_spec.md"),
        dod="Multi-file experiment project + spec document",
        error_code="E10_CODEGEN_FAIL",
        max_retries=2,
    ),
    Stage.RESOURCE_PLANNING: StageContract(
        stage=Stage.RESOURCE_PLANNING,
        input_files=("exp_plan.yaml",),
        output_files=("schedule.json",),
        dod="Resource schedule with GPU/time estimates",
        error_code="E11_SCHED_CONFLICT",
    ),
    # Phase E: Experiment Execution
    Stage.EXPERIMENT_RUN: StageContract(
        stage=Stage.EXPERIMENT_RUN,
        input_files=("schedule.json", "experiment/"),
        output_files=("runs/",),
        dod="All scheduled experiment runs completed with artifacts",
        error_code="E12_RUN_FAIL",
        max_retries=2,
    ),
    Stage.ITERATIVE_REFINE: StageContract(
        stage=Stage.ITERATIVE_REFINE,
        input_files=("runs/",),
        output_files=("refinement_log.json", "experiment_final/"),
        dod="Edit-run-eval loop converged or max iterations reached",
        error_code="E13_REFINE_FAIL",
        max_retries=2,
    ),
    # Phase F: Analysis & Decision
    Stage.RESULT_ANALYSIS: StageContract(
        stage=Stage.RESULT_ANALYSIS,
        input_files=("runs/",),
        output_files=("analysis.md",),
        dod="Metrics analyzed with statistical tests and conclusions",
        error_code="E14_ANALYSIS_ERR",
    ),
    Stage.RESEARCH_DECISION: StageContract(
        stage=Stage.RESEARCH_DECISION,
        input_files=("analysis.md",),
        output_files=("decision.md",),
        dod="PROCEED/PIVOT decision with evidence-based justification",
        error_code="E15_DECISION_FAIL",
    ),
    # Phase G: Paper Writing
    Stage.PAPER_OUTLINE: StageContract(
        stage=Stage.PAPER_OUTLINE,
        input_files=("analysis.md", "decision.md"),
        output_files=("outline.md",),
        dod="Complete paper outline with section-level detail",
        error_code="E16_OUTLINE_FAIL",
    ),
    Stage.PAPER_DRAFT: StageContract(
        stage=Stage.PAPER_DRAFT,
        input_files=("outline.md",),
        output_files=("paper_draft.md",),
        dod="Full paper draft with all sections written",
        error_code="E17_DRAFT_FAIL",
    ),
    Stage.PEER_REVIEW: StageContract(
        stage=Stage.PEER_REVIEW,
        input_files=("paper_draft.md",),
        output_files=("reviews.md",),
        dod=">=2 simulated review perspectives with actionable feedback",
        error_code="E18_REVIEW_FAIL",
    ),
    Stage.PAPER_REVISION: StageContract(
        stage=Stage.PAPER_REVISION,
        input_files=("paper_draft.md", "reviews.md"),
        output_files=("paper_revised.md",),
        dod="All review comments addressed with tracked changes",
        error_code="E19_REVISION_FAIL",
    ),
    # Phase H: Finalization
    Stage.QUALITY_GATE: StageContract(
        stage=Stage.QUALITY_GATE,
        input_files=("paper_revised.md",),
        output_files=("quality_report.json",),
        dod="Quality score meets threshold and approved",
        error_code="E20_GATE_REJECT",
        max_retries=0,
    ),
    Stage.KNOWLEDGE_ARCHIVE: StageContract(
        stage=Stage.KNOWLEDGE_ARCHIVE,
        input_files=(),
        output_files=("archive.md", "bundle_index.json"),
        dod="Retrospective + reproducibility bundle archived",
        error_code="E21_ARCHIVE_FAIL",
    ),
    Stage.EXPORT_PUBLISH: StageContract(
        stage=Stage.EXPORT_PUBLISH,
        input_files=("paper_revised.md",),
        output_files=("paper_final.md", "code/"),
        dod="Final paper exported in target format",
        error_code="E22_EXPORT_FAIL",
    ),
    Stage.CITATION_VERIFY: StageContract(
        stage=Stage.CITATION_VERIFY,
        input_files=("paper_final.md",),  # references.bib is optional (BUG-50)
        output_files=("verification_report.json", "references_verified.bib"),
        dod="All citations verified against real APIs; hallucinated refs flagged",
        error_code="E23_VERIFY_FAIL",
    ),
}
