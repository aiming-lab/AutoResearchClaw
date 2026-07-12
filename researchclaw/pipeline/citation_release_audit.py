"""Disk-only release replay for the Stage 4-24 citation evidence chain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from researchclaw.config import RCConfig
from researchclaw.experiment_runtime.contract import ExperimentContract
from researchclaw.literature.citation_identity import (
    parse_cite_key_registry,
    validate_registry_artifacts,
)
from researchclaw.literature.citation_plan import (
    extract_citation_keys,
    load_final_citation_plan,
    validate_citation_closure_report,
    validate_final_paper_citations,
)
from researchclaw.literature.citation_policy import (
    load_effective_citation_policy,
    resolve_active_config_snapshot,
    validate_citation_allowlist,
)
from researchclaw.literature.citation_support import (
    build_citation_support_closure,
    parse_citation_support_closure,
)
from researchclaw.literature.evidence_cards import load_validated_cards
from researchclaw.literature.experiment_fact_closure import (
    validate_experiment_fact_closure_report,
)
from researchclaw.literature.screening import (
    MAX_SCREEN_CANDIDATES,
    ScreeningContractError,
    normalize_quality_threshold,
    parse_screening_candidates,
    parse_screening_report,
    sha256_text,
)
from researchclaw.pipeline.stage_impls._literature import (
    _candidate_screen_score,
    _extract_topic_keywords,
)


@dataclass(frozen=True)
class CitationAuditError(RuntimeError):
    code: str
    message: str
    path: str

    def __str__(self) -> str:
        return self.message


def audit_citation_evidence(
    run_dir: Path,
    *,
    contract_path: Path,
    contract: ExperimentContract,
) -> None:
    """Replay every deterministic citation/evidence producer from disk."""

    config = _load_active_config(run_dir)
    if config.experiment.claim_scope != contract.claim_scope:
        _raise(
            "citation_contract_scope_mismatch",
            "Active config claim_scope differs from the canonical Stage 9 contract.",
            contract_path.relative_to(run_dir).as_posix(),
        )
    if config.experiment.dataset_origin != contract.dataset_origin:
        _raise(
            "citation_contract_dataset_origin_mismatch",
            "Active config dataset_origin differs from the canonical Stage 9 contract.",
            contract_path.relative_to(run_dir).as_posix(),
        )

    candidates_text = _read(run_dir, "stage-04/candidates.jsonl")
    registry_text = _read(run_dir, "stage-04/cite_key_registry.json")
    references_text = _read(run_dir, "stage-04/references.bib")
    try:
        registry = parse_cite_key_registry(registry_text)
        validate_registry_artifacts(registry, candidates_text, references_text)
        candidates = parse_screening_candidates(candidates_text)
    except Exception as exc:  # noqa: BLE001
        _raise("citation_key_registry_invalid", str(exc), "stage-04/cite_key_registry.json")

    shortlist_text = _read(run_dir, "stage-05/shortlist.jsonl")
    screening_text = _read(run_dir, "stage-05/screening_report.json")
    try:
        shortlist = parse_screening_candidates(shortlist_text)
        report = parse_screening_report(
            screening_text,
            candidates_text_sha256=sha256_text(candidates_text),
            registry_text_sha256=sha256_text(registry_text),
            references_text_sha256=sha256_text(references_text),
            expected_screening_output_path="stage-05/shortlist.jsonl",
            screening_output_text_sha256=sha256_text(shortlist_text),
            expected_minimum_quality_score=normalize_quality_threshold(
                config.research.quality_threshold
            ),
            expected_claim_scope=config.experiment.claim_scope,
            expected_candidate_ids=(row["source_identity"] for row in candidates),
            expected_selected_ids=(row["source_identity"] for row in shortlist),
        )
        _replay_screening_admission(config, candidates, report)
    except Exception as exc:  # noqa: BLE001
        _raise("literature_screening_replay_failed", str(exc), "stage-05/screening_report.json")

    try:
        load_validated_cards(run_dir, config)
    except Exception as exc:  # noqa: BLE001
        _raise("evidence_card_replay_failed", str(exc), "stage-06/cards_manifest.json")

    allowlist_text = _read(run_dir, "stage-06/citation_allowlist.json")
    try:
        validate_citation_allowlist(run_dir, config, allowlist_text)
    except Exception as exc:  # noqa: BLE001
        _raise("citation_allowlist_replay_failed", str(exc), "stage-06/citation_allowlist.json")
    try:
        load_effective_citation_policy(run_dir, config)
    except Exception as exc:  # noqa: BLE001
        _raise("citation_policy_replay_failed", str(exc), "stage-16/citation_policy_effective.json")
    try:
        load_final_citation_plan(run_dir, config)
    except Exception as exc:  # noqa: BLE001
        _raise("citation_plan_replay_failed", str(exc), "stage-16/citation_plan.json")
    try:
        validate_experiment_fact_closure_report(run_dir)
    except Exception as exc:  # noqa: BLE001
        _raise(
            "experiment_fact_closure_replay_failed",
            str(exc),
            "stage-17/experiment_fact_closure_report.json",
        )
    try:
        validate_citation_closure_report(run_dir, config)
    except Exception as exc:  # noqa: BLE001
        _raise(
            "citation_closure_replay_failed",
            str(exc),
            "stage-17/citation_closure_report.json",
        )

    paper_text = _read(run_dir, "stage-23/paper_final_verified.md")
    stage22_text = _read(run_dir, "stage-22/paper_final.md")
    if paper_text != stage22_text:
        _raise(
            "citation_final_paper_mismatch",
            "Stage 23 verified paper differs from canonical Stage 22 paper.",
            "stage-23/paper_final_verified.md",
        )
    try:
        validate_final_paper_citations(run_dir, config, paper_text)
    except Exception as exc:  # noqa: BLE001
        _raise("citation_final_closure_failed", str(exc), "stage-23/paper_final_verified.md")
    _validate_verification_report(run_dir, paper_text)

    support_text = _read(run_dir, "stage-24/citation_support.json")
    try:
        stored_support = parse_citation_support_closure(support_text)
        stored_by_id = {
            row["instance_id"]: row["assessment"]
            for row in stored_support["instances"]
        }

        def _stored_assessor(payload: dict[str, Any]) -> dict[str, str]:
            assessment = stored_by_id.get(str(payload["instance_id"]))
            if assessment is None:
                raise ValueError("support assessment is missing")
            return {
                "verdict": str(assessment["verdict"]),
                "reason": str(assessment["reason"]),
            }

        expected_support = build_citation_support_closure(
            run_dir,
            config,
            paper_text=paper_text,
            assessor=_stored_assessor,
            critic_model=config.paper_revision.critic_model,
        )
        if stored_support != expected_support or not stored_support["valid"]:
            raise ValueError("citation support closure replay mismatch")
        _validate_stage24_bindings(run_dir, stored_support)
    except CitationAuditError:
        raise
    except Exception as exc:  # noqa: BLE001
        _raise("citation_support_replay_failed", str(exc), "stage-24/citation_support.json")


def _load_active_config(run_dir: Path) -> RCConfig:
    pointer = run_dir / "active_config_snapshot.json"
    relative = "config.yaml"
    if pointer.exists():
        data = _strict_json(_read(run_dir, "active_config_snapshot.json"))
        relative = str(data.get("config_source_path") or "")
    if not relative or "\\" in relative:
        _raise("citation_config_binding_invalid", "Unsafe active config path.", "active_config_snapshot.json")
    path = Path(relative)
    if path.is_absolute() or relative != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        _raise("citation_config_binding_invalid", "Unsafe active config path.", "active_config_snapshot.json")
    text = _read(run_dir, relative)
    try:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("config root must be an object")
        config = RCConfig.from_dict(raw, project_root=run_dir, check_paths=False)
        selected, selected_text, _digest = resolve_active_config_snapshot(run_dir, config)
        if selected != relative or selected_text != text:
            raise ValueError("active config replay mismatch")
        return config
    except Exception as exc:  # noqa: BLE001
        _raise("citation_config_binding_invalid", str(exc), relative)


def _replay_screening_admission(
    config: RCConfig,
    candidates: tuple[dict[str, Any], ...],
    report: dict[str, Any],
) -> None:
    keywords = _extract_topic_keywords(config.research.topic, config.research.domains)
    ranked: list[dict[str, Any]] = []
    rejected: list[str] = []
    for source in candidates:
        row = dict(source)
        text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
        overlap = sum(1 for keyword in keywords if keyword in text)
        if overlap < 1:
            rejected.append(str(row["source_identity"]))
            continue
        row["keyword_overlap"] = overlap
        row["screen_rank_score"] = round(_candidate_screen_score(row, keywords), 3)
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            float(row["screen_rank_score"]),
            int(row["keyword_overlap"]),
            int(row.get("citation_count", 0) or 0),
            str(row["source_identity"]),
        ),
        reverse=True,
    )
    admitted = ranked[:MAX_SCREEN_CANDIDATES]
    rejected.extend(str(row["source_identity"]) for row in ranked[MAX_SCREEN_CANDIDATES:])
    if report["prefilter_rejected_candidate_ids"] != rejected:
        raise ScreeningContractError("screening prefilter partition replay mismatch")
    admitted_ids = {str(row["source_identity"]) for row in admitted}
    recorded_admitted = set(report["screened_candidate_ids"]) | set(
        report["unscreened_candidate_ids"]
    )
    if admitted_ids != recorded_admitted:
        raise ScreeningContractError("screening admission replay mismatch")


def _validate_stage24_bindings(run_dir: Path, support: dict[str, Any]) -> None:
    claims = _strict_json(_read(run_dir, "stage-24/claims.json"))
    citations = _strict_json(_read(run_dir, "stage-24/citations.json"))
    truth = _strict_json(_read(run_dir, "stage-24/truth_audit.json"))
    claim_rows = claims.get("claims")
    citation_rows = citations.get("instances")
    if not isinstance(claim_rows, list) or not isinstance(citation_rows, list):
        raise ValueError("Stage 24 claim/citation arrays are missing")
    claims_by_id = {
        str(row.get("id")): row for row in claim_rows if isinstance(row, dict)
    }
    expected_citations: list[tuple[str, str, str]] = []
    for row in support["instances"]:
        claim = claims_by_id.get(row["claim_id"])
        if not isinstance(claim, dict):
            raise ValueError("support claim is missing from claims.json")
        if (
            claim.get("text") != row["claim_text"]
            or claim.get("cited_keys") != [row["cite_key"]]
            or claim.get("status") != "supported"
        ):
            raise ValueError("support claim binding mismatch")
        expected_citations.append(
            (row["instance_id"], row["cite_key"], row["claim_id"])
        )
    actual_citations = [
        (
            str(row.get("instance_id")),
            str(row.get("cite_key")),
            str(row.get("supported_claim_id")),
        )
        for row in citation_rows
        if isinstance(row, dict) and row.get("role") == "claim_support"
    ]
    if actual_citations != expected_citations:
        raise ValueError("citations.json support binding mismatch")
    if (
        truth.get("citation_support_path") != "stage-24/citation_support.json"
        or truth.get("citation_support_sha256") != sha256_text(
            _read(run_dir, "stage-24/citation_support.json")
        )
        or truth.get("citation_support_valid") is not True
        or truth.get("dataset_origin") != support["dataset_origin"]
        or truth.get("dataset_claim_violations") != support["dataset_claim_violations"]
    ):
        raise ValueError("truth_audit citation support binding mismatch")


def _validate_verification_report(run_dir: Path, paper_text: str) -> None:
    report = _strict_json(_read(run_dir, "stage-23/verification_report.json"))
    summary = report.get("summary")
    results = report.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        _raise(
            "citation_verification_replay_failed",
            "Verification summary/results are missing.",
            "stage-23/verification_report.json",
        )
    cited = set(extract_citation_keys(paper_text))
    result_keys: list[str] = []
    statuses: list[str] = []
    for row in results:
        if not isinstance(row, dict):
            _raise(
                "citation_verification_replay_failed",
                "Invalid verification result row.",
                "stage-23/verification_report.json",
            )
        key = str(row.get("cite_key") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        if not key or status not in {"verified", "suspicious", "hallucinated", "skipped"}:
            _raise(
                "citation_verification_replay_failed",
                "Invalid verification result identity or status.",
                "stage-23/verification_report.json",
            )
        result_keys.append(key)
        statuses.append(status)
    if len(result_keys) != len(set(result_keys)) or set(result_keys) != cited:
        _raise(
            "citation_verification_replay_failed",
            "Verification result key closure mismatch.",
            "stage-23/verification_report.json",
        )
    counts = {
        status: statuses.count(status)
        for status in ("verified", "suspicious", "hallucinated", "skipped")
    }
    if (
        summary.get("total") != len(results)
        or any(summary.get(status) != count for status, count in counts.items())
        or summary.get("cited_keys") != sorted(cited)
        or summary.get("verification_complete") is not True
        or summary.get("relevance_complete") is not True
        or summary.get("fatal") is not False
        or summary.get("degraded") is not False
        or counts["verified"] != len(results)
    ):
        _raise(
            "citation_verification_replay_failed",
            "Verification report is incomplete or internally inconsistent.",
            "stage-23/verification_report.json",
        )


def _read(run_dir: Path, relative: str) -> str:
    path = run_dir / relative
    try:
        if path.is_symlink() or not path.is_file():
            _raise("citation_artifact_missing", f"Missing or unsafe artifact: {relative}", relative)
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _raise("citation_artifact_invalid", f"Cannot read {relative}: {exc}", relative)


def _strict_json(text: str) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _raise(code: str, message: str, path: str) -> None:
    raise CitationAuditError(code, message, path)
