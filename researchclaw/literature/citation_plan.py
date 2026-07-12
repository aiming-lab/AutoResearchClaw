"""Strict deterministic v1 citation plans for Stage 16 and Stage 17."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from researchclaw.config import RCConfig
from researchclaw.literature.citation_policy import (
    CitationPolicyContractError,
    load_effective_citation_policy,
    validate_citation_allowlist,
)
from researchclaw.literature.evidence_cards import (
    canonical_json_text,
    load_validated_cards,
)
from researchclaw.literature.experiment_fact_closure import (
    ExperimentFactClosureError,
    build_experiment_fact_closure_report,
    parse_experiment_fact_closure_report,
)
from researchclaw.literature.citation_identity import (
    CitationIdentityError,
    parse_cite_key_registry,
    validate_registry_artifacts,
)
from researchclaw.literature.screening import sha256_text
from researchclaw.pipeline.sectional_validation import extract_citation_keys
from researchclaw.pipeline.manuscript_sections import (
    ManuscriptStructureError,
    parse_manuscript,
)


CITATION_PLAN_SCHEMA_VERSION = 1
CITATION_PLAN_VERSION = 1


class CitationPlanContractError(ValueError):
    """Raised when a citation plan is not closed over retained evidence."""


def build_citation_plan(
    run_dir: Path, config: RCConfig, *, plan_status: str
) -> dict[str, Any]:
    if plan_status not in {"preliminary", "final"}:
        raise CitationPlanContractError("invalid citation plan status")
    allowlist_path = run_dir / "stage-06" / "citation_allowlist.json"
    manifest_path = run_dir / "stage-06" / "cards_manifest.json"
    policy_path = run_dir / "stage-16" / "citation_policy_effective.json"
    try:
        allowlist_text = allowlist_path.read_text(encoding="utf-8")
        manifest_text = manifest_path.read_text(encoding="utf-8")
        policy_text = policy_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPlanContractError(f"cannot read citation-plan source: {exc}") from exc
    try:
        allowlist = validate_citation_allowlist(run_dir, config, allowlist_text)
        policy = load_effective_citation_policy(run_dir, config)
        cards = load_validated_cards(run_dir, config)
    except (CitationPolicyContractError, ValueError) as exc:
        raise CitationPlanContractError(f"invalid citation-plan source: {exc}") from exc

    cards_by_key = {str(card["cite_key"]): card for card in cards}
    target = int(policy["effective_target_unique_sources"])
    selected_keys = list(allowlist["eligible_keys"])[:target]
    if len(selected_keys) < int(policy["effective_min_unique_sources"]):
        raise CitationPlanContractError("citation plan cannot meet effective minimum")
    claims: list[dict[str, Any]] = []
    for ordinal, cite_key in enumerate(selected_keys, start=1):
        card = cards_by_key.get(cite_key)
        if card is None or card["extraction_status"] != "success":
            raise CitationPlanContractError("eligible key lacks successful card")
        excerpts = card["evidence_excerpts"]
        if not excerpts:
            raise CitationPlanContractError("eligible key lacks retained excerpt")
        claims.append(
            {
                "claim_id": f"planned-claim-{ordinal:03d}",
                "section_path": ["Introduction"],
                "claim_text": excerpts[0]["excerpt_text"],
                "claim_type": "background",
                "planned_citations": [
                    {
                        "cite_key": cite_key,
                        "evidence_excerpt_ids": [
                            excerpt["excerpt_id"] for excerpt in excerpts
                        ],
                        "support_status": "abstract_sufficient",
                    }
                ],
            }
        )
    payload = {
        "schema_version": CITATION_PLAN_SCHEMA_VERSION,
        "plan_version": CITATION_PLAN_VERSION,
        "plan_status": plan_status,
        "claim_scope": config.experiment.claim_scope,
        "citation_allowlist_path": "stage-06/citation_allowlist.json",
        "citation_allowlist_sha256": sha256_text(allowlist_text),
        "cards_manifest_path": "stage-06/cards_manifest.json",
        "cards_manifest_sha256": sha256_text(manifest_text),
        "effective_policy_path": "stage-16/citation_policy_effective.json",
        "effective_policy_sha256": sha256_text(policy_text),
        "claims": claims,
    }
    return parse_citation_plan(canonical_json_text(payload))


def parse_citation_plan(text: str) -> dict[str, Any]:
    payload = _parse_object(text, "citation plan")
    _exact_keys(
        payload,
        {
            "schema_version", "plan_version", "plan_status", "claim_scope",
            "citation_allowlist_path", "citation_allowlist_sha256",
            "cards_manifest_path", "cards_manifest_sha256",
            "effective_policy_path", "effective_policy_sha256", "claims",
        },
        "citation plan",
    )
    if payload["schema_version"] != CITATION_PLAN_SCHEMA_VERSION:
        raise CitationPlanContractError("unsupported citation plan schema")
    if payload["plan_version"] != CITATION_PLAN_VERSION:
        raise CitationPlanContractError("unsupported citation plan version")
    if payload["plan_status"] not in {"preliminary", "final"}:
        raise CitationPlanContractError("invalid citation plan status")
    if payload["claim_scope"] not in {"pipeline_validation", "exploratory", "research_release"}:
        raise CitationPlanContractError("invalid citation plan claim_scope")
    expected_paths = {
        "citation_allowlist_path": "stage-06/citation_allowlist.json",
        "cards_manifest_path": "stage-06/cards_manifest.json",
        "effective_policy_path": "stage-16/citation_policy_effective.json",
    }
    for field, expected in expected_paths.items():
        if payload[field] != expected:
            raise CitationPlanContractError(f"noncanonical {field}")
    for field in (
        "citation_allowlist_sha256", "cards_manifest_sha256",
        "effective_policy_sha256",
    ):
        _sha256_field(payload, field)
    if not isinstance(payload["claims"], list) or not payload["claims"]:
        raise CitationPlanContractError("citation plan claims must not be empty")
    claim_ids: set[str] = set()
    planned_keys: set[str] = set()
    for ordinal, claim in enumerate(payload["claims"], start=1):
        if not isinstance(claim, dict):
            raise CitationPlanContractError("planned claim must be an object")
        _exact_keys(
            claim,
            {"claim_id", "section_path", "claim_text", "claim_type", "planned_citations"},
            "planned claim",
        )
        claim_id = _required_string(claim, "claim_id")
        if claim_id != f"planned-claim-{ordinal:03d}" or claim_id in claim_ids:
            raise CitationPlanContractError("planned claim ID sequence mismatch")
        claim_ids.add(claim_id)
        if claim["section_path"] != ["Introduction"]:
            raise CitationPlanContractError("unsupported v1 section_path")
        _required_string(claim, "claim_text")
        if claim["claim_type"] != "background":
            raise CitationPlanContractError("unsupported v1 claim_type")
        citations = claim["planned_citations"]
        if not isinstance(citations, list) or len(citations) != 1:
            raise CitationPlanContractError("v1 claim requires one planned citation")
        citation = citations[0]
        if not isinstance(citation, dict):
            raise CitationPlanContractError("planned citation must be an object")
        _exact_keys(
            citation,
            {"cite_key", "evidence_excerpt_ids", "support_status"},
            "planned citation",
        )
        key = _required_string(citation, "cite_key")
        if key in planned_keys:
            raise CitationPlanContractError("duplicate planned cite_key")
        planned_keys.add(key)
        ids = citation["evidence_excerpt_ids"]
        if not isinstance(ids, list) or not ids or any(
            not isinstance(item, str) or not item.strip() for item in ids
        ) or len(ids) != len(set(ids)):
            raise CitationPlanContractError("invalid evidence_excerpt_ids")
        if citation["support_status"] != "abstract_sufficient":
            raise CitationPlanContractError("final v1 plan requires abstract_sufficient")
    return payload


def validate_citation_plan(
    run_dir: Path, config: RCConfig, text: str, *, plan_status: str
) -> dict[str, Any]:
    stored = parse_citation_plan(text)
    expected = build_citation_plan(run_dir, config, plan_status=plan_status)
    if stored != expected:
        raise CitationPlanContractError("citation plan replay mismatch")
    return stored


def load_final_citation_plan(run_dir: Path, config: RCConfig) -> dict[str, Any]:
    path = run_dir / "stage-16" / "citation_plan.json"
    try:
        if path.is_symlink() or not path.is_file():
            raise CitationPlanContractError("final citation plan is missing or unsafe")
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPlanContractError(f"cannot read final citation plan: {exc}") from exc
    return validate_citation_plan(run_dir, config, text, plan_status="final")


def build_citation_writer_instruction(run_dir: Path, config: RCConfig) -> str:
    """Render the bounded final-plan citation surface for Stage 17."""

    plan = load_final_citation_plan(run_dir, config)
    cards = load_validated_cards(run_dir, config)
    cards_by_key = {str(card["cite_key"]): card for card in cards}
    blocks: list[str] = []
    for claim in plan["claims"]:
        citation = claim["planned_citations"][0]
        cite_key = citation["cite_key"]
        card = cards_by_key.get(cite_key)
        if card is None or card["extraction_status"] != "success":
            raise CitationPlanContractError(
                f"planned key lacks a successful evidence card: {cite_key}"
            )
        excerpts_by_id = {
            excerpt["excerpt_id"]: excerpt
            for excerpt in card["evidence_excerpts"]
        }
        excerpt_lines: list[str] = []
        for excerpt_id in citation["evidence_excerpt_ids"]:
            excerpt = excerpts_by_id.get(excerpt_id)
            if excerpt is None:
                raise CitationPlanContractError(
                    f"planned excerpt is absent from its card: {excerpt_id}"
                )
            excerpt_lines.append(
                f'  - {excerpt_id}: "{excerpt["excerpt_text"]}"'
            )
        blocks.append(
            "\n".join(
                (
                    f"- CLAIM {claim['claim_id']} (section: Introduction)",
                    f"  Allowed wording ceiling: {claim['claim_text']}",
                    f"  Required citation key: [{cite_key}]",
                    "  Retained abstract evidence:",
                    *excerpt_lines,
                )
            )
        )
    if not blocks:
        raise CitationPlanContractError("final citation plan has no writable claims")
    return (
        "\n\nFINAL CITATION PLAN (THE ONLY CITATION AUTHORITY):\n"
        + "\n\n".join(blocks)
        + "\n\nCITATION RULES:\n"
        "- Cite every required key above at least once using exact [cite_key] syntax.\n"
        "- Do not use any citation key that is not listed above.\n"
        "- Do not strengthen a claim beyond its retained abstract evidence.\n"
        "- Do not write a References section; it is generated from the canonical bibliography.\n"
        "- If the evidence does not support a stronger sentence, keep the bounded wording.\n"
    )


def load_canonical_bibliography(run_dir: Path) -> str:
    """Load and replay the immutable registry-bound Stage 4 bibliography."""

    stage04 = run_dir / "stage-04"
    candidates_path = stage04 / "candidates.jsonl"
    registry_path = stage04 / "cite_key_registry.json"
    bibliography_path = stage04 / "references.bib"
    try:
        for path in (candidates_path, registry_path, bibliography_path):
            if path.is_symlink() or not path.is_file():
                raise CitationPlanContractError(
                    f"canonical bibliography source is missing or unsafe: {path.name}"
                )
        candidates_text = candidates_path.read_text(encoding="utf-8")
        registry_text = registry_path.read_text(encoding="utf-8")
        bibliography = bibliography_path.read_text(encoding="utf-8")
        registry = parse_cite_key_registry(registry_text)
        validate_registry_artifacts(registry, candidates_text, bibliography)
    except (OSError, UnicodeDecodeError, CitationIdentityError) as exc:
        raise CitationPlanContractError(f"canonical bibliography is invalid: {exc}") from exc
    return bibliography


def validate_final_paper_citations(
    run_dir: Path, config: RCConfig, paper_text: str
) -> tuple[str, ...]:
    """Replay the final paper against the evidence-bound allowlist and plan."""

    allowlist_path = run_dir / "stage-06" / "citation_allowlist.json"
    try:
        if allowlist_path.is_symlink() or not allowlist_path.is_file():
            raise CitationPlanContractError("citation allowlist is missing or unsafe")
        allowlist_text = allowlist_path.read_text(encoding="utf-8")
        allowlist = validate_citation_allowlist(run_dir, config, allowlist_text)
        plan = load_final_citation_plan(run_dir, config)
    except (OSError, UnicodeDecodeError, CitationPolicyContractError) as exc:
        raise CitationPlanContractError(
            f"cannot validate final paper citations: {exc}"
        ) from exc
    cited = set(extract_citation_keys(paper_text))
    eligible = set(allowlist["eligible_keys"])
    planned = {
        citation["cite_key"]
        for claim in plan["claims"]
        for citation in claim["planned_citations"]
    }
    unknown = sorted(cited - eligible)
    unplanned = sorted(cited - planned)
    missing = sorted(planned - cited)
    if unknown or unplanned or missing:
        raise CitationPlanContractError(
            "final paper citation closure failed: "
            f"unknown={unknown}, unplanned={unplanned}, missing={missing}"
        )
    return tuple(sorted(cited))


def validate_paper_citation_minimum(
    run_dir: Path,
    config: RCConfig,
    paper_text: str,
    *,
    minimum: int,
) -> tuple[str, ...]:
    """Require the current paper to meet policy using eligible planned keys."""

    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise CitationPlanContractError("citation minimum must be a nonnegative integer")
    allowlist_path = run_dir / "stage-06" / "citation_allowlist.json"
    try:
        if allowlist_path.is_symlink() or not allowlist_path.is_file():
            raise CitationPlanContractError("citation allowlist is missing or unsafe")
        allowlist_text = allowlist_path.read_text(encoding="utf-8")
        allowlist = validate_citation_allowlist(run_dir, config, allowlist_text)
        plan = load_final_citation_plan(run_dir, config)
    except (OSError, UnicodeDecodeError, CitationPolicyContractError) as exc:
        raise CitationPlanContractError(
            f"cannot validate paper citation minimum: {exc}"
        ) from exc
    cited = set(extract_citation_keys(paper_text))
    eligible = set(allowlist["eligible_keys"])
    planned = {
        citation["cite_key"]
        for claim in plan["claims"]
        for citation in claim["planned_citations"]
    }
    invalid = sorted(cited - eligible)
    unplanned = sorted(cited - planned)
    counted = sorted(cited & eligible & planned)
    if invalid or unplanned or len(counted) < minimum:
        raise CitationPlanContractError(
            "paper citation minimum failed: "
            f"eligible_count={len(counted)}, minimum={minimum}, "
            f"invalid={invalid}, unplanned={unplanned}"
        )
    return tuple(counted)


def build_citation_closure_report(
    run_dir: Path,
    config: RCConfig,
    *,
    paper_text: str,
    structure_report_text: str,
    experiment_fact_report_text: str,
) -> dict[str, Any]:
    plan_path = run_dir / "stage-16" / "citation_plan.json"
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
        plan = load_final_citation_plan(run_dir, config)
        structure = _parse_object(structure_report_text, "paper structure report")
        experiment = parse_experiment_fact_closure_report(
            experiment_fact_report_text
        )
        expected_experiment = build_experiment_fact_closure_report(
            run_dir, paper_text=paper_text
        )
        if experiment != expected_experiment:
            raise CitationPlanContractError("experiment fact closure replay mismatch")
    except (
        OSError,
        UnicodeDecodeError,
        CitationPlanContractError,
        ExperimentFactClosureError,
    ) as exc:
        raise CitationPlanContractError(f"cannot build citation closure: {exc}") from exc
    planned = [
        citation["cite_key"]
        for claim in plan["claims"]
        for citation in claim["planned_citations"]
    ]
    allowlist_path = run_dir / "stage-06" / "citation_allowlist.json"
    allowlist_text = allowlist_path.read_text(encoding="utf-8")
    allowlist = validate_citation_allowlist(run_dir, config, allowlist_text)
    cited = sorted(extract_citation_keys(paper_text))
    unknown = sorted(set(cited) - set(allowlist["eligible_keys"]))
    unplanned = sorted(set(cited) - set(planned))
    missing = sorted(set(planned) - set(cited))
    structure_valid = False
    experiment_valid = experiment.get("valid") is True
    misplaced: list[str] = []
    try:
        document = parse_manuscript(paper_text, strict=True)
    except ManuscriptStructureError:
        document = None
    if document is not None:
        structure_valid = (
            set(structure) == {
                "schema_version", "valid", "source_sha256", "section_count", "issues"
            }
            and structure.get("schema_version") == 1
            and structure.get("valid") is True
            and structure.get("source_sha256") == sha256_text(paper_text)
            and structure.get("section_count") == len(document.sections)
            and structure.get("issues") == []
        )
        if structure_valid:
            section_keys: dict[str, set[str]] = {}
            for section in document.sections:
                section_keys.setdefault(section.title.casefold(), set()).update(
                    extract_citation_keys(section.body)
                )
            for claim in plan["claims"]:
                assigned = claim["section_path"][-1].casefold()
                for citation in claim["planned_citations"]:
                    if citation["cite_key"] not in section_keys.get(assigned, set()):
                        misplaced.append(citation["cite_key"])
    misplaced = sorted(set(misplaced))
    payload = {
        "schema_version": 1,
        "paper_path": "stage-17/paper_draft.md",
        "paper_sha256": sha256_text(paper_text),
        "citation_plan_path": "stage-16/citation_plan.json",
        "citation_plan_sha256": sha256_text(plan_text),
        "cited_keys": cited,
        "unknown_keys": unknown,
        "unplanned_keys": unplanned,
        "missing_planned_keys": missing,
        "misplaced_planned_keys": misplaced,
        "structure_report_path": "stage-17/paper_structure_report.json",
        "structure_report_sha256": sha256_text(structure_report_text),
        "structure_valid": structure_valid,
        "experiment_fact_closure_report_path": "stage-17/experiment_fact_closure_report.json",
        "experiment_fact_closure_report_sha256": sha256_text(experiment_fact_report_text),
        "experiment_fact_closure_valid": experiment_valid,
        "valid": (
            not unknown and not unplanned and not missing and not misplaced
            and structure_valid and experiment_valid
        ),
    }
    return parse_citation_closure_report(canonical_json_text(payload))


def parse_citation_closure_report(text: str) -> dict[str, Any]:
    payload = _parse_object(text, "citation closure report")
    _exact_keys(
        payload,
        {
            "schema_version", "paper_path", "paper_sha256",
            "citation_plan_path", "citation_plan_sha256", "cited_keys",
            "unknown_keys", "unplanned_keys", "missing_planned_keys",
            "misplaced_planned_keys",
            "structure_report_path", "structure_report_sha256",
            "structure_valid", "experiment_fact_closure_report_path",
            "experiment_fact_closure_report_sha256",
            "experiment_fact_closure_valid", "valid",
        },
        "citation closure report",
    )
    if payload["schema_version"] != 1:
        raise CitationPlanContractError("unsupported citation closure schema")
    expected_paths = {
        "paper_path": "stage-17/paper_draft.md",
        "citation_plan_path": "stage-16/citation_plan.json",
        "structure_report_path": "stage-17/paper_structure_report.json",
        "experiment_fact_closure_report_path": "stage-17/experiment_fact_closure_report.json",
    }
    for field, expected in expected_paths.items():
        if payload[field] != expected:
            raise CitationPlanContractError(f"noncanonical closure {field}")
    for field in (
        "paper_sha256", "citation_plan_sha256", "structure_report_sha256",
        "experiment_fact_closure_report_sha256",
    ):
        _sha256_field(payload, field)
    for field in (
        "cited_keys", "unknown_keys", "unplanned_keys", "missing_planned_keys",
        "misplaced_planned_keys",
    ):
        value = payload[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ) or value != sorted(set(value)):
            raise CitationPlanContractError(f"invalid closure {field}")
    if (
        not isinstance(payload["structure_valid"], bool)
        or not isinstance(payload["experiment_fact_closure_valid"], bool)
        or not isinstance(payload["valid"], bool)
    ):
        raise CitationPlanContractError("closure validity fields must be booleans")
    expected_valid = (
        payload["structure_valid"]
        and payload["experiment_fact_closure_valid"]
        and not payload["unknown_keys"]
        and not payload["unplanned_keys"]
        and not payload["missing_planned_keys"]
        and not payload["misplaced_planned_keys"]
    )
    if payload["valid"] is not expected_valid:
        raise CitationPlanContractError("citation closure valid mismatch")
    return payload


def validate_citation_closure_report(run_dir: Path, config: RCConfig) -> dict[str, Any]:
    paper_path = run_dir / "stage-17" / "paper_draft.md"
    structure_path = run_dir / "stage-17" / "paper_structure_report.json"
    experiment_path = run_dir / "stage-17" / "experiment_fact_closure_report.json"
    report_path = run_dir / "stage-17" / "citation_closure_report.json"
    try:
        paper_text = paper_path.read_text(encoding="utf-8")
        structure_text = structure_path.read_text(encoding="utf-8")
        experiment_text = experiment_path.read_text(encoding="utf-8")
        stored = parse_citation_closure_report(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPlanContractError(f"cannot read citation closure artifacts: {exc}") from exc
    expected = build_citation_closure_report(
        run_dir,
        config,
        paper_text=paper_text,
        structure_report_text=structure_text,
        experiment_fact_report_text=experiment_text,
    )
    if stored != expected or not stored["valid"]:
        raise CitationPlanContractError("citation closure replay failed")
    return stored


def _parse_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CitationPlanContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CitationPlanContractError(f"{label} root must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CitationPlanContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise CitationPlanContractError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CitationPlanContractError(f"{field} must be a nonempty string")
    return value


def _sha256_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CitationPlanContractError(f"invalid {field}")
    return value
