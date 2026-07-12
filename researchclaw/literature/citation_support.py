"""Deterministic Stage 24 citation-support and dataset-origin closure."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from researchclaw.experiment_runtime.contract import find_stage09_contract, load_contract
from researchclaw.literature.citation_plan import load_final_citation_plan
from researchclaw.literature.evidence_cards import load_validated_cards
from researchclaw.literature.experiment_fact_closure import find_dataset_claim_violations
from researchclaw.pipeline.release_artifacts import extract_citation_instances


SUPPORT_CLOSURE_SCHEMA_VERSION = 1
SUPPORT_VERDICTS = {"supported", "unsupported"}


class CitationSupportContractError(ValueError):
    """Raised when citation support cannot be reproduced from canonical inputs."""


AssessmentFn = Callable[[dict[str, Any]], dict[str, str]]


def build_citation_support_closure(
    run_dir: Path,
    config: Any,
    *,
    paper_text: str,
    assessor: AssessmentFn | None,
    critic_model: str,
) -> dict[str, Any]:
    """Build one support obligation for every final-paper citation instance."""

    stage22_paper = run_dir / "stage-22" / "paper_final.md"
    try:
        if stage22_paper.is_symlink() or not stage22_paper.is_file():
            raise CitationSupportContractError("canonical Stage 22 paper is missing or unsafe")
        if stage22_paper.read_text(encoding="utf-8") != paper_text:
            raise CitationSupportContractError("Stage 23 paper is not byte-identical to Stage 22")
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationSupportContractError(f"cannot bind canonical paper: {exc}") from exc

    plan = load_final_citation_plan(run_dir, config)
    cards = load_validated_cards(run_dir, config)
    cards_by_key = {str(card["cite_key"]): card for card in cards}
    plan_by_key: dict[str, dict[str, Any]] = {}
    for claim in plan["claims"]:
        for citation in claim["planned_citations"]:
            key = str(citation["cite_key"])
            if key in plan_by_key:
                raise CitationSupportContractError("duplicate key in final citation plan")
            plan_by_key[key] = {
                "claim": claim,
                "citation": citation,
            }

    verification_path = run_dir / "stage-23" / "verification_report.json"
    verification = _read_object(verification_path, "verification report")
    verification_results = verification.get("results")
    if not isinstance(verification_results, list):
        raise CitationSupportContractError("verification results are missing")
    existence_by_key: dict[str, str] = {}
    for row in verification_results:
        if not isinstance(row, dict):
            raise CitationSupportContractError("invalid verification result")
        key = _required_string(row, "cite_key")
        status = _required_string(row, "status").lower()
        if key in existence_by_key:
            raise CitationSupportContractError("duplicate verification cite_key")
        existence_by_key[key] = status

    instances = extract_citation_instances(paper_text)
    cited_keys = {str(instance["cite_key"]) for instance in instances}
    if set(existence_by_key) != cited_keys:
        raise CitationSupportContractError("verification result key closure mismatch")
    obligations: list[dict[str, Any]] = []
    for instance in instances:
        key = str(instance["cite_key"])
        planned = plan_by_key.get(key)
        card = cards_by_key.get(key)
        if planned is None or card is None or card["extraction_status"] != "success":
            raise CitationSupportContractError(
                f"citation instance lacks planned retained evidence: {key}"
            )
        excerpts_by_id = {
            str(excerpt["excerpt_id"]): excerpt
            for excerpt in card["evidence_excerpts"]
        }
        planned_ids = tuple(planned["citation"]["evidence_excerpt_ids"])
        excerpts: list[dict[str, Any]] = []
        for excerpt_id in planned_ids:
            excerpt = excerpts_by_id.get(str(excerpt_id))
            if excerpt is None:
                raise CitationSupportContractError(
                    f"planned excerpt is absent from evidence card: {excerpt_id}"
                )
            excerpts.append(
                {
                    "excerpt_id": excerpt["excerpt_id"],
                    "source_artifact_path": excerpt["source_artifact_path"],
                    "source_artifact_sha256": excerpt["source_artifact_sha256"],
                    "source_record_id": excerpt["source_record_id"],
                    "json_pointer": excerpt["json_pointer"],
                    "char_start": excerpt["char_start"],
                    "char_end": excerpt["char_end"],
                    "excerpt_text": excerpt["excerpt_text"],
                    "excerpt_sha256": excerpt["excerpt_sha256"],
                }
            )
        assessment_input = {
            "instance_id": instance["instance_id"],
            "cite_key": key,
            "claim_text": instance["claim_text"],
            "citation_context": instance["context"],
            "planned_claim_id": planned["claim"]["claim_id"],
            "planned_claim_text": planned["claim"]["claim_text"],
            "evidence_excerpts": excerpts,
        }
        assessment = assessor(assessment_input) if assessor is not None else {
            "verdict": "unsupported",
            "reason": "No isolated support critic was available.",
        }
        if not isinstance(assessment, dict) or set(assessment) != {"verdict", "reason"}:
            raise CitationSupportContractError("support assessment fields mismatch")
        verdict = _required_string(assessment, "verdict")
        reason = _required_string(assessment, "reason")
        if verdict not in SUPPORT_VERDICTS:
            raise CitationSupportContractError("invalid support assessment verdict")
        existence_status = existence_by_key.get(key, "missing")
        if existence_status != "verified":
            verdict = "unsupported"
            reason = f"Citation existence status is {existence_status}."
        obligations.append(
            {
                "instance_id": instance["instance_id"],
                "cite_key": key,
                "claim_id": f"support-{instance['instance_id']}",
                "claim_text": instance["claim_text"],
                "claim_text_sha256": _sha256(instance["claim_text"]),
                "planned_claim_id": planned["claim"]["claim_id"],
                "existence_status": existence_status,
                "evidence_excerpts": excerpts,
                "assessment": {
                    "verdict": verdict,
                    "reason": reason,
                    "critic_model": critic_model,
                    "context_isolated": True,
                },
            }
        )

    contract_path = find_stage09_contract(run_dir)
    if contract_path is None or contract_path.is_symlink() or not contract_path.is_file():
        raise CitationSupportContractError("canonical experiment contract is missing or unsafe")
    contract = load_contract(contract_path)
    contract_text = contract_path.read_text(encoding="utf-8")
    dataset_violations = list(
        find_dataset_claim_violations(paper_text, contract.dataset_origin)
    )
    payload = {
        "schema_version": SUPPORT_CLOSURE_SCHEMA_VERSION,
        "paper_path": "stage-23/paper_final_verified.md",
        "paper_sha256": _sha256(paper_text),
        "citation_plan_path": "stage-16/citation_plan.json",
        "citation_plan_sha256": _sha256(
            (run_dir / "stage-16" / "citation_plan.json").read_text(encoding="utf-8")
        ),
        "cards_manifest_path": "stage-06/cards_manifest.json",
        "cards_manifest_sha256": _sha256(
            (run_dir / "stage-06" / "cards_manifest.json").read_text(encoding="utf-8")
        ),
        "verification_report_path": "stage-23/verification_report.json",
        "verification_report_sha256": _sha256(
            verification_path.read_text(encoding="utf-8")
        ),
        "experiment_contract_path": contract_path.relative_to(run_dir).as_posix(),
        "experiment_contract_sha256": _sha256(contract_text),
        "dataset_origin": contract.dataset_origin,
        "dataset_claim_violations": dataset_violations,
        "instances": obligations,
        "counts": {
            "total": len(obligations),
            "supported": sum(
                row["assessment"]["verdict"] == "supported" for row in obligations
            ),
            "unsupported": sum(
                row["assessment"]["verdict"] == "unsupported" for row in obligations
            ),
            "dataset_claim_violations": len(dataset_violations),
        },
        "valid": all(row["assessment"]["verdict"] == "supported" for row in obligations)
        and not dataset_violations,
    }
    return parse_citation_support_closure(_canonical_json(payload))


def parse_citation_support_closure(text: str) -> dict[str, Any]:
    payload = _read_json_text(text, "citation support closure")
    expected = {
        "schema_version", "paper_path", "paper_sha256", "citation_plan_path",
        "citation_plan_sha256", "cards_manifest_path", "cards_manifest_sha256",
        "verification_report_path", "verification_report_sha256",
        "experiment_contract_path", "experiment_contract_sha256", "dataset_origin",
        "dataset_claim_violations", "instances", "counts", "valid",
    }
    if set(payload) != expected:
        raise CitationSupportContractError("citation support closure fields mismatch")
    if payload["schema_version"] != SUPPORT_CLOSURE_SCHEMA_VERSION:
        raise CitationSupportContractError("unsupported citation support schema")
    for field, required in {
        "paper_path": "stage-23/paper_final_verified.md",
        "citation_plan_path": "stage-16/citation_plan.json",
        "cards_manifest_path": "stage-06/cards_manifest.json",
        "verification_report_path": "stage-23/verification_report.json",
    }.items():
        if payload[field] != required:
            raise CitationSupportContractError(f"noncanonical {field}")
    _safe_relative_path(payload["experiment_contract_path"])
    for field in (
        "paper_sha256", "citation_plan_sha256", "cards_manifest_sha256",
        "verification_report_sha256", "experiment_contract_sha256",
    ):
        if not isinstance(payload[field], str) or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None:
            raise CitationSupportContractError(f"invalid {field}")
    if payload["dataset_origin"] not in {"synthetic", "public", "local_hardware"}:
        raise CitationSupportContractError("invalid dataset_origin")
    violations = payload["dataset_claim_violations"]
    if not isinstance(violations, list) or any(
        not isinstance(item, str) or not item for item in violations
    ) or violations != sorted(set(violations)):
        raise CitationSupportContractError("invalid dataset claim violations")
    instances = payload["instances"]
    if not isinstance(instances, list):
        raise CitationSupportContractError("instances must be an array")
    seen: set[str] = set()
    for ordinal, row in enumerate(instances):
        _validate_instance(row, ordinal, seen)
    counts = payload["counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "total", "supported", "unsupported", "dataset_claim_violations"
    }:
        raise CitationSupportContractError("invalid support counts")
    expected_counts = {
        "total": len(instances),
        "supported": sum(row["assessment"]["verdict"] == "supported" for row in instances),
        "unsupported": sum(row["assessment"]["verdict"] == "unsupported" for row in instances),
        "dataset_claim_violations": len(violations),
    }
    if counts != expected_counts:
        raise CitationSupportContractError("support counts mismatch")
    expected_valid = not violations and counts["unsupported"] == 0
    if not isinstance(payload["valid"], bool) or payload["valid"] is not expected_valid:
        raise CitationSupportContractError("support valid mismatch")
    return payload


def _validate_instance(row: Any, ordinal: int, seen: set[str]) -> None:
    if not isinstance(row, dict) or set(row) != {
        "instance_id", "cite_key", "claim_id", "claim_text", "claim_text_sha256",
        "planned_claim_id", "existence_status", "evidence_excerpts", "assessment",
    }:
        raise CitationSupportContractError("support instance fields mismatch")
    instance_id = _required_string(row, "instance_id")
    if instance_id != f"cit-{ordinal:04d}" or instance_id in seen:
        raise CitationSupportContractError("support instance sequence mismatch")
    seen.add(instance_id)
    _required_string(row, "cite_key")
    if row["claim_id"] != f"support-{instance_id}":
        raise CitationSupportContractError("support claim ID mismatch")
    claim_text = _required_string(row, "claim_text")
    if row["claim_text_sha256"] != _sha256(claim_text):
        raise CitationSupportContractError("support claim hash mismatch")
    _required_string(row, "planned_claim_id")
    if row["existence_status"] not in {"verified", "suspicious", "hallucinated", "skipped", "missing"}:
        raise CitationSupportContractError("invalid existence status")
    excerpts = row["evidence_excerpts"]
    if not isinstance(excerpts, list) or not excerpts:
        raise CitationSupportContractError("support evidence excerpts are missing")
    excerpt_ids: set[str] = set()
    for excerpt in excerpts:
        required = {
            "excerpt_id", "source_artifact_path", "source_artifact_sha256",
            "source_record_id", "json_pointer", "char_start", "char_end",
            "excerpt_text", "excerpt_sha256",
        }
        if not isinstance(excerpt, dict) or set(excerpt) != required:
            raise CitationSupportContractError("support excerpt fields mismatch")
        excerpt_id = _required_string(excerpt, "excerpt_id")
        if excerpt_id in excerpt_ids:
            raise CitationSupportContractError("duplicate support excerpt ID")
        excerpt_ids.add(excerpt_id)
        text = _required_string(excerpt, "excerpt_text")
        if excerpt["excerpt_sha256"] != _sha256(text):
            raise CitationSupportContractError("support excerpt hash mismatch")
        if excerpt["source_artifact_path"] != "stage-04/candidates.jsonl":
            raise CitationSupportContractError("noncanonical support excerpt source")
        if re.fullmatch(r"[0-9a-f]{64}", str(excerpt["source_artifact_sha256"])) is None:
            raise CitationSupportContractError("invalid support source hash")
        _required_string(excerpt, "source_record_id")
        if excerpt["json_pointer"] != "/abstract":
            raise CitationSupportContractError("invalid support excerpt pointer")
        if isinstance(excerpt["char_start"], bool) or not isinstance(excerpt["char_start"], int):
            raise CitationSupportContractError("invalid excerpt start")
        if isinstance(excerpt["char_end"], bool) or not isinstance(excerpt["char_end"], int):
            raise CitationSupportContractError("invalid excerpt end")
        if excerpt["char_start"] < 0 or excerpt["char_end"] <= excerpt["char_start"]:
            raise CitationSupportContractError("invalid excerpt span")
        if excerpt["char_end"] - excerpt["char_start"] != len(text):
            raise CitationSupportContractError("support excerpt span length mismatch")
    assessment = row["assessment"]
    if not isinstance(assessment, dict) or set(assessment) != {
        "verdict", "reason", "critic_model", "context_isolated"
    }:
        raise CitationSupportContractError("support assessment fields mismatch")
    if assessment["verdict"] not in SUPPORT_VERDICTS:
        raise CitationSupportContractError("invalid support verdict")
    if assessment["verdict"] == "supported" and row["existence_status"] != "verified":
        raise CitationSupportContractError("unverified citation cannot be supported")
    _required_string(assessment, "reason")
    _required_string(assessment, "critic_model")
    if assessment["context_isolated"] is not True:
        raise CitationSupportContractError("support critic context is not isolated")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise CitationSupportContractError(f"{label} is missing or unsafe")
        return _read_json_text(path.read_text(encoding="utf-8"), label)
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationSupportContractError(f"cannot read {label}: {exc}") from exc


def _read_json_text(text: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CitationSupportContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CitationSupportContractError(f"{label} root must be an object")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CitationSupportContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _required_string(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise CitationSupportContractError(f"invalid {field}")
    return item


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CitationSupportContractError("unsafe relative path")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise CitationSupportContractError("unsafe relative path")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
