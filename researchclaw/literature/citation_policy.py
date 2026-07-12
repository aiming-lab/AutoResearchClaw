"""Deterministic citation eligibility and cross-stage policy contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from researchclaw.config import RCConfig
from researchclaw.literature.evidence_cards import (
    canonical_json_text,
    load_validated_card_inputs,
    load_validated_cards,
)
from researchclaw.literature.screening import sha256_text


ALLOWLIST_SCHEMA_VERSION = 1
ELIGIBILITY_POLICY_VERSION = 1
EFFECTIVE_POLICY_SCHEMA_VERSION = 1
EFFECTIVE_POLICY_VERSION = 1
ACTIVE_CONFIG_SCHEMA_VERSION = 1


class CitationPolicyContractError(ValueError):
    """Raised when citation eligibility or effective policy is not replayable."""


def build_citation_allowlist(run_dir: Path, config: RCConfig) -> dict[str, Any]:
    """Recompute citation eligibility from canonical Stage 4-6 artifacts."""
    inputs = load_validated_card_inputs(run_dir, config)
    cards = load_validated_cards(run_dir, config)
    manifest_path = run_dir / "stage-06" / "cards_manifest.json"
    references_path = run_dir / "stage-04" / "references.bib"
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        references_text = references_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPolicyContractError(f"cannot read citation sources: {exc}") from exc

    bib_keys = _bibtex_keys(references_text)
    shortlist_keys = [str(row["cite_key"]) for row in inputs.shortlist]
    if any(key not in bib_keys for key in shortlist_keys):
        raise CitationPolicyContractError("shortlist key is absent from canonical bibliography")

    eligible_keys: list[str] = []
    ineligible: list[dict[str, str]] = []
    for row, card in zip(inputs.shortlist, cards, strict=True):
        cite_key = str(row["cite_key"])
        if card["source_identity"] != row["source_identity"] or card["cite_key"] != cite_key:
            raise CitationPolicyContractError("card/shortlist identity mismatch")
        status = card["extraction_status"]
        if status == "success":
            eligible_keys.append(cite_key)
        else:
            ineligible.append(
                {
                    "cite_key": cite_key,
                    "reason_code": "card_fallback" if status == "fallback" else "card_failed",
                }
            )
    payload = {
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        "shortlist_path": "stage-05/shortlist.jsonl",
        "shortlist_sha256": sha256_text(inputs.shortlist_text),
        "references_path": "stage-04/references.bib",
        "references_sha256": sha256_text(references_text),
        "cards_manifest_path": "stage-06/cards_manifest.json",
        "cards_manifest_sha256": sha256_text(manifest_text),
        "eligible_keys": eligible_keys,
        "ineligible": ineligible,
    }
    return parse_citation_allowlist(canonical_json_text(payload))


def parse_citation_allowlist(text: str) -> dict[str, Any]:
    payload = _parse_object(text, "citation allowlist")
    _exact_keys(
        payload,
        {
            "schema_version",
            "eligibility_policy_version",
            "shortlist_path",
            "shortlist_sha256",
            "references_path",
            "references_sha256",
            "cards_manifest_path",
            "cards_manifest_sha256",
            "eligible_keys",
            "ineligible",
        },
        "citation allowlist",
    )
    if payload["schema_version"] != ALLOWLIST_SCHEMA_VERSION:
        raise CitationPolicyContractError("unsupported allowlist schema_version")
    if payload["eligibility_policy_version"] != ELIGIBILITY_POLICY_VERSION:
        raise CitationPolicyContractError("unsupported eligibility policy version")
    expected_paths = {
        "shortlist_path": "stage-05/shortlist.jsonl",
        "references_path": "stage-04/references.bib",
        "cards_manifest_path": "stage-06/cards_manifest.json",
    }
    for field, expected in expected_paths.items():
        if payload[field] != expected:
            raise CitationPolicyContractError(f"noncanonical {field}")
    for field in ("shortlist_sha256", "references_sha256", "cards_manifest_sha256"):
        _sha256_field(payload, field)
    eligible = _unique_string_list(payload.get("eligible_keys"), "eligible_keys")
    raw_ineligible = payload.get("ineligible")
    if not isinstance(raw_ineligible, list):
        raise CitationPolicyContractError("ineligible must be a list")
    ineligible_keys: list[str] = []
    for item in raw_ineligible:
        if not isinstance(item, dict):
            raise CitationPolicyContractError("ineligible entry must be an object")
        _exact_keys(item, {"cite_key", "reason_code"}, "ineligible entry")
        key = _required_string(item, "cite_key")
        if item["reason_code"] not in {"card_fallback", "card_failed"}:
            raise CitationPolicyContractError("invalid ineligible reason_code")
        ineligible_keys.append(key)
    if len(ineligible_keys) != len(set(ineligible_keys)):
        raise CitationPolicyContractError("duplicate ineligible cite_key")
    if set(eligible) & set(ineligible_keys):
        raise CitationPolicyContractError("eligible/ineligible overlap")
    return payload


def validate_citation_allowlist(
    run_dir: Path, config: RCConfig, text: str
) -> dict[str, Any]:
    stored = parse_citation_allowlist(text)
    expected = build_citation_allowlist(run_dir, config)
    if stored != expected:
        raise CitationPolicyContractError("citation allowlist replay mismatch")
    return stored


def write_active_config_binding(run_dir: Path, snapshot_path: Path) -> None:
    """Bind the exact run-local config snapshot selected by the CLI."""
    try:
        relative = snapshot_path.relative_to(run_dir).as_posix()
    except ValueError as exc:
        raise CitationPolicyContractError("config snapshot is outside run root") from exc
    if not _is_config_snapshot_name(relative):
        raise CitationPolicyContractError("noncanonical config snapshot name")
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise CitationPolicyContractError("config snapshot is missing or unsafe")
    text = snapshot_path.read_text(encoding="utf-8")
    snapshot_hash = sha256_text(text)
    history_path = run_dir / "config_snapshot_history.jsonl"
    history_entries: list[dict[str, Any]] = []
    if history_path.exists():
        if history_path.is_symlink() or not history_path.is_file():
            raise CitationPolicyContractError("config snapshot history is unsafe")
        history_text = history_path.read_text(encoding="utf-8")
        history_entries = _parse_config_snapshot_history(history_text)
        _validate_existing_active_binding(run_dir, history_text, history_entries)
    elif (run_dir / "active_config_snapshot.json").exists():
        raise CitationPolicyContractError(
            "active config pointer exists without snapshot history"
        )
    history_entries.append(
        {
            "schema_version": ACTIVE_CONFIG_SCHEMA_VERSION,
            "ordinal": len(history_entries) + 1,
            "config_source_path": relative,
            "config_source_sha256": snapshot_hash,
        }
    )
    history_text = "".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        for entry in history_entries
    )
    _write_text_atomic(history_path, history_text)
    payload = {
        "schema_version": ACTIVE_CONFIG_SCHEMA_VERSION,
        "config_source_path": relative,
        "config_source_sha256": snapshot_hash,
        "history_path": "config_snapshot_history.jsonl",
        "history_sha256": sha256_text(history_text),
        "history_ordinal": len(history_entries),
    }
    _write_text_atomic(
        run_dir / "active_config_snapshot.json", canonical_json_text(payload)
    )
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise CitationPolicyContractError("checkpoint is unsafe")
        checkpoint = _parse_object(
            checkpoint_path.read_text(encoding="utf-8"), "checkpoint"
        )
        checkpoint["active_config_snapshot_path"] = relative
        checkpoint["active_config_snapshot_sha256"] = snapshot_hash
        checkpoint["config_snapshot_history_sha256"] = sha256_text(history_text)
        _write_text_atomic(checkpoint_path, canonical_json_text(checkpoint))


def resolve_active_config_snapshot(
    run_dir: Path, config: RCConfig
) -> tuple[str, str, str]:
    """Return canonical snapshot path, exact text, and hash for this attempt."""
    pointer_path = run_dir / "active_config_snapshot.json"
    resumed = sorted(run_dir.glob("config.resumed-*.yaml"))
    if pointer_path.exists():
        try:
            if pointer_path.is_symlink() or not pointer_path.is_file():
                raise CitationPolicyContractError("active config pointer is unsafe")
            pointer = _parse_object(
                pointer_path.read_text(encoding="utf-8"), "active config pointer"
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise CitationPolicyContractError(f"cannot read active config pointer: {exc}") from exc
        _exact_keys(
            pointer,
            {
                "schema_version",
                "config_source_path",
                "config_source_sha256",
                "history_path",
                "history_sha256",
                "history_ordinal",
            },
            "active config pointer",
        )
        if pointer["schema_version"] != ACTIVE_CONFIG_SCHEMA_VERSION:
            raise CitationPolicyContractError("unsupported active config schema")
        relative = _required_string(pointer, "config_source_path")
        expected_hash = _sha256_field(pointer, "config_source_sha256")
        if pointer["history_path"] != "config_snapshot_history.jsonl":
            raise CitationPolicyContractError("noncanonical config history path")
        history_hash = _sha256_field(pointer, "history_sha256")
        history_ordinal = _positive_int(pointer, "history_ordinal")
        history_path = run_dir / "config_snapshot_history.jsonl"
        try:
            if history_path.is_symlink() or not history_path.is_file():
                raise CitationPolicyContractError("config snapshot history is missing")
            history_text = history_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CitationPolicyContractError(
                f"cannot read config snapshot history: {exc}"
            ) from exc
        if sha256_text(history_text) != history_hash:
            raise CitationPolicyContractError("config snapshot history hash mismatch")
        history = _parse_config_snapshot_history(history_text)
        if history_ordinal != len(history):
            raise CitationPolicyContractError("active config history ordinal mismatch")
        active_event = history[-1]
        if (
            active_event["config_source_path"] != relative
            or active_event["config_source_sha256"] != expected_hash
        ):
            raise CitationPolicyContractError("active config/history binding mismatch")
    else:
        if (
            resumed
            or (run_dir / "config_snapshot_history.jsonl").exists()
            or (run_dir / "checkpoint.json").exists()
        ):
            raise CitationPolicyContractError(
                "resume state exists without active config pointer"
            )
        relative = "config.yaml"
        expected_hash = ""
    if not _is_config_snapshot_name(relative):
        raise CitationPolicyContractError("noncanonical active config path")
    snapshot = run_dir / relative
    try:
        if snapshot.is_symlink() or not snapshot.is_file():
            raise CitationPolicyContractError("active config snapshot is missing or unsafe")
        text = snapshot.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPolicyContractError(f"cannot read active config snapshot: {exc}") from exc
    digest = sha256_text(text)
    if expected_hash and digest != expected_hash:
        raise CitationPolicyContractError("active config snapshot hash mismatch")
    try:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("config root must be a mapping")
        snapshot_config = RCConfig.from_dict(raw, project_root=run_dir, check_paths=False)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CitationPolicyContractError(f"active config snapshot is invalid: {exc}") from exc
    if snapshot_config.citation_policy != config.citation_policy:
        raise CitationPolicyContractError("active citation policy differs from runtime config")
    if snapshot_config.experiment.claim_scope != config.experiment.claim_scope:
        raise CitationPolicyContractError("active claim_scope differs from runtime config")
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                raise CitationPolicyContractError("checkpoint is unsafe")
            checkpoint = _parse_object(
                checkpoint_path.read_text(encoding="utf-8"), "checkpoint"
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise CitationPolicyContractError(f"cannot read checkpoint: {exc}") from exc
        if (
            checkpoint.get("active_config_snapshot_path") != relative
            or checkpoint.get("active_config_snapshot_sha256") != digest
            or checkpoint.get("config_snapshot_history_sha256") != history_hash
        ):
            raise CitationPolicyContractError("checkpoint config binding mismatch")
    return relative, text, digest


def active_config_checkpoint_fields(run_dir: Path) -> dict[str, str]:
    """Return validated active-config fields that every checkpoint must retain."""
    pointer_path = run_dir / "active_config_snapshot.json"
    history_path = run_dir / "config_snapshot_history.jsonl"
    if not pointer_path.exists() and not history_path.exists():
        return {}
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise CitationPolicyContractError("active config pointer is missing or unsafe")
    if history_path.is_symlink() or not history_path.is_file():
        raise CitationPolicyContractError("config snapshot history is missing or unsafe")
    history_text = history_path.read_text(encoding="utf-8")
    entries = _parse_config_snapshot_history(history_text)
    _validate_existing_active_binding(run_dir, history_text, entries)
    pointer = _parse_object(
        pointer_path.read_text(encoding="utf-8"), "active config pointer"
    )
    return {
        "active_config_snapshot_path": str(pointer["config_source_path"]),
        "active_config_snapshot_sha256": str(pointer["config_source_sha256"]),
        "config_snapshot_history_sha256": str(pointer["history_sha256"]),
    }


def build_effective_citation_policy(run_dir: Path, config: RCConfig) -> dict[str, Any]:
    allowlist_path = run_dir / "stage-06" / "citation_allowlist.json"
    try:
        if allowlist_path.is_symlink() or not allowlist_path.is_file():
            raise CitationPolicyContractError("citation allowlist is missing or unsafe")
        allowlist_text = allowlist_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPolicyContractError(f"cannot read citation allowlist: {exc}") from exc
    allowlist = validate_citation_allowlist(run_dir, config, allowlist_text)
    eligible_count = len(allowlist["eligible_keys"])
    policy = config.citation_policy
    scope = config.experiment.claim_scope
    required = (
        policy.min_unique_sources_research_release
        if scope == "research_release"
        else policy.min_unique_sources_pipeline_validation
    )
    if eligible_count < required:
        raise CitationPolicyContractError(
            f"eligible citation sources {eligible_count} below required minimum {required}"
        )
    effective_min = (
        policy.min_unique_sources_research_release
        if scope == "research_release"
        else min(policy.min_unique_sources_research_release, eligible_count)
    )
    config_path, _config_text, config_hash = resolve_active_config_snapshot(run_dir, config)
    payload = {
        "schema_version": EFFECTIVE_POLICY_SCHEMA_VERSION,
        "policy_version": EFFECTIVE_POLICY_VERSION,
        "claim_scope": scope,
        "eligible_count": eligible_count,
        "effective_min_unique_sources": effective_min,
        "effective_target_unique_sources": min(
            policy.target_unique_sources, eligible_count
        ),
        "citation_allowlist_path": "stage-06/citation_allowlist.json",
        "citation_allowlist_sha256": sha256_text(allowlist_text),
        "config_source_path": config_path,
        "config_source_sha256": config_hash,
    }
    return parse_effective_citation_policy(canonical_json_text(payload))


def parse_effective_citation_policy(text: str) -> dict[str, Any]:
    payload = _parse_object(text, "effective citation policy")
    _exact_keys(
        payload,
        {
            "schema_version",
            "policy_version",
            "claim_scope",
            "eligible_count",
            "effective_min_unique_sources",
            "effective_target_unique_sources",
            "citation_allowlist_path",
            "citation_allowlist_sha256",
            "config_source_path",
            "config_source_sha256",
        },
        "effective citation policy",
    )
    if payload["schema_version"] != EFFECTIVE_POLICY_SCHEMA_VERSION:
        raise CitationPolicyContractError("unsupported effective policy schema")
    if payload["policy_version"] != EFFECTIVE_POLICY_VERSION:
        raise CitationPolicyContractError("unsupported effective policy version")
    if payload["claim_scope"] not in {"pipeline_validation", "exploratory", "research_release"}:
        raise CitationPolicyContractError("invalid effective policy claim_scope")
    for field in (
        "eligible_count",
        "effective_min_unique_sources",
        "effective_target_unique_sources",
    ):
        _nonnegative_int(payload, field)
    if payload["effective_min_unique_sources"] > payload["eligible_count"]:
        raise CitationPolicyContractError("effective minimum exceeds eligible count")
    if payload["effective_target_unique_sources"] > payload["eligible_count"]:
        raise CitationPolicyContractError("effective target exceeds eligible count")
    if payload["effective_target_unique_sources"] < payload["effective_min_unique_sources"]:
        raise CitationPolicyContractError("effective target is below effective minimum")
    if payload["citation_allowlist_path"] != "stage-06/citation_allowlist.json":
        raise CitationPolicyContractError("noncanonical citation allowlist path")
    _sha256_field(payload, "citation_allowlist_sha256")
    if not _is_config_snapshot_name(_required_string(payload, "config_source_path")):
        raise CitationPolicyContractError("noncanonical effective config path")
    _sha256_field(payload, "config_source_sha256")
    return payload


def load_effective_citation_policy(run_dir: Path, config: RCConfig) -> dict[str, Any]:
    path = run_dir / "stage-16" / "citation_policy_effective.json"
    try:
        if path.is_symlink() or not path.is_file():
            raise CitationPolicyContractError("effective citation policy is missing or unsafe")
        stored = parse_effective_citation_policy(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise CitationPolicyContractError(f"cannot read effective citation policy: {exc}") from exc
    expected = build_effective_citation_policy(run_dir, config)
    if stored != expected:
        raise CitationPolicyContractError("effective citation policy replay mismatch")
    return stored


def _bibtex_keys(text: str) -> set[str]:
    keys = re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", text)
    if len(keys) != len(set(keys)):
        raise CitationPolicyContractError("duplicate canonical bibliography key")
    return set(keys)


def _is_config_snapshot_name(value: str) -> bool:
    return value == "config.yaml" or re.fullmatch(
        r"config\.resumed-\d{8}-\d{6}\.yaml", value
    ) is not None


def _parse_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise CitationPolicyContractError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CitationPolicyContractError(f"{label} root must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CitationPolicyContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise CitationPolicyContractError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CitationPolicyContractError(f"{field} must be a nonempty string")
    return value


def _unique_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CitationPolicyContractError(f"{field} must be a list")
    result = tuple(
        item if isinstance(item, str) and item.strip() else _raise_string(field)
        for item in value
    )
    if len(result) != len(set(result)):
        raise CitationPolicyContractError(f"duplicate value in {field}")
    return result


def _raise_string(field: str) -> str:
    raise CitationPolicyContractError(f"{field} entries must be nonempty strings")


def _sha256_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CitationPolicyContractError(f"invalid {field}")
    return value


def _nonnegative_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CitationPolicyContractError(f"{field} must be a nonnegative integer")
    return value


def _positive_int(payload: Mapping[str, Any], field: str) -> int:
    value = _nonnegative_int(payload, field)
    if value < 1:
        raise CitationPolicyContractError(f"{field} must be a positive integer")
    return value


def _parse_config_snapshot_history(text: str) -> list[dict[str, Any]]:
    if not text or not text.endswith("\n"):
        raise CitationPolicyContractError("config snapshot history must end with newline")
    entries: list[dict[str, Any]] = []
    for expected_ordinal, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise CitationPolicyContractError("blank config snapshot history line")
        entry = _parse_object(line, "config snapshot history entry")
        _exact_keys(
            entry,
            {
                "schema_version",
                "ordinal",
                "config_source_path",
                "config_source_sha256",
            },
            "config snapshot history entry",
        )
        if entry["schema_version"] != ACTIVE_CONFIG_SCHEMA_VERSION:
            raise CitationPolicyContractError("unsupported config history schema")
        if _positive_int(entry, "ordinal") != expected_ordinal:
            raise CitationPolicyContractError("config history ordinal is not contiguous")
        if not _is_config_snapshot_name(
            _required_string(entry, "config_source_path")
        ):
            raise CitationPolicyContractError("noncanonical config history path")
        _sha256_field(entry, "config_source_sha256")
        entries.append(entry)
    if not entries:
        raise CitationPolicyContractError("config snapshot history is empty")
    return entries


def _validate_existing_active_binding(
    run_dir: Path,
    history_text: str,
    entries: list[dict[str, Any]],
) -> None:
    pointer_path = run_dir / "active_config_snapshot.json"
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise CitationPolicyContractError(
            "config snapshot history exists without active pointer"
        )
    pointer = _parse_object(
        pointer_path.read_text(encoding="utf-8"), "active config pointer"
    )
    _exact_keys(
        pointer,
        {
            "schema_version",
            "config_source_path",
            "config_source_sha256",
            "history_path",
            "history_sha256",
            "history_ordinal",
        },
        "active config pointer",
    )
    if pointer["schema_version"] != ACTIVE_CONFIG_SCHEMA_VERSION:
        raise CitationPolicyContractError("unsupported active config schema")
    if pointer["history_path"] != "config_snapshot_history.jsonl":
        raise CitationPolicyContractError("noncanonical config history path")
    if pointer["history_sha256"] != sha256_text(history_text):
        raise CitationPolicyContractError("existing config history hash mismatch")
    if _positive_int(pointer, "history_ordinal") != len(entries):
        raise CitationPolicyContractError("existing config history ordinal mismatch")
    active_event = entries[-1]
    if (
        pointer["config_source_path"] != active_event["config_source_path"]
        or pointer["config_source_sha256"]
        != active_event["config_source_sha256"]
    ):
        raise CitationPolicyContractError("existing active config binding mismatch")


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
