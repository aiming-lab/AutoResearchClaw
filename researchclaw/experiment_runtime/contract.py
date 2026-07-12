"""Machine-readable experiment contract for Stages 9-12."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CLAIM_SCOPES = {"pipeline_validation", "exploratory", "research_release"}
DATASET_ORIGINS = {"synthetic", "public", "local_hardware"}
METRIC_DIRECTIONS = {"maximize", "minimize"}


class ContractValidationError(ValueError):
    """Raised when an experiment contract is missing required invariants."""


@dataclass(frozen=True)
class ExperimentContract:
    schema_version: int
    topic: str
    claim_scope: str
    dataset_origin: str
    dataset_name: str | None
    primary_metric: dict[str, Any]
    smoke_budget_sec: int
    run_budget_sec: int
    allowed_inputs: list[dict[str, Any]]
    allowed_outputs: list[dict[str, Any]]
    evaluator: dict[str, Any]
    safety: dict[str, Any]
    sealing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topic": self.topic,
            "claim_scope": self.claim_scope,
            "dataset_origin": self.dataset_origin,
            "dataset_name": self.dataset_name,
            "primary_metric": dict(self.primary_metric),
            "smoke_budget_sec": self.smoke_budget_sec,
            "run_budget_sec": self.run_budget_sec,
            "allowed_inputs": list(self.allowed_inputs),
            "allowed_outputs": list(self.allowed_outputs),
            "evaluator": dict(self.evaluator),
            "safety": dict(self.safety),
            "sealing": dict(self.sealing),
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contract_sha256(path: Path) -> str:
    return sha256_file(path)


def validate_contract_dict(data: dict[str, Any]) -> ExperimentContract:
    errors: list[str] = []

    schema_version = data.get("schema_version")
    if schema_version != 1:
        errors.append("schema_version must be 1")

    topic = str(data.get("topic") or "").strip()
    if not topic:
        errors.append("topic is required")

    claim_scope = str(data.get("claim_scope") or "").strip()
    if claim_scope not in CLAIM_SCOPES:
        errors.append(f"claim_scope must be one of {sorted(CLAIM_SCOPES)}")

    dataset_origin = str(data.get("dataset_origin") or "").strip()
    if dataset_origin not in DATASET_ORIGINS:
        errors.append(f"dataset_origin must be one of {sorted(DATASET_ORIGINS)}")

    if claim_scope == "research_release" and dataset_origin == "synthetic":
        errors.append("research_release cannot use synthetic dataset_origin")

    primary_metric = data.get("primary_metric")
    if not isinstance(primary_metric, dict):
        errors.append("primary_metric must be an object")
        primary_metric = {}
    else:
        if not str(primary_metric.get("key") or "").strip():
            errors.append("primary_metric.key is required")
        if str(primary_metric.get("direction") or "").strip() not in METRIC_DIRECTIONS:
            errors.append("primary_metric.direction must be maximize or minimize")

    smoke_budget_sec = _int_value(data.get("smoke_budget_sec"))
    run_budget_sec = _int_value(data.get("run_budget_sec"))
    if smoke_budget_sec is None or smoke_budget_sec <= 0 or smoke_budget_sec > 120:
        errors.append("smoke_budget_sec must be in 1..120")
        smoke_budget_sec = 0
    if run_budget_sec is None or run_budget_sec <= 0:
        errors.append("run_budget_sec must be positive")
        run_budget_sec = 0
    if smoke_budget_sec and run_budget_sec and run_budget_sec < smoke_budget_sec:
        errors.append("run_budget_sec must be >= smoke_budget_sec")

    evaluator = data.get("evaluator")
    if not isinstance(evaluator, dict):
        errors.append("evaluator must be an object")
        evaluator = {}
    else:
        if evaluator.get("owner") != "scaffold":
            errors.append("evaluator.owner must be scaffold")
        required = evaluator.get("required_result_keys")
        if not isinstance(required, list):
            errors.append("evaluator.required_result_keys must be a list")
        else:
            missing = {"dataset_origin", "metrics"} - {str(x) for x in required}
            if missing:
                errors.append(
                    "evaluator.required_result_keys missing "
                    + ", ".join(sorted(missing))
                )

    allowed_inputs = data.get("allowed_inputs") or []
    allowed_outputs = data.get("allowed_outputs") or []
    if not isinstance(allowed_inputs, list):
        errors.append("allowed_inputs must be a list")
        allowed_inputs = []
    if not isinstance(allowed_outputs, list):
        errors.append("allowed_outputs must be a list")
        allowed_outputs = []

    safety = data.get("safety") or {}
    sealing = data.get("sealing") or {}
    if not isinstance(safety, dict):
        errors.append("safety must be an object")
        safety = {}
    if not isinstance(sealing, dict):
        errors.append("sealing must be an object")
        sealing = {}

    if errors:
        raise ContractValidationError("; ".join(errors))

    dataset_name_raw = data.get("dataset_name")
    dataset_name = str(dataset_name_raw).strip() if dataset_name_raw else None
    return ExperimentContract(
        schema_version=1,
        topic=topic,
        claim_scope=claim_scope,
        dataset_origin=dataset_origin,
        dataset_name=dataset_name,
        primary_metric=primary_metric,
        smoke_budget_sec=int(smoke_budget_sec),
        run_budget_sec=int(run_budget_sec),
        allowed_inputs=allowed_inputs,
        allowed_outputs=allowed_outputs,
        evaluator=evaluator,
        safety=safety,
        sealing=sealing,
    )


def load_contract(path: Path) -> ExperimentContract:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ContractValidationError(f"cannot read contract: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractValidationError("contract root must be an object")
    return validate_contract_dict(raw)


def dump_contract(contract: ExperimentContract, path: Path) -> str:
    path.write_text(
        yaml.safe_dump(contract.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return contract_sha256(path)


def derive_contract(config: Any, plan: dict[str, Any] | None) -> ExperimentContract:
    experiment = getattr(config, "experiment", None)
    claim_scope = str(getattr(experiment, "claim_scope", "pipeline_validation") or "pipeline_validation")
    dataset_origin = str(getattr(experiment, "dataset_origin", "synthetic") or "synthetic")
    time_budget_sec = int(getattr(experiment, "time_budget_sec", 300) or 300)
    smoke_budget_sec = max(1, min(60, time_budget_sec))
    metric_key = str(getattr(experiment, "metric_key", "primary_metric") or "primary_metric")
    metric_direction = str(getattr(experiment, "metric_direction", "minimize") or "minimize")
    dataset_name = (
        "synthetic_pipeline_validation_v1"
        if dataset_origin == "synthetic"
        else _first_dataset_name(plan)
    )
    contract = ExperimentContract(
        schema_version=1,
        topic=str(getattr(getattr(config, "research", None), "topic", "") or ""),
        claim_scope=claim_scope,
        dataset_origin=dataset_origin,
        dataset_name=dataset_name,
        primary_metric={
            "key": metric_key,
            "direction": metric_direction,
            "minimum_valid_value": 0.0,
        },
        smoke_budget_sec=smoke_budget_sec,
        run_budget_sec=time_budget_sec,
        allowed_inputs=[],
        allowed_outputs=[{"path": "results.json", "required": True}],
        evaluator={
            "command": "python main.py",
            "owner": "scaffold",
            "timeout_sec": time_budget_sec,
            "required_result_keys": ["dataset_origin", "metrics"],
        },
        safety={
            "network": "none",
            "env_policy": getattr(getattr(experiment, "sandbox", None), "env_policy", "allowlist"),
            "evidence_policy": "stage12_recomputed_only",
        },
        sealing={
            "candidate_manifest": "selected_candidate_manifest.json",
            "content_hash_algorithm": "sha256",
        },
    )
    return validate_contract_dict(contract.to_dict())


def find_stage09_contract(run_dir: Path) -> Path | None:
    direct = run_dir / "stage-09" / "experiment_contract.yaml"
    if direct.is_file():
        return direct
    if any(
        (direct.parent / marker).exists()
        for marker in ("decision.json", "plan_meta.json", "stage_health.json")
    ):
        return None
    candidates: list[Path] = []
    for path in run_dir.glob("stage-09_v*/experiment_contract.yaml"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: _stage09_version(p.parent.name), reverse=True)[0]


def _stage09_version(name: str) -> int:
    if "_v" not in name:
        return 0
    try:
        return int(name.rsplit("_v", 1)[1])
    except ValueError:
        return -1


def _int_value(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_dataset_name(plan: dict[str, Any] | None) -> str | None:
    if not isinstance(plan, dict):
        return None
    datasets = plan.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, dict):
            name = first.get("name") or first.get("dataset")
            return str(name).strip() if name else None
        return str(first).strip() or None
    if isinstance(datasets, dict) and datasets:
        key = next(iter(datasets.keys()))
        return str(key).strip() or None
    if isinstance(datasets, str):
        return datasets.strip() or None
    return None
