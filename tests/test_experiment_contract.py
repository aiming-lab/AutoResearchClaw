from __future__ import annotations

import copy
from pathlib import Path

import pytest

from researchclaw.config import RCConfig
from researchclaw.experiment_runtime.contract import (
    ContractValidationError,
    derive_contract,
    dump_contract,
    load_contract,
    validate_contract_dict,
)


def _valid_contract() -> dict:
    return {
        "schema_version": 1,
        "topic": "test topic",
        "claim_scope": "pipeline_validation",
        "dataset_origin": "synthetic",
        "dataset_name": None,
        "primary_metric": {
            "key": "detection_f1",
            "direction": "maximize",
            "minimum_valid_value": 0.0,
        },
        "smoke_budget_sec": 60,
        "run_budget_sec": 300,
        "allowed_inputs": [],
        "allowed_outputs": [{"path": "results.json", "required": True}],
        "evaluator": {
            "command": "python main.py",
            "owner": "scaffold",
            "timeout_sec": 300,
            "required_result_keys": ["dataset_origin", "metrics"],
        },
        "safety": {
            "network": "none",
            "env_policy": "allowlist",
            "evidence_policy": "stage12_recomputed_only",
        },
        "sealing": {
            "candidate_manifest": "selected_candidate_manifest.json",
            "content_hash_algorithm": "sha256",
        },
    }


def test_valid_contract_parses() -> None:
    contract = validate_contract_dict(_valid_contract())
    assert contract.claim_scope == "pipeline_validation"
    assert contract.dataset_origin == "synthetic"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", None),
        ("claim_scope", None),
        ("dataset_origin", "unknown"),
    ],
)
def test_contract_rejects_missing_or_unknown_required_fields(field: str, value) -> None:
    data = _valid_contract()
    if value is None:
        data.pop(field)
    else:
        data[field] = value
    with pytest.raises(ContractValidationError):
        validate_contract_dict(data)


def test_contract_rejects_synthetic_research_release() -> None:
    data = _valid_contract()
    data["claim_scope"] = "research_release"
    data["dataset_origin"] = "synthetic"
    with pytest.raises(ContractValidationError, match="research_release"):
        validate_contract_dict(data)


def test_contract_rejects_non_scaffold_evaluator() -> None:
    data = _valid_contract()
    data["evaluator"] = copy.deepcopy(data["evaluator"])
    data["evaluator"]["owner"] = "model"
    with pytest.raises(ContractValidationError, match="evaluator.owner"):
        validate_contract_dict(data)


def test_contract_rejects_smoke_budget_over_limit() -> None:
    data = _valid_contract()
    data["smoke_budget_sec"] = 121
    with pytest.raises(ContractValidationError, match="smoke_budget_sec"):
        validate_contract_dict(data)


def test_contract_rejects_empty_primary_metric_key() -> None:
    data = _valid_contract()
    data["primary_metric"] = copy.deepcopy(data["primary_metric"])
    data["primary_metric"]["key"] = ""
    with pytest.raises(ContractValidationError, match="primary_metric.key"):
        validate_contract_dict(data)


def test_contract_sha256_is_deterministic(tmp_path: Path) -> None:
    contract = validate_contract_dict(_valid_contract())
    p1 = tmp_path / "c1.yaml"
    p2 = tmp_path / "c2.yaml"
    assert dump_contract(contract, p1) == dump_contract(contract, p2)
    assert load_contract(p1).primary_metric["key"] == "detection_f1"


def test_derive_contract_splits_smoke_and_run_budgets(tmp_path: Path) -> None:
    cfg = RCConfig.from_dict(
        {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "topic", "domains": ["security"]},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "none"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 300,
                "metric_key": "detection_f1",
                "metric_direction": "maximize",
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )
    contract = derive_contract(cfg, {"datasets": ["UCI Adult"]})
    assert contract.smoke_budget_sec == 60
    assert contract.run_budget_sec == 300
    assert contract.dataset_origin == "synthetic"
    assert contract.dataset_name == "synthetic_pipeline_validation_v1"


def test_derive_contract_keeps_named_public_dataset(tmp_path: Path) -> None:
    cfg = RCConfig.from_dict(
        {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "topic", "domains": ["security"]},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "none"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 300,
                "metric_key": "detection_f1",
                "metric_direction": "maximize",
                "dataset_origin": "public",
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )
    contract = derive_contract(cfg, {"datasets": ["UCI Adult"]})
    assert contract.dataset_origin == "public"
    assert contract.dataset_name == "UCI Adult"
