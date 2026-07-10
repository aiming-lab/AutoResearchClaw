from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment_runtime.contract import (
    derive_contract,
    dump_contract,
    sha256_file,
)
from researchclaw.pipeline.stage_impls._execution import (
    _execute_experiment_run,
    _load_sealed_candidate,
    _scaffold_sha256,
)
from researchclaw.pipeline.stages import StageStatus


def _cfg(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
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
                "time_budget_sec": 30,
                "metric_key": "detection_f1",
                "metric_direction": "maximize",
                "sandbox": {"python_path": "python"},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_contract(run: Path, cfg: RCConfig) -> Path:
    contract = derive_contract(cfg, {"datasets": ["synthetic traces"]})
    path = run / "stage-09" / "experiment_contract.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_contract(contract, path)
    return path


def _write_selected_candidate(run: Path, cfg: RCConfig) -> Path:
    contract_path = _write_contract(run, cfg)
    selected = run / "stage-10" / "selected_candidate"
    selected.mkdir(parents=True, exist_ok=True)
    main = selected / "main.py"
    main.write_text(
        "import json\n"
        "with open('results.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump({'dataset_origin': 'synthetic', 'metrics': {'detection_f1': 0.5}}, f)\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "contract_sha256": sha256_file(contract_path),
        "scaffold_sha256": _scaffold_sha256(),
        "entry_point": "main.py",
        "files": {"main.py": {"sha256": sha256_file(main)}},
    }
    (run / "stage-10" / "selected_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return selected


def test_stage12_rejects_when_manifest_missing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    _write_contract(run, cfg)
    (run / "stage-10" / "selected_candidate").mkdir(parents=True)

    result = _execute_experiment_run(run / "stage-12", run, cfg, AdapterBundle())

    assert result.status == StageStatus.FAILED
    assert "sealed candidate manifest missing" in (result.error or "")


def test_stage12_loads_valid_sealed_candidate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    selected = _write_selected_candidate(run, cfg)

    assert _load_sealed_candidate(run) == selected


def test_stage12_rejects_contract_hash_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    _write_selected_candidate(run, cfg)
    manifest_path = run / "stage-10" / "selected_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="contract_sha256 mismatch"):
        _load_sealed_candidate(run)


def test_stage12_rejects_scaffold_hash_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    _write_selected_candidate(run, cfg)
    manifest_path = run / "stage-10" / "selected_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scaffold_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="scaffold_sha256 mismatch"):
        _load_sealed_candidate(run)


def test_stage12_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    selected = _write_selected_candidate(run, cfg)
    (selected / "main.py").write_text("print('tampered')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _load_sealed_candidate(run)


def test_stage12_rejects_unmanifested_file(tmp_path: Path) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    selected = _write_selected_candidate(run, cfg)
    (selected / "helper.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unmanifested files"):
        _load_sealed_candidate(run)


@pytest.mark.parametrize("name", ["results.json", "smoke_results.json", "metrics.json"])
def test_stage12_rejects_banned_files(tmp_path: Path, name: str) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    selected = _write_selected_candidate(run, cfg)
    path = selected / name
    path.write_text("{}", encoding="utf-8")
    manifest_path = run / "stage-10" / "selected_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][name] = {"sha256": sha256_file(path)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="may list only Python files|banned files"):
        _load_sealed_candidate(run)


@pytest.mark.parametrize("name", ["runs", "attempts", "candidates", ".smoke_sandbox"])
def test_stage12_rejects_directories(tmp_path: Path, name: str) -> None:
    run = tmp_path / "run"
    cfg = _cfg(tmp_path)
    selected = _write_selected_candidate(run, cfg)
    (selected / name).mkdir()

    with pytest.raises(RuntimeError, match="directories in selected_candidate"):
        _load_sealed_candidate(run)
