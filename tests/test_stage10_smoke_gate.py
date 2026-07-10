from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline._helpers import _collect_experiment_results
from researchclaw.pipeline.stage_impls import _code_generation as codegen
from researchclaw.pipeline.stage_impls._code_generation import (
    _repair_harness_api_misuse,
    _repair_missing_dataset_origin,
    _run_stage10_smoke_gate,
    _seal_selected_candidate,
    _scrub_packaged_experiment_outputs,
)
from researchclaw.experiment_runtime.contract import derive_contract, dump_contract
from researchclaw.pipeline.stages import StageStatus


def _cfg(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {
                "topic": "smoke gate test",
                "domains": ["ml"],
                "daily_paper_count": 2,
                "quality_threshold": 8.2,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {
                "channel": "local",
                "on_stage_start": True,
                "on_stage_fail": False,
                "on_gate_required": True,
            },
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": True, "use_message": True},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake-model",
                "fallback_models": [],
            },
            "security": {"hitl_required_stages": [5, 9, 20]},
            "experiment": {
                "mode": "sandbox",
                "time_budget_sec": 60,
                "metric_key": "detection_f1",
                "metric_direction": "maximize",
                "sandbox": {
                    "python_path": sys.executable,
                    "gpu_required": False,
                    "max_memory_mb": 1024,
                },
                "code_agent": {"enabled": False},
                "opencode": {"enabled": False},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_main(exp_dir: Path, *, include_dataset_origin: bool = True) -> None:
    origin_line = '"dataset_origin": "synthetic",' if include_dataset_origin else ""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "main.py").write_text(
        "\n".join(
            [
                "import json",
                "",
                "def main():",
                "    value = sum([0.25, 0.5])",
                "    print(f'detection_f1: {value}')",
                "    with open('results.json', 'w', encoding='utf-8') as f:",
                f"        json.dump({{{origin_line} 'metrics': {{'detection_f1': value}}}}, f)",
                "",
                "if __name__ == '__main__':",
                "    main()",
            ]
        ),
        encoding="utf-8",
    )


def _write_stage9_contract(run_dir: Path, cfg: RCConfig) -> None:
    stage9 = run_dir / "stage-09"
    stage9.mkdir(parents=True, exist_ok=True)
    dump_contract(
        derive_contract(cfg, {"datasets": ["synthetic traces"]}),
        stage9 / "experiment_contract.yaml",
    )


def test_stage10_smoke_gate_writes_quarantined_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    exp_dir = stage_dir / "experiment"
    _write_main(exp_dir)

    blockers, artifacts = _run_stage10_smoke_gate(stage_dir, exp_dir, _cfg(tmp_path))

    smoke_results = stage_dir / "smoke" / "smoke_results.json"
    payload = json.loads(smoke_results.read_text(encoding="utf-8"))
    collected = _collect_experiment_results(run_dir, metric_key="detection_f1")
    assert blockers == []
    assert artifacts == ["smoke/smoke_results.json"]
    assert payload["status"] == "passed"
    assert payload["dataset_origin"] == "synthetic"
    assert payload["metrics"]["detection_f1"] == 0.75
    assert not (stage_dir / "runs").exists()
    assert not (stage_dir / ".smoke_sandbox").exists()
    assert collected["runs"] == []
    assert collected["metrics_summary"] == {}


def test_smoke_results_and_residual_sandbox_are_not_collected(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    smoke_dir = run_dir / "stage-10" / "smoke"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "smoke_results.json").write_text(
        json.dumps({"metrics": {"detection_f1": 0.99}}),
        encoding="utf-8",
    )
    residual_dir = run_dir / "stage-10" / ".smoke_sandbox" / "_project_0"
    residual_dir.mkdir(parents=True)
    (residual_dir / "results.json").write_text(
        json.dumps({"metrics": {"detection_f1": 0.88}}),
        encoding="utf-8",
    )

    collected = _collect_experiment_results(run_dir, metric_key="detection_f1")

    assert collected["runs"] == []
    assert collected["metrics_summary"] == {}
    assert collected["best_run"] is None


def test_stage10_selected_candidate_is_python_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage9 = run_dir / "stage-09"
    stage10 = run_dir / "stage-10"
    exp_dir = stage10 / "experiment"
    cfg = _cfg(tmp_path)
    stage9.mkdir(parents=True)
    contract_path = stage9 / "experiment_contract.yaml"
    dump_contract(derive_contract(cfg, {"datasets": ["synthetic"]}), contract_path)
    _write_main(exp_dir)
    (exp_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (exp_dir / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    (exp_dir / "results.json").write_text("{}", encoding="utf-8")

    _seal_selected_candidate(stage10, exp_dir, contract_path)

    selected = stage10 / "selected_candidate"
    assert sorted(p.name for p in selected.iterdir()) == ["main.py"]
    manifest = json.loads(
        (stage10 / "selected_candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert sorted(manifest["files"]) == ["main.py"]


def test_stage10_smoke_gate_fails_without_dataset_origin(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run" / "stage-10"
    exp_dir = stage_dir / "experiment"
    _write_main(exp_dir, include_dataset_origin=False)

    blockers, artifacts = _run_stage10_smoke_gate(stage_dir, exp_dir, _cfg(tmp_path))

    smoke_dir = stage_dir / "smoke"
    report = json.loads((smoke_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert artifacts == ["smoke/smoke_report.json"]
    assert any("dataset_origin" in blocker for blocker in blockers)
    assert report["status"] == "failed"
    assert "metrics" not in report
    assert not (smoke_dir / "smoke_results.json").exists()


def test_stage10_smoke_gate_fails_without_primary_metric(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run" / "stage-10"
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "main.py").write_text(
        "\n".join(
            [
                "import json",
                "def main():",
                "    print('other_metric: 0.75')",
                "    with open('results.json', 'w', encoding='utf-8') as f:",
                "        json.dump({'dataset_origin': 'synthetic', "
                "'metrics': {'other_metric': 0.75}}, f)",
                "if __name__ == '__main__':",
                "    main()",
            ]
        ),
        encoding="utf-8",
    )

    blockers, _artifacts = _run_stage10_smoke_gate(
        stage_dir,
        exp_dir,
        _cfg(tmp_path),
    )

    report = json.loads(
        (stage_dir / "smoke" / "smoke_report.json").read_text(encoding="utf-8")
    )
    assert any("detection_f1" in blocker for blocker in blockers)
    assert report["status"] == "failed"
    assert "metrics" not in report


def test_stage10_smoke_gate_failure_writes_no_smoke_results(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run" / "stage-10"
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "main.py").write_text(
        "\n".join(
            [
                "def main():",
                "    raise SystemExit(2)",
                "if __name__ == '__main__':",
                "    main()",
            ]
        ),
        encoding="utf-8",
    )

    blockers, artifacts = _run_stage10_smoke_gate(stage_dir, exp_dir, _cfg(tmp_path))

    smoke_dir = stage_dir / "smoke"
    report = json.loads((smoke_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert artifacts == ["smoke/smoke_report.json"]
    assert any("returncode=2" in blocker for blocker in blockers)
    assert report["status"] == "failed"
    assert "metrics" not in report
    assert not (smoke_dir / "smoke_results.json").exists()


def test_scrub_packaged_experiment_outputs_removes_stale_results(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text("{}", encoding="utf-8")
    (exp_dir / "smoke_results.json").write_text("{}", encoding="utf-8")
    (exp_dir / "runs").mkdir()

    _scrub_packaged_experiment_outputs(exp_dir)

    assert not (exp_dir / "results.json").exists()
    assert not (exp_dir / "smoke_results.json").exists()
    assert not (exp_dir / "runs").exists()


def test_repair_missing_dataset_origin_patches_json_dump_metadata(
    tmp_path: Path,
) -> None:
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    main_py = exp_dir / "main.py"
    main_py.write_text(
        "\n".join(
            [
                "import json",
                "def main():",
                "    results_out = {'metrics': {'detection_f1': 0.75}}",
                "    with open('results.json', 'w', encoding='utf-8') as f:",
                "        json.dump(results_out, f, indent=2)",
            ]
        ),
        encoding="utf-8",
    )

    changed = _repair_missing_dataset_origin(exp_dir)

    repaired = main_py.read_text(encoding="utf-8")
    assert changed is True
    assert "results_out.setdefault('dataset_origin', 'synthetic')" in repaired
    assert "json.dump(results_out, f, indent=2)" in repaired


def test_repair_harness_api_misuse_replaces_elapsed_call(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    main_py = exp_dir / "main.py"
    main_py.write_text(
        "\n".join(
            [
                "from experiment_harness import ExperimentHarness",
                "def main():",
                "    harness = ExperimentHarness(time_budget=60)",
                "    print(harness.elapsed())",
                "    print(harness.progress())",
            ]
        ),
        encoding="utf-8",
    )

    changed = _repair_harness_api_misuse(exp_dir)

    repaired = main_py.read_text(encoding="utf-8")
    assert changed is True
    assert "harness.elapsed()" not in repaired
    assert "harness.progress()" not in repaired
    assert "harness.elapsed" in repaired
    assert "harness.progress" in repaired


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_kwargs: object) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content, model="fake-model")


def test_smoke_blocker_propagates_to_stage_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)
    cfg = _cfg(tmp_path)
    _write_stage9_contract(run_dir, cfg)
    llm = _FakeLLM(
        "\n".join(
            [
                "```filename:main.py",
                "import json",
                "def main():",
                "    value = 0.75",
                "    print(f'detection_f1: {value}')",
                "    with open('results.json', 'w', encoding='utf-8') as f:",
                "        json.dump({'dataset_origin': 'synthetic', "
                "'metrics': {'detection_f1': value}}, f)",
                "if __name__ == '__main__':",
                "    main()",
                "```",
            ]
        )
    )

    def _fake_smoke(_stage_dir: Path, _exp_dir: Path, _config: RCConfig):
        return ["forced smoke failure"], ["smoke/smoke_report.json"]

    monkeypatch.setattr(codegen, "_run_stage10_smoke_gate", _fake_smoke)

    result = codegen._execute_code_generation(
        stage_dir,
        run_dir,
        cfg,
        AdapterBundle(),
        llm=llm,
    )

    assert result.status is StageStatus.FAILED
    assert "forced smoke failure" in (result.error or "")


def test_stage10_repairs_missing_dataset_origin_and_reruns_smoke(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)
    cfg = _cfg(tmp_path)
    _write_stage9_contract(run_dir, cfg)
    llm = _FakeLLM(
        "\n".join(
            [
                "```filename:main.py",
                "import json",
                "def main():",
                "    value = 0.75",
                "    print(f'detection_f1: {value}')",
                "    results_out = {'metrics': {'detection_f1': value}}",
                "    with open('results.json', 'w', encoding='utf-8') as f:",
                "        json.dump(results_out, f, indent=2)",
                "if __name__ == '__main__':",
                "    main()",
                "```",
            ]
        )
    )

    result = codegen._execute_code_generation(
        stage_dir,
        run_dir,
        cfg,
        AdapterBundle(),
        llm=llm,
    )

    smoke_payload = json.loads(
        (stage_dir / "smoke" / "smoke_results.json").read_text(encoding="utf-8")
    )
    repaired_main = (stage_dir / "experiment" / "main.py").read_text(
        encoding="utf-8"
    )
    assert result.status is StageStatus.DONE
    assert smoke_payload["dataset_origin"] == "synthetic"
    assert "setdefault('dataset_origin', 'synthetic')" in repaired_main


def test_stage10_repairs_harness_elapsed_before_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir(parents=True)
    cfg = _cfg(tmp_path)
    _write_stage9_contract(run_dir, cfg)
    llm = _FakeLLM(
        "\n".join(
            [
                "```filename:main.py",
                "import json",
                "from experiment_harness import ExperimentHarness",
                "def main():",
                "    harness = ExperimentHarness(time_budget=60)",
                "    value = 0.75",
                "    print(f'detection_f1: {value}')",
                "    results_out = {'dataset_origin': 'synthetic', "
                "'metrics': {'detection_f1': value}, "
                "'elapsed_seconds': round(harness.elapsed(), 2)}",
                "    with open('results.json', 'w', encoding='utf-8') as f:",
                "        json.dump(results_out, f, indent=2)",
                "if __name__ == '__main__':",
                "    main()",
                "```",
            ]
        )
    )

    result = codegen._execute_code_generation(
        stage_dir,
        run_dir,
        cfg,
        AdapterBundle(),
        llm=llm,
    )

    repaired_main = (stage_dir / "experiment" / "main.py").read_text(
        encoding="utf-8"
    )
    assert result.status is StageStatus.DONE
    assert "harness.elapsed()" not in repaired_main
    assert "harness.elapsed" in repaired_main
