"""SSH execution must preserve project evidence and reject incomplete runs."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import ExperimentConfig
from researchclaw.experiment.sandbox import SandboxResult
from researchclaw.pipeline.stage_impls import _execution
from researchclaw.pipeline.stages import StageStatus


def config():
    return SimpleNamespace(
        experiment=ExperimentConfig(
            mode="ssh_remote", time_budget_sec=60,
            metric_key="accuracy", metric_direction="maximize", max_iterations=1,
        ),
        research=SimpleNamespace(topic="remote execution regression", domains=[]),
    )


def project(run_dir):
    source = run_dir / "stage-10" / "experiment"
    source.mkdir(parents=True)
    (source / "main.py").write_text("from helper import score\nprint(f'accuracy: {score}')\n")
    (source / "helper.py").write_text("score = 0.5\n")
    (source / "dataset.csv").write_text("feature,label\n1,0\n")
    (source / "models").mkdir()
    (source / "models" / "__init__.py").write_text("VERSION = 1\n")
    (source / "fixture.bin").write_bytes(b"\x00\xff\x10")
    return source


def test_stage12_remote_project_and_downloaded_results(tmp_path, monkeypatch):
    source = project(tmp_path)
    stage = tmp_path / "stage-12"
    downloaded = stage / "runs" / "sandbox" / "_ssh_project_1"
    downloaded.mkdir(parents=True)
    raw_results = {"raw_scores": [0.75, 0.85], "mean": 0.8}
    (downloaded / "results.json").write_text(json.dumps(raw_results))
    (downloaded / "raw.csv").write_text("seed,score\n1,0.75\n2,0.85\n")
    backend = Mock(last_run_dir=downloaded)
    backend.run_project.return_value = SandboxResult(0, "accuracy: 0.8\n", "", 6, {"accuracy": 0.8})
    factory = Mock(return_value=backend)
    install = Mock(side_effect=AssertionError("SSH must not install dependencies on the controller"))
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", factory)
    monkeypatch.setattr(_execution, "_ensure_sandbox_deps", install)

    result = _execution._execute_experiment_run(stage, tmp_path, config(), AdapterBundle())

    assert result.status == StageStatus.DONE
    backend.run_project.assert_called_once_with(source, timeout_sec=60)
    backend.run.assert_not_called()
    install.assert_not_called()
    assert not (source / "results.json").exists()
    payload = json.loads((stage / "runs" / "run-1.json").read_text())
    assert payload["structured_results"] == raw_results
    assert payload["stdout"] == "accuracy: 0.8\n"
    assert (stage / payload["artifact_dir"] / "raw.csv").is_file()
    assert json.loads((stage / "runs" / "results.json").read_text()) == raw_results


@pytest.mark.parametrize("returncode,timed_out,metrics,elapsed", [
    (1, False, {"accuracy": 0.9}, 6),  # prints a metric, then crashes
    (74, False, {"accuracy": 0.9}, 6),  # artifact transfer failure
    (124, True, {"accuracy": 0.9}, 60),
    (0, False, {}, 45),  # zero evidence even after a long run
    (0, False, {"accuracy": float("nan")}, 45),
])
def test_stage12_incomplete_remote_run_blocks_pipeline(tmp_path, monkeypatch, returncode, timed_out, metrics, elapsed):
    project(tmp_path)
    stage = tmp_path / "stage-12"
    backend = Mock(last_run_dir=None)
    backend.run_project.return_value = SandboxResult(returncode, "diagnostic output\n", "raw failure\n", elapsed, metrics, timed_out)
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", Mock(return_value=backend))
    result = _execution._execute_experiment_run(stage, tmp_path, config(), AdapterBundle())
    assert result.status == StageStatus.FAILED
    payload = json.loads((stage / "runs" / "run-1.json").read_text())
    assert payload["stderr"] == "raw failure\n"
    assert payload["returncode"] == returncode
    assert "Remote experiment incomplete" in result.error


@pytest.mark.parametrize("returncode,timed_out,expected_best", [(0, False, 0.9), (1, False, 0.5), (124, True, 0.5)])
def test_stage13_executes_remote_candidate_and_only_selects_completed_results(tmp_path, monkeypatch, returncode, timed_out, expected_best):
    project(tmp_path)
    runs = tmp_path / "stage-12" / "runs"
    runs.mkdir(parents=True)
    (runs / "run-1.json").write_text(json.dumps({"status": "completed", "metrics": {"accuracy": 0.5}, "returncode": 0}))
    # A failed run's apparently better metric must not become the baseline.
    (runs / "run-2.json").write_text(json.dumps({"status": "failed", "metrics": {"accuracy": 0.99}, "returncode": 1}))
    stage = tmp_path / "stage-13"
    stage.mkdir()
    backend = Mock()
    backend.run_project.return_value = SandboxResult(returncode, "accuracy: 0.9\n", "", 6, {"accuracy": 0.9}, timed_out)
    monkeypatch.setattr("researchclaw.experiment.factory.create_sandbox", Mock(return_value=backend))
    monkeypatch.setattr(_execution, "_chat_with_prompt", lambda *a, **k: SimpleNamespace(content="```filename:main.py\nfrom helper import score\nprint(f'accuracy: {score}')\n```\n```filename:helper.py\nscore = 0.9\n```"))
    monkeypatch.setattr(_execution, "_detect_runtime_issues", lambda result: None)
    monkeypatch.setattr(_execution, "validate_code", lambda text: SimpleNamespace(ok=True, summary=lambda: "ok"))
    _execution._execute_iterative_refine(stage, tmp_path, config(), AdapterBundle(), llm=Mock())
    backend.run_project.assert_called_once()
    candidate = backend.run_project.call_args.args[0]
    assert (candidate / "helper.py").is_file()
    for destination in (candidate, stage / "experiment_final"):
        assert (destination / "dataset.csv").read_text() == "feature,label\n1,0\n"
        assert (destination / "models" / "__init__.py").read_text() == "VERSION = 1\n"
        assert (destination / "fixture.bin").read_bytes() == b"\x00\xff\x10"
    log = json.loads((stage / "refinement_log.json").read_text())
    assert log["baseline_metric"] == 0.5
    assert log["best_metric"] == expected_best
    assert log["iterations"][0]["improved"] is (expected_best > 0.5)
    # A diagnostic metric must also stay excluded at the downstream merge.
    from researchclaw.pipeline.stage_impls._analysis import _execute_result_analysis
    analysis = tmp_path / "stage-14"
    analysis.mkdir()
    monkeypatch.setattr("researchclaw.experiment.visualize.generate_all_charts", lambda *a, **k: [])
    _execute_result_analysis(analysis, tmp_path, config(), AdapterBundle())
    summary = json.loads((analysis / "experiment_summary.json").read_text())
    assert summary["best_run"]["metrics"]["accuracy"] == expected_best
    assert summary["best_run"]["status"] == "completed"


def test_remote_failed_history_and_structured_results_are_not_aggregated(tmp_path, monkeypatch):
    from researchclaw.pipeline._helpers import _collect_experiment_results
    for name, status, code, metric in (("stage-12", "failed", 1, 0.99), ("stage-12_v1", "completed", 0, 0.5)):
        runs = tmp_path / name / "runs"
        runs.mkdir(parents=True)
        (runs / "run-1.json").write_text(json.dumps({
            "execution_mode": "ssh_remote", "status": status, "returncode": code,
            "timed_out": False, "metrics": {"accuracy": metric},
        }))
        (runs / "results.json").write_text(json.dumps({"source": name}))
    # Tag-based protection also applies to consumers outside Stage 14.
    data = _collect_experiment_results(tmp_path, "accuracy", "maximize")
    assert len(data["runs"]) == 1
    assert data["best_run"]["metrics"]["accuracy"] == 0.5
    assert data["structured_results"] == {"source": "stage-12_v1"}
    # Chart generation is also used directly by final export, without Stage 14.
    from researchclaw.experiment import visualize
    trajectory = Mock(return_value=None)
    monkeypatch.setattr(visualize, "HAS_MATPLOTLIB", True)
    monkeypatch.setattr(visualize, "plot_metric_trajectory", trajectory)
    visualize.generate_all_charts(tmp_path, metric_key="accuracy")
    chart_runs = trajectory.call_args.args[0]
    assert len(chart_runs) == 1
    assert chart_runs[0]["metrics"]["accuracy"] == 0.5


@pytest.mark.parametrize("fix_code,fix_timeout,expected", [(0, False, 0.8), (1, False, 0.5), (-1, True, 0.5)])
def test_analysis_uses_only_successful_final_remote_attempt(tmp_path, monkeypatch, fix_code, fix_timeout, expected):
    from researchclaw.pipeline.stage_impls._analysis import _execute_result_analysis
    runs = tmp_path / "stage-12" / "runs"
    runs.mkdir(parents=True)
    (runs / "run-1.json").write_text(json.dumps({"status": "completed", "returncode": 0, "metrics": {"accuracy": 0.5}}))
    refine = tmp_path / "stage-13"
    refine.mkdir()
    invalid_paired = "PAIRED: proposed vs baseline mean_diff=0.99 std_diff=0.1 t_stat=20 p_value=0.001\n"
    from researchclaw.experiment.sandbox import extract_paired_comparisons
    assert extract_paired_comparisons(invalid_paired)
    (refine / "refinement_log.json").write_text(json.dumps({
        "mode": "ssh_remote", "best_version": "experiment_v1/", "best_metric": expected,
        "iterations": [{"version_dir": "experiment_v1/", "sandbox": {
            "returncode": 1, "timed_out": False, "metrics": {"accuracy": 0.99}, "stdout": invalid_paired,
        }, "sandbox_after_fix": {
            "returncode": fix_code, "timed_out": fix_timeout, "metrics": {"accuracy": 0.8}, "stdout": "",
        }}],
    }))
    analysis = tmp_path / "stage-14"
    analysis.mkdir()
    monkeypatch.setattr("researchclaw.experiment.visualize.generate_all_charts", lambda *a, **k: [])
    _execute_result_analysis(analysis, tmp_path, config(), AdapterBundle())
    summary = json.loads((analysis / "experiment_summary.json").read_text())
    assert summary["best_run"]["metrics"]["accuracy"] == expected
    assert not summary.get("paired_comparisons")
