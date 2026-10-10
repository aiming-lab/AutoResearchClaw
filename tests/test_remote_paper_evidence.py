"""Paper/review evidence rejects incomplete remote runs without deleting logs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.stage_impls._paper_writing import _collect_raw_experiment_metrics


def write_run(root: Path, name: str, payload: dict, stage: str = "stage-12") -> Path:
    path = root / stage / "runs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_refinement(root: Path, iterations: list[dict], mode: str | None = "ssh_remote") -> Path:
    path = root / "stage-13" / "refinement_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"iterations": iterations}
    if mode is not None:
        payload["mode"] = mode
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def attempt(label: str, *, returncode: int = 0, timed_out: bool = False) -> dict:
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "metrics": {f"{label}_metric": 0.73},
        "stdout": f"{label}_raw: 0.73\nPAIRED: {label}_pair\n",
    }


@pytest.mark.parametrize("status,returncode,timed_out", [
    ("failed", 1, False), ("failed", 0, False),
    ("completed", 3, False), ("completed", 0, True),
    (None, 0, False), ("pending", 0, False),
])
def test_incomplete_ssh_run_cannot_supply_metrics_stdout_or_paired(
    tmp_path, status, returncode, timed_out,
):
    path = write_run(tmp_path, "failed", {
        "execution_mode": "ssh_remote", "gpu_available": False,
        "status": status, "returncode": returncode, "timed_out": timed_out,
        "metrics": {"failed_metric": 9.999},
        "stdout": "failed_raw: 9.999\nPAIRED: failed_pair: 9.999\n",
    })
    original = path.read_bytes()
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert block == ""
    assert has_parsed is False
    assert path.read_bytes() == original


def test_completed_ssh_cpu_run_is_accepted_without_gpu(tmp_path):
    write_run(tmp_path, "completed", {
        "execution_mode": "ssh_remote", "gpu_available": False,
        "status": "completed", "returncode": 0, "timed_out": False,
        "metrics": {"accepted_metric": 0.125},
        "stdout": "accepted_raw: 0.125\nPAIRED: accepted_pair: 0.125\n",
    })
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert has_parsed is True
    assert "accepted_metric" in block and "accepted_raw" in block
    assert "accepted_pair" in block
    assert "1 run(s)" in block


def test_successful_and_failed_remote_runs_are_filtered_independently(tmp_path):
    write_run(tmp_path, "good", {
        "execution_mode": "ssh_remote", "status": "completed", "returncode": 0,
        "metrics": {"good_metric": 0.6}, "stdout": "good_raw: 0.6\n",
    })
    write_run(tmp_path, "failed", {
        "execution_mode": "ssh_remote", "status": "failed", "returncode": 1,
        "metrics": {"failed_metric": 99}, "stdout": "failed_raw: 99\n",
    })
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert has_parsed and "good_metric" in block and "good_raw" in block
    assert "failed_metric" not in block and "failed_raw" not in block
    assert "1 run(s)" in block


def test_config_mode_blocks_legacy_remote_payload_without_execution_marker(tmp_path):
    write_run(tmp_path, "legacy", {"metrics": {"legacy_failed": 8}, "stdout": "legacy_raw: 8"})
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path, execution_mode="ssh_remote")
    assert block == "" and has_parsed is False


def test_refinement_mode_identifies_legacy_remote_runs_for_review_callers(tmp_path):
    write_run(tmp_path, "legacy", {"status": "failed", "returncode": 1,
                                    "metrics": {"legacy_failed": 8}})
    write_refinement(tmp_path, [])
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert block == "" and has_parsed is False


def test_remote_mode_discovered_in_later_stage_filters_earlier_attempts(tmp_path):
    write_run(tmp_path, "legacy", {"status": "failed", "returncode": 1,
                                    "metrics": {"earlier_failed": 8}}, stage="stage-12")
    write_run(tmp_path, "remote", {"execution_mode": "ssh_remote", "status": "failed",
                                    "returncode": 1, "metrics": {"later_failed": 9}},
              stage="stage-12_v2")
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert block == "" and has_parsed is False


@pytest.mark.parametrize("repair", [
    attempt("failed_fix", returncode=1), attempt("timed_out_fix", timed_out=True), None,
])
def test_failed_or_invalid_repair_never_falls_back_to_pre_repair_metrics(tmp_path, repair):
    path = write_refinement(tmp_path, [{"sandbox": attempt("pre_repair"),
                                        "sandbox_after_fix": repair}])
    original = path.read_bytes()
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert block == ""
    assert has_parsed is False
    assert path.read_bytes() == original


def test_successful_repair_supersedes_initial_metrics_and_paired_stdout(tmp_path):
    write_refinement(tmp_path, [{"sandbox": attempt("pre_repair", returncode=1),
                                 "sandbox_after_fix": attempt("repaired")}])
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert has_parsed is True
    assert "repaired_metric" in block
    assert "repaired_raw" in block
    assert "repaired_pair" in block
    assert "pre_repair" not in block


def test_remote_run_incomplete_excludes_even_zero_returncode_attempt(tmp_path):
    write_refinement(tmp_path, [{"sandbox": attempt("initial"),
                                 "sandbox_after_fix": attempt("fixed"),
                                 "remote_run_incomplete": True}])
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert block == "" and has_parsed is False


def test_completed_refinement_without_repair_is_accepted(tmp_path):
    write_refinement(tmp_path, [{"sandbox": attempt("completed_refine")}])
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert has_parsed is True
    assert "completed_refine_metric" in block
    assert "completed_refine_pair" in block


def test_run_execution_marker_filters_refinement_without_mode_field(tmp_path):
    write_run(tmp_path, "remote", {"execution_mode": "ssh_remote", "status": "failed",
                                    "returncode": 1, "metrics": {}})
    write_refinement(tmp_path, [{"sandbox": attempt("failed_refine", returncode=1)}], mode=None)
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert block == "" and has_parsed is False


def test_non_remote_runs_keep_existing_permissive_behavior(tmp_path):
    write_run(tmp_path, "local", {"status": "failed", "returncode": 1,
                                  "metrics": {"local_partial": 0.3},
                                  "stdout": "local_raw: 0.3\n"})
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path, execution_mode="sandbox")
    assert has_parsed and "local_partial" in block and "local_raw" in block


def test_explicit_non_remote_source_keeps_its_behavior_in_a_mixed_run(tmp_path):
    write_run(tmp_path, "local", {"execution_mode": "sandbox", "status": "failed",
                                  "returncode": 1, "metrics": {"local_partial": 0.3}})
    write_run(tmp_path, "remote", {"execution_mode": "ssh_remote", "status": "failed",
                                    "returncode": 1, "metrics": {"remote_partial": 0.9}})
    block, has_parsed = _collect_raw_experiment_metrics(tmp_path)
    assert has_parsed and "local_partial" in block
    assert "remote_partial" not in block


def test_non_remote_refinement_keeps_existing_pre_repair_selection(tmp_path):
    write_refinement(tmp_path, [{"sandbox": attempt("legacy_initial"),
                                 "sandbox_after_fix": attempt("legacy_failed_fix", returncode=1)}],
                     mode="sandbox")
    block, _ = _collect_raw_experiment_metrics(tmp_path)
    assert "legacy_initial_metric" in block and "legacy_initial_pair" in block
