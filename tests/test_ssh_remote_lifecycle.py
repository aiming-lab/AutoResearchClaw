"""Remote execution lifecycle tests; all transport stays on the local host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

import pytest

from researchclaw.config import SshRemoteConfig, _parse_experiment_config, validate_config
from researchclaw.experiment.ssh_sandbox import SshRemoteSandbox, _SshResult


@pytest.fixture
def local_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise the shipped supervisor with local file copies instead of SSH."""
    config = SshRemoteConfig(
        host="fake-compute", remote_workdir=str(tmp_path / "remote"),
        remote_python=sys.executable, network_isolation="disabled",
    )
    sandbox = SshRemoteSandbox(config, tmp_path / "local")
    commands: list[str] = []

    def run(command: str, *, timeout_sec: int | None = None):
        commands.append(command)
        cp = subprocess.run(["bash", "-c", command], capture_output=True,
                            text=True, timeout=timeout_sec, check=False)
        return _SshResult(cp.returncode, cp.stdout, cp.stderr)

    def upload(source: Path, destination: str):
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return True

    def download(source: str, destination: Path):
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return True

    monkeypatch.setattr(sandbox, "_ssh_run", run)
    monkeypatch.setattr(sandbox, "_scp_upload", upload)
    monkeypatch.setattr(sandbox, "_scp_download", download)
    return sandbox, commands


def manifest(sandbox: SshRemoteSandbox):
    assert sandbox.last_run_dir is not None
    return json.loads((sandbox.last_run_dir / "_researchclaw_remote.json").read_text())


def test_multi_file_project_collects_raw_artifacts_and_preserves_input(local_remote, tmp_path):
    sandbox, _ = local_remote
    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "helper.py").write_text("value = 7\n")
    (project / ".input").write_text("hidden input")
    (project / "main.py").write_text(
        "from pathlib import Path\nfrom pkg.helper import value\n"
        "import json, os, sys\n"
        "assert Path('.input').read_text() == 'hidden input'\n"
        "assert sys.argv[1:] == ['--label', 'two words']\n"
        "assert os.environ['RUN_LABEL'] == 'explicit env'\n"
        "Path('raw').mkdir()\nPath('raw/measurements.csv').write_text('value\\n7\\n')\n"
        "Path('.generated').write_text('hidden result')\n"
        "Path('results.json').write_text(json.dumps({'value': value}))\n"
        "print('accuracy: 0.95')\nprint('diagnostic', file=sys.stderr)\n"
    )
    result = sandbox.run_project(project, args=["--label", "two words"],
                                 env_overrides={"RUN_LABEL": "explicit env"}, timeout_sec=5)
    assert result.returncode == 0, result.stderr
    assert result.metrics["accuracy"] == 0.95
    run = sandbox.last_run_dir
    assert run is not None
    assert json.loads((run / "results.json").read_text()) == {"value": 7}
    assert (run / "raw/measurements.csv").read_text() == "value\n7\n"
    assert (run / ".generated").read_text() == "hidden result"
    assert (run / "_researchclaw_stdout.log").read_text() == "accuracy: 0.95\n"
    assert "diagnostic" in (run / "_researchclaw_stderr.log").read_text()
    assert not (project / "results.json").exists()
    assert not (project / "_researchclaw_remote.json").exists()
    evidence = manifest(sandbox)
    assert evidence["artifacts_collected"] is True
    assert evidence["remote_retained"] is True
    assert Path(evidence["remote_dir"]).exists()


def test_explicit_cleanup_only_after_successful_collection(local_remote):
    from dataclasses import replace
    sandbox, commands = local_remote
    sandbox.config = replace(sandbox.config, keep_remote=False)
    result = sandbox.run("print('accuracy: 1.0')", timeout_sec=5)
    assert result.returncode == 0
    evidence = manifest(sandbox)
    assert evidence["artifacts_collected"] is True
    assert evidence["remote_retained"] is False
    assert not Path(evidence["remote_dir"]).exists()
    assert any(command.startswith("rm -rf -- ") for command in commands)


def test_execution_failure_preserves_remote_partial_results_and_logs(local_remote):
    from dataclasses import replace
    sandbox, commands = local_remote
    sandbox.config = replace(sandbox.config, keep_remote=False)
    result = sandbox.run(
        "from pathlib import Path\nimport sys\n"
        "Path('partial.csv').write_text('first observation')\n"
        "print('started', flush=True)\nprint('experiment failed', file=sys.stderr)\n"
        "sys.exit(3)\n", timeout_sec=5,
    )
    assert result.returncode == 3
    assert "experiment failed" in result.stderr
    assert "Remote evidence retained" in result.stderr
    assert (sandbox.last_run_dir / "partial.csv").read_text() == "first observation"
    evidence = manifest(sandbox)
    assert evidence["artifacts_collected"] is True
    assert Path(evidence["remote_dir"]).exists()
    assert not any(command.startswith("rm -rf") for command in commands)


def test_upload_failure_never_deletes_remote_evidence(local_remote, monkeypatch):
    sandbox, commands = local_remote
    monkeypatch.setattr(sandbox, "_scp_upload", lambda *args: False)
    result = sandbox.run("raise RuntimeError('must not run')")
    assert result.returncode == -1
    assert "Failed to upload" in result.stderr
    assert len(commands) == 1
    assert Path(manifest(sandbox)["remote_dir"]).exists()


def test_download_failure_is_non_success_and_keeps_partial_local_files(local_remote, monkeypatch):
    sandbox, commands = local_remote

    def partial_download(source, destination):
        shutil.copyfile(Path(source) / "results.json", destination / "results.json")
        return False

    monkeypatch.setattr(sandbox, "_scp_download", partial_download)
    result = sandbox.run(
        "from pathlib import Path\nPath('results.json').write_text('{\"metric\": 1}')\n"
        "print('accuracy: 1.0')", timeout_sec=5,
    )
    assert result.returncode == -1
    assert "Failed to collect" in result.stderr
    assert (sandbox.last_run_dir / "results.json").exists()
    evidence = manifest(sandbox)
    assert evidence["artifacts_collected"] is False
    assert Path(evidence["remote_dir"]).exists()
    assert not any(command.startswith("rm -rf") for command in commands)


def test_missing_completion_evidence_cannot_be_reported_as_success(local_remote, monkeypatch):
    sandbox, _ = local_remote
    real_download = sandbox._scp_download

    def incomplete_download(source, destination):
        real_download(source, destination)
        (destination / "_researchclaw_status.json").unlink()
        return True

    monkeypatch.setattr(sandbox, "_scp_download", incomplete_download)
    result = sandbox.run("print('accuracy: 1.0')", timeout_sec=5)
    assert result.returncode == -1
    assert "completion could not be confirmed" in result.stderr
    assert manifest(sandbox)["remote_retained"] is True


def test_failed_setup_stops_before_experiment_and_collects_setup_logs(local_remote):
    from dataclasses import replace
    sandbox, _ = local_remote
    sandbox.config = replace(sandbox.config, setup_commands=("echo setup-failed >&2; exit 7",))
    result = sandbox.run("print('must not execute')", timeout_sec=5)
    assert result.returncode == 7
    assert "setup-failed" in result.stderr
    assert "must not execute" not in result.stdout
    assert manifest(sandbox)["remote_status"]["phase"] == "setup"


def test_remote_deadline_stops_experiment_and_child_process(local_remote):
    sandbox, _ = local_remote
    result = sandbox.run(
        "from pathlib import Path\nimport subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "print('accuracy: 0.5', flush=True)\ntime.sleep(60)\n", timeout_sec=1,
    )
    assert result.returncode == -1
    assert result.timed_out is True
    assert result.elapsed_sec < 15
    assert "remote deadline" in result.stderr
    evidence = manifest(sandbox)
    assert evidence["remote_retained"] is True
    assert evidence["remote_status"]["timed_out"] is True
    pid = (sandbox.last_run_dir / "child.pid").read_text()
    child_state = subprocess.run(["ps", "-o", "stat=", "-p", pid],
                                 text=True, capture_output=True, check=False).stdout.strip()
    # An unreaped zombie cannot perform computation; live children must be gone.
    assert not child_state or child_state.startswith("Z"), child_state
    if not evidence["remote_termination_confirmed"]:
        assert "termination could not be confirmed" in result.stderr


@pytest.mark.parametrize("confirmed", [True, False])
def test_connection_timeout_attempts_remote_stop_and_records_confirmation(tmp_path, confirmed):
    sandbox = SshRemoteSandbox(SshRemoteConfig(host="fake"), tmp_path)
    results = [_SshResult(0, "", ""), _SshResult(-1, "partial output", "lost connection", True),
               _SshResult(0 if confirmed else 1,
                          json.dumps({"remote_termination_confirmed": confirmed}), "")]
    with mock.patch.object(sandbox, "_ssh_run", side_effect=results) as ssh:
        with mock.patch.object(sandbox, "_scp_upload", return_value=True):
            with mock.patch.object(sandbox, "_scp_download", return_value=False):
                result = sandbox.run("print('hello')", timeout_sec=1)
    assert result.returncode == -1 and result.timed_out
    assert ssh.call_args_list[2].args[0].endswith(" --terminate")
    assert manifest(sandbox)["remote_termination_confirmed"] is confirmed
    if not confirmed:
        assert "termination could not be confirmed" in result.stderr


def test_required_unshare_permission_failure_never_runs_without_isolation(local_remote, tmp_path, monkeypatch):
    from dataclasses import replace
    sandbox, _ = local_remote
    sandbox.config = replace(sandbox.config, network_isolation="required")
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    unshare = tools_dir / "unshare"
    unshare.write_text("#!/bin/sh\necho 'unshare: Operation not permitted' >&2\nexit 1\n")
    unshare.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools_dir) + os.pathsep + os.environ["PATH"])
    result = sandbox.run("print('must not execute')", timeout_sec=5)
    assert result.returncode != 0
    assert "Required network isolation unavailable or not permitted" in result.stderr
    assert "must not execute" not in result.stdout


def test_run_directories_are_unique_and_prior_artifacts_remain(local_remote):
    sandbox, _ = local_remote
    first = sandbox.run("print('accuracy: 0.1')", timeout_sec=5)
    first_dir = sandbox.last_run_dir
    second = sandbox.run("print('accuracy: 0.2')", timeout_sec=5)
    assert first.returncode == second.returncode == 0
    assert sandbox.last_run_dir != first_dir
    assert (first_dir / "_researchclaw_stdout.log").read_text() == "accuracy: 0.1\n"


def test_symlink_entry_point_outside_source_is_rejected_before_transport(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('escaped')")
    (project / "main.py").symlink_to(outside)
    sandbox = SshRemoteSandbox(SshRemoteConfig(host="fake"), tmp_path / "work")
    with mock.patch.object(sandbox, "_ssh_run") as ssh:
        result = sandbox.run_project(project)
    assert result.returncode == -1
    assert "escapes" in result.stderr
    ssh.assert_not_called()


def test_upload_and_download_use_strict_noninteractive_ssh_options(tmp_path):
    sandbox = SshRemoteSandbox(SshRemoteConfig(host="gpu", user="alice", port=2222,
                                              key_path="~/.ssh/example"), tmp_path)
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / ".hidden").write_text("hidden")
    with mock.patch("researchclaw.experiment.ssh_sandbox.subprocess.run",
                    return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
        assert sandbox._scp_upload(tmp_path, "/remote/run")
        upload = run.call_args.args[0]
        assert str(tmp_path / ".hidden") in upload
        assert sandbox._scp_download("/remote/run", tmp_path)
        download = run.call_args.args[0]
    for command in (upload, download):
        assert "StrictHostKeyChecking=yes" in command
        assert "BatchMode=yes" in command
        assert command[command.index("-P") + 1] == "2222"
        assert command[command.index("-i") + 1] == os.path.expanduser("~/.ssh/example")
    assert "alice@gpu:/remote/run/." in download


def test_config_parses_remote_retention_and_explicit_network_policy():
    config = _parse_experiment_config({"ssh_remote": {"keep_remote": False,
                                                     "network_isolation": "disabled"}})
    assert config.ssh_remote.keep_remote is False
    assert config.ssh_remote.network_isolation == "disabled"
    assert SshRemoteConfig().keep_remote is True
    assert SshRemoteConfig().network_isolation == "required"


@pytest.mark.parametrize("field,value", [("network_isolation", "automatic"),
                                         ("keep_remote", "false"),
                                         ("docker_network_policy", "setup_only"),
                                         ("docker_network_policy", "pip_only")])
def test_config_rejects_ambiguous_isolation_and_retention_settings(field, value):
    validation = validate_config({"experiment": {"ssh_remote": {field: value}}}, check_paths=False)
    assert any(f"experiment.ssh_remote.{field}" in error for error in validation.errors)
