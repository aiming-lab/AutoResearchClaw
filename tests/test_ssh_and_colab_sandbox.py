# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
"""Tests for ssh_remote and colab_drive experiment backends."""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest

from researchclaw.config import (
    ColabDriveConfig,
    ExperimentConfig,
    SandboxConfig,
    SshRemoteConfig,
    DockerSandboxConfig,
    CodeAgentConfig,
    BenchmarkAgentConfig,
    FigureAgentConfig,
)
from researchclaw.experiment.ssh_sandbox import (
    SshRemoteSandbox,
    _build_ssh_base,
    _ssh_target,
)
from researchclaw.experiment.colab_sandbox import (
    ColabDriveSandbox,
    COLAB_WORKER_TEMPLATE,
)
from researchclaw.experiment.factory import create_sandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experiment_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        sandbox=SandboxConfig(),
        docker=DockerSandboxConfig(),
        ssh_remote=SshRemoteConfig(),
        colab_drive=ColabDriveConfig(),
        code_agent=CodeAgentConfig(),
        benchmark_agent=BenchmarkAgentConfig(),
        figure_agent=FigureAgentConfig(),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ===========================================================================
# SSH Remote: unit tests
# ===========================================================================

class TestSshTarget:
    def test_with_user(self):
        cfg = SshRemoteConfig(host="gpu.lab.edu", user="alice")
        assert _ssh_target(cfg) == "alice@gpu.lab.edu"

    def test_without_user(self):
        cfg = SshRemoteConfig(host="gpu.lab.edu")
        assert _ssh_target(cfg) == "gpu.lab.edu"


class TestBuildSshBase:
    def test_default_port(self):
        cfg = SshRemoteConfig(host="server", user="bob")
        cmd = _build_ssh_base(cfg)
        assert "ssh" in cmd
        assert "bob@server" in cmd
        assert "-p" not in cmd

    def test_custom_port(self):
        cfg = SshRemoteConfig(host="server", user="bob", port=2222)
        cmd = _build_ssh_base(cfg)
        idx = cmd.index("-p")
        assert cmd[idx + 1] == "2222"

    def test_key_path(self):
        cfg = SshRemoteConfig(host="server", key_path="~/.ssh/my_key")
        cmd = _build_ssh_base(cfg)
        assert "-i" in cmd


class TestSshRemoteSandboxCommands:
    def test_bare_exec_cmd(self, tmp_path: Path):
        cfg = SshRemoteConfig(
            host="server", user="test", gpu_ids=(0, 1),
            remote_python="python3",
        )
        sb = SshRemoteSandbox(cfg, tmp_path)
        cmd = sb._build_bare_exec_cmd("/tmp/rc-test", entry_point="main.py")
        assert "CUDA_VISIBLE_DEVICES=0,1" in cmd
        assert "HOME=/tmp/rc-test" in cmd
        assert "python3 -u main.py" in cmd
        assert "unshare --net" in cmd

    def test_bare_exec_no_gpu(self, tmp_path: Path):
        cfg = SshRemoteConfig(host="server", user="test")
        sb = SshRemoteSandbox(cfg, tmp_path)
        cmd = sb._build_bare_exec_cmd("/tmp/rc-test", entry_point="main.py")
        assert "CUDA_VISIBLE_DEVICES" not in cmd

    def test_docker_exec_cmd(self, tmp_path: Path):
        cfg = SshRemoteConfig(
            host="server", user="test",
            use_docker=True,
            docker_image="myimage:latest",
            docker_network_policy="none",
            docker_memory_limit_mb=4096,
            docker_shm_size_mb=1024,
            gpu_ids=(0,),
        )
        sb = SshRemoteSandbox(cfg, tmp_path)
        cmd = sb._build_docker_exec_cmd("/tmp/rc-test", entry_point="main.py")
        assert "docker run --rm" in cmd
        assert "-v /tmp/rc-test:/workspace" in cmd
        assert "--network none" in cmd
        assert "--memory=4096m" in cmd
        assert "--shm-size=1024m" in cmd
        assert "device=0" in cmd
        assert "myimage:latest" in cmd
        assert cmd.endswith("main.py")

    def test_docker_exec_full_network(self, tmp_path: Path):
        cfg = SshRemoteConfig(
            host="server", use_docker=True,
            docker_network_policy="full",
        )
        sb = SshRemoteSandbox(cfg, tmp_path)
        cmd = sb._build_docker_exec_cmd("/tmp/rc-test", entry_point="main.py")
        assert "--network" not in cmd


class TestSshConnectivityCheck:
    def test_empty_host(self):
        cfg = SshRemoteConfig(host="")
        ok, msg = SshRemoteSandbox.check_ssh_available(cfg)
        assert not ok
        assert "empty" in msg

    def test_unreachable_host(self):
        cfg = SshRemoteConfig(host="nonexistent-host-12345.invalid")
        ok, msg = SshRemoteSandbox.check_ssh_available(cfg)
        assert not ok


class TestSshSandboxRun:
    """Test run() with mocked SSH commands."""

    def test_run_success(self, tmp_path: Path):
        cfg = SshRemoteConfig(host="fake", user="test")
        sb = SshRemoteSandbox(cfg, tmp_path)

        fake_results = [
            mock.Mock(returncode=0, stdout="", stderr=""),      # mkdir
            mock.Mock(returncode=0, stdout="accuracy: 0.95\nloss: 0.05", stderr=""),  # exec
            mock.Mock(returncode=0, stdout="", stderr=""),      # cleanup
        ]
        call_count = [0]

        def fake_ssh_run(command, *, timeout_sec=60):
            from researchclaw.experiment.ssh_sandbox import _SshResult
            idx = min(call_count[0], len(fake_results) - 1)
            r = fake_results[idx]
            call_count[0] += 1
            return _SshResult(
                returncode=r.returncode,
                stdout=r.stdout,
                stderr=r.stderr,
            )

        def fake_scp(local_dir, remote_dir):
            return True

        with mock.patch.object(sb, '_ssh_run', side_effect=fake_ssh_run):
            with mock.patch.object(sb, '_scp_upload', side_effect=fake_scp):
                result = sb.run("print('hello')", timeout_sec=60)

        assert result.returncode == 0
        assert result.metrics.get("accuracy") == 0.95
        assert result.metrics.get("loss") == 0.05

    def test_run_upload_failure(self, tmp_path: Path):
        cfg = SshRemoteConfig(host="fake", user="test")
        sb = SshRemoteSandbox(cfg, tmp_path)

        from researchclaw.experiment.ssh_sandbox import _SshResult

        with mock.patch.object(sb, '_ssh_run', return_value=_SshResult(0, "", "")):
            with mock.patch.object(sb, '_scp_upload', return_value=False):
                result = sb.run("print('hello')")

        assert result.returncode == -1
        assert "Failed to upload" in result.stderr


# ===========================================================================
# Colab Drive: unit tests
# ===========================================================================

class TestColabDriveCheck:
    def test_empty_root(self):
        cfg = ColabDriveConfig(drive_root="")
        ok, msg = ColabDriveSandbox.check_drive_available(cfg)
        assert not ok
        assert "empty" in msg

    def test_nonexistent_root(self):
        cfg = ColabDriveConfig(drive_root="/nonexistent/path/12345")
        ok, msg = ColabDriveSandbox.check_drive_available(cfg)
        assert not ok
        assert "not found" in msg

    def test_existing_root(self, tmp_path: Path):
        cfg = ColabDriveConfig(drive_root=str(tmp_path))
        ok, msg = ColabDriveSandbox.check_drive_available(cfg)
        assert ok


class TestColabDriveSandbox:
    def test_submit_and_collect(self, tmp_path: Path):
        """Simulate the full flow: submit task → worker picks up → collect result."""
        drive_root = tmp_path / "drive"
        drive_root.mkdir()

        cfg = ColabDriveConfig(
            drive_root=str(drive_root),
            poll_interval_sec=1,
            timeout_sec=10,
        )
        sb = ColabDriveSandbox(cfg, tmp_path / "workdir")

        # Simulate worker in a thread: move pending → done with result
        import threading

        def fake_worker():
            pending = drive_root / "pending"
            done = drive_root / "done"
            for _ in range(20):  # poll for up to 20 seconds
                if pending.exists():
                    for task_dir in pending.iterdir():
                        if task_dir.is_dir():
                            done.mkdir(parents=True, exist_ok=True)
                            done_dir = done / task_dir.name
                            task_dir.rename(done_dir)
                            (done_dir / "result.json").write_text(json.dumps({
                                "returncode": 0,
                                "stdout": "primary_metric: 42.0\naccuracy: 0.99",
                                "stderr": "",
                            }))
                            return
                time.sleep(0.5)

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        result = sb.run("print('experiment')", timeout_sec=15)
        worker.join(timeout=5)

        assert result.returncode == 0
        assert result.metrics.get("primary_metric") == 42.0
        assert result.metrics.get("accuracy") == 0.99

    def test_timeout(self, tmp_path: Path):
        """If worker never picks up, should timeout."""
        drive_root = tmp_path / "drive"
        drive_root.mkdir()

        cfg = ColabDriveConfig(
            drive_root=str(drive_root),
            poll_interval_sec=1,
            timeout_sec=3,
        )
        sb = ColabDriveSandbox(cfg, tmp_path / "workdir")
        result = sb.run("print('hello')", timeout_sec=3)

        assert result.timed_out
        assert result.returncode == -1
        assert "did not complete" in result.stderr

    def test_setup_script_written(self, tmp_path: Path):
        drive_root = tmp_path / "drive"
        drive_root.mkdir()

        cfg = ColabDriveConfig(
            drive_root=str(drive_root),
            poll_interval_sec=1,
            timeout_sec=3,
            setup_script="pip install torch -q",
        )
        sb = ColabDriveSandbox(cfg, tmp_path / "workdir")

        # Just submit, don't wait for result
        staging = tmp_path / "workdir" / "_colab_1"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "main.py").write_text("print('hi')")
        sb._write_setup_script(staging)

        setup_sh = staging / "setup.sh"
        assert setup_sh.exists()
        content = setup_sh.read_text()
        assert "pip install torch -q" in content


class TestColabWorkerTemplate:
    def test_template_not_empty(self):
        assert len(COLAB_WORKER_TEMPLATE) > 100

    def test_template_has_key_elements(self):
        assert "pending" in COLAB_WORKER_TEMPLATE
        assert "done" in COLAB_WORKER_TEMPLATE
        assert "result.json" in COLAB_WORKER_TEMPLATE
        assert "drive.mount" in COLAB_WORKER_TEMPLATE


# ===========================================================================
# Factory integration tests
# ===========================================================================

class TestFactoryIntegration:
    def test_ssh_remote_requires_host(self, tmp_path: Path):
        cfg = _make_experiment_config(
            mode="ssh_remote",
            ssh_remote=SshRemoteConfig(host=""),
        )
        with pytest.raises(RuntimeError, match="host"):
            create_sandbox(cfg, tmp_path)

    def test_ssh_remote_checks_connectivity(self, tmp_path: Path):
        cfg = _make_experiment_config(
            mode="ssh_remote",
            ssh_remote=SshRemoteConfig(host="nonexistent.invalid"),
        )
        with pytest.raises(RuntimeError, match="SSH connectivity"):
            create_sandbox(cfg, tmp_path)

    def test_colab_drive_requires_root(self, tmp_path: Path):
        cfg = _make_experiment_config(
            mode="colab_drive",
            colab_drive=ColabDriveConfig(drive_root=""),
        )
        with pytest.raises(RuntimeError, match="empty"):
            create_sandbox(cfg, tmp_path)

    def test_colab_drive_checks_path(self, tmp_path: Path):
        cfg = _make_experiment_config(
            mode="colab_drive",
            colab_drive=ColabDriveConfig(drive_root="/nonexistent/12345"),
        )
        with pytest.raises(RuntimeError, match="not found"):
            create_sandbox(cfg, tmp_path)

    def test_colab_drive_creates_sandbox(self, tmp_path: Path):
        drive_root = tmp_path / "drive"
        drive_root.mkdir()
        cfg = _make_experiment_config(
            mode="colab_drive",
            colab_drive=ColabDriveConfig(drive_root=str(drive_root)),
        )
        sb = create_sandbox(cfg, tmp_path / "workdir")
        assert isinstance(sb, ColabDriveSandbox)


# ===========================================================================
# ACP timeout fix test
# ===========================================================================

class TestAcpTimeoutFix:
    def test_timeout_passed_from_config(self):
        from researchclaw.config import RCConfig, AcpConfig, LlmConfig
        from researchclaw.llm.acp_client import ACPClient, ACPConfig

        acp_cfg = AcpConfig(agent="codex", timeout_sec=1500)
        llm_cfg = LlmConfig(provider="acp", acp=acp_cfg)

        # Simulate RCConfig with just the fields ACPClient.from_rc_config uses
        fake_rc = mock.Mock()
        fake_rc.llm = llm_cfg

        client = ACPClient.from_rc_config(fake_rc)
        assert client.config.timeout_sec == 1500

    def test_timeout_default(self):
        from researchclaw.llm.acp_client import ACPClient

        fake_rc = mock.Mock()
        fake_rc.llm.acp.agent = "claude"
        fake_rc.llm.acp.cwd = "."
        fake_rc.llm.acp.acpx_command = ""
        fake_rc.llm.acp.session_name = "test"
        fake_rc.llm.acp.timeout_sec = 600

        client = ACPClient.from_rc_config(fake_rc)
        assert client.config.timeout_sec == 600
