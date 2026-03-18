"""Tests for DockerSandbox — all mocked, no real Docker needed."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import DockerSandboxConfig, ExperimentConfig
from researchclaw.experiment.docker_sandbox import DockerSandbox, _next_container_name
from researchclaw.experiment.factory import create_sandbox
from researchclaw.experiment.sandbox import SandboxResult


# ── SandboxResult contract ─────────────────────────────────────────────


def test_sandbox_result_fields():
    r = SandboxResult(
        returncode=0,
        stdout="primary_metric: 0.95\n",
        stderr="",
        elapsed_sec=1.2,
        metrics={"primary_metric": 0.95},
        timed_out=False,
    )
    assert r.returncode == 0
    assert r.metrics["primary_metric"] == 0.95
    assert r.timed_out is False


# ── DockerSandbox command building ─────────────────────────────────────


def test_build_run_command_network_none(tmp_path: Path):
    """network_policy='none' → --network none, --user UID:GID."""
    cfg = DockerSandboxConfig(network_policy="none")
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    cmd = sandbox._build_run_command(
        tmp_path / "staging",
        entry_point="main.py",
        container_name="rc-test-1",
    )
    assert "docker" in cmd
    assert "--gpus" in cmd
    assert "--network" in cmd
    assert "none" in cmd
    assert "--memory=8192m" in cmd
    assert "--shm-size=2048m" in cmd
    assert cmd[-1] == "main.py"
    # Should contain --user (non-root)
    assert "--user" in cmd


def test_build_run_command_setup_only(tmp_path: Path):
    """Default network_policy='setup_only' → RC_SETUP_ONLY_NETWORK=1, --cap-add."""
    cfg = DockerSandboxConfig()  # default is setup_only
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    cmd = sandbox._build_run_command(
        tmp_path / "staging",
        entry_point="main.py",
        container_name="rc-test-setup",
    )
    # Should set env var for setup-only network
    assert "-e" in cmd
    env_idx = [i for i, x in enumerate(cmd) if x == "-e"]
    env_values = [cmd[i + 1] for i in env_idx]
    assert "RC_SETUP_ONLY_NETWORK=1" in env_values
    # Should add NET_ADMIN capability
    assert "--cap-add=NET_ADMIN" in cmd
    # Should NOT have --network none (needs network for setup)
    network_indices = [i for i, x in enumerate(cmd) if x == "--network"]
    assert len(network_indices) == 0
    # Should have --user (runs as host user so experiment can write results.json)
    assert "--user" in cmd


def test_build_run_command_full_network(tmp_path: Path):
    """network_policy='full' → no --network none, has --user."""
    cfg = DockerSandboxConfig(network_policy="full")
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    cmd = sandbox._build_run_command(
        tmp_path / "staging",
        entry_point="main.py",
        container_name="rc-test-full",
    )
    # No --network none
    network_indices = [i for i, x in enumerate(cmd) if x == "--network"]
    assert len(network_indices) == 0
    # Should have --user (non-root)
    assert "--user" in cmd


def test_build_run_command_no_gpu(tmp_path: Path):
    cfg = DockerSandboxConfig(gpu_enabled=False, network_policy="none")
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    cmd = sandbox._build_run_command(
        tmp_path / "staging",
        entry_point="main.py",
        container_name="rc-test-2",
    )
    assert "--gpus" not in cmd


def test_build_run_command_specific_gpus(tmp_path: Path):
    cfg = DockerSandboxConfig(gpu_device_ids=(0, 2), network_policy="none")
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    cmd = sandbox._build_run_command(
        tmp_path / "staging",
        entry_point="main.py",
        container_name="rc-test-3",
    )
    assert "--gpus" in cmd
    gpu_idx = cmd.index("--gpus")
    assert "0,2" in cmd[gpu_idx + 1]


# ── Harness injection ─────────────────────────────────────────────────


def test_harness_injection(tmp_path: Path):
    harness_src = Path(__file__).parent.parent / "researchclaw" / "experiment" / "harness_template.py"
    if not harness_src.exists():
        pytest.skip("harness_template.py not found")

    target = tmp_path / "project"
    target.mkdir()
    DockerSandbox._inject_harness(target)
    assert (target / "experiment_harness.py").exists()


# ── Factory ────────────────────────────────────────────────────────────


def test_factory_returns_experiment_sandbox(tmp_path: Path):
    from researchclaw.experiment.sandbox import ExperimentSandbox

    config = ExperimentConfig(mode="sandbox")
    sandbox = create_sandbox(config, tmp_path / "work")
    assert isinstance(sandbox, ExperimentSandbox)


@patch("researchclaw.experiment.docker_sandbox.DockerSandbox.ensure_image", return_value=True)
@patch("researchclaw.experiment.docker_sandbox.DockerSandbox.check_docker_available", return_value=True)
def test_factory_returns_docker_sandbox(mock_avail, mock_image, tmp_path: Path):
    config = ExperimentConfig(mode="docker")
    sandbox = create_sandbox(config, tmp_path / "work")
    assert isinstance(sandbox, DockerSandbox)


@patch("researchclaw.experiment.docker_sandbox.DockerSandbox.check_docker_available", return_value=False)
def test_factory_falls_back_when_docker_unavailable(mock_avail, tmp_path: Path):
    config = ExperimentConfig(mode="docker")
    sandbox = create_sandbox(config, tmp_path / "work")
    # BUG-002: Should fall back to subprocess sandbox instead of raising
    from researchclaw.experiment.sandbox import ExperimentSandbox
    assert isinstance(sandbox, ExperimentSandbox)


@patch("researchclaw.experiment.docker_sandbox.DockerSandbox.ensure_image", return_value=False)
@patch("researchclaw.experiment.docker_sandbox.DockerSandbox.check_docker_available", return_value=True)
def test_factory_raises_when_image_missing(mock_avail, mock_image, tmp_path: Path):
    config = ExperimentConfig(mode="docker")
    with pytest.raises(RuntimeError, match="not found locally"):
        create_sandbox(config, tmp_path / "work")


# ── run() with mocked subprocess ──────────────────────────────────────


@patch("subprocess.run")
def test_docker_run_success(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=0,
        stdout="primary_metric: 0.85\n",
        stderr="",
    )
    cfg = DockerSandboxConfig(network_policy="none")
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    result = sandbox.run("print('hello')", timeout_sec=60)

    assert result.returncode == 0
    assert result.metrics.get("primary_metric") == 0.85
    assert result.timed_out is False


@patch("subprocess.run")
def test_docker_run_timeout(mock_run, tmp_path: Path):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker run", timeout=10)
    cfg = DockerSandboxConfig(network_policy="none")
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    result = sandbox.run("import time; time.sleep(999)", timeout_sec=10)

    assert result.timed_out is True
    assert result.returncode == -1


# ── Dep detection ─────────────────────────────────────────────────────


def test_detect_pip_packages(tmp_path: Path):
    (tmp_path / "main.py").write_text(
        "import torchdiffeq\nimport numpy\nfrom PIL import Image\n"
    )
    detected = DockerSandbox._detect_pip_packages(tmp_path)
    # torchdiffeq and PIL/Pillow are now in builtin → skipped
    # numpy should be skipped (builtin)
    assert "numpy" not in detected
    assert "torchdiffeq" not in detected


def test_detect_pip_packages_finds_unknown(tmp_path: Path):
    """Unknown packages should be detected."""
    (tmp_path / "main.py").write_text(
        "import some_new_package\nimport numpy\n"
    )
    detected = DockerSandbox._detect_pip_packages(tmp_path)
    assert "some_new_package" in detected
    assert "numpy" not in detected


def test_detect_pip_packages_skips_setup_py(tmp_path: Path):
    """setup.py should not be scanned for experiment deps."""
    (tmp_path / "setup.py").write_text("import some_setup_dep\n")
    (tmp_path / "main.py").write_text("import numpy\n")
    detected = DockerSandbox._detect_pip_packages(tmp_path)
    assert "some_setup_dep" not in detected


def test_detect_pip_packages_maps_imports(tmp_path: Path):
    """Known import-to-pip mappings should be applied."""
    (tmp_path / "main.py").write_text(
        "import cv2\nimport wandb\n"
    )
    detected = DockerSandbox._detect_pip_packages(tmp_path)
    assert "opencv-python" in detected
    assert "wandb" in detected


def test_next_container_name_is_thread_safe():
    names: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            name = _next_container_name()
            with lock:
                names.append(name)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(names) == 100
    assert len(names) == len(set(names))


# ── requirements.txt generation ──────────────────────────────────────


def test_write_requirements_txt_from_auto_detect(tmp_path: Path):
    """Auto-detected packages should be written to requirements.txt."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "main.py").write_text("import wandb\nimport optuna\n")

    cfg = DockerSandboxConfig(auto_install_deps=True)
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    sandbox._write_requirements_txt(staging)

    req_path = staging / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text()
    assert "wandb" in content
    assert "optuna" in content


def test_write_requirements_txt_with_pip_pre_install(tmp_path: Path):
    """pip_pre_install packages should be added to requirements.txt."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "main.py").write_text("import numpy\n")

    cfg = DockerSandboxConfig(pip_pre_install=("einops==0.8.0", "kornia"))
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    sandbox._write_requirements_txt(staging)

    req_path = staging / "requirements.txt"
    assert req_path.exists()
    content = req_path.read_text()
    assert "einops==0.8.0" in content
    assert "kornia" in content


def test_write_requirements_txt_respects_existing(tmp_path: Path):
    """If LLM already generated requirements.txt, append only new packages."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "main.py").write_text("import numpy\n")
    (staging / "requirements.txt").write_text("wandb\n")

    cfg = DockerSandboxConfig(pip_pre_install=("wandb", "einops"))
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    sandbox._write_requirements_txt(staging)

    content = (staging / "requirements.txt").read_text()
    # wandb already in existing file, should not be duplicated
    assert content.count("wandb") == 1
    # einops should be appended
    assert "einops" in content


def test_write_requirements_txt_no_packages(tmp_path: Path):
    """No requirements.txt if no packages needed."""
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "main.py").write_text("import numpy\n")

    cfg = DockerSandboxConfig()
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    sandbox._write_requirements_txt(staging)

    assert not (staging / "requirements.txt").exists()


# ── Static checks (mocked) ────────────────────────────────────────────


@patch("subprocess.run")
def test_check_docker_available_true(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    assert DockerSandbox.check_docker_available() is True


@patch("subprocess.run")
def test_check_docker_available_false(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
    assert DockerSandbox.check_docker_available() is False


@patch("subprocess.run", side_effect=FileNotFoundError)
def test_check_docker_available_no_binary(mock_run):
    assert DockerSandbox.check_docker_available() is False


@patch("subprocess.run")
def test_ensure_image_true(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    assert DockerSandbox.ensure_image("researchclaw/experiment:latest") is True


@patch("subprocess.run")
def test_ensure_image_false(mock_run):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
    assert DockerSandbox.ensure_image("nonexistent:latest") is False


# ── Default config values ────────────────────────────────────────────


def test_default_network_policy_is_setup_only():
    """Default network_policy should be 'setup_only', not 'none'."""
    cfg = DockerSandboxConfig()
    assert cfg.network_policy == "setup_only"


def test_default_auto_install_deps_enabled():
    cfg = DockerSandboxConfig()
    assert cfg.auto_install_deps is True


# ── Entry point path traversal validation ─────────────────────────────


@patch("researchclaw.experiment.docker_sandbox.subprocess.run")
def test_run_project_rejects_path_traversal(mock_run: MagicMock, tmp_path: Path):
    """run_project() must reject entry_point with '..' components."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print('hi')")

    cfg = DockerSandboxConfig()
    work = tmp_path / "work"
    sandbox = DockerSandbox(cfg, work)
    # Create escape target so .exists() alone wouldn't catch it
    work.mkdir(parents=True, exist_ok=True)
    (work / "escape.py").write_text("print('escaped!')")

    result = sandbox.run_project(project, entry_point="../escape.py")

    assert result.returncode == -1
    assert ".." in result.stderr
    mock_run.assert_not_called()


@patch("researchclaw.experiment.docker_sandbox.subprocess.run")
def test_run_project_rejects_absolute_path(mock_run: MagicMock, tmp_path: Path):
    """run_project() must reject absolute entry_point paths."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print('hi')")

    cfg = DockerSandboxConfig()
    sandbox = DockerSandbox(cfg, tmp_path / "work")
    result = sandbox.run_project(project, entry_point="/etc/passwd")

    assert result.returncode == -1
    assert "relative" in result.stderr.lower() or "absolute" in result.stderr.lower()
    mock_run.assert_not_called()
