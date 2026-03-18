"""SSH remote sandbox for experiment code execution on remote GPU servers.

Uploads experiment code via scp, executes via ssh, and collects results.
Supports any SSH-accessible machine including cloud VMs, lab servers,
and Colab instances with SSH tunnels.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from researchclaw.config import SshRemoteConfig
from researchclaw.experiment.sandbox import (
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)

logger = logging.getLogger(__name__)


class SshRemoteSandbox:
    """Execute experiment code on a remote machine via SSH.

    Same public API as :class:`ExperimentSandbox` and :class:`DockerSandbox`
    so the pipeline can use any backend transparently.

    Execution model:
      1. Create a unique run directory on the remote host
      2. Upload code (and harness) via scp
      3. Optionally run setup commands (pip install, conda activate, etc.)
      4. Execute the experiment script via ssh
      5. Parse stdout for metrics
      6. Clean up the remote run directory
    """

    def __init__(self, config: SshRemoteConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

    # ------------------------------------------------------------------
    # Public API (matches SandboxProtocol)
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        """Run a single Python code string on the remote host."""
        self._run_counter += 1
        staging = self.workdir / f"_ssh_run_{self._run_counter}"
        staging.mkdir(parents=True, exist_ok=True)

        script_path = staging / "main.py"
        script_path.write_text(code, encoding="utf-8")

        self._inject_harness(staging)

        return self._execute(staging, entry_point="main.py", timeout_sec=timeout_sec)

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> SandboxResult:
        """Run a multi-file experiment project on the remote host."""
        self._run_counter += 1
        staging = self.workdir / f"_ssh_project_{self._run_counter}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        # Pre-copy syntax validation — fail fast before any I/O
        err = validate_entry_point(entry_point)
        if err:
            return SandboxResult(
                returncode=-1, stdout="", stderr=err,
                elapsed_sec=0.0, metrics={},
            )

        self._inject_harness(staging)

        for src_item in project_dir.iterdir():
            dest = staging / src_item.name
            if dest.name == "experiment_harness.py":
                logger.warning(
                    "Project contains experiment_harness.py — skipping (immutable)"
                )
                continue
            if src_item.is_dir():
                shutil.copytree(src_item, dest, dirs_exist_ok=True)
            elif src_item.is_file():
                dest.write_bytes(src_item.read_bytes())

        # Post-copy resolve check — catches symlink-based escapes
        err = validate_entry_point_resolved(staging, entry_point)
        if err:
            return SandboxResult(
                returncode=-1, stdout="", stderr=err,
                elapsed_sec=0.0, metrics={},
            )

        entry = staging / entry_point
        if not entry.exists():
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"Entry point {entry_point} not found in project",
                elapsed_sec=0.0,
                metrics={},
            )

        return self._execute(staging, entry_point=entry_point, timeout_sec=timeout_sec)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def check_ssh_available(config: SshRemoteConfig) -> tuple[bool, str]:
        """Return (ok, message) after testing SSH connectivity."""
        if not config.host:
            return False, "ssh_remote.host is empty"
        cmd = _build_ssh_base(config, extra_opts=["-o", "ConnectTimeout=10"])
        cmd.append("echo researchclaw-ssh-ok")
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15, check=False,
            )
            if cp.returncode == 0 and "researchclaw-ssh-ok" in cp.stdout:
                return True, f"SSH connection to {config.host} OK"
            return False, f"SSH test failed (exit {cp.returncode}): {cp.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, f"SSH connection to {config.host} timed out"
        except FileNotFoundError:
            return False, "ssh command not found on PATH"

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        harness_src = Path(__file__).parent / "harness_template.py"
        if harness_src.exists():
            dest = target_dir / "experiment_harness.py"
            dest.write_text(
                harness_src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute(
        self, staging_dir: Path, *, entry_point: str, timeout_sec: int
    ) -> SandboxResult:
        """Core execution flow for remote experiments.

        Steps:
          1. Create a unique temporary directory on the remote host
          2. Upload experiment files via scp
          3. Run any user-defined setup commands (pip install, etc.)
          4. Execute the experiment (bare Python or Docker container)
          5. Parse metrics from stdout (same format as local sandbox)
          6. Clean up the remote directory regardless of outcome
        """
        cfg = self.config
        run_id = f"rc-{uuid.uuid4().hex[:8]}"
        remote_dir = f"{cfg.remote_workdir}/{run_id}"
        remote_dir_q = shlex.quote(remote_dir)

        # 1. Create remote directory
        mkdir_ok = self._ssh_run(f"mkdir -p {remote_dir_q}")
        if mkdir_ok.returncode != 0:
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"Failed to create remote directory: {mkdir_ok.stderr}",
                elapsed_sec=0.0,
                metrics={},
            )

        # 2. Upload code
        upload_ok = self._scp_upload(staging_dir, remote_dir)
        if not upload_ok:
            self._ssh_run(f"rm -rf {remote_dir_q}", timeout_sec=15)
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"Failed to upload code to {cfg.host}:{remote_dir}",
                elapsed_sec=0.0,
                metrics={},
            )

        # 3. Run setup commands (pip install, conda activate, etc.)
        for setup_cmd in cfg.setup_commands:
            setup_result = self._ssh_run(
                f"cd {remote_dir_q} && {setup_cmd}",
                timeout_sec=cfg.setup_timeout_sec,
            )
            if setup_result.returncode != 0:
                logger.warning(
                    "Setup command failed: %s (exit %d): %s",
                    setup_cmd, setup_result.returncode, setup_result.stderr,
                )

        # 4. Execute experiment
        if cfg.use_docker:
            exec_cmd = self._build_docker_exec_cmd(
                remote_dir, entry_point=entry_point,
            )
        else:
            exec_cmd = self._build_bare_exec_cmd(
                remote_dir, entry_point=entry_point,
            )

        start = time.monotonic()
        result = self._ssh_run(exec_cmd, timeout_sec=timeout_sec)
        elapsed = time.monotonic() - start

        timed_out = result.timed_out

        # 5. Parse metrics from stdout
        metrics = parse_metrics(result.stdout)

        # 6. Clean up remote directory
        self._ssh_run(f"rm -rf {remote_dir_q}", timeout_sec=15)

        return SandboxResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            elapsed_sec=elapsed,
            metrics=metrics,
            timed_out=timed_out,
        )

    def _build_bare_exec_cmd(
        self, remote_dir: str, *, entry_point: str,
    ) -> str:
        """Build command to run Python directly on remote host (with basic sandboxing)."""
        cfg = self.config
        rd = shlex.quote(remote_dir)
        ep = shlex.quote(entry_point)
        py = shlex.quote(cfg.remote_python)

        gpu_env = ""
        if cfg.gpu_ids:
            gpu_env = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in cfg.gpu_ids)} "

        # Security layers:
        # 1. HOME override — prevents reading ~/.ssh, ~/.bashrc, etc.
        # 2. unshare --net — drops network access (Linux only).
        # 3. If unshare unavailable, still runs with HOME override but
        #    logs a warning so the user knows network isolation is missing.
        return (
            f"cd {rd} && "
            f"if command -v unshare >/dev/null 2>&1; then "
            f"HOME={rd} "
            f"{gpu_env}"
            f"unshare --net {py} -u {ep}; "
            f"else "
            f"echo 'WARNING: unshare not available, running without network isolation' >&2; "
            f"HOME={rd} "
            f"{gpu_env}"
            f"{py} -u {ep}; "
            f"fi"
        )

    def _build_docker_exec_cmd(
        self, remote_dir: str, *, entry_point: str,
    ) -> str:
        """Build command to run inside a Docker container on the remote host.

        This is the most secure execution mode: code runs in an isolated
        container with restricted network, memory limits, and no access
        to the host filesystem beyond the experiment directory.
        """
        cfg = self.config
        parts = [
            "docker", "run", "--rm",
            "-v", f"{shlex.quote(remote_dir)}:/workspace",
            "-w", "/workspace",
            f"--memory={cfg.docker_memory_limit_mb}m",
            f"--shm-size={cfg.docker_shm_size_mb}m",
        ]

        # Network isolation
        if cfg.docker_network_policy == "none":
            parts.extend(["--network", "none"])

        # GPU passthrough
        if cfg.gpu_ids:
            device_spec = ",".join(str(g) for g in cfg.gpu_ids)
            parts.extend(["--gpus", f"device={device_spec}"])
        else:
            # Try to pass all GPUs; fails gracefully if none available
            parts.extend(["--gpus", "all"])

        parts.append(shlex.quote(cfg.docker_image))
        parts.extend(["python3", "-u", shlex.quote(entry_point)])

        return " ".join(parts)

    def _ssh_run(
        self, command: str, *, timeout_sec: int | None = None
    ) -> _SshResult:
        """Execute a command on the remote host via ssh."""
        if timeout_sec is None:
            timeout_sec = self.config.timeout_sec
        cmd = _build_ssh_base(self.config) + [command]
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            return _SshResult(
                returncode=cp.returncode,
                stdout=cp.stdout,
                stderr=cp.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return _SshResult(
                returncode=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
        except Exception as exc:  # noqa: BLE001
            return _SshResult(
                returncode=-1,
                stdout="",
                stderr=str(exc),
            )

    def _scp_upload(self, local_dir: Path, remote_dir: str) -> bool:
        """Upload all files from local_dir to remote_dir via scp."""
        cfg = self.config
        target = f"{_ssh_target(cfg)}:{remote_dir}/"

        cmd = ["scp", "-r", "-o", "StrictHostKeyChecking=no"]
        if cfg.port != 22:
            cmd.extend(["-P", str(cfg.port)])
        if cfg.key_path:
            cmd.extend(["-i", os.path.expanduser(cfg.key_path)])

        # Upload all files and directories in the staging directory
        items = [str(f) for f in local_dir.iterdir()]
        if not items:
            return True
        cmd.extend(items)
        cmd.append(target)

        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=cfg.scp_timeout_sec, check=False,
            )
            if cp.returncode != 0:
                logger.error("scp upload failed: %s", cp.stderr.strip())
            return cp.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.error("scp upload error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SshResult:
    __slots__ = ("returncode", "stdout", "stderr", "timed_out")

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        timed_out: bool = False,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def _ssh_target(cfg: SshRemoteConfig) -> str:
    """Build user@host string."""
    if cfg.user:
        return f"{cfg.user}@{cfg.host}"
    return cfg.host


def _build_ssh_base(
    cfg: SshRemoteConfig,
    extra_opts: list[str] | None = None,
) -> list[str]:
    """Build the base ssh command with common options.

    *extra_opts* are inserted **before** the hostname so that SSH
    interprets them as SSH options, not as part of the remote command.
    """
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
    ]
    if cfg.port != 22:
        cmd.extend(["-p", str(cfg.port)])
    if cfg.key_path:
        cmd.extend(["-i", os.path.expanduser(cfg.key_path)])
    if extra_opts:
        cmd.extend(extra_opts)
    cmd.append(_ssh_target(cfg))
    return cmd
