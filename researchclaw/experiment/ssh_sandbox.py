"""SSH remote sandbox for experiment code execution on remote GPU servers.

Uploads experiment code via scp, executes via ssh, and collects results.
Supports any SSH-accessible machine including cloud VMs, lab servers,
and Colab instances with SSH tunnels.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# A small supervisor travels with the project. The experiment deadline lives on
# the compute host, so losing the SSH connection does not remove the deadline.
# Logs and completion evidence are written before they are sent over SSH.
_REMOTE_RUNNER = r'''import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
job = json.loads((root / "_researchclaw_job.json").read_text())
status_path = root / "_researchclaw_status.json"
process_path = root / "_researchclaw_process.json"
signal.signal(signal.SIGHUP, signal.SIG_IGN)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # access denied cannot prove that the group is gone


def terminate_group(pgid):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    for _ in range(20):
        if not group_exists(pgid):
            return True
        time.sleep(0.1)
    return False


def remove_container():
    name = job.get("container_name")
    if not name:
        return True
    try:
        removed = subprocess.run(["docker", "rm", "-f", name],
                                 capture_output=True, timeout=15)
        if removed.returncode == 0:
            return True
        # A failed remove can also mean the --rm container already exited.
        inspected = subprocess.run(["docker", "ps", "-aq", "--filter", "name=^/" + name + "$"],
                                   capture_output=True, timeout=15)
        return inspected.returncode == 0 and not inspected.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False


if "--terminate" in sys.argv:
    confirmed = False
    if status_path.exists():
        status = json.loads(status_path.read_text())
        confirmed = bool(status.get("remote_termination_confirmed"))
    if not confirmed and process_path.exists():
        pid = json.loads(process_path.read_text())["pgid"]
        confirmed = terminate_group(pid)
    confirmed = remove_container() and confirmed
    print(json.dumps({"remote_termination_confirmed": confirmed}))
    sys.exit(0 if confirmed else 1)

status = {"returncode": -1, "timed_out": False,
          "remote_termination_confirmed": False, "phase": "setup"}
try:
    with (root / "_researchclaw_stdout.log").open("wb") as out, \
         (root / "_researchclaw_stderr.log").open("wb") as err:
        commands = [("setup", c, job["setup_timeout_sec"])
                    for c in job["setup_commands"]]
        commands.append(("experiment", job["command"], job["timeout_sec"]))
        for phase, command, deadline in commands:
            status["phase"] = phase
            proc = subprocess.Popen(["bash", "-c", command], cwd=root,
                                    stdout=out, stderr=err, start_new_session=True)
            write_json(process_path, {"pgid": proc.pid})
            try:
                status["returncode"] = proc.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                status["timed_out"] = True
                err.write((phase + " exceeded its remote deadline\n").encode())
                err.flush()
                terminate_group(proc.pid)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                status["returncode"] = -1
            # Also stop children left running after their parent exits.
            terminate_group(proc.pid)
            status["remote_termination_confirmed"] = not group_exists(proc.pid)
            if status["returncode"] != 0 or not status["remote_termination_confirmed"]:
                break
        if job.get("container_name"):
            status["remote_termination_confirmed"] = (
                remove_container() and status["remote_termination_confirmed"])
        if not status["remote_termination_confirmed"]:
            status["returncode"] = -1
            err.write(b"Remote process termination could not be confirmed\n")
except BaseException as exc:
    status["returncode"] = -1
    status["error"] = str(exc)
finally:
    write_json(status_path, status)

for filename, stream in (("_researchclaw_stdout.log", sys.stdout),
                         ("_researchclaw_stderr.log", sys.stderr)):
    path = root / filename
    if path.exists():
        with path.open("rb") as source:
            while True:
                chunk = source.read(65536)
                if not chunk:
                    break
                stream.buffer.write(chunk)
sys.exit(0 if status["returncode"] == 0 else 1)
'''


class SshRemoteSandbox:
    """Execute experiment code on a remote machine via SSH.

    Same public API as :class:`ExperimentSandbox` and :class:`DockerSandbox`
    so the pipeline can use any backend transparently.

    Execution model:
      1. Create a unique run directory on the remote host
      2. Upload code (and harness) via scp
      3. Optionally run setup commands (pip install, conda activate, etc.)
      4. Execute the experiment script via ssh
      5. Collect the complete project, raw artifacts, logs and completion record
      6. Optionally remove the remote copy after verified success and collection
    """

    def __init__(self, config: SshRemoteConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0
        self.last_run_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public API (matches SandboxProtocol)
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        """Run a single Python code string on the remote host."""
        self._run_counter += 1
        staging = self.workdir / f"_ssh_run_{self._run_counter}_{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True, exist_ok=True)
        self.last_run_dir = staging

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
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run a multi-file experiment project on the remote host."""
        self._run_counter += 1
        staging = self.workdir / f"_ssh_project_{self._run_counter}_{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True, exist_ok=True)
        self.last_run_dir = staging

        # Pre-copy syntax validation — fail fast before any I/O
        err = validate_entry_point(entry_point)
        if err:
            return SandboxResult(
                returncode=-1, stdout="", stderr=err,
                elapsed_sec=0.0, metrics={},
            )

        err = validate_entry_point_resolved(project_dir, entry_point)
        if err:
            return SandboxResult(-1, "", err, 0.0, {})

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

        return self._execute(
            staging,
            entry_point=entry_point,
            timeout_sec=timeout_sec,
            entry_args=args,
            env_overrides=env_overrides,
        )

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
        self,
        staging_dir: Path,
        *,
        entry_point: str,
        timeout_sec: int,
        entry_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run, collect evidence, and only clean up an explicitly disposable success."""
        cfg = self.config
        run_id = f"rc-{uuid.uuid4().hex[:8]}"
        remote_dir = f"{cfg.remote_workdir}/{run_id}"
        remote_dir_q = shlex.quote(remote_dir)
        start = time.monotonic()
        evidence: dict[str, object] = {
            "host": cfg.host,
            "remote_dir": remote_dir,
            "artifacts_collected": False,
            "remote_retained": True,
            "remote_termination_confirmed": False,
            "network_isolation": cfg.network_isolation,
            "local_run_dir": str(staging_dir),
        }

        def finish(returncode: int, stdout: str, stderr: str,
                   timed_out: bool = False) -> SandboxResult:
            evidence.update(returncode=returncode, timed_out=timed_out)
            (staging_dir / "_researchclaw_client_stdout.log").write_text(stdout, encoding="utf-8")
            (staging_dir / "_researchclaw_client_stderr.log").write_text(stderr, encoding="utf-8")
            metadata = json.dumps(evidence, indent=2) + "\n"
            (staging_dir / "_researchclaw_remote.json").write_text(metadata, encoding="utf-8")
            if returncode != 0:
                if evidence["remote_retained"] is True:
                    stderr += f"\nRemote evidence retained at {cfg.host}:{remote_dir}"
                else:
                    stderr += f"\nRemote directory creation was not confirmed at {cfg.host}:{remote_dir}"
            return SandboxResult(returncode, stdout, stderr,
                                 time.monotonic() - start, parse_metrics(stdout), timed_out)

        mkdir_ok = self._ssh_run(f"mkdir -p {remote_dir_q}")
        if mkdir_ok.returncode != 0:
            evidence["remote_retained"] = None  # creation was not confirmed
            return finish(-1, "", f"Failed to create remote directory: {mkdir_ok.stderr}",
                          mkdir_ok.timed_out)

        if cfg.use_docker:
            exec_cmd = self._build_docker_exec_cmd(
                remote_dir,
                entry_point=entry_point,
                args=entry_args,
                env_overrides=env_overrides,
                container_name=run_id,
            )
        else:
            exec_cmd = self._build_bare_exec_cmd(
                remote_dir,
                entry_point=entry_point,
                args=entry_args,
                env_overrides=env_overrides,
            )

        (staging_dir / "_researchclaw_runner.py").write_text(_REMOTE_RUNNER, encoding="utf-8")
        (staging_dir / "_researchclaw_job.json").write_text(json.dumps({
            "command": exec_cmd,
            "timeout_sec": timeout_sec,
            "setup_commands": list(cfg.setup_commands),
            "setup_timeout_sec": cfg.setup_timeout_sec,
            "container_name": run_id if cfg.use_docker else None,
        }), encoding="utf-8")
        # Old completion records from a previous project run must not be uploaded.
        for name in ("_researchclaw_status.json", "_researchclaw_process.json",
                     "_researchclaw_stdout.log", "_researchclaw_stderr.log",
                     "_researchclaw_remote.json", "_researchclaw_client_stdout.log",
                     "_researchclaw_client_stderr.log"):
            (staging_dir / name).unlink(missing_ok=True)

        if not self._scp_upload(staging_dir, remote_dir):
            return finish(-1, "", f"Failed to upload code to {cfg.host}:{remote_dir}")

        supervisor = (f"cd {remote_dir_q} && "
                      f"{shlex.quote(cfg.remote_python)} -u _researchclaw_runner.py")
        # Leave time for remote termination and log flush after its own deadline.
        connection_timeout = timeout_sec + len(cfg.setup_commands) * cfg.setup_timeout_sec
        connection_timeout += 30 * (len(cfg.setup_commands) + 1)
        result = self._ssh_run(supervisor, timeout_sec=connection_timeout)
        if result.timed_out or result.returncode in (-1, 255):
            stopped = self._ssh_run(supervisor + " --terminate", timeout_sec=30)
            try:
                evidence["remote_termination_confirmed"] = (
                    stopped.returncode == 0
                    and json.loads(stopped.stdout).get("remote_termination_confirmed") is True
                )
            except (ValueError, AttributeError):
                evidence["remote_termination_confirmed"] = False

        # Each run has its own staging directory. The caller's original project
        # stays untouched and partial transfers remain in last_run_dir.
        collected = staging_dir
        evidence["local_artifacts_dir"] = str(collected)
        if not self._scp_download(remote_dir, collected):
            detail = "Failed to collect remote artifacts; remote execution is not a successful local run"
            if ((result.timed_out or result.returncode in (-1, 255))
                    and not evidence["remote_termination_confirmed"]):
                detail += "; remote process termination could not be confirmed"
            return finish(-1, result.stdout, result.stderr + "\n" + detail, result.timed_out)

        stdout_path = collected / "_researchclaw_stdout.log"
        stderr_path = collected / "_researchclaw_stderr.log"
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else result.stdout
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else result.stderr
        try:
            status = json.loads((collected / "_researchclaw_status.json").read_text(encoding="utf-8"))
            if (not isinstance(status, dict)
                    or type(status.get("returncode")) is not int
                    or type(status.get("timed_out")) is not bool
                    or type(status.get("remote_termination_confirmed")) is not bool):
                raise ValueError("invalid completion record")
        except (OSError, ValueError) as exc:
            return finish(-1, stdout, stderr + f"\nRemote completion could not be confirmed: {exc}",
                          result.timed_out)

        evidence.update(remote_status=status,
                        remote_termination_confirmed=status["remote_termination_confirmed"])
        evidence["artifacts_collected"] = True
        timed_out = result.timed_out or status["timed_out"]
        returncode = status["returncode"]
        if result.returncode != 0 and returncode == 0:
            returncode = result.returncode
            stderr += f"\nSSH execution failed (exit {result.returncode}): {result.stderr}"
        if not status["remote_termination_confirmed"]:
            returncode = -1
            stderr += "\nRemote process termination could not be confirmed"
        if timed_out:
            returncode = -1
        if returncode == 0 and not cfg.keep_remote:
            cleanup = self._ssh_run(f"rm -rf -- {remote_dir_q}", timeout_sec=15)
            if cleanup.returncode == 0:
                evidence["remote_retained"] = False
            else:
                evidence["cleanup_error"] = cleanup.stderr
                stderr += "\nRemote cleanup failed; the remote copy is retained"
        return finish(returncode, stdout, stderr, timed_out)

    def _build_bare_exec_cmd(
        self,
        remote_dir: str,
        *,
        entry_point: str,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Build command to run Python directly on remote host (with basic sandboxing)."""
        cfg = self.config
        rd = shlex.quote(remote_dir)
        ep = shlex.quote(entry_point)
        py = shlex.quote(cfg.remote_python)
        arg_text = " ".join(shlex.quote(arg) for arg in (args or []))
        arg_suffix = f" {arg_text}" if arg_text else ""
        _SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        env_parts = [
            f"{name}={shlex.quote(value)}"
            for name, value in sorted((env_overrides or {}).items())
            if value and _SAFE_ENV_NAME.match(name)
        ]
        env_prefix = (" ".join(env_parts) + " ") if env_parts else ""

        gpu_env = ""
        if cfg.gpu_ids:
            gpu_env = f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in cfg.gpu_ids)} "

        # HOME is a cache/config location, not a filesystem security boundary.
        # Isolation is an explicit policy, never a silent fallback after failure.
        if cfg.network_isolation == "disabled":
            isolation = ""
            preflight = ""
        elif cfg.network_isolation == "required":
            isolation = "unshare --net "
            preflight = (
                "if ! command -v unshare >/dev/null 2>&1 "
                "|| ! unshare --net true; then "
                "echo 'Required network isolation unavailable or not permitted; "
                "set ssh_remote.network_isolation=disabled explicitly to allow networking' >&2; "
                "exit 1; fi; "
            )
        else:
            raise ValueError("ssh_remote.network_isolation must be required or disabled")
        return (f"cd {rd} && {preflight}"
                f"HOME={rd} {gpu_env}{env_prefix}"
                f"{isolation}{py} -u {ep}{arg_suffix}")

    def _build_docker_exec_cmd(
        self,
        remote_dir: str,
        *,
        entry_point: str,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
        container_name: str | None = None,
    ) -> str:
        """Build command to run inside a Docker container on the remote host.

        This is the most secure execution mode: code runs in an isolated
        container with restricted network, memory limits, and no access
        to the host filesystem beyond the experiment directory.
        """
        cfg = self.config
        if cfg.docker_network_policy not in ("none", "full"):
            raise ValueError("ssh_remote.docker_network_policy must be none or full")
        parts = [
            "docker", "run", "--rm",
            "-v", f"{shlex.quote(remote_dir)}:/workspace",
            "-w", "/workspace",
            # BUG-DA8-14: Mirror local Docker sandbox security hardening
            "-e", "HOME=/workspace/.home",
            "-e", "TORCH_HOME=/workspace/.home/.cache/torch",
            "-e", "MPLCONFIGDIR=/tmp/matplotlib",
            f"--memory={cfg.docker_memory_limit_mb}m",
            f"--shm-size={cfg.docker_shm_size_mb}m",
        ]
        if container_name:
            parts.extend(["--name", shlex.quote(container_name)])

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

        _SAFE_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if env_overrides:
            for name, value in sorted(env_overrides.items()):
                if not value or not _SAFE_ENV.match(name):
                    continue
                parts.extend(["-e", shlex.quote(f"{name}={value}")])

        parts.append(shlex.quote(cfg.docker_image))
        parts.extend(["python3", "-u", shlex.quote(entry_point)])
        if args:
            parts.extend(shlex.quote(arg) for arg in args)

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

        cmd = _build_scp_base(cfg)

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
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("scp upload error: %s", exc)
            return False

    def _scp_download(self, remote_dir: str, local_dir: Path) -> bool:
        """Collect every remote file, including nested and hidden artifacts/logs."""
        cmd = _build_scp_base(self.config)
        cmd.extend([f"{_ssh_target(self.config)}:{remote_dir}/.", str(local_dir)])
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=self.config.scp_timeout_sec, check=False)
            if cp.returncode != 0:
                logger.error("scp artifact collection failed: %s", cp.stderr.strip())
            return cp.returncode == 0
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("scp artifact collection error: %s", exc)
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
        "-o", "StrictHostKeyChecking=yes",
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


def _build_scp_base(cfg: SshRemoteConfig) -> list[str]:
    cmd = ["scp", "-r", "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes"]
    if cfg.port != 22:
        cmd.extend(["-P", str(cfg.port)])
    if cfg.key_path:
        cmd.extend(["-i", os.path.expanduser(cfg.key_path)])
    return cmd
