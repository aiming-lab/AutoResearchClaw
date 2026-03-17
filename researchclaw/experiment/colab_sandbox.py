"""Google Drive-based async sandbox for Colab experiment execution.

Execution model:
  1. Write experiment code to a shared Google Drive folder (pending/)
  2. A Colab notebook polls pending/, runs each script, writes results to done/
  3. This sandbox polls done/ until results appear or timeout
  4. Parse metrics from the result file and return

This approach is more robust than direct SSH to Colab because:
  - No SSH tunnel to maintain
  - Colab session timeouts only kill the current experiment, not the pipeline
  - Google Drive sync handles reconnects transparently

Requirements:
  - Google Drive for Desktop installed and syncing (or any Drive mount)
  - A Colab notebook running the worker loop (template provided below)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from researchclaw.config import ColabDriveConfig
from researchclaw.experiment.sandbox import SandboxResult, parse_metrics

logger = logging.getLogger(__name__)

# Template for the Colab worker notebook
COLAB_WORKER_TEMPLATE = '''\
# === ResearchClaw Colab Worker ===
# Run this cell in Google Colab with GPU enabled.
# It polls Google Drive for experiment tasks and executes them.

import os, json, time, subprocess, traceback
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/researchclaw")
PENDING = DRIVE_ROOT / "pending"
RUNNING = DRIVE_ROOT / "running"
DONE = DRIVE_ROOT / "done"

for d in [PENDING, RUNNING, DONE]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Worker ready. Watching {PENDING}")
print("Press Ctrl+C or stop the cell to quit.\\n")

while True:
    for task_dir in sorted(PENDING.iterdir()):
        if not task_dir.is_dir():
            continue

        task_id = task_dir.name
        run_dir = RUNNING / task_id
        done_dir = DONE / task_id

        # Move to running
        task_dir.rename(run_dir)
        print(f"[{task_id}] Running...")

        # Run setup.sh if present
        setup_sh = run_dir / "setup.sh"
        if setup_sh.exists():
            subprocess.run(["bash", str(setup_sh)], cwd=str(run_dir),
                           capture_output=True, timeout=300)

        # Find entry point
        entry = run_dir / "main.py"
        if not entry.exists():
            # Try first .py file
            py_files = sorted(run_dir.glob("*.py"))
            entry = py_files[0] if py_files else None

        result = {"returncode": -1, "stdout": "", "stderr": "entry point not found"}

        if entry:
            try:
                cp = subprocess.run(
                    ["python3", "-u", str(entry)],
                    cwd=str(run_dir),
                    capture_output=True, text=True,
                    timeout=1800,  # 30 min max per experiment
                )
                result = {
                    "returncode": cp.returncode,
                    "stdout": cp.stdout,
                    "stderr": cp.stderr,
                }
            except subprocess.TimeoutExpired as e:
                result = {
                    "returncode": -1,
                    "stdout": (e.stdout or b"").decode("utf-8", errors="replace"),
                    "stderr": "Timed out after 1800s",
                    "timed_out": True,
                }
            except Exception:
                result = {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": traceback.format_exc(),
                }

        # Write result and move to done
        (run_dir / "result.json").write_text(json.dumps(result))
        run_dir.rename(done_dir)
        print(f"[{task_id}] Done (exit {result['returncode']})")

    time.sleep(10)
'''


class ColabDriveSandbox:
    """Execute experiments asynchronously via Google Drive + Colab worker.

    Same public API as ExperimentSandbox/DockerSandbox/SshRemoteSandbox.
    """

    def __init__(self, config: ColabDriveConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

        # Resolve drive root
        self.drive_root = Path(config.drive_root).expanduser().resolve()
        self.pending_dir = self.drive_root / "pending"
        self.done_dir = self.drive_root / "done"

    # ------------------------------------------------------------------
    # Public API (matches SandboxProtocol)
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        self._run_counter += 1
        task_id = f"rc-{uuid.uuid4().hex[:8]}"

        # Stage locally
        staging = self.workdir / f"_colab_{self._run_counter}"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "main.py").write_text(code, encoding="utf-8")

        self._inject_harness(staging)
        self._write_setup_script(staging)

        return self._submit_and_wait(staging, task_id, timeout_sec)

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> SandboxResult:
        self._run_counter += 1
        task_id = f"rc-{uuid.uuid4().hex[:8]}"

        staging = self.workdir / f"_colab_project_{self._run_counter}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        self._inject_harness(staging)

        for src_file in project_dir.iterdir():
            if src_file.is_file():
                dest = staging / src_file.name
                if dest.name == "experiment_harness.py":
                    continue
                dest.write_bytes(src_file.read_bytes())

        if not (staging / entry_point).exists():
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Entry point {entry_point} not found",
                elapsed_sec=0.0, metrics={},
            )

        self._write_setup_script(staging)
        return self._submit_and_wait(staging, task_id, timeout_sec)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def check_drive_available(config: ColabDriveConfig) -> tuple[bool, str]:
        """Check if the Google Drive mount is accessible."""
        if not config.drive_root:
            return False, "colab_drive.drive_root is empty"
        root = Path(config.drive_root).expanduser().resolve()
        try:
            exists = root.exists()
        except OSError:
            exists = False
        if not exists:
            return False, (
                f"Drive root not found: {root}. "
                f"Is Google Drive for Desktop running and syncing?"
            )
        return True, f"Google Drive accessible at {root}"

    @staticmethod
    def write_worker_notebook(output_path: Path) -> None:
        """Write the Colab worker template to a file for the user to upload."""
        output_path.write_text(COLAB_WORKER_TEMPLATE, encoding="utf-8")
        logger.info("Colab worker template written to %s", output_path)

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        harness_src = Path(__file__).parent / "harness_template.py"
        if harness_src.exists():
            dest = target_dir / "experiment_harness.py"
            dest.write_text(
                harness_src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    def _write_setup_script(self, staging: Path) -> None:
        """Write setup.sh if setup_script is configured."""
        if self.config.setup_script:
            setup_path = staging / "setup.sh"
            setup_path.write_text(
                f"#!/bin/bash\nset -e\n{self.config.setup_script}\n",
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # Core: submit task and poll for result
    # ------------------------------------------------------------------

    def _submit_and_wait(
        self, staging: Path, task_id: str, timeout_sec: int,
    ) -> SandboxResult:
        """Submit an experiment task and wait for the Colab worker to complete it.

        Protocol:
          1. Copy experiment files to ``pending/<task_id>/`` in the shared Drive
          2. The Colab worker notebook polls ``pending/``, moves task to
             ``running/``, executes it, writes ``result.json``, moves to ``done/``
          3. This method polls ``done/<task_id>/`` until result appears or timeout

        Google Drive sync latency (typically 5-30s) is handled by the
        configurable poll_interval_sec. If the worker never picks up the task,
        the pending directory is cleaned up on timeout.
        """
        # Ensure directories exist
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)

        # Copy task to pending/
        task_pending = self.pending_dir / task_id
        if task_pending.exists():
            shutil.rmtree(task_pending)
        shutil.copytree(staging, task_pending)
        logger.info("Task %s submitted to %s", task_id, self.pending_dir)

        # Poll done/ for result
        task_done = self.done_dir / task_id
        effective_timeout = max(timeout_sec, self.config.timeout_sec)
        poll_interval = self.config.poll_interval_sec

        start = time.monotonic()
        while time.monotonic() - start < effective_timeout:
            if task_done.exists():
                return self._collect_result(task_done, time.monotonic() - start)
            time.sleep(poll_interval)

        # Timeout — clean up pending task if still there
        elapsed = time.monotonic() - start
        if task_pending.exists():
            shutil.rmtree(task_pending)
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=(
                f"Colab worker did not complete task {task_id} "
                f"within {effective_timeout}s. "
                f"Is the Colab worker notebook running?"
            ),
            elapsed_sec=elapsed,
            metrics={},
            timed_out=True,
        )

    def _collect_result(
        self, task_done: Path, elapsed: float,
    ) -> SandboxResult:
        """Read result.json from the done task directory and clean up.

        The result.json schema (written by the Colab worker):
          {"returncode": int, "stdout": str, "stderr": str, "timed_out"?: bool}

        Metrics are parsed from stdout using the same ``metric_name: value``
        format as the local and Docker sandboxes.
        """
        result_file = task_done / "result.json"
        if not result_file.exists():
            return SandboxResult(
                returncode=-1, stdout="",
                stderr="Colab worker did not write result.json",
                elapsed_sec=elapsed, metrics={},
            )

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Failed to read result.json: {exc}",
                elapsed_sec=elapsed, metrics={},
            )

        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        returncode = data.get("returncode", -1)
        timed_out = data.get("timed_out", False)

        metrics = parse_metrics(stdout)

        # Clean up
        shutil.rmtree(task_done, ignore_errors=True)

        return SandboxResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed,
            metrics=metrics,
            timed_out=timed_out,
        )
