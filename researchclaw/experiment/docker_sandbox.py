"""Docker-based sandbox for experiment code execution with GPU passthrough.

Uses a single-container, three-phase execution model:
  Phase 0: pip install from requirements.txt (if present)
  Phase 1: Run setup.py for dataset downloads (if present)
  Phase 2: Run the experiment script (main.py)

All phases run in the same container, so pip-installed packages
persist into the experiment phase. Network can be disabled after
setup via iptables (``setup_only`` policy).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from researchclaw.config import DockerSandboxConfig
from researchclaw.experiment.sandbox import (
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)

logger = logging.getLogger(__name__)

_CONTAINER_COUNTER = 0
_counter_lock = threading.Lock()


def _next_container_name() -> str:
    global _CONTAINER_COUNTER  # noqa: PLW0603
    with _counter_lock:
        _CONTAINER_COUNTER += 1
        return f"rc-exp-{_CONTAINER_COUNTER}-{os.getpid()}"


# Packages already in the Docker image — skip during auto-detect.
_BUILTIN_PACKAGES = {
    # PyTorch ecosystem
    "torch", "torchvision", "torchaudio", "torchdiffeq",
    # Scientific / ML
    "numpy", "scipy", "sklearn", "pandas", "matplotlib", "seaborn",
    "tqdm", "gymnasium", "networkx",
    # Extended ML ecosystem
    "timm", "einops", "torchmetrics", "albumentations", "kornia",
    "h5py", "tensorboard",
    # HuggingFace / LLM stack
    "transformers", "datasets", "accelerate", "peft", "trl",
    "bitsandbytes", "sentencepiece", "protobuf", "tokenizers",
    "safetensors", "evaluate",
    # Other pre-installed
    "yaml", "PIL", "mujoco",
    # Python stdlib
    "os", "sys", "math", "random", "json", "csv", "re", "time",
    "collections", "itertools", "functools", "pathlib", "typing",
    "dataclasses", "abc", "copy", "io", "logging", "argparse",
    "datetime", "hashlib", "pickle", "subprocess", "shutil",
    "tempfile", "warnings", "unittest", "contextlib", "operator",
    "string", "textwrap", "struct", "statistics", "glob",
    "urllib", "http", "email", "html", "xml",
}

# Map import names to pip package names.
_IMPORT_TO_PIP = {
    "torchdiffeq": "torchdiffeq",
    "torch_geometric": "torch-geometric",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "gym": "gymnasium",
    "ogb": "ogb",
    "dgl": "dgl",
    "lightning": "lightning",
    "pytorch_lightning": "pytorch-lightning",
    "wandb": "wandb",
    "optuna": "optuna",
}


class DockerSandbox:
    """Execute experiment code inside a Docker container.

    Same public API as :class:`ExperimentSandbox` so the pipeline can use
    either backend transparently.

    The container uses ``entrypoint.sh`` which runs three phases in sequence:
      0. ``pip install -r requirements.txt`` (if file present in /workspace)
      1. ``python3 setup.py`` (if file present in /workspace)
      2. ``python3 <entry_point>``

    Network policy controls when network is available:
      - ``"none"``:       No network at any point (``--network none``)
      - ``"setup_only"``: Network during Phase 0+1, disabled via iptables before Phase 2
      - ``"pip_only"``:   Network during Phase 0 only (legacy compat, same as setup_only)
      - ``"full"``:       Network available throughout all phases
    """

    def __init__(self, config: DockerSandboxConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        """Run a single Python code string inside a container."""
        self._run_counter += 1
        staging = self.workdir / f"_docker_run_{self._run_counter}"
        staging.mkdir(parents=True, exist_ok=True)

        script_path = staging / "main.py"
        script_path.write_text(code, encoding="utf-8")

        # Inject experiment harness
        self._inject_harness(staging)

        return self._execute(staging, entry_point="main.py", timeout_sec=timeout_sec)

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> SandboxResult:
        """Run a multi-file experiment project inside a container."""
        self._run_counter += 1
        staging = self.workdir / f"_docker_project_{self._run_counter}"
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

        # Inject harness first (immutable)
        self._inject_harness(staging)

        # Copy project files and subdirectories (skip harness overwrite)
        import shutil as _shutil
        for src_item in project_dir.iterdir():
            dest = staging / src_item.name
            if src_item.name == "experiment_harness.py":
                logger.warning(
                    "Project contains experiment_harness.py — skipping (immutable)"
                )
                continue
            if src_item.is_file():
                dest.write_bytes(src_item.read_bytes())
            elif src_item.is_dir() and not src_item.name.startswith((".", "__")):
                _shutil.copytree(src_item, dest, dirs_exist_ok=True)

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
    def check_docker_available() -> bool:
        """Return True if the Docker daemon is reachable."""
        try:
            cp = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return cp.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def check_nvidia_runtime() -> bool:
        """Return True if the NVIDIA Container Toolkit is available."""
        try:
            cp = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all",
                 "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
                 "nvidia-smi"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            return cp.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def ensure_image(image: str) -> bool:
        """Return True if *image* exists locally (does NOT pull)."""
        try:
            cp = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return cp.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        harness_src = Path(__file__).parent / "harness_template.py"
        if harness_src.exists():
            dest = target_dir / "experiment_harness.py"
            dest.write_text(harness_src.read_text(encoding="utf-8"), encoding="utf-8")
            logger.debug("Injected experiment harness into %s", target_dir)
        else:
            logger.warning("Harness template not found at %s", harness_src)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute(
        self, staging_dir: Path, *, entry_point: str, timeout_sec: int
    ) -> SandboxResult:
        """Core execution: single container, three-phase via entrypoint.sh."""
        cfg = self.config
        container_name = _next_container_name()

        # Auto-generate requirements.txt if packages need installing
        if cfg.network_policy in ("pip_only", "setup_only", "full"):
            self._write_requirements_txt(staging_dir)

        # Build the docker run command
        cmd = self._build_run_command(
            staging_dir,
            entry_point=entry_point,
            container_name=container_name,
        )

        start = time.monotonic()
        timed_out = False
        try:
            logger.debug("Docker run command: %s", cmd)
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
            elapsed = time.monotonic() - start
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = -1
            # Force-kill the container on timeout
            self._kill_container(container_name)
            elapsed = time.monotonic() - start
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"Docker execution error: {exc}",
                elapsed_sec=elapsed,
                metrics={},
            )
        finally:
            # Always clean up the container regardless of how we exit.
            # docker rm -f is idempotent: safe even if container was
            # already removed by --rm, already dead, or never created.
            if not cfg.keep_containers:
                self._remove_container(container_name)

        # Parse metrics from stdout
        metrics = parse_metrics(stdout)

        # Try to read structured results.json from staging dir (volume-mounted)
        results_json_path = staging_dir / "results.json"
        if results_json_path.exists():
            try:
                structured = json.loads(
                    results_json_path.read_text(encoding="utf-8")
                )
                if isinstance(structured, dict):
                    for k, v in structured.items():
                        if k not in metrics:
                            try:
                                metrics[k] = float(v)
                            except (TypeError, ValueError):
                                pass
            except (json.JSONDecodeError, OSError):
                pass

        return SandboxResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed,
            metrics=metrics,
            timed_out=timed_out,
        )

    def _build_run_command(
        self,
        staging_dir: Path,
        *,
        entry_point: str,
        container_name: str,
    ) -> list[str]:
        """Build the ``docker run`` command list.

        The container uses ``entrypoint.sh`` which handles:
          Phase 0: pip install requirements.txt
          Phase 1: python3 setup.py
          Phase 2: python3 <entry_point>

        Network policy determines --network and RC_SETUP_ONLY_NETWORK env.
        """
        cfg = self.config
        cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "-v", f"{staging_dir}:/workspace",
            "-w", "/workspace",
            f"--memory={cfg.memory_limit_mb}m",
            f"--shm-size={cfg.shm_size_mb}m",
        ]

        # --- Network policy ---
        if cfg.network_policy == "none":
            # Fully isolated — no network at any point
            cmd.extend(["--network", "none"])
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        elif cfg.network_policy in ("setup_only", "pip_only"):
            # Network during Phase 0+1, disabled via iptables before Phase 2.
            # Run as host user so experiment can write results.json to volume.
            # iptables requires NET_ADMIN but will gracefully degrade if
            # the user lacks root — network remains available but the code
            # has already been validated by the pipeline security check.
            cmd.extend(["-e", "RC_SETUP_ONLY_NETWORK=1"])
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
            cmd.extend(["--cap-add=NET_ADMIN"])
        elif cfg.network_policy == "full":
            # Full network throughout — for development/debugging
            cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

        # Mount pre-cached datasets
        # Priority: /opt/datasets (system) > ~/.cache/datasets (user)
        datasets_host = Path("/opt/datasets")
        user_datasets = Path.home() / ".cache" / "datasets"
        if datasets_host.is_dir():
            cmd.extend(["-v", f"{datasets_host}:/workspace/data:ro"])
        elif user_datasets.is_dir():
            cmd.extend(["-v", f"{user_datasets}:/workspace/data:rw"])
        else:
            # Create user-level cache so containers can download datasets
            user_datasets.mkdir(parents=True, exist_ok=True)
            cmd.extend(["-v", f"{user_datasets}:/workspace/data:rw"])

        # Mount HuggingFace model cache (read-only for model weights).
        # BUG-103 fix: Don't set HF_HOME to the read-only mount — the
        # transformers library writes token/telemetry files under HF_HOME.
        # Instead, use HF_HUB_CACHE for read-only model access and let
        # HF_HOME default to a writable location inside the container.
        hf_mounted = False
        _hf_hub_cache = "/home/researcher/.cache/huggingface/hub"
        hf_home_env = os.environ.get("HF_HOME", "").strip()
        if hf_home_env:
            xdg_hf = Path(hf_home_env).resolve()
            if xdg_hf.is_dir():
                cmd.extend(["-v", f"{xdg_hf}:{_hf_hub_cache}:ro"])
                cmd.extend(["-e", f"HF_HUB_CACHE={_hf_hub_cache}"])
                hf_mounted = True
        if not hf_mounted:
            hf_cache_host = Path.home() / ".cache" / "huggingface"
            if hf_cache_host.is_dir():
                cmd.extend(["-v", f"{hf_cache_host}:{_hf_hub_cache}:ro"])
                cmd.extend(["-e", f"HF_HUB_CACHE={_hf_hub_cache}"])

        # BUG-107 fix: Set TORCH_HOME to writable location so torchvision
        # can download pretrained model weights (e.g., Inception-v3 for FID).
        cmd.extend(["-e", "TORCH_HOME=/workspace/.cache/torch"])

        # Pass HF token if available (for gated model downloads)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            cmd.extend(["-e", f"HF_TOKEN={hf_token}"])

        # GPU passthrough
        if cfg.gpu_enabled:
            if cfg.gpu_device_ids:
                device_spec = ",".join(str(d) for d in cfg.gpu_device_ids)
                cmd.extend(["--gpus", f"device={device_spec}"])
            else:
                cmd.extend(["--gpus", "all"])

        # Image + entry point (passed as CMD arg to entrypoint.sh)
        cmd.append(cfg.image)
        cmd.append(entry_point)

        return cmd

    def _write_requirements_txt(self, staging_dir: Path) -> None:
        """Generate requirements.txt in staging dir from auto-detected imports
        and explicit pip_pre_install, unless one already exists (LLM-generated).
        """
        req_path = staging_dir / "requirements.txt"

        # If the LLM already generated a requirements.txt, respect it but
        # append any pip_pre_install packages not already listed.
        existing_reqs: set[str] = set()
        if req_path.exists():
            for line in req_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name (before any version specifier)
                    pkg = re.split(r"[><=!~\[]", line)[0].strip().lower()
                    existing_reqs.add(pkg)

        # Collect additional packages to install
        packages: list[str] = []

        # From config pip_pre_install
        for pkg in self.config.pip_pre_install:
            pkg_base = re.split(r"[><=!~\[]", pkg)[0].strip().lower()
            if pkg_base not in existing_reqs:
                packages.append(pkg)
                existing_reqs.add(pkg_base)

        # Auto-detect from imports
        if self.config.auto_install_deps:
            detected = self._detect_pip_packages(staging_dir)
            for pkg in detected:
                pkg_base = pkg.lower()
                if pkg_base not in existing_reqs:
                    packages.append(pkg)
                    existing_reqs.add(pkg_base)

        if not packages and not req_path.exists():
            return  # Nothing to install

        if packages:
            mode = "a" if req_path.exists() else "w"
            with open(req_path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n# Auto-detected by ResearchClaw\n")
                for pkg in packages:
                    f.write(pkg + "\n")
            logger.info("requirements.txt updated: %s", packages)

    @staticmethod
    def _detect_pip_packages(staging_dir: Path) -> list[str]:
        """Scan Python files for import statements and return pip package names."""
        import_re = re.compile(
            r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE
        )
        # Exclude local project modules (any .py file in staging_dir)
        local_modules = {
            pyf.stem for pyf in staging_dir.glob("*.py")
        }
        detected: list[str] = []
        for pyf in staging_dir.glob("*.py"):
            if pyf.name == "setup.py":
                continue  # Don't scan setup.py for experiment deps
            text = pyf.read_text(encoding="utf-8", errors="replace")
            for m in import_re.finditer(text):
                top_module = m.group(1).split(".")[0]
                if top_module in _BUILTIN_PACKAGES:
                    continue
                if top_module in local_modules:
                    continue  # Skip local project modules
                pip_name = _IMPORT_TO_PIP.get(top_module, top_module)
                if pip_name not in detected:
                    detected.append(pip_name)

        return detected

    @staticmethod
    def _kill_container(name: str) -> None:
        try:
            subprocess.run(
                ["docker", "kill", name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _remove_container(name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
