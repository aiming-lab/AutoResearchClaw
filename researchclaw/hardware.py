"""Hardware detection for GPU-aware experiment execution."""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

# VRAM threshold (MB) — GPUs with less than this are "limited"
_HIGH_VRAM_THRESHOLD_MB = 8192

# Words that indicate a log/status line rather than a metric
LOG_WORDS: frozenset[str] = frozenset({
    "running", "loading", "saving", "processing", "starting",
    "finished", "completed", "initializing", "downloading",
    "training", "evaluating", "epoch", "step", "iteration",
    "experiment", "warning", "error", "info", "debug",
    "experiments", "using", "setting", "creating", "building",
    "computing", "reading", "writing", "opening", "closing",
})

# Maximum word count for a plausible metric name
_MAX_METRIC_NAME_WORDS = 6


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware capabilities of the local machine."""

    has_gpu: bool
    gpu_type: str  # "cuda" | "mps" | "cpu"
    gpu_name: str  # e.g. "NVIDIA RTX 4090" / "Apple M3 Pro" / "CPU only"
    vram_mb: int | None  # NVIDIA only; None for MPS/CPU
    tier: str  # "high" | "limited" | "cpu_only"
    warning: str  # User-facing warning message (empty if tier=high)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    """Detect local GPU hardware and return a HardwareProfile.

    Detection order:
    1. NVIDIA GPU via ``nvidia-smi``
    2. macOS Apple Silicon (MPS) via platform check
    3. Fallback to CPU-only
    """
    # --- Try NVIDIA ---
    profile = _detect_nvidia()
    if profile is not None:
        return profile

    # --- Try macOS MPS (Apple Silicon) ---
    profile = _detect_mps()
    if profile is not None:
        return profile

    # --- CPU only ---
    return HardwareProfile(
        has_gpu=False,
        gpu_type="cpu",
        gpu_name="CPU only",
        vram_mb=None,
        tier="cpu_only",
        warning=(
            "No GPU detected. Only CPU-based experiments (NumPy, sklearn) are supported. "
            "For deep learning research ideas, please use a machine with a GPU or a remote GPU server."
        ),
    )


def _detect_nvidia() -> HardwareProfile | None:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None

        # Parse first GPU line: "NVIDIA GeForce RTX 4090, 24564"
        line = result.stdout.strip().splitlines()[0].strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return None

        gpu_name = parts[0]
        try:
            vram_mb = int(float(parts[1]))
        except (ValueError, IndexError):
            vram_mb = 0

        if vram_mb >= _HIGH_VRAM_THRESHOLD_MB:
            tier = "high"
            warning = ""
        else:
            tier = "limited"
            warning = (
                f"Local GPU ({gpu_name}, {vram_mb} MB VRAM) has limited memory. "
                "Complex deep learning experiments may be slow or run out of memory. "
                "Consider using a remote GPU server for best results."
            )

        return HardwareProfile(
            has_gpu=True,
            gpu_type="cuda",
            gpu_name=gpu_name,
            vram_mb=vram_mb,
            tier=tier,
            warning=warning,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _detect_mps() -> HardwareProfile | None:
    """Detect macOS Apple Silicon GPU (MPS)."""
    if platform.system() != "Darwin":
        return None

    if platform.machine() != "arm64":
        return None

    # Get chip name via sysctl
    gpu_name = "Apple Silicon GPU"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return HardwareProfile(
        has_gpu=True,
        gpu_type="mps",
        gpu_name=gpu_name,
        vram_mb=None,  # MPS shares system memory
        tier="limited",
        warning=(
            f"macOS GPU detected ({gpu_name}). PyTorch MPS backend is available "
            "but has limited performance compared to NVIDIA CUDA GPUs. "
            "For large-scale experiments, consider using a remote GPU server."
        ),
    )


def ensure_torch_available(python_path: str, gpu_type: str) -> bool:
    """Check if PyTorch is importable; attempt install if not.

    Returns True if torch is available after this call.
    """
    from pathlib import Path

    python = Path(python_path)
    if not python.is_absolute():
        python = Path.cwd() / python

    # Check if already installed
    try:
        result = subprocess.run(
            [str(python), "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info("PyTorch %s already available at %s", version, python)
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    # Not installed — attempt install
    if gpu_type == "cpu":
        logger.info("No GPU available; skipping PyTorch installation")
        return False

    logger.info("PyTorch not found. Attempting install for %s...", gpu_type)
    pip_cmd = [str(python), "-m", "pip", "install", "--quiet", "torch"]

    try:
        result = subprocess.run(
            pip_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        if result.returncode == 0:
            logger.info("PyTorch installed successfully")
            return True
        logger.warning("PyTorch installation failed: %s", result.stderr[:300])
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("PyTorch installation error: %s", exc)
        return False


def is_metric_name(name: str) -> bool:
    """Return True if *name* looks like a metric name rather than a log line.

    Used to filter stdout lines when parsing ``name: value`` metric output.
    """
    words = name.lower().split()
    if len(words) > _MAX_METRIC_NAME_WORDS:
        return False
    if any(w in LOG_WORDS for w in words):
        return False
    return True
