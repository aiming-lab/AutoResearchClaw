"""Hardware detection for GPU-aware experiment execution."""

from __future__ import annotations

import logging
import platform
import shlex
import subprocess
from collections.abc import Mapping
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
    """Detected hardware capabilities of an execution environment."""

    has_gpu: bool
    gpu_type: str  # "cuda" | "mps" | "cpu" | "unknown"
    gpu_name: str  # e.g. "NVIDIA RTX 4090" / "Apple M3 Pro" / "CPU only"
    vram_mb: int | None  # NVIDIA only; None for MPS/CPU
    tier: str  # "high" | "limited" | "cpu_only" | "unknown"
    warning: str  # User-facing warning message (empty if tier=high)
    execution_mode: str = "local"
    detection_status: str = "detected"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def format_hardware_prompt(
    profile: Mapping[str, object] | None, *, remote: bool = False
) -> str:
    """Describe a saved execution profile without probing the coordinator.

    A missing or incompatible profile stays unknown. In particular, a local
    MPS profile is never evidence about an SSH execution host.
    """
    location = "remote SSH execution host" if remote else "execution environment"
    source = profile.get("execution_mode") if profile else None
    incompatible = bool(profile) and (
        (remote and (
            profile.get("gpu_type") == "mps"
            or source != "ssh_remote"
            or profile.get("detection_status") not in {"detected", "failed"}
        ))
        or (not remote and source == "ssh_remote")
    )
    if not profile or incompatible:
        return (
            f"- Hardware: unknown for the {location}; no usable Stage 1 profile.\n"
            "- GPU availability, GPU count, and VRAM: unknown.\n"
            "- Do not assume CUDA, MPS, or a particular GPU model or memory size.\n"
            "- Plan a small CPU-compatible experiment until the execution hardware is verified."
        )

    lines = [f"- Hardware profile: Stage 1, {location}"]
    gpu_type = str(profile.get("gpu_type") or "unknown")
    gpu_name = str(profile.get("gpu_name") or "unknown")
    has_gpu = profile.get("has_gpu")
    if gpu_type == "unknown" or profile.get("detection_status") == "failed":
        lines.extend([
            "- GPU availability, GPU count, and VRAM: unknown (detection incomplete).",
            "- Do not infer CPU-only hardware from a failed hardware check.",
            "- Plan conservatively until the execution hardware is verified.",
        ])
    elif has_gpu is True and gpu_type in {"cuda", "mps"}:
        lines.append(f"- GPU: {gpu_name} ({gpu_type})")
        vram = profile.get("vram_mb")
        if isinstance(vram, (int, float)) and not isinstance(vram, bool) and vram > 0:
            lines.append(f"- VRAM: {vram:g} MB")
        elif gpu_type == "mps":
            lines.append("- VRAM: unknown; MPS uses shared system memory.")
        else:
            lines.append("- VRAM: unknown; do not assume a large-memory GPU.")
        lines.append("- Use one GPU for this experiment; do not require distributed training.")
        lines.append(f"- PyTorch device, if PyTorch is available: torch.device('{gpu_type}')")
        if profile.get("tier") == "limited":
            lines.append("- Keep models, datasets, batch sizes, and training duration lightweight.")
        if gpu_type == "cuda":
            lines.extend([
                "- Precision: use FP32 by default; enable FP16 with loss scaling only when appropriate.",
                "- Do not assume native BF16 or FlashAttention-2 support. Check GPU capability and installed kernels first.",
            ])
            if "v100" in gpu_name.lower():
                lines.append(
                    "- V100 compatibility: do not use bf16=True or torch.bfloat16 by default; "
                    "do not request attn_implementation='flash_attention_2'. Use FP32/FP16 "
                    "and a compatible eager or SDPA attention implementation."
                )
    elif has_gpu is False and gpu_type == "cpu":
        lines.extend([
            f"- GPU: none reported ({gpu_name}).",
            "- Use CPU execution; do not require CUDA or MPS.",
        ])
    else:
        lines.append("- GPU availability and VRAM: unknown; design conservatively.")
    warning = profile.get("warning")
    if warning:
        lines.append(f"- Hardware advisory: {warning}")
    return "\n".join(lines)


def detect_hardware(ssh_config: object | None = None) -> HardwareProfile:
    """Detect GPU hardware and return a HardwareProfile.

    When *ssh_config* is provided (an ``SshRemoteConfig`` with ``host`` set),
    hardware detection runs on the remote machine via SSH instead of locally.

    Detection order:
    1. NVIDIA GPU via ``nvidia-smi`` (remote or local)
    2. macOS Apple Silicon (MPS) via platform check (local only)
    3. Fallback to CPU-only
    """
    # --- Remote detection via SSH ---
    if ssh_config is not None:
        host = getattr(ssh_config, "host", "")
        if host:
            profile = _detect_nvidia_remote(ssh_config)
            if profile is not None:
                return profile
        return HardwareProfile(
            has_gpu=False,
            gpu_type="unknown",
            gpu_name="Remote hardware unknown",
            vram_mb=None,
            tier="unknown",
            warning=(
                "Remote GPU detection failed; check the SSH host and NVIDIA runtime. "
                "This is not evidence that the remote host is CPU-only."
            ),
            execution_mode="ssh_remote",
            detection_status="failed",
        )

    # --- Try local NVIDIA ---
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


def _detect_nvidia_remote(ssh_config: object) -> HardwareProfile | None:
    """Detect NVIDIA GPU on a remote host via SSH."""
    host = getattr(ssh_config, "host", "")
    user = getattr(ssh_config, "user", "")
    port = getattr(ssh_config, "port", 22)
    key_path = getattr(ssh_config, "key_path", "")

    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    if port and port != 22:
        ssh_cmd.extend(["-p", str(port)])
    target = f"{user}@{host}" if user else host
    ssh_cmd.append(target)
    query_cmd = ["nvidia-smi"]
    gpu_ids = getattr(ssh_config, "gpu_ids", ())
    if gpu_ids:
        query_cmd.extend(["-i", ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)])
    query_cmd.extend(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    ssh_cmd.append(shlex.join(query_cmd))

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return None

        output_lines = result.stdout.strip().splitlines()
        if not output_lines:
            return None
        line = output_lines[0].strip()
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return None

        gpu_name = parts[0]
        try:
            detected_vram = int(float(parts[1]))
            vram_mb = detected_vram if detected_vram > 0 else None
        except (ValueError, IndexError):
            vram_mb = None

        tier = "high" if vram_mb is not None and vram_mb >= _HIGH_VRAM_THRESHOLD_MB else "limited"
        warning = "" if tier == "high" else (
            f"Remote GPU ({gpu_name}) memory is unknown."
            if vram_mb is None
            else f"Remote GPU ({gpu_name}, {vram_mb} MB VRAM) has limited memory."
        )
        return HardwareProfile(
            has_gpu=True,
            gpu_type="cuda",
            gpu_name=f"{gpu_name} (remote: {host})",
            vram_mb=vram_mb,
            tier=tier,
            warning=warning,
            execution_mode="ssh_remote",
            detection_status="detected",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Remote hardware detection failed for %s: %s", host, exc)
        return None


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
