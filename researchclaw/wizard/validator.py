"""Environment detection and recommendation for the setup wizard."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentReport:
    """Report of detected environment capabilities."""

    has_gpu: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    has_docker: bool = False
    docker_version: str = ""
    has_python: bool = True
    python_version: str = ""
    has_latex: bool = False
    available_memory_gb: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_gpu": self.has_gpu,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "has_docker": self.has_docker,
            "docker_version": self.docker_version,
            "has_python": self.has_python,
            "python_version": self.python_version,
            "has_latex": self.has_latex,
            "available_memory_gb": round(self.available_memory_gb, 1),
            "recommendations": self.recommendations,
        }


def detect_environment() -> EnvironmentReport:
    """Detect local environment and generate recommendations."""
    import sys
    import subprocess

    report = EnvironmentReport()
    report.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Docker
    if shutil.which("docker"):
        report.has_docker = True
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=5
            )
            report.docker_version = result.stdout.strip()
        except Exception:
            pass

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            report.has_gpu = True
            report.gpu_name = torch.cuda.get_device_name(0)
            report.gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

    # LaTeX
    report.has_latex = shutil.which("pdflatex") is not None

    # Memory
    try:
        import psutil
        report.available_memory_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass

    # Recommendations
    if not report.has_docker:
        report.recommendations.append(
            "Install Docker for experiment isolation (recommended)"
        )
    if not report.has_gpu:
        report.recommendations.append(
            "No GPU detected — use 'simulated' mode or remote GPU server"
        )
    if not report.has_latex:
        report.recommendations.append(
            "Install LaTeX (texlive) for PDF paper export"
        )

    if report.has_gpu and report.has_docker:
        report.recommendations.append(
            "Environment ready for full Docker GPU experiments"
        )

    return report
