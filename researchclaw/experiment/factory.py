"""Factory for creating sandbox backends based on experiment config."""

from __future__ import annotations

import logging
from pathlib import Path

from researchclaw.config import ExperimentConfig
from researchclaw.experiment.sandbox import ExperimentSandbox, SandboxProtocol

logger = logging.getLogger(__name__)


def create_sandbox(config: ExperimentConfig, workdir: Path) -> SandboxProtocol:
    """Return the appropriate sandbox backend for *config.mode*.

    - ``"sandbox"`` → :class:`ExperimentSandbox` (subprocess)
    - ``"docker"``  → :class:`DockerSandbox`  (Docker container)
    """
    if config.mode == "docker":
        from researchclaw.experiment.docker_sandbox import DockerSandbox

        docker_cfg = config.docker

        if not DockerSandbox.check_docker_available():
            raise RuntimeError(
                "Docker daemon is not reachable. "
                "Start Docker or switch to mode: sandbox."
            )

        if not DockerSandbox.ensure_image(docker_cfg.image):
            raise RuntimeError(
                f"Docker image '{docker_cfg.image}' not found locally. "
                f"Build it: docker build -t {docker_cfg.image} researchclaw/docker/"
            )

        if docker_cfg.gpu_enabled:
            logger.info("Docker sandbox: GPU passthrough enabled")

        return DockerSandbox(docker_cfg, workdir)

    # Default: subprocess sandbox
    return ExperimentSandbox(config.sandbox, workdir)
