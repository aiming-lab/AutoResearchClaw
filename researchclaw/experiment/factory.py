"""Factory for creating sandbox backends based on experiment config."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from researchclaw.config import ExperimentConfig
from researchclaw.experiment.sandbox import ExperimentSandbox, SandboxProtocol

if TYPE_CHECKING:
    from researchclaw.experiment.agentic_sandbox import AgenticSandbox

logger = logging.getLogger(__name__)


def _write_sandbox_metadata(
    metadata_dir: Path | None,
    *,
    requested_backend: str,
    actual_backend: str,
    fallback_used: bool,
    env_policy: dict[str, object] | None = None,
) -> None:
    """Persist machine-readable sandbox/env metadata for release_check.

    release_check fails closed when this metadata is missing, so every
    sandbox construction on the pipeline path must record its decision —
    including (especially) fallbacks.
    """
    if metadata_dir is None:
        return
    try:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        (metadata_dir / "sandbox_metadata.json").write_text(
            json.dumps(
                {
                    "requested_backend": requested_backend,
                    "actual_backend": actual_backend,
                    "fallback_used": fallback_used,
                    "generated": now,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if env_policy is not None:
            (metadata_dir / "environment_policy.json").write_text(
                json.dumps({**env_policy, "generated": now}, indent=2),
                encoding="utf-8",
            )
    except OSError:
        logger.warning("Could not persist sandbox metadata to %s", metadata_dir)


def _subprocess_env_policy(config: ExperimentConfig) -> dict[str, object]:
    sb = config.sandbox
    policy = getattr(sb, "env_policy", "allowlist") or "allowlist"
    return {
        "policy": policy,
        "allowlist": list(getattr(sb, "env_allowlist", ()) or ()),
        "inherit_all": policy == "inherit_all",
    }


def create_sandbox(
    config: ExperimentConfig,
    workdir: Path,
    *,
    metadata_dir: Path | None = None,
) -> SandboxProtocol:
    """Return the appropriate sandbox backend for *config.mode*.

    - ``"sandbox"`` → :class:`ExperimentSandbox` (subprocess)
    - ``"docker"``  → :class:`DockerSandbox`  (Docker container)

    When *metadata_dir* is given (pipeline stage dirs), the requested/actual
    backend and environment policy are persisted for release_check.
    """
    if config.mode == "docker":
        from researchclaw.experiment.docker_sandbox import DockerSandbox

        docker_cfg = config.docker

        if not DockerSandbox.check_docker_available():
            # Fail closed by default: silently downgrading Docker isolation
            # to a host subprocess is a release blocker, not a convenience.
            if not getattr(config.sandbox, "allow_docker_fallback", False):
                raise RuntimeError(
                    "Docker daemon is not reachable and "
                    "experiment.sandbox.allow_docker_fallback is false. "
                    "Start Docker, or explicitly opt in to the unsafe "
                    "subprocess fallback (release_check will flag it)."
                )
            logger.warning(
                "Docker daemon is not reachable — falling back to subprocess "
                "sandbox (allow_docker_fallback=true). This run is NOT "
                "release-eligible: fallback_used=true is recorded."
            )
            _write_sandbox_metadata(
                metadata_dir,
                requested_backend="docker",
                actual_backend="subprocess",
                fallback_used=True,
                env_policy=_subprocess_env_policy(config),
            )
            return ExperimentSandbox(config.sandbox, workdir)

        if not DockerSandbox.ensure_image(docker_cfg.image):
            raise RuntimeError(
                f"Docker image '{docker_cfg.image}' not found locally. "
                f"Build it: docker build -t {docker_cfg.image} researchclaw/docker/"
            )

        if docker_cfg.gpu_enabled:
            logger.info("Docker sandbox: GPU passthrough enabled")

        _write_sandbox_metadata(
            metadata_dir,
            requested_backend="docker",
            actual_backend="docker",
            fallback_used=False,
            env_policy={"policy": "container_isolated", "image": docker_cfg.image},
        )
        return DockerSandbox(docker_cfg, workdir)

    if config.mode == "ssh_remote":
        from researchclaw.experiment.ssh_sandbox import SshRemoteSandbox

        ssh_cfg = config.ssh_remote
        if not ssh_cfg.host:
            raise RuntimeError(
                "ssh_remote mode requires experiment.ssh_remote.host in config."
            )

        ok, msg = SshRemoteSandbox.check_ssh_available(ssh_cfg)
        if not ok:
            raise RuntimeError(f"SSH connectivity check failed: {msg}")

        logger.info("SSH remote sandbox: %s", msg)
        _write_sandbox_metadata(
            metadata_dir,
            requested_backend="ssh_remote",
            actual_backend="ssh_remote",
            fallback_used=False,
            env_policy={"policy": "remote_host", "host": ssh_cfg.host},
        )
        return SshRemoteSandbox(ssh_cfg, workdir)

    if config.mode == "colab_drive":
        from researchclaw.experiment.colab_sandbox import ColabDriveSandbox

        colab_cfg = config.colab_drive
        ok, msg = ColabDriveSandbox.check_drive_available(colab_cfg)
        if not ok:
            raise RuntimeError(f"Colab Drive check failed: {msg}")

        logger.info("Colab Drive sandbox: %s", msg)

        # Write worker template for user convenience
        worker_path = Path(colab_cfg.drive_root).expanduser() / "colab_worker.py"
        if not worker_path.exists():
            ColabDriveSandbox.write_worker_notebook(worker_path)
            logger.info(
                "Colab worker template written to %s — "
                "upload this to Colab and run it.",
                worker_path,
            )

        _write_sandbox_metadata(
            metadata_dir,
            requested_backend="colab_drive",
            actual_backend="colab_drive",
            fallback_used=False,
            env_policy={"policy": "remote_colab"},
        )
        return ColabDriveSandbox(colab_cfg, workdir)

    if config.mode in ("collider_agent", "biology_agent", "stat_agent"):
        _write_sandbox_metadata(
            metadata_dir,
            requested_backend=config.mode,
            actual_backend=config.mode,
            fallback_used=False,
            env_policy={"policy": "agent_container"},
        )

    if config.mode == "collider_agent":
        from researchclaw.experiment.collider_agent_sandbox import ColliderAgentSandbox

        ca_cfg = config.collider_agent
        return ColliderAgentSandbox(ca_cfg, workdir)

    if config.mode == "biology_agent":
        from researchclaw.experiment.biology_agent_sandbox import BiologyAgentSandbox

        ba_cfg = config.biology_agent
        return BiologyAgentSandbox(ba_cfg, workdir)

    if config.mode == "stat_agent":
        from researchclaw.experiment.stat_agent_sandbox import StatAgentSandbox

        sa_cfg = config.stat_agent
        return StatAgentSandbox(sa_cfg, workdir)

    if config.mode != "sandbox":
        raise RuntimeError(
            f"Unsupported experiment mode for create_sandbox(): {config.mode}"
        )

    _write_sandbox_metadata(
        metadata_dir,
        requested_backend="sandbox",
        actual_backend="subprocess",
        fallback_used=False,
        env_policy=_subprocess_env_policy(config),
    )
    return ExperimentSandbox(config.sandbox, workdir)


def create_agentic_sandbox(
    config: ExperimentConfig,
    workdir: Path,
    skills_dir: Path | None = None,
) -> "AgenticSandbox":  # noqa: F821
    """Return an :class:`AgenticSandbox` for agentic experiment mode.

    Validates that Docker is available before returning.
    """
    from researchclaw.experiment.agentic_sandbox import AgenticSandbox

    if not AgenticSandbox.check_docker_available():
        raise RuntimeError(
            "Docker daemon is not reachable. "
            "Agentic mode requires Docker. Start Docker first."
        )

    agentic_cfg = config.agentic
    if agentic_cfg.gpu_enabled:
        logger.info("Agentic sandbox: GPU passthrough enabled")

    return AgenticSandbox(agentic_cfg, workdir, skills_dir=skills_dir)
