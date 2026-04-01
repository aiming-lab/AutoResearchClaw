"""SSH remote executor: upload code, run experiments, download results."""

from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any

from researchclaw.servers.registry import ServerEntry

logger = logging.getLogger(__name__)


class SSHExecutor:
    """Execute experiments on remote servers via SSH/rsync."""

    def __init__(self, server: ServerEntry) -> None:
        self.server = server
        self.host = server.host

    async def upload_code(self, local_dir: Path, remote_dir: str) -> None:
        """Upload experiment code via rsync."""
        local = str(local_dir.resolve()) + "/"
        remote = f"{self.host}:{remote_dir}/"
        logger.info("Uploading %s -> %s", local, remote)
        proc = await asyncio.create_subprocess_exec(
            "rsync", "-az", "--delete",
            "-e", "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no",
            local, remote,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"rsync upload failed: {stderr.decode().strip()}")

    async def run_experiment(
        self,
        remote_dir: str,
        command: str,
        timeout: int = 3600,
    ) -> dict[str, Any]:
        """Run an experiment command on the remote server."""
        full_cmd = f"cd {shlex.quote(remote_dir)} && {command}"
        logger.info("Running on %s: %s", self.host, full_cmd)
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            self.host, full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"success": False, "error": f"Timeout after {timeout}s", "returncode": -1}

        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode,
        }

    async def download_results(self, remote_dir: str, local_dir: Path) -> None:
        """Download experiment results via rsync."""
        local_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{self.host}:{remote_dir}/"
        local = str(local_dir.resolve()) + "/"
        logger.info("Downloading %s -> %s", remote, local)
        proc = await asyncio.create_subprocess_exec(
            "rsync", "-az",
            "-e", "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no",
            remote, local,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"rsync download failed: {stderr.decode().strip()}")

    async def cleanup(self, remote_dir: str) -> None:
        """Remove remote experiment directory."""
        logger.info("Cleaning up %s:%s", self.host, remote_dir)
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            self.host, f"rm -rf {shlex.quote(remote_dir)}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
