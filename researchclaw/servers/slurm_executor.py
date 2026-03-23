"""Slurm HPC executor: submit, monitor, and cancel batch jobs."""

from __future__ import annotations

import asyncio
import logging
import textwrap
from typing import Any

from researchclaw.servers.registry import ServerEntry

logger = logging.getLogger(__name__)


class SlurmExecutor:
    """Submit and manage Slurm batch jobs via SSH."""

    def __init__(self, server: ServerEntry) -> None:
        if server.server_type != "slurm":
            raise ValueError(f"Server {server.name} is not a slurm server")
        self.server = server
        self.host = server.host

    def _generate_sbatch_script(
        self,
        command: str,
        job_name: str = "researchclaw",
        resources: dict[str, Any] | None = None,
    ) -> str:
        """Generate an sbatch submission script."""
        res = resources or {}
        gpus = res.get("gpus", 1)
        mem = res.get("mem_gb", 16)
        time_limit = res.get("time", "01:00:00")
        partition = res.get("partition", "")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --mem={mem}G",
            f"#SBATCH --time={time_limit}",
            "#SBATCH --output=slurm-%j.out",
            "#SBATCH --error=slurm-%j.err",
        ]
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        lines.append("")
        lines.append(command)
        return "\n".join(lines)

    async def submit_job(
        self,
        command: str,
        remote_dir: str,
        job_name: str = "researchclaw",
        resources: dict[str, Any] | None = None,
    ) -> str:
        """Submit a Slurm job and return the job ID."""
        script = self._generate_sbatch_script(command, job_name, resources)
        # Write script and submit via SSH
        import shlex as _shlex
        ssh_cmd = (
            f"cd {_shlex.quote(remote_dir)} && "
            f"cat <<'EOFSCRIPT' > _job.sh\n{script}\nEOFSCRIPT\n"
            f"&& sbatch _job.sh"
        )
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            self.host, ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {stderr.decode().strip()}")

        # Parse "Submitted batch job 12345"
        output = stdout.decode().strip()
        parts = output.split()
        if len(parts) >= 4 and parts[-1].isdigit():
            job_id = parts[-1]
            logger.info("Submitted Slurm job %s on %s", job_id, self.server.name)
            return job_id
        raise RuntimeError(f"Could not parse sbatch output: {output}")

    async def check_job(self, job_id: str) -> dict[str, Any]:
        """Check job status via squeue/sacct."""
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            self.host,
            f"squeue -j {job_id} -h -o '%T' 2>/dev/null || sacct -j {job_id} -n -o State -P 2>/dev/null",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        state = stdout.decode().strip().split("\n")[0].strip() if stdout else "UNKNOWN"
        return {"job_id": job_id, "state": state}

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            self.host, f"scancel {job_id}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        logger.info("Cancelled Slurm job %s on %s", job_id, self.server.name)
