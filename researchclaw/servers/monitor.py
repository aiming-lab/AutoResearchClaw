"""Server monitor: check health and resource usage of registered servers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from researchclaw.servers.registry import ServerEntry, ServerRegistry

logger = logging.getLogger(__name__)


class ServerMonitor:
    """Monitor health and resource usage of registered servers."""

    def __init__(self, registry: ServerRegistry) -> None:
        self.registry = registry
        self._status_cache: dict[str, dict[str, Any]] = {}

    async def check_status(self, server: ServerEntry) -> dict[str, Any]:
        """Check a single server's status via SSH (nvidia-smi, free, uptime)."""
        try:
            result = await _ssh_command(server.host, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null; echo '---'; free -m | head -2; echo '---'; uptime")
            status = _parse_status_output(result, server)
            status["reachable"] = True
        except Exception as exc:
            logger.warning("Cannot reach server %s: %s", server.name, exc)
            status = {"reachable": False, "error": str(exc)}

        self._status_cache[server.name] = status
        return status

    async def check_all(self) -> dict[str, dict[str, Any]]:
        """Check all servers concurrently."""
        servers = self.registry.list_all()
        tasks = [self.check_status(s) for s in servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, dict[str, Any]] = {}
        for server, result in zip(servers, results):
            if isinstance(result, Exception):
                out[server.name] = {"reachable": False, "error": str(result)}
            else:
                out[server.name] = result
        return out

    def get_cached(self, name: str) -> dict[str, Any] | None:
        """Return cached status for a server."""
        return self._status_cache.get(name)

    def get_gpu_usage(self, server: ServerEntry) -> dict[str, Any]:
        """Return cached GPU usage for a server (sync convenience)."""
        cached = self._status_cache.get(server.name, {})
        return cached.get("gpu", {})


async def _ssh_command(host: str, command: str) -> str:
    """Run a command on a remote host via SSH."""
    proc = await asyncio.create_subprocess_exec(
        "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
        host, command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"SSH command failed (rc={proc.returncode}): {stderr.decode().strip()}")
    return stdout.decode()


def _parse_status_output(raw: str, server: ServerEntry) -> dict[str, Any]:
    """Parse combined nvidia-smi + free + uptime output."""
    sections = raw.split("---")
    status: dict[str, Any] = {"server": server.name, "host": server.host}

    # GPU section
    if len(sections) >= 1:
        gpu_lines = [l.strip() for l in sections[0].strip().splitlines() if l.strip()]
        gpus = []
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "utilization_pct": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                })
        status["gpu"] = {"count": len(gpus), "devices": gpus}

    # Memory section
    if len(sections) >= 2:
        mem_lines = sections[1].strip().splitlines()
        if len(mem_lines) >= 2:
            parts = mem_lines[1].split()
            if len(parts) >= 4:
                status["memory"] = {
                    "total_mb": int(parts[1]),
                    "used_mb": int(parts[2]),
                    "free_mb": int(parts[3]),
                }

    # Uptime section
    if len(sections) >= 3:
        status["uptime"] = sections[2].strip()

    return status
