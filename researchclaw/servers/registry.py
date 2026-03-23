"""Server registry: manage available compute servers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ServerEntry:
    """A compute server that can run experiments."""

    def __init__(
        self,
        name: str,
        host: str,
        server_type: str = "ssh",
        gpu: str = "",
        vram_gb: int = 0,
        priority: int = 1,
        scheduler: str = "",
        cloud_provider: str = "",
        cloud_instance_type: str = "",
        cost_per_hour: float = 0.0,
    ) -> None:
        self.name = name
        self.host = host
        self.server_type = server_type  # ssh | slurm | cloud
        self.gpu = gpu
        self.vram_gb = vram_gb
        self.priority = priority
        self.scheduler = scheduler  # slurm | pbs | lsf
        self.cloud_provider = cloud_provider  # aws | gcp | azure
        self.cloud_instance_type = cloud_instance_type
        self.cost_per_hour = cost_per_hour

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host,
            "server_type": self.server_type,
            "gpu": self.gpu,
            "vram_gb": self.vram_gb,
            "priority": self.priority,
            "scheduler": self.scheduler,
            "cloud_provider": self.cloud_provider,
            "cloud_instance_type": self.cloud_instance_type,
            "cost_per_hour": self.cost_per_hour,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServerEntry:
        return cls(
            name=data["name"],
            host=data.get("host", ""),
            server_type=data.get("server_type", "ssh"),
            gpu=data.get("gpu", ""),
            vram_gb=int(data.get("vram_gb", 0)),
            priority=int(data.get("priority", 1)),
            scheduler=data.get("scheduler", ""),
            cloud_provider=data.get("cloud_provider", ""),
            cloud_instance_type=data.get("cloud_instance_type", ""),
            cost_per_hour=float(data.get("cost_per_hour", 0.0)),
        )


class ServerRegistry:
    """Registry of available compute servers."""

    def __init__(self, servers: list[ServerEntry] | None = None) -> None:
        self._servers: dict[str, ServerEntry] = {}
        for s in (servers or []):
            self._servers[s.name] = s

    def add(self, server: ServerEntry) -> None:
        """Register a new server."""
        self._servers[server.name] = server
        logger.info("Registered server: %s (%s)", server.name, server.host)

    def remove(self, name: str) -> None:
        """Remove a server from the registry."""
        if name not in self._servers:
            raise KeyError(f"Unknown server: {name}")
        del self._servers[name]

    def get(self, name: str) -> ServerEntry:
        """Get a server by name."""
        if name not in self._servers:
            raise KeyError(f"Unknown server: {name}")
        return self._servers[name]

    def list_all(self) -> list[ServerEntry]:
        """Return all registered servers sorted by priority (lower = higher priority)."""
        return sorted(self._servers.values(), key=lambda s: s.priority)

    def get_available(self, exclude: set[str] | None = None) -> list[ServerEntry]:
        """Return servers not in the exclude set, sorted by priority."""
        excluded = exclude or set()
        return [s for s in self.list_all() if s.name not in excluded]

    def get_best_match(
        self,
        requirements: dict[str, Any] | None = None,
        prefer_free: bool = True,
    ) -> ServerEntry | None:
        """Find the best server matching resource requirements.

        Args:
            requirements: dict with optional keys: min_vram_gb, server_type, gpu
            prefer_free: prefer servers with cost_per_hour == 0
        """
        reqs = requirements or {}
        candidates = self.list_all()

        # Filter by minimum VRAM
        min_vram = reqs.get("min_vram_gb", 0)
        if min_vram:
            candidates = [s for s in candidates if s.vram_gb >= min_vram]

        # Filter by server type
        stype = reqs.get("server_type")
        if stype:
            candidates = [s for s in candidates if s.server_type == stype]

        # Filter by GPU model substring
        gpu_req = reqs.get("gpu")
        if gpu_req:
            candidates = [s for s in candidates if gpu_req.lower() in s.gpu.lower()]

        if not candidates:
            return None

        # Sort: prefer free servers, then by priority
        if prefer_free:
            candidates.sort(key=lambda s: (s.cost_per_hour > 0, s.priority))

        return candidates[0]

    @property
    def count(self) -> int:
        return len(self._servers)
