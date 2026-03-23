"""Cloud executor: stub for AWS/GCP/Azure GPU instance management."""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.servers.registry import ServerEntry

logger = logging.getLogger(__name__)


class CloudExecutor:
    """Manage cloud GPU instances for experiment execution.

    This is a stub implementation. Actual cloud provider APIs (boto3, google-cloud,
    azure-mgmt) are imported lazily to avoid hard dependencies.
    """

    def __init__(self, server: ServerEntry) -> None:
        if server.server_type != "cloud":
            raise ValueError(f"Server {server.name} is not a cloud server")
        self.server = server
        self.provider = server.cloud_provider

    async def launch_instance(self) -> dict[str, Any]:
        """Launch a cloud GPU instance."""
        logger.info(
            "Launching %s instance (%s) for %s",
            self.provider,
            self.server.cloud_instance_type,
            self.server.name,
        )
        # Stub: actual implementation would call provider SDK
        return {
            "provider": self.provider,
            "instance_type": self.server.cloud_instance_type,
            "status": "stub_launched",
            "instance_id": f"stub-{self.server.name}",
            "cost_per_hour": self.server.cost_per_hour,
        }

    async def terminate_instance(self, instance_id: str) -> None:
        """Terminate a cloud instance."""
        logger.info("Terminating instance %s on %s", instance_id, self.provider)

    async def get_instance_status(self, instance_id: str) -> dict[str, Any]:
        """Check instance status."""
        return {"instance_id": instance_id, "status": "stub_unknown"}
