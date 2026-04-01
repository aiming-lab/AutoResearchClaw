"""Task dispatcher: route experiment tasks to the best available server."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from researchclaw.servers.registry import ServerEntry, ServerRegistry
from researchclaw.servers.monitor import ServerMonitor
from researchclaw.servers.ssh_executor import SSHExecutor
from researchclaw.servers.slurm_executor import SlurmExecutor

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """Dispatch experiment tasks to the best available server."""

    def __init__(
        self,
        registry: ServerRegistry,
        monitor: ServerMonitor,
        prefer_free: bool = True,
        failover: bool = True,
    ) -> None:
        self.registry = registry
        self.monitor = monitor
        self.prefer_free = prefer_free
        self.failover = failover
        self._tasks: dict[str, dict[str, Any]] = {}
        self._busy_servers: set[str] = set()

    async def dispatch(self, task: dict[str, Any]) -> str:
        """Dispatch a task to the best available server.

        Args:
            task: dict with keys: command, local_dir, requirements (optional)

        Returns:
            task_id for tracking
        """
        task_id = uuid.uuid4().hex[:12]
        requirements = task.get("requirements", {})

        # Find best server
        server = self.registry.get_best_match(
            requirements=requirements,
            prefer_free=self.prefer_free,
        )
        if server is None:
            self._tasks[task_id] = {"status": "queued", "task": task, "error": "No matching server"}
            logger.warning("No server available for task %s, queued", task_id)
            return task_id

        # Dispatch based on server type
        self._tasks[task_id] = {
            "status": "dispatched",
            "server": server.name,
            "task": task,
        }
        self._busy_servers.add(server.name)
        logger.info("Dispatched task %s to %s (%s)", task_id, server.name, server.server_type)
        return task_id

    async def execute_task(self, task_id: str) -> dict[str, Any]:
        """Execute a dispatched task on its assigned server."""
        info = self._tasks.get(task_id)
        if not info or info["status"] != "dispatched":
            return {"success": False, "error": "Task not dispatched"}

        server = self.registry.get(info["server"])
        task = info["task"]
        remote_dir = f"/tmp/researchclaw_{task_id}"

        try:
            if server.server_type == "slurm":
                executor = SlurmExecutor(server)
                job_id = await executor.submit_job(
                    command=task["command"],
                    remote_dir=remote_dir,
                    resources=task.get("requirements"),
                )
                info["status"] = "running"
                info["job_id"] = job_id
                return {"success": True, "job_id": job_id}
            else:
                # Default: SSH executor
                executor = SSHExecutor(server)  # type: ignore[assignment]
                result = await executor.run_experiment(
                    remote_dir=remote_dir,
                    command=task["command"],
                    timeout=task.get("timeout", 3600),
                )
                info["status"] = "completed" if result["success"] else "failed"
                info["result"] = result
                return result
        except Exception as exc:
            logger.error("Task %s failed: %s", task_id, exc)
            info["status"] = "failed"
            info["error"] = str(exc)

            # Failover: try another server (non-recursive, single attempt)
            if self.failover:
                tried = {server.name}
                alt = self.registry.get_best_match(
                    requirements=task.get("requirements"),
                    prefer_free=self.prefer_free,
                )
                if alt and alt.name not in tried:
                    logger.info("Failing over task %s to %s", task_id, alt.name)
                    info["server"] = alt.name
                    info["status"] = "dispatched"
                    try:
                        alt_server = self.registry.get(alt.name)
                        result = await alt_server.run_experiment(
                            remote_dir=task.get("remote_dir", ""),
                            command=task.get("command", ""),
                            timeout=task.get("timeout", 3600),
                        )
                        info["status"] = "completed"
                        return result
                    except Exception as alt_exc:
                        logger.error("Failover also failed: %s", alt_exc)

            return {"success": False, "error": str(exc)}
        finally:
            self._busy_servers.discard(server.name)

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a task."""
        info = self._tasks.get(task_id)
        if not info:
            return {"task_id": task_id, "status": "unknown"}
        return {
            "task_id": task_id,
            "status": info["status"],
            "server": info.get("server"),
            "error": info.get("error"),
        }
