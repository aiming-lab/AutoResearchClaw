"""Tests for multi-server resource scheduling (C2): Registry, Monitor, Dispatcher, Executors."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researchclaw.servers.registry import ServerEntry, ServerRegistry
from researchclaw.servers.monitor import ServerMonitor, _parse_status_output
from researchclaw.servers.dispatcher import TaskDispatcher
from researchclaw.servers.ssh_executor import SSHExecutor
from researchclaw.servers.slurm_executor import SlurmExecutor
from researchclaw.servers.cloud_executor import CloudExecutor


# ── fixtures ──────────────────────────────────────────────────────


def _make_server(
    name: str = "s1",
    host: str = "gpu1.local",
    server_type: str = "ssh",
    vram_gb: int = 24,
    priority: int = 1,
    cost: float = 0.0,
    scheduler: str = "",
    cloud_provider: str = "",
) -> ServerEntry:
    return ServerEntry(
        name=name,
        host=host,
        server_type=server_type,
        gpu="RTX 4090",
        vram_gb=vram_gb,
        priority=priority,
        cost_per_hour=cost,
        scheduler=scheduler,
        cloud_provider=cloud_provider,
    )


@pytest.fixture
def registry() -> ServerRegistry:
    return ServerRegistry([
        _make_server("local", "localhost", vram_gb=48, priority=1),
        _make_server("cloud1", "cloud.host", server_type="cloud", vram_gb=80, priority=3, cost=2.0, cloud_provider="aws"),
        _make_server("hpc", "hpc.host", server_type="slurm", vram_gb=40, priority=2, scheduler="slurm"),
    ])


# ══════════════════════════════════════════════════════════════════
# ServerEntry tests
# ══════════════════════════════════════════════════════════════════


class TestServerEntry:
    def test_to_dict_roundtrip(self) -> None:
        s = _make_server()
        d = s.to_dict()
        s2 = ServerEntry.from_dict(d)
        assert s2.name == s.name
        assert s2.vram_gb == s.vram_gb

    def test_defaults(self) -> None:
        s = ServerEntry.from_dict({"name": "x"})
        assert s.server_type == "ssh"
        assert s.priority == 1


# ══════════════════════════════════════════════════════════════════
# ServerRegistry tests
# ══════════════════════════════════════════════════════════════════


class TestServerRegistry:
    def test_list_all_sorted_by_priority(self, registry: ServerRegistry) -> None:
        servers = registry.list_all()
        priorities = [s.priority for s in servers]
        assert priorities == sorted(priorities)

    def test_count(self, registry: ServerRegistry) -> None:
        assert registry.count == 3

    def test_add_server(self) -> None:
        reg = ServerRegistry()
        reg.add(_make_server("new"))
        assert reg.count == 1
        assert reg.get("new").name == "new"

    def test_remove_server(self, registry: ServerRegistry) -> None:
        registry.remove("local")
        assert registry.count == 2

    def test_remove_unknown_raises(self, registry: ServerRegistry) -> None:
        with pytest.raises(KeyError):
            registry.remove("ghost")

    def test_get_unknown_raises(self, registry: ServerRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("ghost")

    def test_get_available_excludes(self, registry: ServerRegistry) -> None:
        avail = registry.get_available(exclude={"local"})
        names = [s.name for s in avail]
        assert "local" not in names
        assert len(names) == 2

    def test_get_best_match_by_vram(self, registry: ServerRegistry) -> None:
        best = registry.get_best_match({"min_vram_gb": 40})
        assert best is not None
        assert best.vram_gb >= 40

    def test_get_best_match_by_type(self, registry: ServerRegistry) -> None:
        best = registry.get_best_match({"server_type": "slurm"})
        assert best is not None
        assert best.server_type == "slurm"

    def test_get_best_match_prefers_free(self, registry: ServerRegistry) -> None:
        best = registry.get_best_match(prefer_free=True)
        assert best is not None
        assert best.cost_per_hour == 0.0

    def test_get_best_match_none_when_impossible(self, registry: ServerRegistry) -> None:
        best = registry.get_best_match({"min_vram_gb": 999})
        assert best is None

    def test_get_best_match_by_gpu(self, registry: ServerRegistry) -> None:
        best = registry.get_best_match({"gpu": "RTX"})
        assert best is not None

    def test_get_best_match_no_requirements(self, registry: ServerRegistry) -> None:
        best = registry.get_best_match()
        assert best is not None
        assert best.name == "local"


# ══════════════════════════════════════════════════════════════════
# ServerMonitor tests
# ══════════════════════════════════════════════════════════════════


class TestServerMonitor:
    def test_parse_status_output(self) -> None:
        raw = "75, 8000, 24576\n---\n              total        used        free\nMem:          64000       32000       32000\n---\n 10:00:00 up 5 days"
        server = _make_server()
        status = _parse_status_output(raw, server)
        assert status["gpu"]["count"] == 1
        assert status["gpu"]["devices"][0]["utilization_pct"] == 75
        assert status["memory"]["total_mb"] == 64000
        assert "uptime" in status

    def test_parse_status_no_gpu(self) -> None:
        raw = "\n---\n              total        used        free\nMem:          64000       32000       32000\n---\nup 1 day"
        server = _make_server()
        status = _parse_status_output(raw, server)
        assert status["gpu"]["count"] == 0

    def test_get_cached_none(self, registry: ServerRegistry) -> None:
        monitor = ServerMonitor(registry)
        assert monitor.get_cached("local") is None

    def test_get_gpu_usage_empty(self, registry: ServerRegistry) -> None:
        monitor = ServerMonitor(registry)
        assert monitor.get_gpu_usage(_make_server()) == {}

    def test_check_status_unreachable(self, registry: ServerRegistry) -> None:
        monitor = ServerMonitor(registry)
        with patch("researchclaw.servers.monitor._ssh_command", side_effect=RuntimeError("unreachable")):
            status = asyncio.run(monitor.check_status(_make_server()))
        assert status["reachable"] is False

    def test_check_all(self, registry: ServerRegistry) -> None:
        monitor = ServerMonitor(registry)
        with patch("researchclaw.servers.monitor._ssh_command", side_effect=RuntimeError("unreachable")):
            results = asyncio.run(monitor.check_all())
        assert len(results) == 3
        for name, status in results.items():
            assert status["reachable"] is False


# ══════════════════════════════════════════════════════════════════
# SSHExecutor tests
# ══════════════════════════════════════════════════════════════════


class TestSSHExecutor:
    def test_init(self) -> None:
        server = _make_server()
        exe = SSHExecutor(server)
        assert exe.host == "gpu1.local"

    def test_run_experiment_timeout(self) -> None:
        server = _make_server()
        exe = SSHExecutor(server)

        async def _run() -> dict:
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                proc = AsyncMock()
                proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
                proc.kill = AsyncMock()
                proc.wait = AsyncMock()
                mock_exec.return_value = proc
                return await exe.run_experiment("/tmp/test", "echo hello", timeout=1)

        result = asyncio.run(_run())
        assert result["success"] is False
        assert "Timeout" in result["error"]


# ══════════════════════════════════════════════════════════════════
# SlurmExecutor tests
# ══════════════════════════════════════════════════════════════════


class TestSlurmExecutor:
    def test_init_wrong_type_raises(self) -> None:
        server = _make_server(server_type="ssh")
        with pytest.raises(ValueError, match="not a slurm"):
            SlurmExecutor(server)

    def test_generate_sbatch_script(self) -> None:
        server = _make_server(server_type="slurm", scheduler="slurm")
        exe = SlurmExecutor(server)
        script = exe._generate_sbatch_script("python main.py", resources={"gpus": 2, "mem_gb": 32})
        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH --mem=32G" in script
        assert "python main.py" in script

    def test_sbatch_script_default_resources(self) -> None:
        server = _make_server(server_type="slurm", scheduler="slurm")
        exe = SlurmExecutor(server)
        script = exe._generate_sbatch_script("echo hi")
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --time=01:00:00" in script

    def test_submit_job_parses_output(self) -> None:
        server = _make_server(server_type="slurm", scheduler="slurm")
        exe = SlurmExecutor(server)

        async def _run() -> str:
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                proc = AsyncMock()
                proc.communicate = AsyncMock(return_value=(b"Submitted batch job 12345\n", b""))
                proc.returncode = 0
                mock_exec.return_value = proc
                return await exe.submit_job("echo hi", "/tmp/test")

        job_id = asyncio.run(_run())
        assert job_id == "12345"


# ══════════════════════════════════════════════════════════════════
# CloudExecutor tests
# ══════════════════════════════════════════════════════════════════


class TestCloudExecutor:
    def test_init_wrong_type_raises(self) -> None:
        server = _make_server(server_type="ssh")
        with pytest.raises(ValueError, match="not a cloud"):
            CloudExecutor(server)

    def test_launch_instance_stub(self) -> None:
        server = _make_server(server_type="cloud", cloud_provider="aws")
        exe = CloudExecutor(server)
        result = asyncio.run(exe.launch_instance())
        assert result["status"] == "stub_launched"
        assert result["provider"] == "aws"


# ══════════════════════════════════════════════════════════════════
# TaskDispatcher tests
# ══════════════════════════════════════════════════════════════════


class TestTaskDispatcher:
    def test_dispatch_returns_task_id(self, registry: ServerRegistry) -> None:
        monitor = ServerMonitor(registry)
        disp = TaskDispatcher(registry, monitor)
        task_id = asyncio.run(disp.dispatch({"command": "echo hi", "local_dir": "/tmp"}))
        assert len(task_id) == 12

    def test_dispatch_no_server_queues(self) -> None:
        reg = ServerRegistry()
        monitor = ServerMonitor(reg)
        disp = TaskDispatcher(reg, monitor)
        task_id = asyncio.run(disp.dispatch({"command": "echo hi"}))
        status = disp.get_task_status(task_id)
        assert status["status"] == "queued"

    def test_get_task_status_unknown(self, registry: ServerRegistry) -> None:
        monitor = ServerMonitor(registry)
        disp = TaskDispatcher(registry, monitor)
        status = disp.get_task_status("nonexistent")
        assert status["status"] == "unknown"
