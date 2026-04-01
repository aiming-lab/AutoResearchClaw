"""Run data collector — scans artifacts/ for pipeline state."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunSnapshot:
    """Point-in-time snapshot of a pipeline run."""

    run_id: str
    path: str
    status: str = "unknown"
    current_stage: int = 0
    current_stage_name: str = ""
    total_stages: int = 23
    start_time: str = ""
    elapsed_sec: float = 0.0
    is_active: bool = False
    topic: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    stages_completed: list[str] = field(default_factory=list)
    last_log_lines: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": self.path,
            "status": self.status,
            "current_stage": self.current_stage,
            "current_stage_name": self.current_stage_name,
            "total_stages": self.total_stages,
            "start_time": self.start_time,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "is_active": self.is_active,
            "topic": self.topic,
            "metrics": self.metrics,
            "stages_completed": self.stages_completed,
            "error": self.error,
        }


class DashboardCollector:
    """Collect run state from artifacts/ directory."""

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        max_log_lines: int = 200,
    ) -> None:
        self._artifacts = Path(artifacts_dir)
        self._max_log_lines = max_log_lines

    def collect_all(self) -> list[RunSnapshot]:
        """Scan all rc-* directories and return snapshots."""
        if not self._artifacts.exists():
            return []

        runs: list[RunSnapshot] = []
        for d in sorted(self._artifacts.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("rc-"):
                try:
                    snap = self._collect_run(d)
                    runs.append(snap)
                except Exception as exc:
                    logger.debug("Failed to collect %s: %s", d, exc)
        return runs

    def collect_run(self, run_dir: str | Path) -> RunSnapshot:
        """Collect a single run."""
        return self._collect_run(Path(run_dir))

    def _collect_run(self, run_dir: Path) -> RunSnapshot:
        snap = RunSnapshot(run_id=run_dir.name, path=str(run_dir))

        # --- checkpoint.json ---
        ckpt_path = run_dir / "checkpoint.json"
        if ckpt_path.exists():
            try:
                with ckpt_path.open() as f:
                    ckpt = json.load(f)
                snap.current_stage = ckpt.get("stage", 0)
                snap.current_stage_name = ckpt.get("stage_name", "")
                snap.status = ckpt.get("status", "running")
                snap.topic = ckpt.get("topic", "")
                snap.start_time = ckpt.get("start_time", "")
            except Exception:
                pass

        # --- heartbeat.json ---
        hb_path = run_dir / "heartbeat.json"
        if hb_path.exists():
            try:
                with hb_path.open() as f:
                    hb = json.load(f)
                last_ts = hb.get("timestamp", 0)
                snap.is_active = (time.time() - last_ts) < 60
                if snap.is_active:
                    snap.status = "running"
            except Exception:
                pass

        # --- stage directories ---
        snap.stages_completed = sorted(
            [d.name for d in run_dir.iterdir()
             if d.is_dir() and d.name.startswith("stage-")]
        )

        # --- experiment metrics (results.json) ---
        for results_path in run_dir.rglob("results.json"):
            try:
                with results_path.open() as f:
                    snap.metrics = json.load(f)
                break
            except Exception:
                pass

        # --- last log lines ---
        log_path = run_dir / "pipeline.log"
        if log_path.exists():
            try:
                lines = log_path.read_text(errors="replace").splitlines()
                snap.last_log_lines = lines[-self._max_log_lines:]
            except Exception:
                pass

        return snap
