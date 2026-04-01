"""Exploration branch management for Co-Pilot mode."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BranchManager:
    """Manage exploration branches during Co-Pilot sessions."""

    def __init__(self, run_dir: Path, max_branches: int = 3):
        self.run_dir = run_dir
        self.max_branches = max_branches
        self._branches_dir = run_dir / "branches"

    def create_branch(
        self,
        name: str,
        from_stage: int,
    ) -> str:
        """Create a new exploration branch by copying state up to from_stage."""
        if len(self.list_branches()) >= self.max_branches:
            raise ValueError(
                f"Maximum branches ({self.max_branches}) reached. "
                f"Delete a branch before creating a new one."
            )

        branch_dir = self._branches_dir / name
        if branch_dir.exists():
            raise ValueError(f"Branch '{name}' already exists.")

        branch_dir.mkdir(parents=True, exist_ok=True)

        # Copy stage directories up to from_stage
        for stage_num in range(1, from_stage + 1):
            src = self.run_dir / f"stage-{stage_num:02d}"
            if src.exists():
                dest = branch_dir / f"stage-{stage_num:02d}"
                shutil.copytree(src, dest, dirs_exist_ok=True)

        # Write branch metadata
        meta = {
            "name": name,
            "from_stage": from_stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        (branch_dir / "branch_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        logger.info("Created branch '%s' from stage %d", name, from_stage)
        return str(branch_dir)

    def list_branches(self) -> list[dict[str, Any]]:
        """List all branches with their metadata."""
        if not self._branches_dir.exists():
            return []

        branches = []
        for branch_dir in sorted(self._branches_dir.iterdir()):
            if not branch_dir.is_dir():
                continue
            meta_path = branch_dir / "branch_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["path"] = str(branch_dir)
                    branches.append(meta)
                except (json.JSONDecodeError, OSError):
                    branches.append({
                        "name": branch_dir.name,
                        "path": str(branch_dir),
                        "status": "unknown",
                    })
            else:
                branches.append({
                    "name": branch_dir.name,
                    "path": str(branch_dir),
                    "status": "unknown",
                })
        return branches

    def switch_branch(self, name: str) -> Path:
        """Get the directory path for a branch (for resuming execution)."""
        branch_dir = self._branches_dir / name
        if not branch_dir.exists():
            raise ValueError(f"Branch '{name}' does not exist.")
        return branch_dir

    def delete_branch(self, name: str) -> None:
        """Delete a branch and its data."""
        branch_dir = self._branches_dir / name
        if not branch_dir.exists():
            raise ValueError(f"Branch '{name}' does not exist.")
        shutil.rmtree(branch_dir)
        logger.info("Deleted branch '%s'", name)

    def compare_branches(
        self,
        branch_a: str,
        branch_b: str,
    ) -> dict[str, Any]:
        """Compare results between two branches."""
        dir_a = self._branches_dir / branch_a
        dir_b = self._branches_dir / branch_b

        if not dir_a.exists():
            return {"error": f"Branch '{branch_a}' does not exist."}
        if not dir_b.exists():
            return {"error": f"Branch '{branch_b}' does not exist."}

        # Compare experiment summaries if available
        result: dict[str, Any] = {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "stages_a": self._count_stages(dir_a),
            "stages_b": self._count_stages(dir_b),
        }

        summary_a = self._read_experiment_summary(dir_a)
        summary_b = self._read_experiment_summary(dir_b)

        if summary_a and summary_b:
            result["metrics_a"] = summary_a.get("metrics_summary", {})
            result["metrics_b"] = summary_b.get("metrics_summary", {})

        return result

    @staticmethod
    def _count_stages(branch_dir: Path) -> int:
        """Count completed stages in a branch directory."""
        count = 0
        for d in branch_dir.iterdir():
            if d.is_dir() and d.name.startswith("stage-"):
                count += 1
        return count

    @staticmethod
    def _read_experiment_summary(
        branch_dir: Path,
    ) -> dict[str, Any] | None:
        """Read experiment summary from a branch."""
        summary_path = branch_dir / "stage-14" / "experiment_summary.json"
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
