"""Track MetaClaw skill effectiveness across pipeline runs.

Records which skills were active during each stage and correlates
with stage success/failure to identify high/low-value skills.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillEffectivenessRecord:
    """One record of a skill's effectiveness in a pipeline stage."""

    skill_name: str
    stage_name: str
    run_id: str
    stage_success: bool
    timestamp: str


class SkillFeedbackStore:
    """JSONL-backed store for skill effectiveness records."""

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: SkillEffectivenessRecord) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def append_many(self, records: list[SkillEffectivenessRecord]) -> None:
        if not records:
            return
        with self._path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        logger.info("Recorded %d skill effectiveness entries", len(records))

    def load_all(self) -> list[SkillEffectivenessRecord]:
        if not self._path.exists():
            return []
        records: list[SkillEffectivenessRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(
                    SkillEffectivenessRecord(
                        skill_name=data["skill_name"],
                        stage_name=data["stage_name"],
                        run_id=data["run_id"],
                        stage_success=data["stage_success"],
                        timestamp=data["timestamp"],
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return records

    def compute_skill_stats(self) -> dict[str, dict[str, int | float]]:
        """Compute success rate per skill across all recorded runs.

        Returns:
            Dict mapping skill_name to {total, successes, success_rate}.
        """
        records = self.load_all()
        stats: dict[str, dict[str, int]] = {}
        for rec in records:
            if rec.skill_name not in stats:
                stats[rec.skill_name] = {"total": 0, "successes": 0}
            stats[rec.skill_name]["total"] += 1
            if rec.stage_success:
                stats[rec.skill_name]["successes"] += 1

        result: dict[str, dict[str, int | float]] = {}
        for name, counts in stats.items():
            total = counts["total"]
            successes = counts["successes"]
            result[name] = {
                "total": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else 0.0,
            }
        return result


def record_stage_skills(
    store: SkillFeedbackStore,
    stage_name: str,
    run_id: str,
    stage_success: bool,
    active_skills: list[str],
) -> None:
    """Record effectiveness of all active skills for a completed stage."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = [
        SkillEffectivenessRecord(
            skill_name=skill,
            stage_name=stage_name,
            run_id=run_id,
            stage_success=stage_success,
            timestamp=now,
        )
        for skill in active_skills
    ]
    store.append_many(records)
