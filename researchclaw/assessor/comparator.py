"""Historical score comparison and tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HistoryComparator:
    """Track and compare paper quality scores across runs."""

    def __init__(self, history_dir: Path | None = None):
        self._history_dir = history_dir
        self._entries: list[dict[str, Any]] = []
        if history_dir:
            self._load_history()

    def _load_history(self) -> None:
        """Load score history from disk."""
        if self._history_dir is None:
            return
        history_file = self._history_dir / "quality_history.json"
        if not history_file.exists():
            return
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._entries = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load quality history: %s", exc)

    def record(
        self,
        run_id: str,
        topic: str,
        scores: dict[str, Any],
    ) -> None:
        """Record a quality assessment result."""
        entry = {
            "run_id": run_id,
            "topic": topic,
            "overall": scores.get("overall", 0.0),
            "scores": scores.get("scores", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        self._save_history()

    def _save_history(self) -> None:
        """Persist history to disk."""
        if self._history_dir is None:
            return
        self._history_dir.mkdir(parents=True, exist_ok=True)
        history_file = self._history_dir / "quality_history.json"
        history_file.write_text(
            json.dumps(self._entries, indent=2), encoding="utf-8"
        )

    def compare(
        self,
        current_scores: dict[str, Any],
        previous_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Compare current scores with a previous run or best historical."""
        if not self._entries:
            return {
                "comparison": "no_history",
                "message": "No previous runs to compare against.",
            }

        if previous_run_id:
            prev = next(
                (e for e in self._entries if e["run_id"] == previous_run_id),
                None,
            )
        else:
            prev = max(self._entries, key=lambda e: e.get("overall", 0))

        if prev is None:
            return {
                "comparison": "not_found",
                "message": f"Run '{previous_run_id}' not found in history.",
            }

        current_overall = current_scores.get("overall", 0.0)
        prev_overall = prev.get("overall", 0.0)
        delta = round(current_overall - prev_overall, 2)

        dim_deltas = {}
        current_dims = current_scores.get("scores", {})
        prev_dims = prev.get("scores", {})
        for dim in set(current_dims) | set(prev_dims):
            cur = current_dims.get(dim, 0.0)
            prv = prev_dims.get(dim, 0.0)
            dim_deltas[dim] = round(cur - prv, 2)

        trend = "improved" if delta > 0.5 else ("declined" if delta < -0.5 else "stable")

        return {
            "comparison": "success",
            "previous_run_id": prev.get("run_id", "unknown"),
            "current_overall": current_overall,
            "previous_overall": prev_overall,
            "delta": delta,
            "trend": trend,
            "dimension_deltas": dim_deltas,
        }

    def get_best_run(self) -> dict[str, Any] | None:
        """Return the highest-scoring historical run."""
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.get("overall", 0))

    def get_history(self) -> list[dict[str, Any]]:
        """Return all historical entries."""
        return list(self._entries)
