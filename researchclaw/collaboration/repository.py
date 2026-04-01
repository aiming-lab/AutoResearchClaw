"""Shared knowledge repository for cross-instance collaboration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARTIFACT_TYPES = (
    "literature_summary",
    "experiment_results",
    "code_template",
    "review_feedback",
)


class ResearchRepository:
    """File-system-based shared knowledge repository.

    Artifacts are organized by run_id and type, enabling cross-instance
    search and import of research outputs.
    """

    def __init__(self, repo_dir: str | Path = ".researchclaw/shared") -> None:
        self._repo_dir = Path(repo_dir)

    @property
    def repo_dir(self) -> Path:
        """Return the repository directory path."""
        return self._repo_dir

    def publish(self, run_id: str, artifacts: dict[str, Any]) -> int:
        """Publish research artifacts to the shared repository.

        Args:
            run_id: Unique identifier for the pipeline run.
            artifacts: Dict mapping artifact type to content.
                Supported types: literature_summary, experiment_results,
                code_template, review_feedback.

        Returns:
            Number of artifacts published.
        """
        run_dir = self._repo_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for artifact_type, content in artifacts.items():
            if artifact_type not in ARTIFACT_TYPES:
                logger.warning("Unknown artifact type: %s", artifact_type)
                continue

            artifact_path = run_dir / f"{artifact_type}.json"
            payload = {
                "run_id": run_id,
                "type": artifact_type,
                "content": content,
                "published_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            artifact_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            count += 1

        logger.info("Published %d artifacts for run %s", count, run_id)
        return count

    def search(
        self,
        query: str,
        artifact_type: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for artifacts matching a query.

        Simple keyword-based search (case-insensitive substring match).

        Args:
            query: Search query string.
            artifact_type: Filter by type (optional).
            max_results: Maximum number of results.

        Returns:
            List of matching artifact dicts.
        """
        if not self._repo_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        for run_dir in sorted(self._repo_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            for artifact_file in run_dir.glob("*.json"):
                if artifact_type and artifact_file.stem != artifact_type:
                    continue

                try:
                    payload = json.loads(
                        artifact_file.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    continue

                content_str = json.dumps(payload.get("content", ""), default=str).lower()
                if query_lower in content_str:
                    results.append(payload)
                    if len(results) >= max_results:
                        return results

        return results

    def list_runs(self) -> list[str]:
        """List all run IDs in the repository.

        Returns:
            List of run ID strings, most recent first.
        """
        if not self._repo_dir.exists():
            return []
        return sorted(
            [d.name for d in self._repo_dir.iterdir() if d.is_dir()],
            reverse=True,
        )

    def get_run_artifacts(self, run_id: str) -> dict[str, Any]:
        """Get all artifacts for a specific run.

        Args:
            run_id: The run identifier.

        Returns:
            Dict mapping artifact type to content.
        """
        run_dir = self._repo_dir / run_id
        if not run_dir.exists():
            return {}

        artifacts: dict[str, Any] = {}
        for artifact_file in run_dir.glob("*.json"):
            try:
                payload = json.loads(artifact_file.read_text(encoding="utf-8"))
                artifacts[artifact_file.stem] = payload.get("content")
            except (json.JSONDecodeError, OSError):
                continue

        return artifacts

    def import_literature(self, source_run_id: str) -> list[dict[str, Any]]:
        """Import literature summaries from another run.

        Args:
            source_run_id: The source run ID to import from.

        Returns:
            List of literature summary dicts.
        """
        artifacts = self.get_run_artifacts(source_run_id)
        content = artifacts.get("literature_summary")
        if content is None:
            return []

        if isinstance(content, list):
            return content
        return [content]

    def import_code_template(
        self,
        source_run_id: str,
        pattern: str,
    ) -> str | None:
        """Import a code template from another run.

        Args:
            source_run_id: The source run ID.
            pattern: Substring to search for in the code template.

        Returns:
            Matching code template string, or None.
        """
        artifacts = self.get_run_artifacts(source_run_id)
        content = artifacts.get("code_template")
        if content is None:
            return None

        content_str = str(content)
        if pattern.lower() in content_str.lower():
            return content_str

        return None
