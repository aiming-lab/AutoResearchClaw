"""Git-native experiment version management inspired by autoresearch."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentGitManager:
    """Git-native experiment versioning, inspired by autoresearch.

    Every successful experiment is a commit; failed experiments are reset.
    This enables git log as an experiment journal and easy rollback.
    """

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir: Path = repo_dir
        self._active_branch: str | None = None
        self._original_branch: str | None = self._detect_current_branch()

    def create_experiment_branch(self, tag: str) -> str:
        branch = f"experiment/{tag}"
        result = self._run_git(["checkout", "-b", branch])
        if result is None or result.returncode != 0:
            self._log_git_failure("create_experiment_branch", result)
            return ""
        self._active_branch = branch
        return branch

    def commit_experiment(
        self, run_id: str, metrics: dict[str, object], description: str
    ) -> str:
        add_result = self._run_git(["add", "-A"])
        if add_result is None or add_result.returncode != 0:
            self._log_git_failure("git add", add_result)
            return ""

        message = self._format_commit_message(
            run_id=run_id, metrics=metrics, description=description
        )
        commit_result = self._run_git(["commit", "-m", message])
        if commit_result is None or commit_result.returncode != 0:
            self._log_git_failure("git commit", commit_result)
            return ""

        hash_result = self._run_git(["rev-parse", "HEAD"])
        if hash_result is None or hash_result.returncode != 0:
            self._log_git_failure("git rev-parse HEAD", hash_result)
            return ""
        return self._clean_output(hash_result.stdout)

    def discard_experiment(self, run_id: str, reason: str) -> bool:
        logger.info("Discarding experiment %s: %s", run_id, reason)
        result = self._run_git(["reset", "--hard", "HEAD"])
        if result is None or result.returncode != 0:
            self._log_git_failure("discard_experiment", result)
            return False
        return True

    def get_experiment_history(self) -> list[dict[str, str]]:
        result = self._run_git(["log", "--oneline", "--fixed-strings", "--grep", "experiment("])
        if result is None or result.returncode != 0:
            self._log_git_failure("git log", result)
            return []

        history: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parsed = self._parse_experiment_log_line(line)
            if parsed is not None:
                history.append(parsed)
        return history

    def is_git_repo(self) -> bool:
        """Check whether repo_dir is inside a git repository."""
        result = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return result is not None and result.returncode == 0

    def get_current_branch(self) -> str:
        """Return the name of the current branch, or '' on failure."""
        name = self._detect_current_branch()
        return name or ""

    def return_to_original_branch(self) -> bool:
        """Switch back to the branch that was active when the manager was created."""
        if not self._original_branch:
            return False
        result = self._run_git(["checkout", self._original_branch])
        if result is None or result.returncode != 0:
            self._log_git_failure("return_to_original_branch", result)
            return False
        self._active_branch = self._original_branch
        return True

    def get_experiment_diff(self) -> str:
        """Return the git diff of uncommitted changes (for logging/debugging)."""
        result = self._run_git(["diff", "--stat"])
        if result is None or result.returncode != 0:
            return ""
        return result.stdout.strip()

    def clean_untracked(self) -> bool:
        """Remove untracked files in the experiment workspace."""
        result = self._run_git(["clean", "-fd"])
        return result is not None and result.returncode == 0

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            logger.debug("Running git command: git %s", " ".join(args))
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Git operation failed (%s): %s", " ".join(args), exc)
            return None

    @staticmethod
    def _format_commit_message(
        *, run_id: str, metrics: dict[str, object], description: str
    ) -> str:
        metrics_json = json.dumps(metrics, sort_keys=True)
        return f"experiment({run_id}): {description}\n\nMetrics: {metrics_json}"

    @staticmethod
    def _clean_output(output: str) -> str:
        return output.strip()

    @staticmethod
    def _parse_experiment_log_line(line: str) -> dict[str, str] | None:
        pattern = re.compile(r"^([0-9a-fA-F]+)\s+experiment\(([^)]+)\):\s*(.*)$")
        match = pattern.match(line.strip())
        if match is None:
            return None
        commit_hash, run_id, message = match.groups()
        return {"hash": commit_hash, "run_id": run_id, "message": message}

    @staticmethod
    def _log_git_failure(
        operation: str, result: subprocess.CompletedProcess[str] | None
    ) -> None:
        if result is None:
            logger.warning("Git operation failed for %s", operation)
            return
        stderr = result.stderr.strip()
        if stderr:
            logger.warning("Git operation failed for %s: %s", operation, stderr)
        else:
            logger.warning(
                "Git operation failed for %s with code %s", operation, result.returncode
            )

    def _detect_current_branch(self) -> str | None:
        """Detect the current git branch name, or None if not in a repo."""
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if result is None or result.returncode != 0:
            return None
        name = result.stdout.strip()
        return name if name else None
