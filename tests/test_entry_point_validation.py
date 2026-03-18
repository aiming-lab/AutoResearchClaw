"""Tests for entry point path traversal validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from researchclaw.experiment.sandbox import (
    ExperimentSandbox,
    validate_entry_point,
    validate_entry_point_resolved,
)


# ── Unit tests: validate_entry_point (syntax) ─────────────────────────


class TestValidateEntryPoint:
    """Syntax-only checks — no filesystem needed."""

    def test_valid_entry_point(self) -> None:
        assert validate_entry_point("main.py") is None

    def test_valid_nested_entry_point(self) -> None:
        assert validate_entry_point("src/train.py") is None

    def test_valid_dot_slash_prefix(self) -> None:
        assert validate_entry_point("./main.py") is None

    def test_valid_dot_in_middle(self) -> None:
        assert validate_entry_point("src/./train.py") is None

    def test_valid_deeply_nested(self) -> None:
        assert validate_entry_point("a/b/c/d/main.py") is None

    def test_rejects_absolute_path(self) -> None:
        err = validate_entry_point("/etc/passwd")
        assert err is not None
        assert "relative" in err.lower() or "absolute" in err.lower()

    def test_rejects_path_traversal(self) -> None:
        err = validate_entry_point("../../../etc/passwd")
        assert err is not None
        assert ".." in err

    def test_rejects_dotdot_in_middle(self) -> None:
        err = validate_entry_point("src/../../etc/passwd")
        assert err is not None
        assert ".." in err

    def test_rejects_empty_string(self) -> None:
        err = validate_entry_point("")
        assert err is not None
        assert "empty" in err.lower()

    def test_rejects_whitespace_only(self) -> None:
        err = validate_entry_point("   ")
        assert err is not None
        assert "empty" in err.lower()


# ── Unit tests: validate_entry_point_resolved (containment) ───────────


class TestValidateEntryPointResolved:
    """Resolve-based checks — needs a real staging directory."""

    def test_valid_path_passes(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("pass")
        assert validate_entry_point_resolved(tmp_path, "main.py") is None

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        """A symlink pointing outside staging must be caught."""
        escape_target = tmp_path / "outside" / "secret.py"
        escape_target.parent.mkdir()
        escape_target.write_text("print('escaped!')")

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "legit.py").symlink_to(escape_target)

        err = validate_entry_point_resolved(staging, "legit.py")
        assert err is not None
        assert "escapes" in err.lower()

    def test_nested_valid_path_passes(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "train.py").write_text("pass")
        assert validate_entry_point_resolved(tmp_path, "src/train.py") is None


# ── Integration tests: ExperimentSandbox.run_project() ────────────────


class TestExperimentSandboxEntryPointValidation:
    """Verify validation is wired into ExperimentSandbox.run_project()."""

    def _make_sandbox(self, tmp_path: Path) -> ExperimentSandbox:
        from researchclaw.config import SandboxConfig

        cfg = SandboxConfig()
        return ExperimentSandbox(cfg, tmp_path / "work")

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "main.py").write_text("print('hi')")

        sandbox = self._make_sandbox(tmp_path)
        # Create escape target so .exists() alone wouldn't catch it
        work = tmp_path / "work"
        work.mkdir(parents=True, exist_ok=True)
        (work / "escape.py").write_text("print('escaped!')")

        with patch("subprocess.run") as mock_run:
            result = sandbox.run_project(project, entry_point="../escape.py")

        assert result.returncode == -1
        assert ".." in result.stderr
        mock_run.assert_not_called()

    def test_rejects_absolute_path(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "main.py").write_text("print('hi')")

        sandbox = self._make_sandbox(tmp_path)

        with patch("subprocess.run") as mock_run:
            result = sandbox.run_project(project, entry_point="/etc/passwd")

        assert result.returncode == -1
        assert "relative" in result.stderr.lower() or "absolute" in result.stderr.lower()
        mock_run.assert_not_called()

    # NOTE: A symlink integration test is not included here because the
    # copy loop (write_bytes/read_bytes) follows symlinks and creates
    # regular files in staging.  The resolve check is defense-in-depth
    # for future copy mechanism changes; see
    # TestValidateEntryPointResolved.test_symlink_escape_rejected for
    # the unit-level proof that the function catches symlink escapes.
