"""Tests for Overleaf sync (C4): Sync engine, Conflict resolver, Watcher, Formatter."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.overleaf.sync import OverleafSync
from researchclaw.overleaf.conflict import ConflictResolver, _extract_conflicts, _resolve_content
from researchclaw.overleaf.watcher import FileWatcher
from researchclaw.overleaf.formatter import LatexFormatter


# ══════════════════════════════════════════════════════════════════
# ConflictResolver tests
# ══════════════════════════════════════════════════════════════════


class TestConflictResolver:
    def test_no_conflicts(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text("\\section{Intro}\nHello world\n")
        resolver = ConflictResolver()
        assert not resolver.has_conflicts(tmp_path)

    def test_has_conflicts(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            \\section{Intro}
            <<<<<<< HEAD
            Our method is great.
            =======
            Our method is good.
            >>>>>>> remote
        """)
        (tmp_path / "paper.tex").write_text(content)
        resolver = ConflictResolver()
        assert resolver.has_conflicts(tmp_path)

    def test_detect_conflicts(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            <<<<<<< HEAD
            line A
            =======
            line B
            >>>>>>> remote
        """)
        (tmp_path / "main.tex").write_text(content)
        resolver = ConflictResolver()
        conflicts = resolver.detect(tmp_path)
        assert len(conflicts) == 1
        assert conflicts[0]["ours"] == "line A"
        assert conflicts[0]["theirs"] == "line B"

    def test_resolve_ours(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            \\section{Intro}
            <<<<<<< HEAD
            AI version
            =======
            Human version
            >>>>>>> remote
            \\section{End}
        """)
        (tmp_path / "paper.tex").write_text(content)
        resolver = ConflictResolver()
        resolved = resolver.resolve(tmp_path, strategy="ours")
        assert len(resolved) == 1
        text = (tmp_path / "paper.tex").read_text()
        assert "AI version" in text
        assert "Human version" not in text
        assert "<<<<<<" not in text

    def test_resolve_theirs(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            <<<<<<< HEAD
            AI text
            =======
            Human text
            >>>>>>> remote
        """)
        (tmp_path / "paper.tex").write_text(content)
        resolver = ConflictResolver()
        resolver.resolve(tmp_path, strategy="theirs")
        text = (tmp_path / "paper.tex").read_text()
        assert "Human text" in text
        assert "AI text" not in text

    def test_multiple_conflicts(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            <<<<<<< HEAD
            A1
            =======
            B1
            >>>>>>> remote
            middle
            <<<<<<< HEAD
            A2
            =======
            B2
            >>>>>>> remote
        """)
        (tmp_path / "paper.tex").write_text(content)
        resolver = ConflictResolver()
        conflicts = resolver.detect(tmp_path)
        assert len(conflicts) == 2


class TestConflictHelpers:
    def test_extract_conflicts_empty(self) -> None:
        assert _extract_conflicts("no conflicts here") == []

    def test_resolve_content_ours(self) -> None:
        content = "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> remote\n"
        resolved = _resolve_content(content, "ours")
        assert "ours" in resolved
        assert "theirs" not in resolved

    def test_resolve_content_theirs(self) -> None:
        content = "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> remote\n"
        resolved = _resolve_content(content, "theirs")
        assert "theirs" in resolved
        assert "ours" not in resolved


# ══════════════════════════════════════════════════════════════════
# FileWatcher tests
# ══════════════════════════════════════════════════════════════════


class TestFileWatcher:
    def test_no_changes_initially(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text("content")
        watcher = FileWatcher(tmp_path)
        assert watcher.check_changes() == []

    def test_detect_new_file(self, tmp_path: Path) -> None:
        watcher = FileWatcher(tmp_path)
        (tmp_path / "new.tex").write_text("new content")
        changes = watcher.check_changes()
        assert "new.tex" in changes

    def test_detect_modified_file(self, tmp_path: Path) -> None:
        f = tmp_path / "paper.tex"
        f.write_text("v1")
        watcher = FileWatcher(tmp_path)
        # Modify
        import time
        time.sleep(0.05)
        f.write_text("v2")
        changes = watcher.check_changes()
        assert "paper.tex" in changes

    def test_detect_deleted_file(self, tmp_path: Path) -> None:
        f = tmp_path / "paper.tex"
        f.write_text("content")
        watcher = FileWatcher(tmp_path)
        f.unlink()
        changes = watcher.check_changes()
        assert "paper.tex" in changes

    def test_only_watches_extensions(self, tmp_path: Path) -> None:
        watcher = FileWatcher(tmp_path, extensions=(".tex",))
        (tmp_path / "readme.md").write_text("markdown")
        changes = watcher.check_changes()
        assert changes == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        watcher = FileWatcher(tmp_path / "nonexistent")
        assert watcher.check_changes() == []


# ══════════════════════════════════════════════════════════════════
# LatexFormatter tests
# ══════════════════════════════════════════════════════════════════


class TestLatexFormatter:
    def test_normalize_paths(self) -> None:
        content = r"\includegraphics[width=0.5\textwidth]{/home/user/artifacts/rc-123/figures/plot.png}"
        result = LatexFormatter.normalize_paths(content)
        assert "figures/plot.png" in result
        assert "/home/user" not in result

    def test_ensure_document_class_adds(self) -> None:
        content = "\\begin{document}\nHello\n\\end{document}"
        result = LatexFormatter.ensure_document_class(content)
        assert "\\documentclass" in result

    def test_ensure_document_class_noop(self) -> None:
        content = "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}"
        result = LatexFormatter.ensure_document_class(content)
        assert result.count("\\documentclass") == 1

    def test_strip_local_comments(self) -> None:
        content = "Normal line\n% RESEARCHCLAW: internal note\nAnother line\n"
        result = LatexFormatter.strip_local_comments(content)
        assert "RESEARCHCLAW" not in result
        assert "Normal line" in result
        assert "Another line" in result

    def test_fix_encoding(self) -> None:
        content = "\\documentclass{article}\n\\begin{document}\n"
        result = LatexFormatter.fix_encoding(content)
        assert "\\usepackage[utf8]{inputenc}" in result

    def test_fix_encoding_noop(self) -> None:
        content = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n"
        result = LatexFormatter.fix_encoding(content)
        assert result.count("inputenc") == 1

    def test_format_for_overleaf(self, tmp_path: Path) -> None:
        tex = tmp_path / "paper.tex"
        tex.write_text("\\documentclass{article}\n% RESEARCHCLAW: test\n\\begin{document}\nHello\n\\end{document}\n")
        formatter = LatexFormatter()
        result = formatter.format_for_overleaf(tex)
        assert "RESEARCHCLAW" not in result
        assert "inputenc" in result


# ══════════════════════════════════════════════════════════════════
# OverleafSync tests (mock git)
# ══════════════════════════════════════════════════════════════════


class TestOverleafSync:
    def test_init(self) -> None:
        sync = OverleafSync(git_url="https://git.overleaf.com/abc123")
        assert sync.git_url == "https://git.overleaf.com/abc123"
        assert sync.branch == "main"
        assert sync.local_dir is None

    def test_get_status_before_setup(self) -> None:
        sync = OverleafSync(git_url="https://git.overleaf.com/abc123")
        status = sync.get_status()
        assert status["local_dir"] is None
        assert status["last_sync"] is None

    def test_push_before_setup_raises(self, tmp_path: Path) -> None:
        sync = OverleafSync(git_url="https://git.overleaf.com/abc123")
        with pytest.raises(RuntimeError, match="setup"):
            sync.push_paper(tmp_path / "paper.tex")

    def test_pull_before_setup_raises(self) -> None:
        sync = OverleafSync(git_url="https://git.overleaf.com/abc123")
        with pytest.raises(RuntimeError, match="setup"):
            sync.pull_changes()

    def test_resolve_before_setup_raises(self) -> None:
        sync = OverleafSync(git_url="https://git.overleaf.com/abc123")
        with pytest.raises(RuntimeError, match="setup"):
            sync.resolve_conflicts()

    @patch("researchclaw.overleaf.sync.subprocess.run")
    def test_setup_clones(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        sync = OverleafSync(git_url="https://git.overleaf.com/abc123")
        local = sync.setup(tmp_path)
        assert local == tmp_path / "overleaf_repo"
        # git clone was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "clone" in args
