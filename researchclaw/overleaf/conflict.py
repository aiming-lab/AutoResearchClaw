"""Section-level conflict detection and resolution for LaTeX files."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# LaTeX section commands in order of depth
_SECTION_PATTERN = re.compile(
    r"^\\(part|chapter|section|subsection|subsubsection)\b",
    re.MULTILINE,
)
_CONFLICT_MARKER = re.compile(r"^<<<<<<<\s", re.MULTILINE)


class ConflictResolver:
    """Detect and resolve merge conflicts at the LaTeX section level."""

    def has_conflicts(self, repo_dir: Path) -> bool:
        """Check if there are unresolved merge conflicts."""
        for tex in repo_dir.glob("**/*.tex"):
            content = tex.read_text(encoding="utf-8", errors="replace")
            if _CONFLICT_MARKER.search(content):
                return True
        return False

    def detect(self, repo_dir: Path) -> list[dict[str, str]]:
        """Find all conflict regions and which sections they belong to."""
        conflicts: list[dict[str, str]] = []
        for tex in repo_dir.glob("**/*.tex"):
            content = tex.read_text(encoding="utf-8", errors="replace")
            file_conflicts = _extract_conflicts(content)
            for c in file_conflicts:
                c["file"] = str(tex.relative_to(repo_dir))
            conflicts.extend(file_conflicts)
        return conflicts

    def resolve(self, repo_dir: Path, strategy: str = "ours") -> list[str]:
        """Resolve all conflicts in .tex files using the given strategy.

        strategy: "ours" keeps the local (AI) version,
                  "theirs" keeps the remote (human) version
        """
        resolved_files: list[str] = []
        for tex in repo_dir.glob("**/*.tex"):
            content = tex.read_text(encoding="utf-8", errors="replace")
            if not _CONFLICT_MARKER.search(content):
                continue
            resolved = _resolve_content(content, strategy)
            tex.write_text(resolved, encoding="utf-8")
            resolved_files.append(str(tex.relative_to(repo_dir)))
            logger.info("Resolved conflicts in %s (strategy=%s)", tex.name, strategy)
        return resolved_files


def _extract_conflicts(content: str) -> list[dict[str, str]]:
    """Extract conflict regions from file content."""
    conflicts: list[dict[str, str]] = []
    in_conflict = False
    ours_lines: list[str] = []
    theirs_lines: list[str] = []
    current = ours_lines

    for line in content.splitlines():
        if line.startswith("<<<<<<<"):
            in_conflict = True
            ours_lines = []
            theirs_lines = []
            current = ours_lines
        elif line.startswith("=======") and in_conflict:
            current = theirs_lines
        elif line.startswith(">>>>>>>") and in_conflict:
            conflicts.append({
                "ours": "\n".join(ours_lines),
                "theirs": "\n".join(theirs_lines),
            })
            in_conflict = False
        elif in_conflict:
            current.append(line)
    return conflicts


def _resolve_content(content: str, strategy: str) -> str:
    """Replace conflict markers with the chosen side."""
    lines = content.splitlines(keepends=True)
    result: list[str] = []
    state = "normal"  # normal | ours | theirs

    for line in lines:
        if line.startswith("<<<<<<<"):
            state = "ours"
        elif line.startswith("=======") and state == "ours":
            state = "theirs"
        elif line.startswith(">>>>>>>") and state == "theirs":
            state = "normal"
        else:
            if state == "normal":
                result.append(line)
            elif state == "ours" and strategy == "ours":
                result.append(line)
            elif state == "theirs" and strategy == "theirs":
                result.append(line)
    return "".join(result)
