from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from json import dumps, loads
from pathlib import Path
from shutil import copy2, copytree, rmtree
from typing import Any


class PaperRepairError(ValueError):
    """Raised when a paper-repair session cannot be created or applied."""


TRACKED_STAGE_PATHS: dict[str, tuple[str, ...]] = {
    "stage-22": (
        "paper.tex",
        "paper.pdf",
        "paper_final.md",
        "paper_final_latex.md",
        "references.bib",
        "references_verified.bib",
        "neurips_2025.sty",
        "charts",
    ),
    "stage-23": (
        "paper_final_verified.md",
        "references_verified.bib",
        "verification_report.json",
        "charts",
    ),
}


@dataclass(frozen=True)
class TrackedItem:
    relative_path: str
    kind: str
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackedItem:
        return cls(
            relative_path=str(data.get("relative_path", "")),
            kind=str(data.get("kind", "file")),
            exists=bool(data.get("exists", False)),
        )


@dataclass(frozen=True)
class ApplyEntry:
    backup_id: str
    applied_at: str
    note: str
    backup_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApplyEntry:
        return cls(
            backup_id=str(data.get("backup_id", "")),
            applied_at=str(data.get("applied_at", "")),
            note=str(data.get("note", "")),
            backup_dir=str(data.get("backup_dir", "")),
        )


@dataclass(frozen=True)
class PaperRepairSession:
    source_run_dir: str
    session_dir: str
    workspace_dir: str
    created_at: str
    tracked_items: tuple[TrackedItem, ...]
    apply_history: tuple[ApplyEntry, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_dir": self.source_run_dir,
            "session_dir": self.session_dir,
            "workspace_dir": self.workspace_dir,
            "created_at": self.created_at,
            "tracked_items": [item.to_dict() for item in self.tracked_items],
            "apply_history": [entry.to_dict() for entry in self.apply_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperRepairSession:
        return cls(
            source_run_dir=str(data.get("source_run_dir", "")),
            session_dir=str(data.get("session_dir", "")),
            workspace_dir=str(data.get("workspace_dir", "")),
            created_at=str(data.get("created_at", "")),
            tracked_items=tuple(
                TrackedItem.from_dict(item)
                for item in data.get("tracked_items", [])
                if isinstance(item, dict)
            ),
            apply_history=tuple(
                ApplyEntry.from_dict(item)
                for item in data.get("apply_history", [])
                if isinstance(item, dict)
            ),
        )


def init_paper_repair(
    run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    source_run_dir = Path(run_dir).resolve()
    if not source_run_dir.exists():
        raise PaperRepairError(f"Run directory not found: {source_run_dir}")

    tracked_items = _collect_tracked_items(source_run_dir)
    if not tracked_items:
        raise PaperRepairError(
            "No paper-export artifacts found under stage-22 or stage-23. "
            "Expected files such as paper.tex or paper_final_verified.md."
        )

    session_dir = Path(output_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = session_dir / "workspace"
    if workspace_dir.exists():
        raise PaperRepairError(
            f"Repair workspace already exists: {workspace_dir}. "
            "Use a fresh output directory for each repair session."
        )
    workspace_dir.mkdir(parents=True, exist_ok=False)

    for item in tracked_items:
        if not item.exists:
            continue
        source_path = source_run_dir / item.relative_path
        target_path = workspace_dir / item.relative_path
        _copy_path(source_path, target_path, item.kind)

    session = PaperRepairSession(
        source_run_dir=str(source_run_dir),
        session_dir=str(session_dir),
        workspace_dir=str(workspace_dir),
        created_at=_utc_now(),
        tracked_items=tracked_items,
    )

    session_json = session_dir / "paper-repair.json"
    session_json.write_text(dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
    readme_path = session_dir / "README.md"
    readme_path.write_text(_render_repair_readme(session), encoding="utf-8")

    return {
        "session_json": str(session_json),
        "readme": str(readme_path),
        "workspace": str(workspace_dir),
    }


def apply_paper_repair(
    session_json_path: str | Path,
    *,
    note: str | None = None,
) -> dict[str, str]:
    session_path = Path(session_json_path).resolve()
    session = _load_session(session_path)
    source_run_dir = Path(session.source_run_dir)
    workspace_dir = Path(session.workspace_dir)
    if not workspace_dir.exists():
        raise PaperRepairError(f"Repair workspace not found: {workspace_dir}")

    backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = Path(session.session_dir) / "backups" / backup_id
    backup_dir.mkdir(parents=True, exist_ok=False)

    for item in session.tracked_items:
        target_path = source_run_dir / item.relative_path
        backup_path = backup_dir / item.relative_path
        if target_path.exists():
            _copy_path(target_path, backup_path, _path_kind(target_path))

        workspace_path = workspace_dir / item.relative_path
        if workspace_path.exists():
            _copy_path(workspace_path, target_path, _path_kind(workspace_path))

    entry = ApplyEntry(
        backup_id=backup_id,
        applied_at=_utc_now(),
        note=(note or "").strip(),
        backup_dir=str(backup_dir),
    )
    rewritten = PaperRepairSession(
        source_run_dir=session.source_run_dir,
        session_dir=session.session_dir,
        workspace_dir=session.workspace_dir,
        created_at=session.created_at,
        tracked_items=session.tracked_items,
        apply_history=session.apply_history + (entry,),
    )
    session_path.write_text(dumps(rewritten.to_dict(), indent=2) + "\n", encoding="utf-8")
    return {
        "session_json": str(session_path),
        "backup_dir": str(backup_dir),
        "published_run_dir": str(source_run_dir),
    }


def rollback_paper_repair(
    session_json_path: str | Path,
    *,
    backup_id: str | None = None,
) -> dict[str, str]:
    session_path = Path(session_json_path).resolve()
    session = _load_session(session_path)
    if not session.apply_history:
        raise PaperRepairError("No published repair exists yet, so there is nothing to roll back.")

    entry = _select_backup_entry(session, backup_id)
    backup_dir = Path(entry.backup_dir)
    if not backup_dir.exists():
        raise PaperRepairError(f"Backup directory not found: {backup_dir}")

    source_run_dir = Path(session.source_run_dir)
    for item in session.tracked_items:
        source_path = backup_dir / item.relative_path
        target_path = source_run_dir / item.relative_path
        if source_path.exists():
            _copy_path(source_path, target_path, _path_kind(source_path))
        elif not item.exists and target_path.exists():
            _remove_path(target_path)

    remaining_history = tuple(
        history_entry
        for history_entry in session.apply_history
        if history_entry.backup_id != entry.backup_id
    )
    rewritten = PaperRepairSession(
        source_run_dir=session.source_run_dir,
        session_dir=session.session_dir,
        workspace_dir=session.workspace_dir,
        created_at=session.created_at,
        tracked_items=session.tracked_items,
        apply_history=remaining_history,
    )
    session_path.write_text(dumps(rewritten.to_dict(), indent=2) + "\n", encoding="utf-8")
    return {
        "session_json": str(session_path),
        "rolled_back_backup": entry.backup_id,
        "published_run_dir": str(source_run_dir),
    }


def _collect_tracked_items(run_dir: Path) -> tuple[TrackedItem, ...]:
    items: list[TrackedItem] = []
    for stage_name, relative_paths in TRACKED_STAGE_PATHS.items():
        stage_dir = run_dir / stage_name
        if not stage_dir.exists():
            continue
        for relative_path in relative_paths:
            full_path = stage_dir / relative_path
            items.append(
                TrackedItem(
                    relative_path=f"{stage_name}/{relative_path}",
                    kind="directory" if full_path.is_dir() else "file",
                    exists=full_path.exists(),
                )
            )
    return tuple(items)


def _load_session(session_json_path: Path) -> PaperRepairSession:
    if not session_json_path.exists():
        raise PaperRepairError(f"Paper repair JSON not found: {session_json_path}")
    data = loads(session_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PaperRepairError("Paper repair JSON must decode to a mapping.")
    return PaperRepairSession.from_dict(data)


def _select_backup_entry(
    session: PaperRepairSession,
    backup_id: str | None,
) -> ApplyEntry:
    if backup_id:
        for entry in session.apply_history:
            if entry.backup_id == backup_id:
                return entry
        raise PaperRepairError(f"Backup id not found in repair session: {backup_id}")
    return session.apply_history[-1]


def _copy_path(source_path: Path, target_path: Path, kind: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        _remove_path(target_path)
    if kind == "directory":
        copytree(source_path, target_path)
        return
    copy2(source_path, target_path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        rmtree(path)
        return
    path.unlink()


def _path_kind(path: Path) -> str:
    return "directory" if path.is_dir() else "file"


def _render_repair_readme(session: PaperRepairSession) -> str:
    lines = [
        "# Paper Repair Workspace",
        "",
        "This workspace is a post-export repair lane for a completed AutoResearchClaw run.",
        "",
        f"- Source run: `{session.source_run_dir}`",
        f"- Created at: `{session.created_at}`",
        f"- Workspace root: `{session.workspace_dir}`",
        "",
        "## Tracked Artifacts",
    ]
    for item in session.tracked_items:
        state = "present" if item.exists else "missing in source run"
        lines.append(f"- `{item.relative_path}` ({item.kind}, {state})")
    lines.extend(
        [
            "",
            "## Workflow",
            "1. Edit files under `workspace/`.",
            "2. Publish repairs back to the source run with:",
            "   `python -m autoresearchclaw paper-repair-apply --repair-json <session>/paper-repair.json`",
            "3. If needed, roll back the most recent publish with:",
            "   `python -m autoresearchclaw paper-repair-rollback --repair-json <session>/paper-repair.json`",
            "",
            "Each publish snapshots the original files under `backups/<timestamp>/` before overwriting them.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
