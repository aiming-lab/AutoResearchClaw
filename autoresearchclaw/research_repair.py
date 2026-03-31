from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from json import dumps, loads
import os
from pathlib import Path
from shutil import copy2, copytree
import subprocess
import sys
from typing import Any

import yaml


class ResearchRepairError(ValueError):
    """Raised when a research-repair session cannot be created or prepared."""


STAGE_NAME_BY_NUMBER: dict[int, str] = {
    1: "TOPIC_INIT",
    2: "PROBLEM_DECOMPOSE",
    3: "SEARCH_STRATEGY",
    4: "LITERATURE_COLLECT",
    5: "LITERATURE_SCREEN",
    6: "KNOWLEDGE_EXTRACT",
    7: "SYNTHESIS",
    8: "HYPOTHESIS_GEN",
    9: "EXPERIMENT_DESIGN",
    10: "CODE_GENERATION",
    11: "RESOURCE_PLANNING",
    12: "EXPERIMENT_RUN",
    13: "ITERATIVE_REFINE",
    14: "RESULT_ANALYSIS",
    15: "RESEARCH_DECISION",
    16: "PAPER_OUTLINE",
    17: "PAPER_DRAFT",
    18: "PEER_REVIEW",
    19: "PAPER_REVISION",
    20: "QUALITY_GATE",
    21: "KNOWLEDGE_ARCHIVE",
    22: "EXPORT_PUBLISH",
    23: "CITATION_VERIFY",
}
STAGE_NUMBER_BY_NAME: dict[str, int] = {
    name.upper(): number for number, name in STAGE_NAME_BY_NUMBER.items()
}

FIXED_CONTEXT_PATHS: tuple[str, ...] = (
    "checkpoint.json",
    "pipeline_summary.json",
    "experiment_diagnosis.json",
    "repair_prompt.txt",
    "quality_warning.txt",
    "experiment_summary_best.json",
    "analysis_best.md",
    "stage-09/exp_plan.yaml",
    "stage-12/runs/results.json",
    "stage-20/quality_report.json",
    "stage-23/paper_final_verified.md",
    "stage-23/verification_report.json",
)
LATEST_GLOB_PATHS: tuple[str, ...] = (
    "stage-14*/experiment_summary.json",
    "stage-14*/analysis.md",
    "stage-15*/decision.md",
)

WSL_PASSTHROUGH_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
)
REPAIR_RUN_ROOT_ENV_VAR = "AUTORESEARCHCLAW_REPAIR_RUN_ROOT"


@dataclass(frozen=True)
class ContextItem:
    relative_path: str
    kind: str
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextItem:
        return cls(
            relative_path=str(data.get("relative_path", "")),
            kind=str(data.get("kind", "file")),
            exists=bool(data.get("exists", False)),
        )


@dataclass(frozen=True)
class LaunchEntry:
    launched_at: str
    child_run_dir: str
    generated_config_path: str
    launch_script: str
    command_preview: str
    target_stage_name: str
    target_stage_number: int
    launch_log: str = ""
    inherited_stage_dirs: tuple[str, ...] = field(default_factory=tuple)
    executed: bool = False
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaunchEntry:
        return cls(
            launched_at=str(data.get("launched_at", "")),
            child_run_dir=str(data.get("child_run_dir", "")),
            generated_config_path=str(data.get("generated_config_path", "")),
            launch_script=str(data.get("launch_script", "")),
            launch_log=str(data.get("launch_log", "")),
            command_preview=str(data.get("command_preview", "")),
            target_stage_name=str(data.get("target_stage_name", "")),
            target_stage_number=int(data.get("target_stage_number", 0)),
            inherited_stage_dirs=tuple(data.get("inherited_stage_dirs") or ()),
            executed=bool(data.get("executed", False)),
            pid=int(data["pid"]) if data.get("pid") is not None else None,
        )


@dataclass(frozen=True)
class ReusePolicy:
    hard_reuse_stage_dirs: tuple[str, ...] = field(default_factory=tuple)
    soft_context_paths: tuple[str, ...] = field(default_factory=tuple)
    rerun_from_stage_name: str = ""
    rerun_from_stage_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReusePolicy:
        return cls(
            hard_reuse_stage_dirs=tuple(data.get("hard_reuse_stage_dirs") or ()),
            soft_context_paths=tuple(data.get("soft_context_paths") or ()),
            rerun_from_stage_name=str(data.get("rerun_from_stage_name", "")),
            rerun_from_stage_number=int(data.get("rerun_from_stage_number", 0)),
        )


@dataclass(frozen=True)
class ResearchRepairSession:
    source_run_dir: str
    source_run_id: str
    session_dir: str
    workspace_dir: str
    created_at: str
    base_config_path: str
    upstream_root: str
    target_stage_name: str
    target_stage_number: int
    repair_reason: str
    context_items: tuple[ContextItem, ...]
    feedback_path: str
    reuse_policy: ReusePolicy
    launch_history: tuple[LaunchEntry, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_dir": self.source_run_dir,
            "source_run_id": self.source_run_id,
            "session_dir": self.session_dir,
            "workspace_dir": self.workspace_dir,
            "created_at": self.created_at,
            "base_config_path": self.base_config_path,
            "upstream_root": self.upstream_root,
            "target_stage_name": self.target_stage_name,
            "target_stage_number": self.target_stage_number,
            "repair_reason": self.repair_reason,
            "context_items": [item.to_dict() for item in self.context_items],
            "feedback_path": self.feedback_path,
            "reuse_policy": self.reuse_policy.to_dict(),
            "launch_history": [item.to_dict() for item in self.launch_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchRepairSession:
        return cls(
            source_run_dir=str(data.get("source_run_dir", "")),
            source_run_id=str(data.get("source_run_id", "")),
            session_dir=str(data.get("session_dir", "")),
            workspace_dir=str(data.get("workspace_dir", "")),
            created_at=str(data.get("created_at", "")),
            base_config_path=str(data.get("base_config_path", "")),
            upstream_root=str(data.get("upstream_root", "")),
            target_stage_name=str(data.get("target_stage_name", "")),
            target_stage_number=int(data.get("target_stage_number", 0)),
            repair_reason=str(data.get("repair_reason", "")),
            context_items=tuple(
                ContextItem.from_dict(item)
                for item in data.get("context_items", [])
                if isinstance(item, dict)
            ),
            feedback_path=str(data.get("feedback_path", "")),
            reuse_policy=ReusePolicy.from_dict(
                data.get("reuse_policy") if isinstance(data.get("reuse_policy"), dict) else {}
            ),
            launch_history=tuple(
                LaunchEntry.from_dict(item)
                for item in data.get("launch_history", [])
                if isinstance(item, dict)
            ),
        )


def init_research_repair(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "config.arc.yaml",
    target_stage: str = "EXPERIMENT_DESIGN",
    reason: str | None = None,
    feedback: list[str] | tuple[str, ...] = (),
    upstream_root: str | Path = ".",
) -> dict[str, str]:
    source_run_dir = Path(run_dir).resolve()
    if not source_run_dir.exists():
        raise ResearchRepairError(f"Run directory not found: {source_run_dir}")

    base_config_path = Path(config_path).resolve()
    if not base_config_path.exists():
        raise ResearchRepairError(f"Config not found: {base_config_path}")

    upstream_root_path = Path(upstream_root).resolve()
    if not upstream_root_path.exists():
        raise ResearchRepairError(f"Upstream root not found: {upstream_root_path}")

    stage_number, stage_name = _normalize_stage_ref(target_stage)
    source_run_id = _read_source_run_id(source_run_dir)
    context_items = _collect_context_items(source_run_dir)
    reuse_policy = _build_reuse_policy(
        context_items=context_items,
        target_stage_number=stage_number,
        target_stage_name=stage_name,
    )

    session_dir = Path(output_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = session_dir / "workspace"
    if workspace_dir.exists():
        raise ResearchRepairError(
            f"Research repair workspace already exists: {workspace_dir}. "
            "Use a fresh output directory for each repair session."
        )
    workspace_dir.mkdir(parents=True, exist_ok=False)

    context_root = workspace_dir / "context"
    for item in context_items:
        if not item.exists:
            continue
        source_path = source_run_dir / item.relative_path
        target_path = context_root / item.relative_path
        _copy_path(source_path, target_path, item.kind)

    repair_config_path = workspace_dir / "repair-config.yaml"
    copy2(base_config_path, repair_config_path)

    feedback_path = workspace_dir / "feedback.md"
    feedback_path.write_text(
        _render_feedback_template(
            source_run_id=source_run_id,
            target_stage_name=stage_name,
            reason=(reason or "").strip(),
            feedback=list(feedback),
        ),
        encoding="utf-8",
    )

    repair_reason = (
        (reason or "").strip()
        or "Human review concluded that the completed run needs more data, more experiments, or stronger protocol coverage."
    )
    session = ResearchRepairSession(
        source_run_dir=str(source_run_dir),
        source_run_id=source_run_id,
        session_dir=str(session_dir),
        workspace_dir=str(workspace_dir),
        created_at=_utc_now(),
        base_config_path=str(base_config_path),
        upstream_root=str(upstream_root_path),
        target_stage_name=stage_name,
        target_stage_number=stage_number,
        repair_reason=repair_reason,
        context_items=context_items,
        feedback_path=str(feedback_path),
        reuse_policy=reuse_policy,
    )

    session_json = session_dir / "research-repair.json"
    session_json.write_text(dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
    readme_path = session_dir / "README.md"
    readme_path.write_text(_render_repair_readme(session), encoding="utf-8")

    return {
        "session_json": str(session_json),
        "readme": str(readme_path),
        "workspace": str(workspace_dir),
        "feedback": str(feedback_path),
        "repair_config": str(repair_config_path),
    }


def prepare_research_repair_run(
    session_json_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    extra_feedback: list[str] | tuple[str, ...] = (),
    auto_approve: bool = False,
    skip_preflight: bool = False,
    execute: bool = False,
) -> dict[str, str]:
    session_path = Path(session_json_path).resolve()
    session = _load_session(session_path)
    workspace_dir = Path(session.workspace_dir)
    repair_config_path = workspace_dir / "repair-config.yaml"
    if not repair_config_path.exists():
        raise ResearchRepairError(f"Repair config not found: {repair_config_path}")

    feedback_text = _read_feedback(Path(session.feedback_path), extra_feedback)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    child_run_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _default_child_run_dir(session, timestamp=timestamp)
    )
    if child_run_dir.exists() and any(child_run_dir.iterdir()):
        raise ResearchRepairError(
            f"Child run directory already exists and is not empty: {child_run_dir}"
        )
    child_run_dir.mkdir(parents=True, exist_ok=True)
    inherited_stage_dirs = _copy_prerequisite_stage_dirs(
        source_run_dir=Path(session.source_run_dir),
        child_run_dir=child_run_dir,
        target_stage_number=session.target_stage_number,
    )

    generated_dir = Path(session.session_dir) / "generated-runs" / timestamp
    generated_dir.mkdir(parents=True, exist_ok=False)

    config_data = _load_yaml(repair_config_path)
    generated_config_path = generated_dir / "repair-config.generated.yaml"
    _write_generated_config(
        config_data,
        generated_config_path,
        session=session,
        feedback_text=feedback_text,
        child_run_dir=child_run_dir,
    )

    metadata = {
        "generated_at": _utc_now(),
        "parent_run_dir": session.source_run_dir,
        "parent_run_id": session.source_run_id,
        "target_stage_name": session.target_stage_name,
        "target_stage_number": session.target_stage_number,
        "repair_reason": session.repair_reason,
        "feedback_path": session.feedback_path,
        "feedback_excerpt": feedback_text[:1200],
        "generated_config_path": str(generated_config_path),
        "inherited_stage_dirs": list(inherited_stage_dirs),
        "reuse_policy": session.reuse_policy.to_dict(),
        "soft_context_note": (
            "Parent-run downstream analysis/draft artifacts are provided only as "
            "reference context. They are not authoritative outputs for this child run."
        ),
        "compact_repair_brief": _build_compact_repair_brief(
            session=session,
            feedback_text=feedback_text,
        ),
    }
    metadata_path = child_run_dir / "research_repair_parent.json"
    metadata_path.write_text(dumps(metadata, indent=2) + "\n", encoding="utf-8")

    inner_command = _build_inner_launch_command(
        upstream_root=Path(session.upstream_root),
        generated_config_path=generated_config_path,
        child_run_dir=child_run_dir,
        stage_name=session.target_stage_name,
        auto_approve=auto_approve,
        skip_preflight=skip_preflight,
    )
    command_preview = _wrap_launch_command_for_display(inner_command)
    launch_script = generated_dir / "launch.sh"
    launch_script.write_text(command_preview + "\n", encoding="utf-8")
    launch_log = generated_dir / "launch.log"

    pid: int | None = None
    if execute:
        pid = _launch_command(
            inner_command,
            launch_log,
            upstream_root=Path(session.upstream_root),
        )

    launch_entry = LaunchEntry(
        launched_at=_utc_now(),
        child_run_dir=str(child_run_dir),
        generated_config_path=str(generated_config_path),
        launch_script=str(launch_script),
        launch_log=str(launch_log),
        command_preview=command_preview,
        target_stage_name=session.target_stage_name,
        target_stage_number=session.target_stage_number,
        inherited_stage_dirs=inherited_stage_dirs,
        executed=execute,
        pid=pid,
    )
    rewritten = ResearchRepairSession(
        source_run_dir=session.source_run_dir,
        source_run_id=session.source_run_id,
        session_dir=session.session_dir,
        workspace_dir=session.workspace_dir,
        created_at=session.created_at,
        base_config_path=session.base_config_path,
        upstream_root=session.upstream_root,
        target_stage_name=session.target_stage_name,
        target_stage_number=session.target_stage_number,
        repair_reason=session.repair_reason,
        context_items=session.context_items,
        feedback_path=session.feedback_path,
        reuse_policy=session.reuse_policy,
        launch_history=session.launch_history + (launch_entry,),
    )
    session_path.write_text(dumps(rewritten.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "session_json": str(session_path),
        "child_run_dir": str(child_run_dir),
        "generated_config": str(generated_config_path),
        "launch_script": str(launch_script),
        "launch_log": str(launch_log),
        "command_preview": command_preview,
        "metadata": str(metadata_path),
        "pid": "" if pid is None else str(pid),
    }


def _normalize_stage_ref(stage_ref: str) -> tuple[int, str]:
    raw = str(stage_ref).strip()
    if not raw:
        raise ResearchRepairError("Target stage must not be empty.")
    if raw.isdigit():
        stage_number = int(raw)
        stage_name = STAGE_NAME_BY_NUMBER.get(stage_number)
        if stage_name is None:
            raise ResearchRepairError(f"Unknown stage number: {stage_number}")
        return stage_number, stage_name
    stage_name = raw.upper()
    stage_number = STAGE_NUMBER_BY_NAME.get(stage_name)
    if stage_number is None:
        valid = ", ".join(STAGE_NAME_BY_NUMBER.values())
        raise ResearchRepairError(
            f"Unknown stage name '{raw}'. Valid stage names: {valid}"
        )
    return stage_number, stage_name


def _read_source_run_id(run_dir: Path) -> str:
    if run_dir.name.strip():
        return run_dir.name.strip()
    summary_path = run_dir / "pipeline_summary.json"
    if summary_path.exists():
        try:
            data = loads(summary_path.read_text(encoding="utf-8"))
            run_id = data.get("run_id")
            if isinstance(run_id, str) and run_id.strip():
                return run_id.strip()
        except (OSError, ValueError):
            pass
    return run_dir.name


def _collect_context_items(run_dir: Path) -> tuple[ContextItem, ...]:
    relative_paths: list[str] = []
    for relative_path in FIXED_CONTEXT_PATHS:
        relative_paths.append(relative_path)
    for pattern in LATEST_GLOB_PATHS:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            relative_paths.append(matches[-1].relative_to(run_dir).as_posix())

    deduped: list[str] = []
    seen: set[str] = set()
    for relative_path in relative_paths:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        deduped.append(relative_path)

    items: list[ContextItem] = []
    for relative_path in deduped:
        full_path = run_dir / relative_path
        items.append(
            ContextItem(
                relative_path=relative_path,
                kind="directory" if full_path.is_dir() else "file",
                exists=full_path.exists(),
            )
        )
    return tuple(items)


def _build_reuse_policy(
    *,
    context_items: tuple[ContextItem, ...],
    target_stage_number: int,
    target_stage_name: str,
) -> ReusePolicy:
    hard_reuse = tuple(
        f"stage-{number:02d}" for number in range(1, target_stage_number)
    )
    soft_context: list[str] = []
    for item in context_items:
        rel = item.relative_path
        if rel.startswith("stage-"):
            stage_prefix = rel.split("/", 1)[0]
            number_text = stage_prefix.replace("stage-", "").split("_", 1)[0]
            number_text = number_text.split("-", 1)[0]
            try:
                stage_number = int(number_text)
            except ValueError:
                stage_number = 0
            if stage_number >= target_stage_number:
                soft_context.append(rel)
        elif rel in {
            "checkpoint.json",
            "pipeline_summary.json",
            "experiment_diagnosis.json",
            "experiment_summary_best.json",
            "analysis_best.md",
            "repair_prompt.txt",
        }:
            soft_context.append(rel)
    deduped_soft: list[str] = []
    seen: set[str] = set()
    for rel in soft_context:
        if rel in seen:
            continue
        seen.add(rel)
        deduped_soft.append(rel)
    return ReusePolicy(
        hard_reuse_stage_dirs=hard_reuse,
        soft_context_paths=tuple(deduped_soft),
        rerun_from_stage_name=target_stage_name,
        rerun_from_stage_number=target_stage_number,
    )


def _render_feedback_template(
    *,
    source_run_id: str,
    target_stage_name: str,
    reason: str,
    feedback: list[str],
) -> str:
    lines = [
        "# Research Repair Feedback",
        "",
        f"- Parent run: `{source_run_id}`",
        f"- Target stage: `{target_stage_name}`",
        f"- Reason: `{reason or 'Add the human repair reason here.'}`",
        "",
        "## Human Repair Request",
    ]
    if feedback:
        for item in feedback:
            item_text = str(item).strip()
            if item_text:
                lines.append(f"- {item_text}")
    else:
        lines.extend(
            [
                "- State exactly what was insufficient in the completed run.",
                "- Say what must be added: more data, more seeds, more conditions, or stronger protocol checks.",
                "- If real local assets are required, say so explicitly.",
                "- If the previous run should be considered invalid unless those changes happen, say that too.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_repair_readme(session: ResearchRepairSession) -> str:
    lines = [
        "# Research Repair Workspace",
        "",
        "This workspace is for run-level repair, not post-export paper cleanup.",
        "",
        f"- Source run: `{session.source_run_dir}`",
        f"- Source run id: `{session.source_run_id}`",
        f"- Target rollback stage: `{session.target_stage_name}`",
        f"- Base config: `{session.base_config_path}`",
        f"- Created at: `{session.created_at}`",
        "",
        "## Reuse Policy",
        "- Hard reuse: parent stages before the target stage are copied directly into the child run.",
        f"- Hard-reused stage dirs: `{', '.join(session.reuse_policy.hard_reuse_stage_dirs)}`",
        "- Soft reuse: downstream analysis / decision / paper artifacts are copied into `workspace/context/` only as draft reference material.",
        f"- Soft-context artifacts: `{', '.join(session.reuse_policy.soft_context_paths)}`",
        "- Authoritative rerun boundary: all stages from the target stage onward must be regenerated from the new evidence.",
        "",
        "## What This Is For",
        "- Human review says the completed run is not strong enough yet.",
        "- Instead of only editing the exported paper, create a child run that goes back to the experiment stages.",
        "- Typical reasons: not enough data, not enough seeds, wrong protocol, or real assets were not used.",
        "",
        "## Workspace Files",
        f"- `workspace/repair-config.yaml`: editable config seed for the child run.",
        f"- `workspace/feedback.md`: human repair instructions that will be preserved in child-run repair metadata and exposed as a compact repair brief.",
        f"- `workspace/context/`: copied reference artifacts from the parent run.",
        "",
        "## Workflow",
        "1. Edit `workspace/feedback.md` and, if needed, `workspace/repair-config.yaml`.",
        "2. Prepare a child run with:",
        "   `python -m autoresearchclaw research-repair-run --repair-json <session>/research-repair.json`",
        "3. Add `--execute` only when you explicitly want to launch the new upstream run.",
        "",
        "The prepared child run keeps a parent pointer via `research_repair_parent.json` so the repair lineage stays auditable.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _load_session(session_json_path: Path) -> ResearchRepairSession:
    if not session_json_path.exists():
        raise ResearchRepairError(f"Research repair JSON not found: {session_json_path}")
    data = loads(session_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ResearchRepairError("Research repair JSON must decode to a mapping.")
    return ResearchRepairSession.from_dict(data)


def _read_feedback(feedback_path: Path, extra_feedback: list[str] | tuple[str, ...]) -> str:
    if not feedback_path.exists():
        raise ResearchRepairError(f"Feedback file not found: {feedback_path}")
    feedback_text = feedback_path.read_text(encoding="utf-8").strip()
    extras = [str(item).strip() for item in extra_feedback if str(item).strip()]
    if extras:
        feedback_text = feedback_text.rstrip() + "\n\n## CLI Additions\n" + "\n".join(
            f"- {item}" for item in extras
        )
    return feedback_text.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResearchRepairError(f"Could not read YAML config: {path}") from exc
    if not isinstance(data, dict):
        raise ResearchRepairError(f"Config must decode to a mapping: {path}")
    return data


def _write_generated_config(
    config_data: dict[str, Any],
    target_path: Path,
    *,
    session: ResearchRepairSession,
    feedback_text: str,
    child_run_dir: Path,
) -> None:
    research = config_data.setdefault("research", {})
    if not isinstance(research, dict):
        raise ResearchRepairError("Config field `research` must be a mapping.")
    original_topic = str(research.get("topic", "")).strip()
    research["topic"] = _build_repair_topic(
        session=session,
        original_topic=original_topic,
        feedback_text=feedback_text,
    )
    project = config_data.setdefault("project", {})
    if isinstance(project, dict):
        project_name = str(project.get("name", "research-repair")).strip() or "research-repair"
        if not project_name.endswith("-repair"):
            project["name"] = f"{project_name}-repair"
    _apply_repair_runtime_defaults(
        config_data,
        session=session,
        generated_config_path=target_path,
        child_run_dir=child_run_dir,
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _build_repair_topic(
    *,
    session: ResearchRepairSession,
    original_topic: str,
    feedback_text: str,
) -> str:
    base_topic = _normalize_single_line(original_topic)
    if not base_topic:
        base_topic = (
            "Engineering-drawing circle localization with explicit rule evidence "
            "and learned heatmaps."
        )

    repair_focus = _first_repair_focus_line(feedback_text) or _normalize_single_line(
        session.repair_reason
    )
    if repair_focus:
        return (
            f"{base_topic}\n\n"
            f"Repair focus: rerun from {session.target_stage_name} and strengthen "
            f"{repair_focus}."
        ).strip()
    return (
        f"{base_topic}\n\n"
        f"Repair focus: rerun from {session.target_stage_name} with stronger "
        "real-data coverage, seeds, and experiment protocol."
    ).strip()


def _normalize_single_line(text: str) -> str:
    normalized = " ".join(str(text).split()).strip()
    return normalized


def _first_repair_focus_line(feedback_text: str) -> str:
    for raw_line in feedback_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("- "):
            continue
        payload = line[2:].strip()
        lowered = payload.lower()
        if lowered.startswith(("parent run:", "target stage:", "reason:")):
            continue
        return _normalize_single_line(payload)
    return ""


def _apply_repair_runtime_defaults(
    config_data: dict[str, Any],
    *,
    session: ResearchRepairSession,
    generated_config_path: Path,
    child_run_dir: Path,
) -> None:
    llm = config_data.setdefault("llm", {})
    if isinstance(llm, dict):
        acp = llm.setdefault("acp", {})
        if isinstance(acp, dict):
            timestamp = generated_config_path.parent.name.strip() or "repair"
            stage_slug = session.target_stage_name.lower().replace("_", "-")
            acp["session_name"] = f"researchclaw-{stage_slug}-{timestamp}"
            current_timeout = _safe_int(acp.get("timeout_sec"), 1800)
            acp["timeout_sec"] = max(current_timeout, 3200)
            current_retries = _safe_int(acp.get("reconnect_retries"), 2)
            acp["reconnect_retries"] = max(current_retries, 6)
            acp["reconnect_backoff_sec"] = 3.0
            acp["verbose"] = True
            acp["capture_status_on_failure"] = True
            acp["archive_failed_prompt_files"] = True
            acp["debug_log_path"] = _to_wsl_path(child_run_dir / "acp_debug.jsonl")
            if session.target_stage_number >= STAGE_NUMBER_BY_NAME["CODE_GENERATION"]:
                acp["stateless_prompt"] = True

    experiment = config_data.setdefault("experiment", {})
    if isinstance(experiment, dict):
        code_agent = experiment.setdefault("code_agent", {})
        if isinstance(code_agent, dict):
            code_agent["architecture_planning"] = False
            code_agent["review_max_rounds"] = 0
            if session.target_stage_number >= STAGE_NUMBER_BY_NAME["CODE_GENERATION"]:
                code_agent["fallback_to_legacy_on_acp_failure"] = False


def _default_child_run_dir(
    session: ResearchRepairSession,
    *,
    timestamp: str,
) -> Path:
    suffix = f"{session.source_run_id}-repair-{timestamp}"
    override = os.environ.get(REPAIR_RUN_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override).resolve() / suffix

    if sys.platform.startswith("win"):
        detected_root = _detect_windows_wsl_run_root()
        if detected_root is not None:
            return detected_root / suffix

    return Path(session.source_run_dir).resolve().parent / suffix


def _detect_windows_wsl_run_root() -> Path | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        probe = subprocess.run(
            [
                "wsl",
                "bash",
                "-lc",
                'mkdir -p "$HOME/.autoresearchclaw/artifacts" && wslpath -w "$HOME/.autoresearchclaw/artifacts"',
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return None
    if probe.returncode != 0:
        return None
    output = (probe.stdout or "").strip().splitlines()
    if not output:
        return None
    candidate = output[-1].strip()
    if not candidate:
        return None
    return Path(candidate).resolve()


def _safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_inner_launch_command(
    *,
    upstream_root: Path,
    generated_config_path: Path,
    child_run_dir: Path,
    stage_name: str,
    auto_approve: bool,
    skip_preflight: bool,
) -> str:
    upstream_root_wsl = _to_wsl_path(upstream_root)
    generated_config_wsl = _to_wsl_path(generated_config_path)
    child_run_dir_wsl = _to_wsl_path(child_run_dir)
    exe_wsl = _to_wsl_path(upstream_root / ".venv" / "bin" / "researchclaw")
    command_parts = [f"cd { _sh_quote(upstream_root_wsl) }", 'export PATH="$HOME/bin:$PATH"']
    tmp_bin = upstream_root / ".tmp_bin"
    if tmp_bin.exists():
        tmp_bin_wsl = _to_wsl_path(tmp_bin)
        command_parts.append(
            f"export PATH={_sh_quote(tmp_bin_wsl)}:\"$PATH\""
        )
    command_parts.append(
        f"{_sh_quote(exe_wsl)} run --config {_sh_quote(generated_config_wsl)} --output {_sh_quote(child_run_dir_wsl)} --from-stage {stage_name}"
    )
    if auto_approve:
        command_parts[-1] += " --auto-approve"
    if skip_preflight:
        command_parts[-1] += " --skip-preflight"
    inner = " && ".join(command_parts)
    return inner


def _build_compact_repair_brief(
    *,
    session: ResearchRepairSession,
    feedback_text: str,
) -> str:
    feedback_lines: list[str] = []
    for raw_line in feedback_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        payload = line[2:].strip()
        lowered = payload.lower()
        if lowered.startswith(("parent run:", "target stage:", "reason:")):
            continue
        feedback_lines.append(_normalize_single_line(payload))
        if len(feedback_lines) >= 5:
            break

    lines = [
        "## Repair Context",
        f"- Parent run: `{session.source_run_id}`",
        f"- Authoritative rerun starts at: `{session.target_stage_name}`",
        f"- Repair reason: {_normalize_single_line(session.repair_reason)}",
    ]
    if session.reuse_policy.hard_reuse_stage_dirs:
        lines.append(
            "- Hard reuse: "
            + ", ".join(session.reuse_policy.hard_reuse_stage_dirs)
        )
    lines.append(
        "- Downstream parent analysis and paper artifacts are soft context only."
    )
    if feedback_lines:
        lines.append("- Human requirements:")
        lines.extend(f"- {item}" for item in feedback_lines)
    return "\n".join(lines).strip()


def _wrap_launch_command_for_display(inner_command: str) -> str:
    if sys.platform.startswith("win"):
        return f"wsl bash -lc {_sh_quote(inner_command)}"
    return f"bash -lc {_sh_quote(inner_command)}"


def _launch_command(
    inner_command: str,
    launch_log: Path,
    *,
    upstream_root: Path | None = None,
) -> int:
    launch_env, forwarded_env = _build_launch_env(upstream_root=upstream_root)
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    with launch_log.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {inner_command}\n\n")
        if forwarded_env:
            log_handle.write(
                "# Forwarded to child process via WSLENV: "
                + ", ".join(forwarded_env)
                + "\n\n"
            )
        log_handle.flush()
        if sys.platform.startswith("win"):
            process = subprocess.Popen(
                ["wsl", "bash", "-lc", inner_command],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=launch_env,
            )
        else:
            process = subprocess.Popen(
                ["bash", "-lc", inner_command],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=launch_env,
            )
    return int(process.pid)


def _build_launch_env(
    *,
    upstream_root: Path | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    env = dict(os.environ)
    forwarded: list[str] = []
    for name in WSL_PASSTHROUGH_ENV_VARS:
        value = env.get(name, "")
        if value:
            forwarded.append(name)

    asset_env = _discover_runtime_asset_env(upstream_root)
    for name, value in asset_env.items():
        if value:
            env[name] = value
            forwarded.append(name)

    if not sys.platform.startswith("win"):
        return env, tuple(_dedupe_preserve_order(forwarded))

    forwarded = _dedupe_preserve_order(forwarded)
    if not forwarded:
        return env, ()

    wslenv_entries = [item for item in env.get("WSLENV", "").split(":") if item]
    existing_names = {item.split("/", 1)[0] for item in wslenv_entries}
    for name in forwarded:
        if name not in existing_names:
            wslenv_entries.append(name)
    env["WSLENV"] = ":".join(wslenv_entries)
    return env, tuple(forwarded)


def _discover_runtime_asset_env(upstream_root: Path | None) -> dict[str, str]:
    if upstream_root is None:
        return {}
    if sys.platform.startswith("win"):
        return _discover_runtime_asset_env_via_wsl(upstream_root)
    return {}


def _discover_runtime_asset_env_via_wsl(upstream_root: Path) -> dict[str, str]:
    upstream_root_wsl = _to_wsl_path(upstream_root)
    exe_wsl = _to_wsl_path(upstream_root / ".venv" / "bin" / "python")
    script = r"""
from config import build_default_config
from pathlib import Path
import json

cfg = build_default_config()
specs = cfg.build_dataset_specs()
payload = {"VECTRA_REPO_ROOT": str(Path.cwd())}

simple = specs.get("engineering_primitives_simple_scenes_noslot_v1_local_20260326", {})
if isinstance(simple, dict):
    payload["VECTRA_SIMPLE_DATASET_ROOT"] = str(simple.get("dataset_root", ""))
    payload["VECTRA_SIMPLE_ASSET_ROOT"] = str(simple.get("dataset_root", ""))
    payload["VECTRA_SIMPLE_MANIFEST_PATH"] = str(simple.get("manifest_path", ""))
    caches = simple.get("cache_roots", {})
    if isinstance(caches, dict):
        payload["VECTRA_SIMPLE_HEATMAP_DIR"] = str(caches.get("learned", ""))

page = specs.get("page_minus_titleblock", {})
if isinstance(page, dict):
    page_root = Path(str(page.get("dataset_root", ""))).expanduser()
    payload["VECTRA_PAGE_DATASET_ROOT"] = str(page_root)
    payload["VECTRA_PAGE_IMAGE_DIR"] = str(page_root / "train2017")
    payload["VECTRA_PAGE_SIDECAR_DIR"] = str(page_root / "sidecars" / "train2017")
    split_json = Path(str(page.get("split_manifest_path", ""))).expanduser()
    payload["VECTRA_PAGE_SPLIT_JSON"] = str(split_json)
    if str(split_json):
        one_drive_png_root = split_json.parent.parent
        payload["VECTRA_ONE_DRIVE_PNG_ROOT"] = str(one_drive_png_root)
        payload["VECTRA_PAGE_GT_SOLID_CSV"] = str(split_json.parent / "gt" / "train2017_solid.csv")
        payload["VECTRA_PAGE_GT_DASHED_CSV"] = str(split_json.parent / "gt" / "train2017_dashed.csv")

probe = specs.get("DeepPatent2_negative_clutter_probe", {})
if isinstance(probe, dict):
    payload["VECTRA_DEEPPATENT_DATASET_ROOT"] = str(probe.get("dataset_root", ""))

clean = {k: v for k, v in payload.items() if v and v != "."}
print(json.dumps(clean, ensure_ascii=False))
""".strip()

    command = (
        f"cd {_sh_quote(upstream_root_wsl)} && "
        f"{_sh_quote(exe_wsl)} - <<'PY'\n{script}\nPY"
    )
    try:
        probe = subprocess.run(
            ["wsl", "bash", "-lc", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return {}
    if probe.returncode != 0:
        return {}
    lines = [line.strip() for line in (probe.stdout or "").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        payload = loads(lines[-1])
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in payload.items()
        if str(value).strip()
    }


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _copy_path(source_path: Path, target_path: Path, kind: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "directory":
        copytree(source_path, target_path)
        return
    copy2(source_path, target_path)


def _copy_prerequisite_stage_dirs(
    *,
    source_run_dir: Path,
    child_run_dir: Path,
    target_stage_number: int,
) -> tuple[str, ...]:
    inherited: list[str] = []
    for stage_number in range(1, target_stage_number):
        stage_dir_name = f"stage-{stage_number:02d}"
        source_stage_dir = source_run_dir / stage_dir_name
        if not source_stage_dir.exists():
            continue
        target_stage_dir = child_run_dir / stage_dir_name
        if target_stage_dir.exists():
            continue
        copytree(source_stage_dir, target_stage_dir)
        inherited.append(stage_dir_name)
    return tuple(inherited)


def _to_wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    normalized = resolved.replace("/", "\\")
    lowered = normalized.lower()
    wsl_prefixes = ("\\\\wsl$\\", "\\\\wsl.localhost\\")
    for prefix in wsl_prefixes:
        if lowered.startswith(prefix):
            parts = normalized.split("\\")
            # UNC layout: \\wsl$\Distro\path\inside\wsl
            if len(parts) >= 5:
                remainder = "/".join(segment for segment in parts[4:] if segment)
                return "/" + remainder if remainder else "/"
    if ":" not in resolved:
        return resolved.replace("\\", "/")
    drive, rest = resolved.split(":", 1)
    return f"/mnt/{drive.lower()}{rest.replace('\\', '/')}"


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
