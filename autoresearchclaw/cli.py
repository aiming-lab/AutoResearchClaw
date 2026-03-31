from __future__ import annotations

import argparse
import sys

from .paper_repair import (
    PaperRepairError,
    apply_paper_repair,
    init_paper_repair,
    rollback_paper_repair,
)
from .research_repair import (
    ResearchRepairError,
    init_research_repair,
    prepare_research_repair_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearchclaw",
        description="Manual paper-level and research-level repair workflows for completed AutoResearchClaw runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper_init = subparsers.add_parser(
        "paper-repair-init",
        help="Create an editable post-export paper repair workspace from a completed run.",
    )
    paper_init.add_argument("--run-dir", required=True, help="Completed run directory with stage-22 or stage-23 paper artifacts.")
    paper_init.add_argument(
        "--output-dir",
        default="artifacts/paper-repair",
        help="Directory where the paper repair workspace and manifest will be written.",
    )

    paper_apply = subparsers.add_parser(
        "paper-repair-apply",
        help="Publish repaired paper artifacts back into the source run.",
    )
    paper_apply.add_argument("--repair-json", required=True, help="Path to a paper-repair.json manifest.")
    paper_apply.add_argument("--note", help="Optional note describing the published paper fix.")

    paper_rollback = subparsers.add_parser(
        "paper-repair-rollback",
        help="Restore the most recent published paper repair snapshot into the source run.",
    )
    paper_rollback.add_argument("--repair-json", required=True, help="Path to a paper-repair.json manifest.")
    paper_rollback.add_argument("--backup-id", help="Optional backup id to roll back to. Defaults to the most recent publish.")

    research_init = subparsers.add_parser(
        "research-repair-init",
        help="Create a run-level repair workspace that can send a completed run back to experiment stages.",
    )
    research_init.add_argument("--run-dir", required=True, help="Existing AutoResearchClaw run directory to repair.")
    research_init.add_argument(
        "--output-dir",
        default="artifacts/research-repair",
        help="Directory where the research-repair workspace and manifest will be written.",
    )
    research_init.add_argument(
        "--config",
        default="config.arc.yaml",
        help="Base config to copy into the repair workspace for the child run.",
    )
    research_init.add_argument(
        "--target-stage",
        default="EXPERIMENT_DESIGN",
        help="Stage number or stage name to restart from, such as 9, CODE_GENERATION, or EXPERIMENT_RUN.",
    )
    research_init.add_argument("--reason", help="Short human reason for why the completed run should be repaired.")
    research_init.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Initial repair feedback bullet to seed into workspace/feedback.md. Repeatable.",
    )
    research_init.add_argument(
        "--upstream-root",
        default=".",
        help="Path to the AutoResearchClaw checkout used for child runs.",
    )

    research_run = subparsers.add_parser(
        "research-repair-run",
        help="Prepare, and optionally launch, a child run from a research-repair workspace.",
    )
    research_run.add_argument("--repair-json", required=True, help="Path to a research-repair.json manifest.")
    research_run.add_argument("--output-dir", help="Optional explicit child run output directory.")
    research_run.add_argument(
        "--feedback",
        action="append",
        default=[],
        help="Additional repair feedback bullet to append before generating the child run. Repeatable.",
    )
    research_run.add_argument(
        "--auto-approve",
        action="store_true",
        help="Launch the child run with --auto-approve so the child pipeline will not stop at quality gates.",
    )
    research_run.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Pass --skip-preflight to the child run command.",
    )
    research_run.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the child run. Without this flag, only launch metadata and scripts are prepared.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "paper-repair-init":
        return _run_paper_repair_init(args.run_dir, args.output_dir)
    if args.command == "paper-repair-apply":
        return _run_paper_repair_apply(args.repair_json, args.note)
    if args.command == "paper-repair-rollback":
        return _run_paper_repair_rollback(args.repair_json, args.backup_id)
    if args.command == "research-repair-init":
        return _run_research_repair_init(
            args.run_dir,
            args.output_dir,
            args.config,
            args.target_stage,
            args.reason,
            list(args.feedback),
            args.upstream_root,
        )
    if args.command == "research-repair-run":
        return _run_research_repair_run(
            args.repair_json,
            args.output_dir,
            list(args.feedback),
            bool(args.auto_approve),
            bool(args.skip_preflight),
            bool(args.execute),
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_paper_repair_init(run_dir: str, output_dir: str) -> int:
    try:
        outputs = init_paper_repair(run_dir, output_dir)
    except (PaperRepairError, OSError) as exc:
        print(f"Paper repair init failed: {exc}", file=sys.stderr)
        return 1

    print("Paper repair workspace created")
    print(f"Workspace: {outputs['workspace']}")
    print(f"Session JSON: {outputs['session_json']}")
    print(f"README: {outputs['readme']}")
    return 0


def _run_paper_repair_apply(repair_json: str, note: str | None) -> int:
    try:
        outputs = apply_paper_repair(repair_json, note=note)
    except (PaperRepairError, OSError) as exc:
        print(f"Paper repair publish failed: {exc}", file=sys.stderr)
        return 1

    print("Paper repair published")
    print(f"Run dir: {outputs['published_run_dir']}")
    print(f"Backup dir: {outputs['backup_dir']}")
    print(f"Session JSON: {outputs['session_json']}")
    return 0


def _run_paper_repair_rollback(repair_json: str, backup_id: str | None) -> int:
    try:
        outputs = rollback_paper_repair(repair_json, backup_id=backup_id)
    except (PaperRepairError, OSError) as exc:
        print(f"Paper repair rollback failed: {exc}", file=sys.stderr)
        return 1

    print("Paper repair rolled back")
    print(f"Run dir: {outputs['published_run_dir']}")
    print(f"Rolled back backup: {outputs['rolled_back_backup']}")
    print(f"Session JSON: {outputs['session_json']}")
    return 0


def _run_research_repair_init(
    run_dir: str,
    output_dir: str,
    config_path: str,
    target_stage: str,
    reason: str | None,
    feedback: list[str],
    upstream_root: str,
) -> int:
    try:
        outputs = init_research_repair(
            run_dir,
            output_dir,
            config_path=config_path,
            target_stage=target_stage,
            reason=reason,
            feedback=feedback,
            upstream_root=upstream_root,
        )
    except (ResearchRepairError, OSError) as exc:
        print(f"Research repair init failed: {exc}", file=sys.stderr)
        return 1

    print("Research repair workspace created")
    print(f"Workspace: {outputs['workspace']}")
    print(f"Session JSON: {outputs['session_json']}")
    print(f"Feedback: {outputs['feedback']}")
    print(f"Repair config: {outputs['repair_config']}")
    print(f"README: {outputs['readme']}")
    return 0


def _run_research_repair_run(
    repair_json: str,
    output_dir: str | None,
    feedback: list[str],
    auto_approve: bool,
    skip_preflight: bool,
    execute: bool,
) -> int:
    try:
        outputs = prepare_research_repair_run(
            repair_json,
            output_dir=output_dir,
            extra_feedback=feedback,
            auto_approve=auto_approve,
            skip_preflight=skip_preflight,
            execute=execute,
        )
    except (ResearchRepairError, OSError) as exc:
        print(f"Research repair launch preparation failed: {exc}", file=sys.stderr)
        return 1

    print("Research repair child run prepared")
    print(f"Child run dir: {outputs['child_run_dir']}")
    print(f"Generated config: {outputs['generated_config']}")
    print(f"Launch script: {outputs['launch_script']}")
    print(f"Metadata: {outputs['metadata']}")
    print("Command preview:")
    print(outputs["command_preview"])
    if execute and outputs.get("pid"):
        print(f"Process pid: {outputs['pid']}")
    print(f"Session JSON: {outputs['session_json']}")
    return 0
