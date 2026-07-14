"""ResearchClaw CLI — run the 23-stage autonomous research pipeline."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

from researchclaw.adapters import AdapterBundle
from researchclaw.config import (
    CONFIG_SEARCH_ORDER,
    EXAMPLE_CONFIG,
    RCConfig,
    resolve_config_path,
)
from researchclaw.health import print_doctor_report, run_doctor, write_doctor_report


# ---------------------------------------------------------------------------
# OpenCode installation helpers
# ---------------------------------------------------------------------------

def _is_opencode_installed() -> bool:
    """Check if the ``opencode`` CLI is available on PATH."""
    opencode_cmd = shutil.which("opencode")
    if opencode_cmd is None:
        return False
    try:
        r = subprocess.run(
            [opencode_cmd, "--version"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _is_npm_installed() -> bool:
    """Check if ``npm`` is available on PATH."""
    return shutil.which("npm") is not None


def _install_opencode() -> bool:
    """Install OpenCode globally via npm.  Returns True on success."""
    print("  Installing opencode-ai (this may take a minute)...")
    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        print("  npm is not installed. Cannot install OpenCode.")
        return False
    try:
        r = subprocess.run(
            [npm_cmd, "i", "-g", "opencode-ai@latest"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            print("  OpenCode installed successfully!")
            return True
        else:
            print(f"  Installation failed (exit {r.returncode}):")
            if r.stderr:
                for line in r.stderr.strip().splitlines()[:5]:
                    print(f"    {line}")
            return False
    except subprocess.TimeoutExpired:
        print("  Installation timed out.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  Installation failed: {exc}")
        return False


def _prompt_opencode_install() -> bool:
    """Interactively prompt the user to install OpenCode.

    Returns True if OpenCode is now available (already installed or
    just installed successfully).  Returns False otherwise.
    """
    if _is_opencode_installed():
        return True

    if not sys.stdin.isatty():
        return False

    print()
    print("=" * 60)
    print("  OpenCode Beast Mode  (Recommended)")
    print("=" * 60)
    print()
    print("  OpenCode is an AI coding agent that dramatically improves")
    print("  experiment code generation for complex research tasks.")
    print()
    print("  With OpenCode enabled, ResearchClaw can generate multi-file")
    print("  experiment projects with custom architectures, training")
    print("  loops, and ablation studies — far beyond single-file limits.")
    print()

    if not _is_npm_installed():
        print("  Node.js/npm is required but not installed.")
        print("  To install OpenCode later:")
        print("    1. Install Node.js: https://nodejs.org/")
        print("    2. Run: npm i -g opencode-ai@latest")
        print("    — or: researchclaw setup")
        print()
        return False

    try:
        answer = input("  Install OpenCode now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if answer in ("", "y", "yes"):
        success = _install_opencode()
        if not success:
            print("  You can retry later with: researchclaw setup")
        return success
    else:
        print("  Skipped. You can install later with: researchclaw setup")
        return False


def _resolve_config_or_exit(args: argparse.Namespace) -> Path | None:
    """Resolve config path from args, printing helpful errors on failure.

    Returns the resolved Path on success, or None if the config cannot be found
    (after printing an error message to stderr).
    """
    path = resolve_config_path(getattr(args, "config", None))
    if path is not None and not path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        return None
    if path is None:
        search_list = ", ".join(CONFIG_SEARCH_ORDER)
        print(
            f"Error: no config file found (searched: {search_list}).\n"
            f"Run 'researchclaw init' to create one from the example template.",
            file=sys.stderr,
        )
        return None
    return path


def _generate_run_id(topic: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:6]
    return f"rc-{ts}-{topic_hash}"


def cmd_run(args: argparse.Namespace) -> int:
    resolved = _resolve_config_or_exit(args)
    if resolved is None:
        return 1
    config_path = resolved
    topic = cast(str | None, args.topic)
    output = cast(str | None, args.output)
    from_stage_name = cast(str | None, args.from_stage)
    to_stage_name = cast(str | None, getattr(args, "to_stage", None))
    auto_approve = cast(bool, args.auto_approve)
    incremental_experiment = cast(bool, getattr(args, "incremental_experiment", False))
    skip_preflight = cast(bool, args.skip_preflight)
    resume = cast(bool, args.resume)
    skip_noncritical = cast(bool, args.skip_noncritical_stage)
    no_graceful_degradation = cast(bool, args.no_graceful_degradation)
    hitl_mode = cast(str | None, getattr(args, "mode", None))

    kb_root_path = None
    profile_override = cast(str | None, getattr(args, "profile", None))
    config = RCConfig.load(
        config_path, check_paths=False, profile_override=profile_override
    )

    # If the user deployed a profile, force every detector call to agree
    # with it — otherwise Stage 10/18/22 may keyword-detect a different
    # domain from the topic and mix prompt styles.
    if config.project.profile:
        try:
            from researchclaw.domains.detector import set_forced_profile
            set_forced_profile(config.project.profile)
        except Exception:  # noqa: BLE001
            pass

    # Override incremental_experiment if CLI flag is set
    if incremental_experiment:
        import dataclasses

        config = dataclasses.replace(
            config,
            experiment=dataclasses.replace(
                config.experiment,
                collider_agent=dataclasses.replace(
                    config.experiment.collider_agent, incremental=True
                ),
            ),
        )

    # Override graceful_degradation if CLI flag is set
    if no_graceful_degradation:
        import dataclasses as _dc_gd

        new_research = _dc_gd.replace(config.research, graceful_degradation=False)
        config = _dc_gd.replace(config, research=new_research)

    # Derive gate behavior from project.mode (CLI --auto-approve overrides)
    mode = config.project.mode.lower()
    if auto_approve:
        # Explicit CLI flag takes precedence over config mode
        stop_on_gate = False
    elif mode == "full-auto":
        auto_approve = True
        stop_on_gate = False
    else:
        # "semi-auto" and "docs-first" should block on gates
        stop_on_gate = True

    if topic:
        import dataclasses

        new_research = dataclasses.replace(config.research, topic=topic)
        config = dataclasses.replace(config, research=new_research)

    # --- LLM Preflight ---
    if not skip_preflight:
        from researchclaw.llm import create_llm_client

        client = create_llm_client(config)
        print("Preflight check...", end=" ", flush=True)
        ok, msg = client.preflight()
        if ok:
            print(msg)
        else:
            print(f"FAILED — {msg}", file=sys.stderr)
            return 1

    run_id = _generate_run_id(config.research.topic)
    run_dir = Path(output or f"artifacts/{run_id}")

    # BUG-119 / BUG-216: When --resume or --from-stage is used without
    # --output, search for the most recent existing run directory that
    # matches the topic.  Without this, --from-stage creates a new empty
    # directory that has no prior stage artifacts.
    if (resume or from_stage_name) and not output:
        topic_hash = hashlib.sha256(config.research.topic.encode()).hexdigest()[:6]
        artifacts_root = Path("artifacts")
        if artifacts_root.is_dir():
            candidates = sorted(
                (
                    d for d in artifacts_root.iterdir()
                    if d.is_dir()
                    and d.name.startswith("rc-")
                    and d.name.endswith(f"-{topic_hash}")
                    and (d / "checkpoint.json").exists()
                ),
                key=lambda d: d.name,
                reverse=True,  # newest first (timestamp in name)
            )
            if candidates:
                run_dir = candidates[0]
                run_id = run_dir.name
                print(f"Found existing run: {run_dir}")
            elif from_stage_name:
                print(
                    f"Error: --from-stage {from_stage_name} requires prior "
                    f"stage artifacts, but no existing run found for topic "
                    f"hash '{topic_hash}'. Use --output to specify the run "
                    f"directory containing prior artifacts.",
                    file=sys.stderr,
                )
                return 1
            else:
                print(
                    "Warning: --resume specified but no checkpoint found "
                    f"for topic hash '{topic_hash}'. Starting new run.",
                    file=sys.stderr,
                )

    run_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot — copy the active YAML into the run directory so each
    # task's run_dir is self-describing (profile, topic, experiment mode,
    # LLM provider). On resume, snapshot under a timestamped name to avoid
    # overwriting the original.
    try:
        snapshot_path = run_dir / "config.yaml"
        if snapshot_path.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            snapshot_path = run_dir / f"config.resumed-{ts}.yaml"
        shutil.copy2(config_path, snapshot_path)
    except Exception as _snap_exc:  # noqa: BLE001
        print(f"Warning: config snapshot failed: {_snap_exc}", file=sys.stderr)

    if config.knowledge_base.root:
        kb_root_path = Path(config.knowledge_base.root)
        kb_root_path.mkdir(parents=True, exist_ok=True)

    adapters = AdapterBundle()

    # --- HITL session setup ---
    hitl_session = None
    try:
        from researchclaw.hitl.config import HITLConfig
        from researchclaw.hitl.session import HITLSession
        from researchclaw.hitl.presets import get_preset

        hitl_config = None
        if hitl_mode:
            # CLI --mode flag takes precedence
            hitl_config = get_preset(hitl_mode)
            if hitl_config is None:
                hitl_config = HITLConfig(enabled=True, mode=hitl_mode)
        elif hasattr(config, "hitl") and config.hitl is not None:
            hitl_config = config.hitl
        # If HITL is enabled, auto_approve should be False
        if hitl_config and hitl_config.enabled:
            auto_approve = False
            stop_on_gate = False  # HITL handles gates directly
    except ImportError:
        hitl_config = None

    from researchclaw.pipeline.runner import execute_pipeline, read_checkpoint
    from researchclaw.pipeline.stages import Stage

    # --- Determine start stage ---
    from_stage = Stage.TOPIC_INIT
    if from_stage_name:
        try:
            from_stage = Stage[from_stage_name.upper()]
        except KeyError:
            valid = ", ".join(s.name for s in Stage)
            print(
                f"Error: unknown stage '{from_stage_name}'. "
                f"Valid stages: {valid}",
                file=sys.stderr,
            )
            return 1
    elif resume:
        resumed = read_checkpoint(run_dir)
        if resumed is not None:
            from_stage = resumed
            print(f"Resuming from checkpoint: Stage {int(from_stage)}: {from_stage.name}")

    # --- Determine stop stage ---
    to_stage: Stage | None = None
    if to_stage_name:
        try:
            to_stage = Stage[to_stage_name.upper()]
        except KeyError:
            valid = ", ".join(s.name for s in Stage)
            print(
                f"Error: unknown stage '{to_stage_name}'. "
                f"Valid stages: {valid}",
                file=sys.stderr,
            )
            return 1
        if int(to_stage) < int(from_stage):
            print(
                f"Error: --to-stage {to_stage.name} (stage {int(to_stage)}) "
                f"must be >= --from-stage {from_stage.name} (stage {int(from_stage)})",
                file=sys.stderr,
            )
            return 1

    # --- Create HITL session and wire to adapters ---
    if hitl_config and hitl_config.enabled:
        try:
            hitl_session = HITLSession(
                run_id=run_id,
                config=hitl_config,
                run_dir=run_dir,
            )
            # Check for scripted intervention file (env var or CLI flag)
            interventions_file = os.environ.get("HITL_INTERVENTIONS_FILE", "")
            if not interventions_file:
                interventions_file = getattr(args, "interventions", None) or ""
            if interventions_file and Path(interventions_file).is_file():
                from researchclaw.hitl.adapters.scripted_adapter import (
                    ScriptedHITLAdapter,
                )

                scripted = ScriptedHITLAdapter.from_file(interventions_file)
                hitl_session.set_input_callback(scripted.collect_input)
                print(f"  HITL:    scripted ({len(scripted.pending_stages)} interventions)")
            else:
                # Wire CLI adapter for interactive input
                from researchclaw.hitl.adapters.cli_adapter import CLIAdapter

                cli_adapter = CLIAdapter(run_dir=run_dir)
                hitl_session.set_input_callback(cli_adapter.collect_input)
            adapters.hitl = hitl_session
        except Exception as _hitl_exc:
            import logging
            logging.getLogger(__name__).warning(
                "HITL session setup failed: %s", _hitl_exc
            )

    from researchclaw import __version__
    print(f"ResearchClaw v{__version__} — Starting pipeline")
    print(f"  Run ID:  {run_id}")
    print(f"  Topic:   {config.research.topic}")
    print(f"  Output:  {run_dir}")
    print(f"  Mode:    {config.project.mode}")
    if hitl_config and hitl_config.enabled:
        print(f"  HITL:    {hitl_config.mode}")
    if config.project.profile:
        print(f"  Profile: {config.project.profile}")
    print(f"  From:    Stage {int(from_stage)}: {from_stage.name}")
    if to_stage:
        print(f"  To:      Stage {int(to_stage)}: {to_stage.name}")

    # Hint: OpenCode beast mode
    exp_cfg = getattr(config, "experiment", None)
    oc_cfg = getattr(exp_cfg, "opencode", None)
    if oc_cfg and getattr(oc_cfg, "enabled", False) and not _is_opencode_installed():
        print()
        print("  Hint: OpenCode beast mode is enabled but not installed.")
        print("        Run 'researchclaw setup' to install for better code generation.")

    print()

    results = execute_pipeline(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        adapters=adapters,
        from_stage=from_stage,
        to_stage=to_stage,
        auto_approve_gates=auto_approve,
        stop_on_gate=stop_on_gate,
        skip_noncritical=skip_noncritical,
        kb_root=kb_root_path,
    )

    done = sum(1 for r in results if r.status.value == "done")
    paused = sum(1 for r in results if r.status.value == "paused")
    failed = sum(1 for r in results if r.status.value == "failed")

    # --- Complete HITL session ---
    if hitl_session is not None:
        hitl_session.complete()
        if hitl_session.interventions_count > 0:
            print(
                f"  HITL: {hitl_session.interventions_count} interventions, "
                f"{hitl_session.total_human_time_sec:.0f}s human time"
            )

    if paused:
        print(
            f"\nPipeline paused: {done}/{len(results)} stages done, "
            f"{paused} paused, {failed} failed"
        )
    else:
        print(f"\nPipeline complete: {done}/{len(results)} stages done, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from researchclaw.config import validate_config
    import yaml

    resolved = _resolve_config_or_exit(args)
    if resolved is None:
        return 1
    config_path = resolved
    no_check_paths = cast(bool, args.no_check_paths)

    with config_path.open(encoding="utf-8") as f:
        loaded = cast(object, yaml.safe_load(f))

    if loaded is None:
        data: dict[str, object] = {}
    elif isinstance(loaded, dict):
        loaded_map = cast(Mapping[object, object], loaded)
        data = {str(key): value for key, value in loaded_map.items()}
    else:
        print("Config validation FAILED:")
        print("  Error: Config root must be a mapping")
        return 1

    result = validate_config(data, check_paths=not no_check_paths)
    if result.ok:
        print("Config validation passed")
        for w in result.warnings:
            print(f"  Warning: {w}")
        return 0
    else:
        print("Config validation FAILED:")
        for e in result.errors:
            print(f"  Error: {e}")
        return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    resolved = _resolve_config_or_exit(args)
    if resolved is None:
        return 1
    config_path = resolved
    output = cast(str | None, args.output)

    report = run_doctor(config_path)
    print_doctor_report(report)
    if output:
        write_doctor_report(report, Path(output))
    return 0 if report.overall == "pass" else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Summarise resolved config: profile, prompt bank, stages, extras."""
    from researchclaw.pipeline._domain import _prompt_bank_domain_from_config
    from researchclaw.prompts import PromptManager

    resolved = _resolve_config_or_exit(args)
    if resolved is None:
        return 1
    config_path = resolved
    profile_override = cast(str | None, getattr(args, "profile", None))
    try:
        config = RCConfig.load(
            config_path,
            check_paths=False,
            profile_override=profile_override,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    bank = _prompt_bank_domain_from_config(config)
    profile_id = str(getattr(config.project, "profile", "") or "").strip() or "(none)"
    topic = str(getattr(config.research, "topic", "") or "").strip() or "(unset)"
    domains_raw = getattr(config.research, "domains", ()) or ()
    domains = ", ".join(str(d) for d in domains_raw) or "(none)"

    extras_cfg = getattr(config.prompts, "extra_prompts", ()) or ()
    extras_pairs: dict[str, str] = {}
    for item in extras_cfg:
        try:
            stage_key, value = item
        except (TypeError, ValueError):
            continue
        extras_pairs[str(stage_key)] = str(value)

    pm = PromptManager(
        config.prompts.custom_file or None,
        domain=bank,
        extra_prompts=extras_pairs or None,
    )
    resolved_extras = pm.extra_prompts()
    hyp_roles = sorted(pm.debate_roles_hypothesis().keys())
    ana_roles = sorted(pm.debate_roles_analysis().keys())

    bank_label = {"ml": "ML (machine learning)", "hep_ph": "HEP-ph (particle phenomenology)"}.get(bank, bank)

    print("=" * 68)
    print("ResearchClaw configuration summary")
    print("=" * 68)
    print(f"  Config file      : {config_path}")
    print(f"  Project profile  : {profile_id}")
    print(f"  Research topic   : {topic}")
    print(f"  Declared domains : {domains}")
    print(f"  Resolved bank    : {bank}  [{bank_label}]")
    if config.prompts.custom_file:
        print(f"  Prompt overrides : {config.prompts.custom_file}")
    print()
    print("  Debate roles (hypothesis): " + ", ".join(hyp_roles))
    print("  Debate roles (analysis)  : " + ", ".join(ana_roles))
    print()
    print("Pipeline stages  (mark [EXTRA] = prompts.extra_prompts configured)")
    print("-" * 68)
    for stage in pm.stage_names():
        marker = " [EXTRA]" if stage in resolved_extras else ""
        print(f"  - {stage}{marker}")
    if resolved_extras:
        print()
        print("Configured extra prompts")
        print("-" * 68)
        for stage, text in resolved_extras.items():
            preview = text.replace("\n", " ")
            if len(preview) > 72:
                preview = preview[:72] + "..."
            print(f"  {stage}: {preview}")
    else:
        print()
        print("  (no prompts.extra_prompts configured — any stage above can")
        print("   receive custom guidance via config.yaml -> prompts.extra_prompts)")
    print("=" * 68)
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    """C1: Multi-project management commands."""
    from researchclaw.project.manager import ProjectManager

    action = cast(str, args.project_action)
    config_path = Path(cast(str, args.config))
    config = RCConfig.load(config_path, check_paths=False)
    pm = ProjectManager(Path(config.multi_project.projects_dir))

    if action == "list":
        projects = pm.list_all()
        if not projects:
            print("No projects found.")
        for p in projects:
            marker = " *" if pm.active and pm.active.name == p.name else ""
            print(f"  {p.name} [{p.status}]{marker}")
        return 0
    elif action == "status":
        status = pm.get_status()
        print(f"Total projects: {status['total']}")
        print(f"Active: {status.get('active', 'none')}")
        return 0
    elif action == "create":
        name = cast(str, args.name)
        topic = cast(str | None, getattr(args, "topic", None))
        proj = pm.create(name, str(config_path), topic=topic or "")
        print(f"Created project: {proj.name}")
        return 0
    elif action == "switch":
        name = cast(str, args.name)
        pm.switch(name)
        print(f"Switched to project: {name}")
        return 0
    elif action == "compare":
        names = cast(list[str], args.names)
        if len(names) != 2:
            print("Error: compare requires exactly 2 project names", file=sys.stderr)
            return 1
        result = pm.compare(names[0], names[1])
        print(f"Comparing {names[0]} vs {names[1]}:")
        for k, v in result.get("metric_diff", {}).items():
            print(f"  {k}: delta={v['delta']:.4f}")
        return 0
    else:
        print(f"Unknown project action: {action}", file=sys.stderr)
        return 1


def cmd_mcp(args: argparse.Namespace) -> int:
    """C3: MCP integration commands."""
    import asyncio

    start = cast(bool, args.start)
    if start:
        from researchclaw.mcp.server import ResearchClawMCPServer

        server = ResearchClawMCPServer()
        print("Starting MCP server...")
        asyncio.run(server.start())
        return 0
    else:
        from researchclaw.mcp.tools import list_tool_names

        names = list_tool_names()
        print("Available MCP tools:")
        for name in names:
            print(f"  {name}")
        return 0


def cmd_overleaf(args: argparse.Namespace) -> int:
    """C4: Overleaf sync commands."""
    config_path = Path(cast(str, args.config))
    config = RCConfig.load(config_path, check_paths=False)

    if not config.overleaf.enabled:
        print("Overleaf sync is not enabled in config.", file=sys.stderr)
        return 1

    from researchclaw.overleaf.sync import OverleafSync

    sync = OverleafSync(
        git_url=config.overleaf.git_url,
        branch=config.overleaf.branch,
    )

    do_sync = cast(bool, args.sync)
    do_status = cast(bool, args.status)

    if do_status:
        status = sync.get_status()
        for k, v in status.items():
            print(f"  {k}: {v}")
        return 0
    elif do_sync:
        run_dir = Path(cast(str, args.run_dir))
        if not run_dir.exists():
            print(f"Error: run_dir not found: {run_dir}", file=sys.stderr)
            return 1
        sync.setup(run_dir)
        sync.pull_changes()
        print("Overleaf sync complete.")
        return 0
    else:
        print("Use --sync or --status", file=sys.stderr)
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI web server."""
    config_path = Path(cast(str, args.config))
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    config = RCConfig.load(config_path, check_paths=False)
    host = cast(str, args.host) or config.server.host
    port = int(cast(int, args.port) or config.server.port)

    try:
        from researchclaw.server.app import create_app
        import uvicorn
    except ImportError as exc:
        print(
            f"Error: web dependencies not installed — pip install researchclaw[web]\n{exc}",
            file=sys.stderr,
        )
        return 1

    app = create_app(config, monitor_dir=args.monitor_dir)
    uvicorn.run(app, host=host, port=port)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Start dashboard-only server (no pipeline control)."""
    config_path = Path(cast(str, args.config))
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    config = RCConfig.load(config_path, check_paths=False)
    host = cast(str, args.host) or config.server.host
    port = int(cast(int, args.port) or config.server.port)

    try:
        from researchclaw.server.app import create_app
        import uvicorn
    except ImportError as exc:
        print(
            f"Error: web dependencies not installed — pip install researchclaw[web]\n{exc}",
            file=sys.stderr,
        )
        return 1

    app = create_app(config, dashboard_only=True, monitor_dir=args.monitor_dir)
    uvicorn.run(app, host=host, port=port)
    return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    """Run the interactive setup wizard."""
    from researchclaw.wizard.quickstart import QuickStartWizard

    wizard = QuickStartWizard()
    output = cast(str | None, args.output)

    import yaml

    config = wizard.run_interactive()
    if output:
        Path(output).write_text(yaml.dump(config, default_flow_style=False))
        print(f"Config written to {output}")
    else:
        print(yaml.dump(config, default_flow_style=False))
    return 0


_PROVIDER_CHOICES = {
    "1": ("openai", "OPENAI_API_KEY"),
    "2": ("openrouter", "OPENROUTER_API_KEY"),
    "3": ("deepseek", "DEEPSEEK_API_KEY"),
    "4": ("minimax", "MINIMAX_API_KEY"),
    "5": ("acp", ""),
    "6": ("ollama", ""),
    "7": ("minimax-global", "MINIMAX_API_KEY"),
    "8": ("minimax-anthropic", "MINIMAX_API_KEY"),
    "9": ("minimax-anthropic-cn", "MINIMAX_API_KEY"),
}

_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "minimax": "https://api.minimaxi.com/v1",
    "minimax-global": "https://api.minimax.io/v1",
    "minimax-anthropic": "https://api.minimax.io/anthropic",
    "minimax-anthropic-cn": "https://api.minimaxi.com/anthropic",
    "ollama": "http://localhost:11434/v1",
}

_MINIMAX_MODELS = (
    "MiniMax-M3",
    ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"],
)

_PROVIDER_MODELS = {
    "openai": ("gpt-4o", ["gpt-4.1", "gpt-4o-mini"]),
    "openrouter": (
        "anthropic/claude-3.5-sonnet",
        ["google/gemini-pro-1.5", "meta-llama/llama-3.1-70b-instruct"],
    ),
    "deepseek": ("deepseek-chat", ["deepseek-reasoner"]),
    "minimax": _MINIMAX_MODELS,
    "minimax-global": _MINIMAX_MODELS,
    "minimax-anthropic": _MINIMAX_MODELS,
    "minimax-anthropic-cn": _MINIMAX_MODELS,
    "ollama": ("llama3.2", ["mistral", "qwen2.5:7b"]),
}


def cmd_init(args: argparse.Namespace) -> int:
    force = cast(bool, args.force)
    dest = Path("config.arc.yaml")

    if dest.exists() and not force:
        print(f"{dest} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    # Look for the example config: first in repo root (relative to package),
    # then in CWD (for development), then bundled in the package data dir.
    _candidates = [
        Path(__file__).resolve().parent.parent / EXAMPLE_CONFIG,  # repo root
        Path.cwd() / EXAMPLE_CONFIG,                              # cwd fallback
        Path(__file__).resolve().parent / "data" / EXAMPLE_CONFIG, # packaged
    ]
    example = next((p for p in _candidates if p.exists()), None)
    if example is None:
        print(
            f"Error: example config not found.\n"
            f"Searched: {', '.join(str(c) for c in _candidates)}",
            file=sys.stderr,
        )
        return 1

    # Interactive provider prompt (TTY only, else default to openai)
    choice = "1"
    if sys.stdin.isatty():
        print("Select LLM provider:")
        print("  1) openai       (requires OPENAI_API_KEY)")
        print("  2) openrouter   (requires OPENROUTER_API_KEY)")
        print("  3) deepseek     (requires DEEPSEEK_API_KEY)")
        print("  4) minimax-cn-openai        (requires MINIMAX_API_KEY)")
        print("  5) acp          (local AI agent — no API key needed)")
        print("  6) ollama       (local Ollama server — no API key needed)")
        print("  7) minimax-global-openai    (requires MINIMAX_API_KEY)")
        print("  8) minimax-global-anthropic (requires MINIMAX_API_KEY)")
        print("  9) minimax-cn-anthropic     (requires MINIMAX_API_KEY)")
        try:
            raw = input("Choice [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if raw in _PROVIDER_CHOICES:
            choice = raw

    provider, api_key_env = _PROVIDER_CHOICES[choice]

    content = example.read_text(encoding="utf-8")

    # String-based replacement to preserve YAML comments
    content = content.replace(
        'provider: "openai-compatible"', f'provider: "{provider}"'
    )

    if provider == "acp":
        # ACP doesn't need base_url or api_key
        content = content.replace(
            'base_url: "https://api.openai.com/v1"', 'base_url: ""'
        )
        content = content.replace('api_key_env: "OPENAI_API_KEY"', 'api_key_env: ""')
        # Beast Mode is incompatible with ACP (opencode_bridge prepends "openai/")
        content = content.replace(
            '    enabled: true                # Master switch (default: true)',
            '    enabled: false               # Master switch (disabled for ACP)',
        )
    elif provider == "ollama":
        # Ollama runs locally — set base_url, clear api_key_env, set dummy key
        base_url = _PROVIDER_URLS["ollama"]
        content = content.replace(
            'base_url: "https://api.openai.com/v1"', f'base_url: "{base_url}"'
        )
        content = content.replace('api_key_env: "OPENAI_API_KEY"', 'api_key_env: ""')
        content = content.replace('api_key: ""', 'api_key: "ollama"')
    else:
        base_url = _PROVIDER_URLS.get(provider, "https://api.openai.com/v1")
        content = content.replace(
            'base_url: "https://api.openai.com/v1"', f'base_url: "{base_url}"'
        )
        if api_key_env:
            content = content.replace(
                'api_key_env: "OPENAI_API_KEY"', f'api_key_env: "{api_key_env}"'
            )

    if provider in _PROVIDER_MODELS:
        primary, fallbacks = _PROVIDER_MODELS[provider]
        content = content.replace('primary_model: "gpt-4o"', f'primary_model: "{primary}"')
        # Replace fallback models block
        old_fallbacks = '  fallback_models:\n    - "gpt-4.1"\n    - "gpt-4o-mini"'
        new_fallbacks = "  fallback_models:\n" + "".join(
            f'    - "{m}"\n' for m in fallbacks
        )
        content = content.replace(old_fallbacks, new_fallbacks.rstrip("\n"))

    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest} (provider: {provider})")

    if provider == "acp":
        print("\nNext steps:")
        print("  1. Ensure your ACP agent is installed and on PATH")
        print("  2. Edit config.arc.yaml to set llm.acp.agent if needed")
        print("  3. Run: researchclaw doctor")
    elif provider == "ollama":
        print("\nNext steps:")
        print("  1. Ensure Ollama is running: ollama serve")
        print("  2. Pull your model: ollama pull llama3.2")
        print("  3. Edit config.arc.yaml to set llm.primary_model to your model name")
        print("  4. Run: researchclaw doctor")
    elif provider in ("minimax-anthropic", "minimax-anthropic-cn"):
        print("\nNext steps:")
        print('  1. Install the adapter: pip install "researchclaw[anthropic]"')
        print("  2. Export your API key: export MINIMAX_API_KEY=...")
        print("  3. Edit config.arc.yaml to customize your settings")
        print("  4. Run: researchclaw doctor")
    else:
        env_var = api_key_env or "OPENAI_API_KEY"
        print(f"\nNext steps:")
        print(f"  1. Export your API key: export {env_var}=sk-...")
        print("  2. Edit config.arc.yaml to customize your settings")
        print("  3. Run: researchclaw doctor")

    # Offer OpenCode installation
    _prompt_opencode_install()

    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Post-install setup — check and install optional tools."""
    print("ResearchClaw — Environment Setup\n")

    # 1. OpenCode
    if _is_opencode_installed():
        try:
            opencode_cmd = shutil.which("opencode") or "opencode"
            r = subprocess.run(
                [opencode_cmd, "--version"],
                capture_output=True, text=True, timeout=15,
            )
            ver = r.stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001
            ver = "unknown"
        print(f"  [OK] OpenCode is installed (version: {ver})")
    else:
        installed = _prompt_opencode_install()
        if installed:
            print("  [OK] OpenCode is now available")
        else:
            print("  [--] OpenCode not installed (beast mode will be unavailable)")

    # 2. Docker (informational)
    print()
    if shutil.which("docker"):
        print("  [OK] Docker is available (sandbox execution enabled)")
    else:
        print("  [--] Docker not found (experiment sandbox unavailable)")
        print("       Install: https://docs.docker.com/get-docker/")

    # 3. LaTeX (informational)
    if shutil.which("pdflatex"):
        print("  [OK] LaTeX is available (PDF paper compilation enabled)")
    else:
        print("  [--] LaTeX not found (paper will be exported as .tex only)")
        print("       Install: sudo apt install texlive-full  (or equivalent)")

    print()
    print("Run 'researchclaw doctor' for a full environment health check.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from researchclaw.report import generate_report, write_report

    run_dir = Path(cast(str, args.run_dir))
    output = cast(str | None, args.output)

    try:
        report = generate_report(run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(report)
    if output:
        write_report(run_dir, Path(output))
        print(f"\nReport written to {output}")
    return 0


# ── Research Enhancement commands (Agent D) ───────────────────────


def cmd_trends(args: argparse.Namespace) -> int:
    """Research trend tracking commands."""
    config_path = Path(cast(str, args.config))
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    config = RCConfig.load(config_path, check_paths=False)

    import asyncio

    from researchclaw.trends.feeds import FeedManager
    from researchclaw.trends.trend_analyzer import TrendAnalyzer

    domains = cast(list[str] | None, args.domains) or list(config.research.domains)
    if not domains:
        domains = ["machine learning"]

    feed_manager = FeedManager(
        sources=config.trends.sources,
        s2_api_key=config.llm.s2_api_key,
    )

    if cast(bool, args.digest):
        from researchclaw.trends.daily_digest import DailyDigest

        digest = DailyDigest(feed_manager)
        result = asyncio.run(digest.generate(domains, config.trends.max_papers_per_day))
        print(result)
        return 0

    if cast(bool, args.analyze):
        papers = feed_manager.fetch_recent_papers(domains, max_papers=50)
        analyzer = TrendAnalyzer()
        analysis = analyzer.analyze(papers, config.trends.trend_window_days)
        print(analyzer.generate_trend_report(analysis))
        return 0

    if cast(bool, args.suggest_topics):
        from researchclaw.trends.auto_topic import AutoTopicGenerator
        from researchclaw.trends.opportunity_finder import OpportunityFinder

        papers = feed_manager.fetch_recent_papers(domains, max_papers=50)
        analyzer = TrendAnalyzer()
        finder = OpportunityFinder()
        generator = AutoTopicGenerator(analyzer, finder)
        candidates = asyncio.run(generator.generate_candidates(domains, papers))
        print(generator.format_candidates(candidates))
        return 0

    print("Usage: researchclaw trends --digest|--analyze|--suggest-topics")
    return 0


def cmd_calendar(args: argparse.Namespace) -> int:
    """Conference deadline calendar commands."""
    from researchclaw.calendar.deadlines import ConferenceCalendar
    from researchclaw.calendar.planner import SubmissionPlanner

    calendar = ConferenceCalendar.load_builtin()
    domains = cast(list[str] | None, args.domains)

    if cast(bool, args.upcoming):
        print(calendar.format_upcoming(domains=domains))
        return 0

    plan_venue = cast(str | None, args.plan)
    if plan_venue:
        planner = SubmissionPlanner(calendar)
        print(planner.format_plan(plan_venue))
        return 0

    print("Usage: researchclaw calendar --upcoming|--plan <venue>")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    """List, validate, or install skills."""
    from researchclaw.skills.loader import load_skill_from_skillmd, load_skills_from_directory
    from researchclaw.skills.registry import SkillRegistry

    action = args.skills_action or "list"
    user_dir = Path.home() / ".researchclaw" / "skills"

    if action == "list":
        # Build full registry to show all available skills
        custom_dirs: list[str] = []
        if user_dir.is_dir():
            custom_dirs.append(str(user_dir))
        project_skills = Path.cwd() / ".claude" / "skills"
        if project_skills.is_dir():
            custom_dirs.append(str(project_skills))

        registry = SkillRegistry(custom_dirs=custom_dirs)
        skills = registry.list_all()
        if not skills:
            print("No skills loaded.")
            return 0

        # Group by category
        by_cat: dict[str, list] = {}
        for s in skills:
            by_cat.setdefault(s.category, []).append(s)
        for cat in sorted(by_cat):
            print(f"\n[{cat}]")
            for s in sorted(by_cat[cat], key=lambda x: x.name):
                stages = ",".join(str(x) for x in s.applicable_stages) if s.applicable_stages else "all"
                src = "builtin"
                if s.source_dir:
                    sd = str(s.source_dir)
                    if ".researchclaw" in sd:
                        src = "user"
                    elif ".claude" in sd:
                        src = "project"
                    elif ".metaclaw" in sd:
                        src = "metaclaw"
                print(f"  {s.name:<35} stages={stages:<12} ({src})")

        print(f"\nTotal: {len(skills)} skills")
        print(f"\nSkill directories:")
        print(f"  builtin:  researchclaw/skills/builtin/")
        print(f"  user:     {user_dir}/")
        print(f"  project:  .claude/skills/")
        return 0

    elif action == "install":
        # Install a skill from a directory or URL
        source = getattr(args, "source", None)
        if not source:
            print("Usage: researchclaw skills install <path-to-skill-dir>")
            return 1
        source_path = Path(source).expanduser().resolve()
        skill_md = source_path / "SKILL.md"
        if not skill_md.exists():
            # Maybe the path IS the SKILL.md
            if source_path.name == "SKILL.md" and source_path.exists():
                source_path = source_path.parent
                skill_md = source_path / "SKILL.md"
            else:
                print(f"Error: no SKILL.md found in {source_path}")
                return 1

        skill = load_skill_from_skillmd(skill_md)
        if not skill:
            print(f"Error: failed to parse {skill_md}")
            return 1

        # Copy to user skills directory
        target = user_dir / skill.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, target, dirs_exist_ok=True)
        print(f"Installed skill '{skill.name}' -> {target}")
        return 0

    elif action == "validate":
        source = getattr(args, "source", None)
        if not source:
            print("Usage: researchclaw skills validate <path-to-SKILL.md>")
            return 1
        path = Path(source).expanduser().resolve()
        if path.is_dir():
            path = path / "SKILL.md"
        if not path.exists():
            print(f"Error: {path} not found")
            return 1
        skill = load_skill_from_skillmd(path)
        if not skill:
            print(f"FAIL: Could not parse {path}")
            return 1
        print(f"OK: {skill.name}")
        print(f"  description: {skill.description[:80]}")
        print(f"  category:    {skill.category}")
        print(f"  stages:      {skill.applicable_stages or 'all'}")
        print(f"  keywords:    {skill.trigger_keywords[:5]}")
        print(f"  body:        {len(skill.body)} chars")
        return 0

    print("Usage: researchclaw skills [list|install|validate]")
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser (without parsing)."""
    parser = argparse.ArgumentParser(
        prog="researchclaw",
        description="ResearchClaw — Autonomous Research Pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the 23-stage research pipeline")
    _ = run_p.add_argument("--topic", "-t", help="Override research topic")
    _ = run_p.add_argument(
        "--config", "-c", default=None,
        help="Config file (default: auto-detect config.arc.yaml or config.yaml)",
    )
    _ = run_p.add_argument("--output", "-o", help="Output directory")
    _ = run_p.add_argument(
        "--from-stage", help="Start from a specific stage (e.g. PAPER_OUTLINE)"
    )
    _ = run_p.add_argument(
        "--to-stage", help="Stop after this stage completes (e.g. EXPERIMENT_DESIGN)"
    )
    _ = run_p.add_argument(
        "--auto-approve", action="store_true", help="Auto-approve gate stages"
    )
    _ = run_p.add_argument(
        "--incremental-experiment",
        action="store_true",
        help=(
            "Enable non-destructive Stage 12 re-entry for hep-ph collider runs: "
            "snapshot the existing collider_workspace, treat the new "
            "collider_plan.md as a delta, and merge new results.json into "
            "the prior one. Use with --from-stage CODE_GENERATION or "
            "--from-stage EXPERIMENT_RUN."
        ),
    )
    _ = run_p.add_argument(
        "--mode", "-m",
        choices=["full-auto", "gate-only", "checkpoint", "step-by-step",
                 "co-pilot", "express", "thorough", "learning"],
        default=None,
        help="HITL intervention mode (overrides config)",
    )
    _ = run_p.add_argument(
        "--skip-preflight", action="store_true", help="Skip LLM preflight check"
    )
    _ = run_p.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    _ = run_p.add_argument(
        "--skip-noncritical-stage", action="store_true",
        help="Skip noncritical stages on failure instead of aborting"
    )
    _ = run_p.add_argument(
        "--no-graceful-degradation", action="store_true",
        help="Disable graceful degradation: fail pipeline on quality gate failure"
    )
    _ = run_p.add_argument(
        "--interventions",
        default=None,
        help="Path to scripted HITL interventions JSON file",
    )
    _ = run_p.add_argument(
        "--profile", "-p",
        default=None,
        help=(
            "Domain profile id to deploy (e.g. hep_ph, ml_vision). The profile "
            "supplies config-level defaults (experiment.mode, export.target_conference, "
            "pip_packages, ...) and forces the domain adapter. Run `researchclaw "
            "profile list` to see available profiles. Overrides project.profile."
        ),
    )
    val_p = sub.add_parser("validate", help="Validate config file")
    _ = val_p.add_argument(
        "--config", "-c", default=None,
        help="Config file (default: auto-detect config.arc.yaml or config.yaml)",
    )
    _ = val_p.add_argument(
        "--no-check-paths", action="store_true", help="Skip path existence checks"
    )

    doc_p = sub.add_parser("doctor", help="Check environment and configuration health")
    _ = doc_p.add_argument(
        "--config", "-c", default=None,
        help="Config file (default: auto-detect config.arc.yaml or config.yaml)",
    )
    _ = doc_p.add_argument("--output", "-o", help="Write JSON report to file")

    init_p = sub.add_parser("init", help="Create config.arc.yaml from example template")
    _ = init_p.add_argument(
        "--force", action="store_true", help="Overwrite existing config.arc.yaml"
    )

    _ = sub.add_parser("setup", help="Check and install optional tools (OpenCode, etc.)")

    # Introspection: show resolved domain / prompt bank / stage extras
    info_p = sub.add_parser(
        "info",
        help="Show resolved config: profile, prompt bank, stage list, extras",
        description=(
            "Summarise the current run configuration — project profile, research "
            "topic, resolved prompt bank (ml or hep_ph), debate role roster, and "
            "the full pipeline stage list with [EXTRA] markers for any stages "
            "that have custom guidance configured via prompts.extra_prompts."
        ),
    )
    _ = info_p.add_argument(
        "--config", "-c", default=None,
        help="Config file (default: auto-detect config.arc.yaml or config.yaml)",
    )
    _ = info_p.add_argument(
        "--profile", "-p", default=None,
        help="Override project.profile for this invocation",
    )

    rpt_p = sub.add_parser("report", help="Generate human-readable run report")
    _ = rpt_p.add_argument(
        "--run-dir", required=True, help="Path to run artifacts directory"
    )
    _ = rpt_p.add_argument("--output", "-o", help="Write report to file")

    # A: Web platform
    srv_p = sub.add_parser("serve", help="Start the web server")
    _ = srv_p.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    _ = srv_p.add_argument("--host", default="", help="Host to bind (default from config)")
    _ = srv_p.add_argument("--port", type=int, default=0, help="Port (default from config)")
    _ = srv_p.add_argument("--monitor-dir", help="Artifacts dir to monitor")

    dash_p = sub.add_parser("dashboard", help="Start dashboard-only server")
    _ = dash_p.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    _ = dash_p.add_argument("--host", default="", help="Host to bind")
    _ = dash_p.add_argument("--port", type=int, default=0, help="Port")
    _ = dash_p.add_argument("--monitor-dir", help="Artifacts dir to monitor")

    wiz_p = sub.add_parser("wizard", help="Run the setup wizard")
    _ = wiz_p.add_argument("--output", "-o", help="Write config to file")

    # C1: Multi-project management
    proj_p = sub.add_parser("project", help="Multi-project management")
    _ = proj_p.add_argument(
        "project_action",
        choices=["list", "status", "create", "switch", "compare"],
        help="Project action",
    )
    _ = proj_p.add_argument("--name", "-n", help="Project name")
    _ = proj_p.add_argument("--names", nargs="*", help="Project names (for compare)")
    _ = proj_p.add_argument("--topic", "-t", help="Research topic")
    _ = proj_p.add_argument(
        "--config", "-c", default="config.yaml", help="Config file path"
    )

    # C3: MCP integration
    mcp_p = sub.add_parser("mcp", help="MCP integration")
    _ = mcp_p.add_argument(
        "--start", action="store_true", help="Start MCP server"
    )

    # C4: Overleaf sync
    ovl_p = sub.add_parser("overleaf", help="Overleaf bidirectional sync")
    _ = ovl_p.add_argument("--sync", action="store_true", help="Run sync")
    _ = ovl_p.add_argument("--status", action="store_true", help="Show status")
    _ = ovl_p.add_argument("--run-dir", help="Run artifacts directory")
    _ = ovl_p.add_argument(
        "--config", "-c", default="config.yaml", help="Config file path"
    )

    # D1: Research trend tracking
    trends_p = sub.add_parser("trends", help="Research trend tracking")
    _ = trends_p.add_argument("--digest", action="store_true", help="Generate daily digest")
    _ = trends_p.add_argument("--analyze", action="store_true", help="Analyze trends")
    _ = trends_p.add_argument(
        "--suggest-topics", action="store_true", help="Suggest research topics"
    )
    _ = trends_p.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    _ = trends_p.add_argument("--domains", nargs="+", help="Override domains")

    # Domain profiles (deployable — bundle prompts + infra defaults per domain)
    prof_p = sub.add_parser(
        "profile",
        help="List, inspect, create, validate, edit, or delete domain profiles",
        description=(
            "Manage domain profiles (hep_ph, ml_vision, ...). Profiles bundle "
            "prompt specialisations and deployment defaults (experiment mode, "
            "target conference, Docker image, pip packages). `create` launches "
            "an interactive wizard with autocomplete suggestions."
        ),
    )
    _ = prof_p.add_argument(
        "profile_action",
        nargs="?",
        default="list",
        choices=[
            "list", "show", "path", "dirs", "schema",
            "create", "validate", "edit", "delete",
        ],
        help=(
            "Action — list: show all profiles; show: print a profile's YAML; "
            "path: print a profile's file path; dirs: list search directories; "
            "schema: dump autocomplete vocabularies (JSON); create: interactive "
            "wizard; validate: check a profile for errors; edit: open profile "
            "in $EDITOR; delete: remove a user-created profile (default: list)"
        ),
    )
    _ = prof_p.add_argument(
        "profile_id",
        nargs="?",
        default=None,
        help="Profile id (required for show/path/validate/edit/delete; "
             "optional for create — wizard will ask)",
    )
    # `create` wizard + non-interactive path
    _ = prof_p.add_argument(
        "--from-yaml",
        default=None,
        metavar="FILE",
        help="create: read profile spec from YAML/JSON file (non-interactive)",
    )
    _ = prof_p.add_argument(
        "--target-dir",
        default=None,
        metavar="DIR",
        help="create: write the new profile into this directory "
             "(default: ~/.researchclaw/profiles/)",
    )
    _ = prof_p.add_argument(
        "--force", "-f",
        action="store_true",
        help="create/delete: overwrite existing or skip confirmation",
    )
    _ = prof_p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="delete: skip the confirmation prompt",
    )
    _ = prof_p.add_argument(
        "--json",
        action="store_true",
        help="schema/list: emit machine-readable JSON",
    )
    _ = prof_p.add_argument(
        "--non-interactive",
        action="store_true",
        help="create: fail instead of prompting (requires --from-yaml or flags)",
    )
    # Direct spec flags for scripted `create` (alternative to --from-yaml)
    _ = prof_p.add_argument("--display-name", default=None)
    _ = prof_p.add_argument("--parent", default=None,
                             help="Parent domain (ml, hep_ph, physics, ...)")
    _ = prof_p.add_argument("--mode", default=None,
                             help="preferred_experiment_mode")
    _ = prof_p.add_argument("--project-mode", default=None,
                             help="preferred_project_mode (docs-first | semi-auto | full-auto)")
    _ = prof_p.add_argument("--venue", default=None,
                             help="preferred_target_conference (e.g. jhep, neurips_2025)")
    _ = prof_p.add_argument("--metric-key", default=None)
    _ = prof_p.add_argument("--metric-direction", default=None,
                             choices=[None, "maximize", "minimize"])
    _ = prof_p.add_argument("--docker-image", default=None)
    _ = prof_p.add_argument("--gpu", action="store_true", default=None,
                             help="Mark gpu_required: true")
    _ = prof_p.add_argument("--time-budget", type=int, default=None,
                             dest="time_budget", metavar="SECS")
    _ = prof_p.add_argument("--max-iter", type=int, default=None,
                             dest="max_iter")
    _ = prof_p.add_argument("--paradigm", default=None,
                             help="experiment_paradigm (comparison, simulation, ...)")
    _ = prof_p.add_argument("--entry-point", default=None,
                             dest="entry_point")
    _ = prof_p.add_argument("--pip", action="append", default=None,
                             metavar="PKG",
                             help="Add a pip package (repeatable)")
    _ = prof_p.add_argument("--library", action="append", default=None,
                             metavar="LIB", dest="libraries",
                             help="Add a core library (repeatable)")
    _ = prof_p.add_argument("--keyword", action="append", default=None,
                             metavar="KW", dest="keywords",
                             help="Add a paper keyword (repeatable)")

    # Skills management
    sk_p = sub.add_parser("skills", help="List, install, or validate skills")
    _ = sk_p.add_argument("skills_action", nargs="?", default="list",
                          choices=["list", "install", "validate"],
                          help="Action to perform (default: list)")
    _ = sk_p.add_argument("source", nargs="?", default=None,
                          help="Path for install/validate")

    # D4: Conference deadline calendar
    cal_p = sub.add_parser("calendar", help="Conference deadline calendar")
    _ = cal_p.add_argument("--upcoming", action="store_true", help="Show upcoming deadlines")
    _ = cal_p.add_argument("--plan", help="Generate submission timeline for a venue")
    _ = cal_p.add_argument("--domains", nargs="+", help="Filter by domain")

    # HITL: Attach to running pipeline
    attach_p = sub.add_parser("attach", help="Attach to a running/paused pipeline for HITL interaction")
    _ = attach_p.add_argument("run_dir", help="Path to run artifacts directory")

    # HITL: Check pipeline status
    status_p = sub.add_parser("status", help="Show pipeline and HITL status")
    _ = status_p.add_argument("run_dir", help="Path to run artifacts directory")

    # HITL: Approve a gate
    approve_p = sub.add_parser("approve", help="Approve the current HITL gate")
    _ = approve_p.add_argument("run_dir", help="Path to run artifacts directory")
    _ = approve_p.add_argument("--message", "-m", default="", help="Approval note")

    # HITL: Reject a gate
    reject_p = sub.add_parser("reject", help="Reject the current HITL gate")
    _ = reject_p.add_argument("run_dir", help="Path to run artifacts directory")
    _ = reject_p.add_argument("--reason", "-r", default="", help="Rejection reason")

    # HITL: Inject guidance
    guide_p = sub.add_parser("guide", help="Inject guidance for a pipeline stage")
    _ = guide_p.add_argument("run_dir", help="Path to run artifacts directory")
    _ = guide_p.add_argument("--stage", "-s", type=int, required=True, help="Target stage number")
    _ = guide_p.add_argument("--message", "-m", required=True, help="Guidance text")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = cast(str | None, args.command)

    if command == "run":
        return cmd_run(args)
    elif command == "validate":
        return cmd_validate(args)
    elif command == "doctor":
        return cmd_doctor(args)
    elif command == "init":
        return cmd_init(args)
    elif command == "setup":
        return cmd_setup(args)
    elif command == "info":
        return cmd_info(args)
    elif command == "report":
        return cmd_report(args)
    elif command == "serve":
        return cmd_serve(args)
    elif command == "dashboard":
        return cmd_dashboard(args)
    elif command == "wizard":
        return cmd_wizard(args)
    elif command == "project":
        return cmd_project(args)
    elif command == "mcp":
        return cmd_mcp(args)
    elif command == "overleaf":
        return cmd_overleaf(args)
    elif command == "trends":
        return cmd_trends(args)
    elif command == "calendar":
        return cmd_calendar(args)
    elif command == "skills":
        return cmd_skills(args)
    elif command == "profile":
        return cmd_profile(args)
    elif command == "attach":
        return cmd_attach(args)
    elif command == "status":
        return cmd_status(args)
    elif command == "approve":
        return cmd_hitl_approve(args)
    elif command == "reject":
        return cmd_hitl_reject(args)
    elif command == "guide":
        return cmd_hitl_guide(args)
    else:
        parser.print_help()
        return 0


# ---------------------------------------------------------------------------
# Profile subcommand
# ---------------------------------------------------------------------------


# ---- Wizard helpers (TTY prompts + autocomplete pick-lists) ---------------

def _is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def _prompt_str(label: str, *, default: str = "", example: str = "") -> str:
    hint_parts: list[str] = []
    if default:
        hint_parts.append(f"default: {default}")
    if example:
        hint_parts.append(f"e.g. {example}")
    hint = f" [{' | '.join(hint_parts)}]" if hint_parts else ""
    try:
        raw = input(f"{label}{hint}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def _prompt_int(label: str, *, default: int = 0, example: str = "") -> int:
    while True:
        raw = _prompt_str(label, default=str(default) if default else "", example=example)
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"  ! '{raw}' is not an integer. Try again.")


def _prompt_bool(label: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = _prompt_str(f"{label} {suffix}").lower()
    if not raw:
        return default
    return raw in ("y", "yes", "true", "1")


def _prompt_choice(
    label: str,
    choices: list[str],
    *,
    default: str = "",
    allow_custom: bool = False,
) -> str:
    """Numbered pick-list. Accepts 1-based number, the value itself, or blank for default."""
    print(f"{label}:")
    for idx, val in enumerate(choices, start=1):
        marker = "*" if val == default else " "
        print(f"  {marker} {idx:>2}. {val}")
    tail = "  (press Enter for default"
    if default:
        tail += f" '{default}'"
    tail += "; type a value not on the list" if allow_custom else ""
    tail += ")"
    print(tail)
    while True:
        raw = _prompt_str("  >").strip()
        if not raw:
            return default
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(choices):
                return choices[i - 1]
            print(f"  ! pick 1..{len(choices)}")
            continue
        if raw in choices:
            return raw
        if allow_custom:
            return raw
        print(f"  ! '{raw}' is not one of the listed options. Try again.")


def _prompt_multi_select(
    label: str,
    choices: list[str],
    *,
    default_selected: list[str] | None = None,
    allow_custom: bool = True,
) -> list[str]:
    """Numbered multi-select. Type '1,2,5' to pick, '+torch,numpy' to add custom."""
    default_selected = list(default_selected or [])
    print(f"{label}:")
    for idx, val in enumerate(choices, start=1):
        marker = "*" if val in default_selected else " "
        print(f"  {marker} {idx:>2}. {val}")
    print(
        "  (comma-separated numbers or names; prefix with '+' to add custom; "
        "Enter = keep the marked defaults; '-' = clear all)"
    )
    raw = _prompt_str("  >").strip()
    if not raw:
        return default_selected
    if raw == "-":
        return []
    selected: list[str] = list(default_selected)
    for tok in (t.strip() for t in raw.split(",") if t.strip()):
        if tok.startswith("+") and allow_custom:
            custom = tok[1:].strip()
            if custom and custom not in selected:
                selected.append(custom)
            continue
        if tok.isdigit():
            i = int(tok)
            if 1 <= i <= len(choices):
                val = choices[i - 1]
                if val not in selected:
                    selected.append(val)
            continue
        if tok in choices:
            if tok not in selected:
                selected.append(tok)
        elif allow_custom:
            if tok not in selected:
                selected.append(tok)
    return selected


_PARENT_DEFAULT_HINTS: dict[str, dict[str, Any]] = {
    "ml": {
        "mode": "docker", "venue": "neurips_2025",
        "metric_key": "accuracy", "metric_dir": "maximize",
        "docker_image": "researchclaw/sandbox-ml:latest",
        "gpu": True, "time_budget": 1800, "max_iter": 5,
    },
    "hep_ph": {
        "mode": "collider_agent", "venue": "jhep",
        "metric_key": "exclusion_95cl", "metric_dir": "maximize",
        "docker_image": "", "gpu": False,
        "time_budget": 7200, "max_iter": 5,
    },
    "physics": {
        "mode": "docker", "venue": "prd",
        "metric_key": "primary_metric", "metric_dir": "maximize",
        "docker_image": "researchclaw/sandbox-generic:latest",
        "gpu": False, "time_budget": 3600, "max_iter": 5,
    },
}


def _wizard_build_spec(initial_id: str = "") -> dict[str, Any]:
    """Interactive profile wizard. Returns a spec dict ready for create_profile."""
    from researchclaw.domains.deploy import (
        keyword_suggestions,
        library_suggestions,
        schema_vocabularies,
        validate_profile_id,
    )

    vocab = schema_vocabularies()
    print("\n=== Create domain profile (interactive) ===")
    print("Press Ctrl+C at any time to abort. Every field has a default or can be left blank.\n")

    # Profile id
    pid = initial_id.strip()
    while True:
        pid = _prompt_str("Profile id (lowercase, digits, underscore)",
                          default=pid, example="my_domain, hep_dark_matter")
        err = validate_profile_id(pid)
        if err is None:
            break
        print(f"  ! {err}")
        pid = ""

    display = _prompt_str(
        "Display name", default=pid.replace("_", " ").title(),
        example="Dark Matter Phenomenology",
    )

    parent = _prompt_choice(
        "Parent domain",
        vocab["parent_domains"],
        default="ml",
        allow_custom=True,
    )
    hints = _PARENT_DEFAULT_HINTS.get(parent, _PARENT_DEFAULT_HINTS["ml"])

    mode = _prompt_choice(
        "Preferred experiment mode (fills experiment.mode)",
        vocab["experiment_modes"],
        default=hints["mode"],
    )
    pmode = _prompt_choice(
        "Preferred project mode (fills project.mode)",
        vocab["project_modes"],
        default="full-auto",
    )
    venue = _prompt_choice(
        "Preferred target conference (fills export.target_conference)",
        vocab["target_conferences"],
        default=hints["venue"],
    )

    tbudget = _prompt_int(
        "Default time budget in seconds (0 = leave unset)",
        default=int(hints["time_budget"]),
        example="1800 for quick ML, 7200 for HEP runs",
    )
    max_iter = _prompt_int(
        "Default max pipeline iterations (0 = leave unset)",
        default=int(hints["max_iter"]),
        example="3, 5, 10",
    )
    metric_key = _prompt_str(
        "Default metric key", default=hints["metric_key"],
        example="accuracy, f1, exclusion_95cl, chi2",
    )
    metric_dir = _prompt_choice(
        "Metric direction",
        vocab["metric_directions"],
        default=hints["metric_dir"],
    )

    docker_image = _prompt_choice(
        "Docker image (blank = no default)",
        vocab["docker_images"] + [""],
        default=hints["docker_image"],
        allow_custom=True,
    )
    gpu_required = _prompt_bool("Require GPU?", default=bool(hints["gpu"]))

    paradigm = _prompt_choice(
        "Experiment paradigm",
        vocab["paradigms"],
        default="comparison",
    )

    pip_packages = _prompt_multi_select(
        "Pip packages pre-installed in sandbox",
        library_suggestions(parent),
        default_selected=library_suggestions(parent)[:5],
    )
    core_libs = _prompt_multi_select(
        "Core libraries (documented in the profile for prompts)",
        library_suggestions(parent),
        default_selected=pip_packages[:4],
    )
    keywords = _prompt_multi_select(
        "Paper keywords (help the detector + LLM)",
        keyword_suggestions(parent),
        default_selected=keyword_suggestions(parent)[:3],
    )

    entry_point = _prompt_str("Entry point filename", default="main.py",
                              example="main.py, run.py, experiment.py")

    spec: dict[str, Any] = {
        "domain_id": pid,
        "display_name": display,
        "parent_domain": parent,
        "preferred_experiment_mode": mode,
        "preferred_project_mode": pmode,
        "preferred_target_conference": venue,
        "default_time_budget_sec": tbudget,
        "default_max_iterations": max_iter,
        "default_metric_key": metric_key,
        "default_metric_direction": metric_dir,
        "docker_image": docker_image,
        "gpu_required": gpu_required,
        "pip_packages": pip_packages,
        "core_libraries": core_libs,
        "paper_keywords": keywords,
        "experiment_paradigm": paradigm,
        "entry_point": entry_point,
    }
    return spec


def _spec_from_flags(args: argparse.Namespace) -> dict[str, Any]:
    """Build a partial spec dict from CLI flags (scripted / non-interactive)."""
    spec: dict[str, Any] = {}
    pid = cast(str | None, args.profile_id)
    if pid:
        spec["domain_id"] = pid
    mapping = {
        "display_name": args.display_name,
        "parent_domain": args.parent,
        "preferred_experiment_mode": args.mode,
        "preferred_project_mode": args.project_mode,
        "preferred_target_conference": args.venue,
        "default_metric_key": args.metric_key,
        "default_metric_direction": args.metric_direction,
        "docker_image": args.docker_image,
        "default_time_budget_sec": args.time_budget,
        "default_max_iterations": args.max_iter,
        "experiment_paradigm": args.paradigm,
        "entry_point": args.entry_point,
    }
    for key, val in mapping.items():
        if val is not None and val != "":
            spec[key] = val
    if args.gpu is True:
        spec["gpu_required"] = True
    if args.pip:
        spec["pip_packages"] = list(args.pip)
    if args.libraries:
        spec["core_libraries"] = list(args.libraries)
    if args.keywords:
        spec["paper_keywords"] = list(args.keywords)
    return spec


# ---- Main dispatcher -------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    """List / show / locate / create / validate / edit / delete domain profiles."""
    from researchclaw.domains.deploy import (
        create_profile,
        default_user_profile_dir,
        delete_profile,
        describe_profile,
        is_package_profile,
        list_deployable_profiles,
        profile_search_dirs,
        resolve_profile_path,
        schema_vocabularies,
        validate_profile_data,
        validate_profile_id,
        writable_profile_dirs,
    )

    action = cast(str, args.profile_action or "list")
    profile_id = cast(str | None, args.profile_id)

    # ---- list ---------------------------------------------------------------
    if action == "list":
        entries = list_deployable_profiles()
        if getattr(args, "json", False):
            import json as _json
            print(_json.dumps(entries, indent=2))
            return 0
        if not entries:
            print("No profiles found.")
            return 0
        width = max(len(e["profile_id"]) for e in entries)
        for entry in entries:
            mode = entry.get("preferred_experiment_mode") or "—"
            venue = entry.get("preferred_target_conference") or "—"
            print(
                f"  {entry['profile_id']:<{width}}  "
                f"{entry['display_name']:<40}  mode={mode:<16} venue={venue}"
            )
        print(f"\nTotal: {len(entries)} profiles")
        print(
            "\nDeploy with:\n"
            "  researchclaw run --profile <id>\n"
            "or set project.profile in config.yaml\n"
            "\nCreate a new profile:\n"
            "  researchclaw profile create                # interactive wizard\n"
            "  researchclaw profile create my_id --from-yaml spec.yaml\n"
            "\nAdd a new domain by dropping <id>.yaml into:\n"
            "  ./profiles/                          (project-local)\n"
            "  ~/.researchclaw/profiles/            (user-wide)\n"
            "  researchclaw/domains/profiles/       (package — read-only)"
        )
        return 0

    # ---- dirs ---------------------------------------------------------------
    if action == "dirs":
        search = profile_search_dirs()
        writable = [p.resolve() for p in writable_profile_dirs()]
        print("Profile search directories (first hit wins):")
        for idx, d in enumerate(search, 1):
            flag = "writable" if d.resolve() in writable else "read-only"
            present = "exists" if d.is_dir() else "missing"
            print(f"  {idx}. {d}   [{flag}, {present}]")
        print("\nDefault target for `profile create`:")
        print(f"  {default_user_profile_dir()}")
        print("\nOverride with RESEARCHCLAW_PROFILES_DIR, or pass --target-dir.")
        return 0

    # ---- schema -------------------------------------------------------------
    if action == "schema":
        vocab = schema_vocabularies()
        if getattr(args, "json", False):
            import json as _json
            print(_json.dumps(vocab, indent=2))
            return 0
        print("Accepted values for profile fields:\n")
        for key, val in vocab.items():
            if isinstance(val, list):
                print(f"{key}:")
                for v in val:
                    print(f"  - {v}")
            elif isinstance(val, dict):
                print(f"{key} (by parent_domain):")
                for parent, items in val.items():
                    print(f"  {parent}: {', '.join(items)}")
            print()
        return 0

    # ---- create -------------------------------------------------------------
    if action == "create":
        spec_source = cast(str | None, args.from_yaml)
        target_dir = (
            Path(cast(str, args.target_dir)).expanduser().resolve()
            if args.target_dir else None
        )

        if spec_source:
            spec_path = Path(spec_source).expanduser().resolve()
            if not spec_path.is_file():
                print(f"Error: --from-yaml file not found: {spec_path}", file=sys.stderr)
                return 1
            try:
                import yaml as _yaml
                spec = _yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # noqa: BLE001
                print(f"Error: could not parse {spec_path}: {exc}", file=sys.stderr)
                return 1
            if not isinstance(spec, dict):
                print(f"Error: {spec_path} must be a YAML/JSON mapping.", file=sys.stderr)
                return 1
            if profile_id and not spec.get("domain_id"):
                spec["domain_id"] = profile_id
        else:
            spec = _spec_from_flags(args)
            need_wizard = not args.non_interactive
            have_enough = bool(spec.get("domain_id"))
            if need_wizard and _is_tty():
                try:
                    spec = _wizard_build_spec(initial_id=spec.get("domain_id", ""))
                except (KeyboardInterrupt, EOFError):
                    print("\nAborted.")
                    return 130
            elif not have_enough:
                print(
                    "Error: profile create needs either --from-yaml, a TTY for "
                    "the interactive wizard, or at least --id/profile_id + "
                    "--display-name via flags.",
                    file=sys.stderr,
                )
                return 1

        # Validate before writing.
        errors = validate_profile_data(spec)
        if errors:
            print("Profile data failed validation:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1

        # Show summary + confirm if interactive.
        pid = spec["domain_id"]
        dest = target_dir or default_user_profile_dir()
        print("\n=== Profile summary ===")
        for key in (
            "domain_id", "display_name", "parent_domain",
            "preferred_experiment_mode", "preferred_project_mode",
            "preferred_target_conference",
            "default_time_budget_sec", "default_max_iterations",
            "default_metric_key", "default_metric_direction",
            "docker_image", "gpu_required", "experiment_paradigm",
            "entry_point",
        ):
            val = spec.get(key, "")
            if val in (None, "", 0, False):
                continue
            print(f"  {key}: {val}")
        for key in ("pip_packages", "core_libraries", "paper_keywords"):
            vals = spec.get(key) or []
            if vals:
                print(f"  {key}: {', '.join(vals)}")
        print(f"  -> will write to: {dest / (pid + '.yaml')}")

        if _is_tty() and not args.force and not args.non_interactive:
            if not _prompt_bool("Write profile?", default=True):
                print("Aborted.")
                return 0

        try:
            written = create_profile(spec, target_dir=target_dir, force=args.force)
        except FileExistsError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print("Tip: pass --force to overwrite, or pick a different id.",
                  file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"\n✓ Created profile '{pid}' at {written}")
        print("\nNext steps:")
        print(f"  researchclaw profile show {pid}")
        print(f"  researchclaw run --profile {pid}")
        print("  (or set project.profile: " + pid + " in config.yaml)")
        return 0

    # ---- validate -----------------------------------------------------------
    if action == "validate":
        if not profile_id:
            print("Error: 'profile validate' requires a profile id.", file=sys.stderr)
            return 1
        try:
            from researchclaw.domains.deploy import load_profile_yaml
            data = load_profile_yaml(profile_id)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        errors = validate_profile_data(data)
        if errors:
            print(f"✗ Profile '{profile_id}' has {len(errors)} issue(s):")
            for e in errors:
                print(f"  - {e}")
            return 1
        print(f"✓ Profile '{profile_id}' is valid.")
        return 0

    # ---- edit ---------------------------------------------------------------
    if action == "edit":
        if not profile_id:
            print("Error: 'profile edit' requires a profile id.", file=sys.stderr)
            return 1
        path = resolve_profile_path(profile_id)
        if path is None:
            print(f"Error: profile '{profile_id}' not found.", file=sys.stderr)
            return 1
        if is_package_profile(profile_id):
            print(
                f"Note: {path} is a bundled profile (read-only). Copy it into "
                f"{default_user_profile_dir()} first to override.",
                file=sys.stderr,
            )
        editor = os.environ.get("EDITOR", "").strip()
        if not editor:
            print(f"$EDITOR not set — open the file manually:\n  {path}")
            return 0
        import subprocess
        rc = subprocess.call([editor, str(path)])
        return rc

    # ---- delete -------------------------------------------------------------
    if action == "delete":
        if not profile_id:
            print("Error: 'profile delete' requires a profile id.", file=sys.stderr)
            return 1
        err = validate_profile_id(profile_id)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        path = resolve_profile_path(profile_id)
        if path is None:
            print(f"Error: profile '{profile_id}' not found.", file=sys.stderr)
            return 1
        if is_package_profile(profile_id):
            print(
                f"Refusing to delete bundled profile at {path}.\n"
                f"To hide it, drop a shadowing <id>.yaml into "
                f"{default_user_profile_dir()} or ./profiles/.",
                file=sys.stderr,
            )
            return 1
        if not args.yes and not args.force and _is_tty():
            if not _prompt_bool(f"Delete profile '{profile_id}' at {path}?",
                                default=False):
                print("Aborted.")
                return 0
        try:
            removed = delete_profile(profile_id)
        except (PermissionError, ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"✓ Deleted {removed}")
        return 0

    # ---- show / path (need profile_id) --------------------------------------
    if not profile_id:
        print(
            f"Error: 'profile {action}' requires a profile id.\n"
            "Run 'researchclaw profile list' to see available profiles.",
            file=sys.stderr,
        )
        return 1

    path = resolve_profile_path(profile_id)
    if path is None:
        print(f"Error: profile '{profile_id}' not found.", file=sys.stderr)
        return 1

    if action == "path":
        print(path)
        return 0

    if action == "show":
        info = describe_profile(profile_id)
        print(f"# Profile: {info['display_name']} ({info['profile_id']})")
        print(f"# Source:  {info['source_path']}")
        bundle = " [bundled, read-only]" if is_package_profile(profile_id) else ""
        print(f"# Status:  {'user-created' if not bundle else 'bundled'}{bundle}\n")
        print("# Deployment summary:")
        print(f"#   experiment.mode          = {info['preferred_experiment_mode'] or '(unset)'}")
        print(f"#   project.mode             = {info['preferred_project_mode'] or '(unset)'}")
        print(f"#   export.target_conference = {info['preferred_target_conference'] or '(unset)'}")
        print(f"#   docker.image             = {info['docker_image'] or '(unset)'}")
        print(f"#   docker.gpu_enabled       = {info['gpu_required']}")
        if info["pip_packages"]:
            print(f"#   pip_pre_install          = {', '.join(info['pip_packages'])}")
        print("\n# --- raw YAML ---")
        print(path.read_text(encoding="utf-8").rstrip())
        return 0

    print(f"Unknown profile action: {action}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# HITL subcommands
# ---------------------------------------------------------------------------


def cmd_attach(args: argparse.Namespace) -> int:
    """Attach to a running/paused pipeline for interactive HITL."""
    run_dir = Path(cast(str, args.run_dir))
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    from researchclaw.hitl.store import HITLStore
    from researchclaw.hitl.tui.panel import show_pipeline_status, show_intervention_log

    store = HITLStore(run_dir)
    waiting = store.load_waiting()

    # Show current status
    session_data = store.load_session()
    mode = session_data.get("mode", "unknown") if session_data else "unknown"
    show_pipeline_status(run_dir, mode=mode)
    print()

    if waiting is None:
        print("  Pipeline is not waiting for input.")
        print("  Use 'researchclaw status' for full details.")
        return 0

    # Pipeline is waiting — enter interactive mode
    print(f"  Pipeline is paused at Stage {waiting['stage']} ({waiting.get('stage_name', '?')})")
    print(f"  Reason: {waiting.get('reason', '?')}")
    print()

    from researchclaw.hitl.intervention import WaitingState, PauseReason
    from researchclaw.hitl.adapters.cli_adapter import CLIAdapter

    ws = WaitingState.from_dict(waiting)
    adapter = CLIAdapter(run_dir=run_dir)
    human_input = adapter.collect_input(ws)

    # Write response for the pipeline process to pick up
    import json

    response_path = run_dir / "hitl" / "response.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(human_input.to_dict(), indent=2), encoding="utf-8"
    )
    print(f"\n  Response saved. Pipeline will pick it up automatically.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show pipeline and HITL status."""
    run_dir = Path(cast(str, args.run_dir))
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    from researchclaw.hitl.store import HITLStore
    from researchclaw.hitl.tui.panel import show_pipeline_status, show_intervention_log

    store = HITLStore(run_dir)
    session_data = store.load_session()
    mode = session_data.get("mode", "unknown") if session_data else "N/A"

    show_pipeline_status(run_dir, mode=mode)
    print()

    summary = store.get_summary()
    print(f"  HITL interventions: {summary['intervention_count']}")
    print(f"  Chat sessions: stages {summary['chat_stages']}")
    print(f"  Guidance injected: stages {summary['guidance_stages']}")
    print(f"  Snapshots: {summary['snapshot_count']}")

    if store.is_waiting():
        waiting = store.load_waiting()
        if waiting:
            print(f"\n  ⚠ WAITING for input at Stage {waiting['stage']}")
            print(f"    Reason: {waiting.get('reason', '?')}")
            print(f"    Since: {waiting.get('since', '?')}")
            print(f"    Use 'researchclaw attach {run_dir}' to respond.")

    print()
    show_intervention_log(run_dir)
    return 0


def cmd_hitl_approve(args: argparse.Namespace) -> int:
    """Approve the current HITL gate (non-interactive)."""
    run_dir = Path(cast(str, args.run_dir))
    message = cast(str, args.message)

    import json

    response = {"action": "approve", "message": message}
    response_path = run_dir / "hitl" / "response.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(response, indent=2), encoding="utf-8"
    )
    print(f"  Approved. Response saved to {response_path}")
    return 0


def cmd_hitl_reject(args: argparse.Namespace) -> int:
    """Reject the current HITL gate (non-interactive)."""
    run_dir = Path(cast(str, args.run_dir))
    reason = cast(str, args.reason)

    import json

    response = {"action": "reject", "message": reason}
    response_path = run_dir / "hitl" / "response.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(response, indent=2), encoding="utf-8"
    )
    print(f"  Rejected. Response saved to {response_path}")
    return 0


def cmd_hitl_guide(args: argparse.Namespace) -> int:
    """Inject guidance for a pipeline stage."""
    run_dir = Path(cast(str, args.run_dir))
    stage = cast(int, args.stage)
    message = cast(str, args.message)

    from researchclaw.hitl.store import HITLStore

    store = HITLStore(run_dir)
    store.save_guidance(stage, message)

    # Also write to stage dir
    stage_dir = run_dir / f"stage-{stage:02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "hitl_guidance.md").write_text(message, encoding="utf-8")

    print(f"  Guidance saved for Stage {stage} ({len(message)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
