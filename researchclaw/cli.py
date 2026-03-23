"""ResearchClaw CLI — run the 23-stage autonomous research pipeline."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import cast

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
    auto_approve = cast(bool, args.auto_approve)
    skip_preflight = cast(bool, args.skip_preflight)
    resume = cast(bool, args.resume)
    skip_noncritical = cast(bool, args.skip_noncritical_stage)
    no_graceful_degradation = cast(bool, args.no_graceful_degradation)

    kb_root_path = None
    config = RCConfig.load(config_path, check_paths=False)

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

    # BUG-119: When --resume without --output, search for the most recent
    # existing run directory that matches the topic and has a checkpoint.
    if resume and not output:
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
                print(f"Found existing run to resume: {run_dir}")
            else:
                print(
                    "Warning: --resume specified but no checkpoint found "
                    f"for topic hash '{topic_hash}'. Starting new run.",
                    file=sys.stderr,
                )

    run_dir.mkdir(parents=True, exist_ok=True)

    if config.knowledge_base.root:
        kb_root_path = Path(config.knowledge_base.root)
        kb_root_path.mkdir(parents=True, exist_ok=True)

    adapters = AdapterBundle()

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

    from researchclaw import __version__
    print(f"ResearchClaw v{__version__} — Starting pipeline")
    print(f"  Run ID:  {run_id}")
    print(f"  Topic:   {config.research.topic}")
    print(f"  Output:  {run_dir}")
    print(f"  Mode:    {config.project.mode}")
    print(f"  From:    Stage {int(from_stage)}: {from_stage.name}")

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
        auto_approve_gates=auto_approve,
        stop_on_gate=stop_on_gate,
        skip_noncritical=skip_noncritical,
        kb_root=kb_root_path,
    )

    done = sum(1 for r in results if r.status.value == "done")
    failed = sum(1 for r in results if r.status.value == "failed")
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
}

_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "minimax": "https://api.minimax.io/v1",
}

_PROVIDER_MODELS = {
    "openai": ("gpt-4o", ["gpt-4.1", "gpt-4o-mini"]),
    "openrouter": (
        "anthropic/claude-3.5-sonnet",
        ["google/gemini-pro-1.5", "meta-llama/llama-3.1-70b-instruct"],
    ),
    "deepseek": ("deepseek-chat", ["deepseek-reasoner"]),
    "minimax": ("MiniMax-M2.5", ["MiniMax-M2.5-highspeed"]),
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
        print("  4) minimax      (requires MINIMAX_API_KEY)")
        print("  5) acp          (local AI agent — no API key needed)")
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

def main(argv: list[str] | None = None) -> int:
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
        "--auto-approve", action="store_true", help="Auto-approve gate stages"
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

    # D4: Conference deadline calendar
    cal_p = sub.add_parser("calendar", help="Conference deadline calendar")
    _ = cal_p.add_argument("--upcoming", action="store_true", help="Show upcoming deadlines")
    _ = cal_p.add_argument("--plan", help="Generate submission timeline for a venue")
    _ = cal_p.add_argument("--domains", nargs="+", help="Filter by domain")

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
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
