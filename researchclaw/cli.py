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
    if shutil.which("opencode") is None:
        return False
    try:
        r = subprocess.run(
            ["opencode", "--version"],
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
    try:
        r = subprocess.run(
            ["npm", "i", "-g", "opencode-ai@latest"],
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

_PROVIDER_CHOICES = {
    "1": ("openai", "OPENAI_API_KEY"),
    "2": ("openrouter", "OPENROUTER_API_KEY"),
    "3": ("deepseek", "DEEPSEEK_API_KEY"),
    "4": ("chatgpt", ""),
    "5": ("acp", ""),
}

_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

_PROVIDER_MODELS = {
    "openai": ("gpt-4o", ["gpt-4.1", "gpt-4o-mini"]),
    "openrouter": (
        "anthropic/claude-3.5-sonnet",
        ["google/gemini-pro-1.5", "meta-llama/llama-3.1-70b-instruct"],
    ),
    "deepseek": ("deepseek-chat", ["deepseek-reasoner"]),
    "chatgpt": ("gpt-4o", ["gpt-4.1", "gpt-4o-mini"]),
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
        print("  4) chatgpt      (ChatGPT Plus/Pro subscription — browser login)")
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

    if provider in ("acp", "chatgpt"):
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

    if provider == "chatgpt":
        print("\nNext steps:")
        print("  1. Install ChatGPT extras: pip install 'researchclaw[chatgpt]'")
        print("  2. Login with your subscription: researchclaw login")
        print("  3. Edit config.arc.yaml to customize your settings")
        print("  4. Run: researchclaw doctor")
    elif provider == "acp":
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
            r = subprocess.run(
                ["opencode", "--version"],
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


_CHATGPT_TOS_DISCLAIMER = (
    "DISCLAIMER: This feature uses OpenAI's public OAuth endpoints to "
    "authenticate with your ChatGPT Plus/Pro subscription. By proceeding, "
    "you confirm that your use complies with OpenAI's Terms of Service "
    "(https://openai.com/policies/terms-of-use). ResearchClaw is not "
    "affiliated with OpenAI."
)


def cmd_login(args: argparse.Namespace) -> int:
    """Authenticate with ChatGPT Plus/Pro subscription via browser OAuth."""
    from researchclaw.llm.chatgpt_oauth import load_auth, run_oauth_flow

    existing = load_auth()
    if existing and not existing.expired:
        print("Already logged in with a valid ChatGPT session.")
        print(f"  Account: {existing.account_id or 'unknown'}")
        print("  Use 'researchclaw logout' to clear credentials.")
        return 0

    print(_CHATGPT_TOS_DISCLAIMER)
    print()

    if sys.stdin.isatty():
        try:
            confirm = input("Continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = ""
        if confirm not in ("y", "yes"):
            print("Login cancelled.")
            return 1

    tokens = run_oauth_flow()
    print(f"\nLogged in successfully!")
    print(f"  Account: {tokens.account_id or 'unknown'}")
    print(f"  Tokens stored in OS keychain (service: researchclaw-chatgpt)")
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    """Clear stored ChatGPT authentication credentials."""
    from researchclaw.llm.chatgpt_oauth import clear_auth

    clear_auth()
    print("ChatGPT credentials cleared from OS keychain.")
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

    _ = sub.add_parser("login", help="Login with ChatGPT Plus/Pro subscription (browser OAuth)")
    _ = sub.add_parser("logout", help="Clear stored ChatGPT credentials")

    rpt_p = sub.add_parser("report", help="Generate human-readable run report")
    _ = rpt_p.add_argument(
        "--run-dir", required=True, help="Path to run artifacts directory"
    )
    _ = rpt_p.add_argument("--output", "-o", help="Write report to file")
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
    elif command == "login":
        return cmd_login(args)
    elif command == "logout":
        return cmd_logout(args)
    elif command == "report":
        return cmd_report(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
