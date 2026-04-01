"""ResearchClaw config loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import sys
import yaml

DEFAULT_PYTHON_PATH = (
    ".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python3"
)

CONFIG_SEARCH_ORDER: tuple[str, ...] = ("config.arc.yaml", "config.yaml")


def _safe_int(val: Any, default: int) -> int:
    """Convert value to int, handling None/null YAML values."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


_VALID_NETWORK_POLICIES = {"none", "setup_only", "pip_only", "full"}


def _validate_network_policy(val: object, default: str = "setup_only") -> str:
    """Validate network_policy, falling back to *default* on bad values."""
    s = str(val).strip().lower() if val else default
    if s not in _VALID_NETWORK_POLICIES:
        import logging as _cfg_log

        _cfg_log.getLogger(__name__).warning(
            "Invalid network_policy %r, using %r",
            val,
            default,
        )
        return default
    return s


def _safe_float(val: Any, default: float) -> float:
    """Convert value to float, handling None/null YAML values.

    BUG-DA8-11: Also rejects NaN/Inf which YAML can produce via .nan/.inf.
    """
    if val is None:
        return default
    try:
        import math

        result = float(val)
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


EXAMPLE_CONFIG = "config.researchclaw.example.yaml"


def resolve_config_path(explicit: str | None) -> Path | None:
    """Return first existing config from search order, or explicit path if given."""
    if explicit is not None:
        return Path(explicit)
    for name in CONFIG_SEARCH_ORDER:
        candidate = Path(name)
        if candidate.exists():
            return candidate
    return None


REQUIRED_FIELDS = (
    "project.name",
    "research.topic",
    "runtime.timezone",
    "notifications.channel",
    "knowledge_base.root",
    "llm.base_url",
    "llm.api_key_env",
)
KB_SUBDIRS = (
    "questions",
    "literature",
    "experiments",
    "findings",
    "decisions",
    "reviews",
)
PROJECT_MODES = {"docs-first", "semi-auto", "full-auto"}
KB_BACKENDS = {"markdown", "obsidian"}
EXPERIMENT_MODES = {
    "simulated",
    "sandbox",
    "docker",
    "ssh_remote",
    "colab_drive",
    "agentic",
}
CLI_AGENT_PROVIDERS = {"llm", "claude_code", "codex"}


def _get_by_path(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    mode: str = "docs-first"


@dataclass(frozen=True)
class ResearchConfig:
    topic: str
    domains: tuple[str, ...] = ()
    daily_paper_count: int = 0
    quality_threshold: float = 0.0
    graceful_degradation: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    timezone: str
    max_parallel_tasks: int = 1
    approval_timeout_hours: int = 12
    retry_limit: int = 0


@dataclass(frozen=True)
class NotificationsConfig:
    channel: str
    target: str = ""
    on_stage_start: bool = False
    on_stage_fail: bool = False
    on_gate_required: bool = True


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    backend: str
    root: str
    obsidian_vault: str = ""


@dataclass(frozen=True)
class OpenClawBridgeConfig:
    use_cron: bool = False
    use_message: bool = False
    use_memory: bool = False
    use_sessions_spawn: bool = False
    use_web_fetch: bool = False
    use_browser: bool = False


@dataclass(frozen=True)
class AcpConfig:
    """ACP (Agent Client Protocol) settings."""

    agent: str = "claude"
    cwd: str = "."
    acpx_command: str = ""
    session_name: str = "researchclaw"
    timeout_sec: int = 1800


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    base_url: str = ""
    wire_api: str = "chat_completions"
    api_key_env: str = ""
    api_key: str = ""
    primary_model: str = ""
    fallback_models: tuple[str, ...] = ()
    s2_api_key: str = ""
    notes: str = ""
    acp: AcpConfig = field(default_factory=AcpConfig)


@dataclass(frozen=True)
class SecurityConfig:
    hitl_required_stages: tuple[int, ...] = (5, 9, 20)
    allow_publish_without_approval: bool = False
    redact_sensitive_logs: bool = True


@dataclass(frozen=True)
class SandboxConfig:
    python_path: str = DEFAULT_PYTHON_PATH
    gpu_required: bool = False
    allowed_imports: tuple[str, ...] = (
        "math",
        "random",
        "json",
        "csv",
        "numpy",
        "torch",
        "sklearn",
    )
    max_memory_mb: int = 4096


@dataclass(frozen=True)
class SshRemoteConfig:
    host: str = ""
    user: str = ""
    port: int = 22
    key_path: str = ""
    gpu_ids: tuple[int, ...] = ()
    remote_workdir: str = "/tmp/researchclaw_experiments"
    remote_python: str = "python3"
    setup_commands: tuple[str, ...] = ()
    use_docker: bool = False
    docker_image: str = "researchclaw/experiment:latest"
    docker_network_policy: str = "none"
    docker_memory_limit_mb: int = 8192
    docker_shm_size_mb: int = 2048
    timeout_sec: int = 600  # default 10 min for experiment execution
    scp_timeout_sec: int = 300  # default 5 min for file uploads
    setup_timeout_sec: int = 300  # default 5 min for setup commands


@dataclass(frozen=True)
class ColabDriveConfig:
    """Configuration for Google Drive-based async Colab execution."""

    drive_root: str = ""  # local mount path, e.g. ~/Google Drive/MyDrive/researchclaw
    poll_interval_sec: int = 30
    timeout_sec: int = 3600
    setup_script: str = ""  # commands to run before experiment, written to setup.sh


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Configuration for Docker-based experiment sandbox."""

    image: str = "researchclaw/experiment:latest"
    gpu_enabled: bool = True
    gpu_device_ids: tuple[int, ...] = ()
    memory_limit_mb: int = 8192
    network_policy: str = "setup_only"  # none | setup_only | pip_only | full
    pip_pre_install: tuple[str, ...] = ()
    auto_install_deps: bool = True
    shm_size_mb: int = 2048
    container_python: str = "/usr/bin/python3"
    keep_containers: bool = False


@dataclass(frozen=True)
class AgenticConfig:
    """Configuration for the agentic experiment mode.

    Launches a coding agent (e.g. Claude Code) inside a Docker container
    with full shell access so it can run arbitrary CLI commands, write code,
    and iteratively complete the experiment.
    """

    image: str = "researchclaw/experiment:latest"
    agent_cli: str = "claude"
    agent_install_cmd: str = "npm install -g @anthropic-ai/claude-code"
    network_policy: str = "full"  # Agent needs network access
    timeout_sec: int = 1800  # 30 min per session
    memory_limit_mb: int = 8192
    gpu_enabled: bool = False
    mount_skills: bool = True
    allow_shell_commands: bool = True
    max_turns: int = 50


@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for the advanced multi-phase code generation agent."""

    enabled: bool = True
    # Phase 1: Blueprint planning (deep implementation blueprint)
    architecture_planning: bool = True
    # Phase 2: Sequential file generation (one-by-one following blueprint)
    sequential_generation: bool = True
    # Phase 2.5: Hard validation gates (AST-based)
    hard_validation: bool = True
    hard_validation_max_repairs: int = 4
    # Phase 3: Execution-in-the-loop (run → parse error → fix)
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60
    # Phase 4: Solution tree search (off by default — higher cost)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2
    tree_search_eval_timeout_sec: int = 120
    # Phase 5: Multi-agent review dialog
    review_max_rounds: int = 2


@dataclass(frozen=True)
class OpenCodeConfig:
    """OpenCode 'Beast Mode' — external AI coding agent for complex experiments.

    Requires: npm i -g opencode-ai@latest
    """

    enabled: bool = True
    auto: bool = True  # Auto-trigger without user confirmation
    complexity_threshold: float = 0.2  # 0.0-1.0
    model: str = ""  # Empty = use llm.primary_model
    timeout_sec: int = 600  # Max seconds for opencode run
    max_retries: int = 1
    workspace_cleanup: bool = True


@dataclass(frozen=True)
class BenchmarkAgentConfig:
    """Configuration for the BenchmarkAgent multi-agent system."""

    enabled: bool = True
    # Surveyor
    enable_hf_search: bool = True
    max_hf_results: int = 10
    # Surveyor — web search
    enable_web_search: bool = True
    max_web_results: int = 5
    web_search_min_local: int = 3  # skip web search when local benchmarks >= this
    # Selector
    tier_limit: int = 2
    min_benchmarks: int = 1
    min_baselines: int = 2
    prefer_cached: bool = True
    # Orchestrator
    max_iterations: int = 2


@dataclass(frozen=True)
class FigureAgentConfig:
    """Configuration for the FigureAgent multi-agent system."""

    enabled: bool = True
    # Planner
    min_figures: int = 3
    max_figures: int = 8
    # Orchestrator
    max_iterations: int = 3  # max CodeGen→Renderer→Critic retry loops
    # Renderer security
    render_timeout_sec: int = 30
    use_docker: bool | None = None  # None = auto-detect, True/False to force
    docker_image: str = "researchclaw/experiment:latest"
    # Code generation output format
    output_format: str = "python"  # "python" (matplotlib) or "latex" (TikZ/PGFPlots)
    # Nano Banana (Gemini image generation)
    gemini_api_key: str = ""  # or set GEMINI_API_KEY / GOOGLE_API_KEY env var
    gemini_model: str = "gemini-2.5-flash-image"
    nano_banana_enabled: bool = True  # enable/disable Gemini image generation
    # Critic
    strict_mode: bool = False
    # Output
    dpi: int = 300


@dataclass(frozen=True)
class ExperimentRepairConfig:
    """Experiment repair loop — diagnose and fix failed experiments before paper writing.

    When enabled, after Stage 14 (result_analysis) the pipeline:
    1. Diagnoses experiment failures (missing deps, crashes, OOM, time guard, etc.)
    2. Assesses experiment quality (full_paper / preliminary_study / technical_report)
    3. If quality is insufficient, generates targeted repair prompts
    4. Re-runs experiment with fixes, up to ``max_cycles`` times
    5. Selects best results across all cycles for paper writing
    """

    enabled: bool = True
    max_cycles: int = 3
    min_completion_rate: float = 0.5  # At least 50% conditions must complete
    min_conditions: int = 2  # At least 2 conditions for a valid experiment
    use_opencode: bool = True  # Use OpenCode agent for repairs (vs LLM prompt)
    timeout_sec_per_cycle: int = 600  # Max time per repair cycle


@dataclass(frozen=True)
class CliAgentConfig:
    """CLI-based code generation backend for Stages 10 & 13.

    provider: "llm"          — use existing LLM chat API (default, backward-compatible)
              "claude_code"  — Claude Code CLI (``claude -p``)
              "codex"        — OpenAI Codex CLI (``codex exec``)

    Auth for claude_code: ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL env vars.
    Auth for codex:       OPENAI_API_KEY env var.
    """

    provider: str = "llm"
    binary_path: str = ""  # auto-detected via PATH if empty
    model: str = ""  # model override for the CLI agent
    max_budget_usd: float = 5.0
    timeout_sec: int = 600
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "simulated"
    time_budget_sec: int = 300
    max_iterations: int = 10
    max_refine_duration_sec: int = 0  # 0 = auto (3× time_budget_sec)
    metric_key: str = "primary_metric"
    metric_direction: str = "minimize"
    keep_threshold: float = 0.0
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    docker: DockerSandboxConfig = field(default_factory=DockerSandboxConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    ssh_remote: SshRemoteConfig = field(default_factory=SshRemoteConfig)
    colab_drive: ColabDriveConfig = field(default_factory=ColabDriveConfig)
    code_agent: CodeAgentConfig = field(default_factory=CodeAgentConfig)
    opencode: OpenCodeConfig = field(default_factory=OpenCodeConfig)
    benchmark_agent: BenchmarkAgentConfig = field(default_factory=BenchmarkAgentConfig)
    figure_agent: FigureAgentConfig = field(default_factory=FigureAgentConfig)
    repair: ExperimentRepairConfig = field(default_factory=ExperimentRepairConfig)
    cli_agent: CliAgentConfig = field(default_factory=CliAgentConfig)


@dataclass(frozen=True)
class MetaClawPRMConfig:
    """PRM quality gate settings for MetaClaw bridge."""

    enabled: bool = False
    api_base: str = ""
    api_key_env: str = ""
    api_key: str = ""
    model: str = "gpt-5.4"
    votes: int = 3
    temperature: float = 0.6
    gate_stages: tuple[int, ...] = (5, 9, 15, 20)


@dataclass(frozen=True)
class MetaClawLessonToSkillConfig:
    """Settings for converting lessons into MetaClaw skills."""

    enabled: bool = True
    min_severity: str = "warning"
    max_skills_per_run: int = 3


@dataclass(frozen=True)
class MetaClawBridgeConfig:
    """MetaClaw integration bridge configuration."""

    enabled: bool = False
    proxy_url: str = "http://localhost:30000"
    skills_dir: str = "~/.metaclaw/skills"
    fallback_url: str = ""
    fallback_api_key: str = ""
    prm: MetaClawPRMConfig = field(default_factory=MetaClawPRMConfig)
    lesson_to_skill: MetaClawLessonToSkillConfig = field(
        default_factory=MetaClawLessonToSkillConfig
    )


@dataclass(frozen=True)
class WebSearchConfig:
    """Configuration for web search and crawling capabilities."""

    enabled: bool = True
    tavily_api_key: str = ""
    tavily_api_key_env: str = "TAVILY_API_KEY"
    enable_scholar: bool = True
    enable_crawling: bool = True
    enable_pdf_extraction: bool = True
    max_web_results: int = 10
    max_scholar_results: int = 10
    max_crawl_urls: int = 5


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for paper export and LaTeX generation."""

    target_conference: str = "neurips_2025"
    authors: str = "Anonymous"
    bib_file: str = "references"


@dataclass(frozen=True)
class PromptsConfig:
    """Configuration for prompt externalization."""

    custom_file: str = ""  # Path to custom prompts YAML (empty = use defaults)


# ── Agent B: Intelligence & Memory configs ────────────────────────


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for the persistent evolutionary memory system."""

    enabled: bool = True
    store_dir: str = ".researchclaw/memory"
    embedding_model: str = "text-embedding-3-small"
    max_entries_per_category: int = 500
    decay_half_life_days: int = 90
    confidence_threshold: float = 0.3
    inject_at_stages: tuple[int, ...] = (1, 9, 10, 17)


@dataclass(frozen=True)
class SkillsConfig:
    """Configuration for the dynamic skills library."""

    enabled: bool = True
    builtin_dir: str = ""  # empty = use package default
    custom_dirs: tuple[str, ...] = ()
    external_dirs: tuple[str, ...] = ()
    auto_match: bool = True
    max_skills_per_stage: int = 3
    fallback_matching: bool = True


@dataclass(frozen=True)
class KnowledgeGraphConfig:
    """Configuration for the research knowledge graph."""

    enabled: bool = False
    store_path: str = ".researchclaw/knowledge_graph"
    max_entities: int = 10000
    auto_update: bool = True


# ── Web platform configs (Agent A) ──────────────────────────────


@dataclass(frozen=True)
class ServerConfig:
    """Web server configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: tuple[str, ...] = ("*",)
    auth_token: str = ""  # empty = no authentication
    voice_enabled: bool = False
    whisper_model: str = "whisper-1"
    whisper_api_url: str = ""  # empty = use OpenAI default


@dataclass(frozen=True)
class DashboardConfig:
    """Dashboard configuration."""

    enabled: bool = True
    refresh_interval_sec: int = 5
    max_log_lines: int = 1000
    browser_notifications: bool = True


# ── Agent C: Infrastructure configs ────────────────────────────────


@dataclass(frozen=True)
class MultiProjectConfig:
    """C1: Multi-project parallel management."""

    enabled: bool = False
    projects_dir: str = ".researchclaw/projects"
    max_concurrent: int = 2
    shared_knowledge: bool = True


@dataclass(frozen=True)
class ServerEntryConfig:
    """Single compute server entry for C2."""

    name: str = ""
    host: str = ""
    server_type: str = "ssh"
    gpu: str = ""
    vram_gb: int = 0
    priority: int = 1
    cost_per_hour: float = 0.0
    scheduler: str = ""
    cloud_provider: str = ""


@dataclass(frozen=True)
class ServersConfig:
    """C2: Multi-server resource scheduling."""

    enabled: bool = False
    servers: tuple[ServerEntryConfig, ...] = ()
    prefer_free: bool = True
    failover: bool = True
    monitor_interval_sec: int = 60


@dataclass(frozen=True)
class MCPIntegrationConfig:
    """C3: MCP standardized integration."""

    server_enabled: bool = False
    server_port: int = 3000
    server_transport: str = "stdio"
    external_servers: tuple[dict, ...] = ()


@dataclass(frozen=True)
class OverleafConfig:
    """C4: Overleaf bidirectional sync."""

    enabled: bool = False
    git_url: str = ""
    branch: str = "main"
    auto_push: bool = True
    auto_pull: bool = False
    poll_interval_sec: int = 300


COPILOT_MODES = ("co-pilot", "auto-pilot", "zero-touch")


@dataclass(frozen=True)
class TrendsConfig:
    """D1: Research trend tracking."""

    enabled: bool = False
    domains: tuple[str, ...] = ()
    daily_digest: bool = True
    digest_time: str = "08:00"
    max_papers_per_day: int = 20
    trend_window_days: int = 30
    sources: tuple[str, ...] = ("arxiv", "semantic_scholar")


@dataclass(frozen=True)
class CoPilotConfig:
    """D2: Interactive co-pilot mode."""

    mode: str = "auto-pilot"
    pause_at_gates: bool = True
    pause_at_every_stage: bool = False
    feedback_timeout_sec: int = 3600
    allow_branching: bool = True
    max_branches: int = 3


@dataclass(frozen=True)
class QualityAssessorConfig:
    """D3: Paper quality assessor."""

    enabled: bool = True
    dimensions: tuple[str, ...] = (
        "novelty",
        "rigor",
        "clarity",
        "impact",
        "experiments",
    )
    venue_recommendation: bool = True
    score_history: bool = True


@dataclass(frozen=True)
class CalendarConfig:
    """D4: Conference deadline calendar."""

    enabled: bool = False
    target_venues: tuple[str, ...] = ()
    reminder_days_before: tuple[int, ...] = (30, 14, 7, 3, 1)
    auto_plan: bool = True


@dataclass(frozen=True)
class RCConfig:
    project: ProjectConfig
    research: ResearchConfig
    runtime: RuntimeConfig
    notifications: NotificationsConfig
    knowledge_base: KnowledgeBaseConfig
    openclaw_bridge: OpenClawBridgeConfig
    llm: LlmConfig
    security: SecurityConfig = field(default_factory=SecurityConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    metaclaw_bridge: MetaClawBridgeConfig = field(default_factory=MetaClawBridgeConfig)
    # Agent B: Intelligence & Memory
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    # Agent C: Infrastructure
    multi_project: MultiProjectConfig = field(default_factory=MultiProjectConfig)
    compute_servers: ServersConfig = field(default_factory=ServersConfig)
    mcp: MCPIntegrationConfig = field(default_factory=MCPIntegrationConfig)
    overleaf: OverleafConfig = field(default_factory=OverleafConfig)
    # Agent A: Web platform
    server: ServerConfig = field(default_factory=ServerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    # Agent D: Research Enhancement
    trends: TrendsConfig = field(default_factory=TrendsConfig)
    copilot: CoPilotConfig = field(default_factory=CoPilotConfig)
    quality_assessor: QualityAssessorConfig = field(
        default_factory=QualityAssessorConfig
    )
    calendar: CalendarConfig = field(default_factory=CalendarConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        project_root: Path | None = None,
        check_paths: bool = True,
    ) -> RCConfig:
        result = validate_config(
            data, project_root=project_root, check_paths=check_paths
        )
        if not result.ok:
            raise ValueError("; ".join(result.errors))

        project = data["project"]
        research = data["research"]
        runtime = data["runtime"]
        notifications = data["notifications"]
        knowledge_base = data["knowledge_base"]
        bridge = data.get("openclaw_bridge") or {}
        llm = data["llm"]
        security = data.get("security") or {}
        experiment = data.get("experiment") or {}
        export = data.get("export") or {}
        prompts = data.get("prompts") or {}
        web_search = data.get("web_search") or {}
        metaclaw = data.get("metaclaw_bridge") or {}
        memory_data = data.get("memory") or {}
        skills_data = data.get("skills") or {}
        knowledge_graph_data = data.get("knowledge_graph") or {}
        multi_project = data.get("multi_project") or {}
        compute_servers = data.get("compute_servers") or {}
        mcp_data = data.get("mcp") or {}
        overleaf = data.get("overleaf") or {}
        server = data.get("server") or {}
        dashboard_data = data.get("dashboard") or {}
        trends_data = data.get("trends") or {}
        copilot_data = data.get("copilot") or {}
        quality_assessor_data = data.get("quality_assessor") or {}
        calendar_data = data.get("calendar") or {}

        return cls(
            project=ProjectConfig(
                name=project["name"], mode=project.get("mode", "docs-first")
            ),
            research=ResearchConfig(
                topic=research["topic"],
                domains=tuple(research.get("domains") or ()),
                daily_paper_count=int(research.get("daily_paper_count", 0)),
                quality_threshold=float(research.get("quality_threshold", 0.0)),
                graceful_degradation=bool(research.get("graceful_degradation", True)),
            ),
            runtime=RuntimeConfig(
                timezone=runtime["timezone"],
                max_parallel_tasks=int(runtime.get("max_parallel_tasks", 1)),
                approval_timeout_hours=int(runtime.get("approval_timeout_hours", 12)),
                retry_limit=int(runtime.get("retry_limit", 0)),
            ),
            notifications=NotificationsConfig(
                channel=notifications["channel"],
                target=notifications.get("target", ""),
                on_stage_start=bool(notifications.get("on_stage_start", False)),
                on_stage_fail=bool(notifications.get("on_stage_fail", False)),
                on_gate_required=bool(notifications.get("on_gate_required", True)),
            ),
            knowledge_base=KnowledgeBaseConfig(
                backend=knowledge_base.get("backend", "markdown"),
                root=knowledge_base["root"],
                obsidian_vault=knowledge_base.get("obsidian_vault", ""),
            ),
            openclaw_bridge=OpenClawBridgeConfig(
                use_cron=bool(bridge.get("use_cron", False)),
                use_message=bool(bridge.get("use_message", False)),
                use_memory=bool(bridge.get("use_memory", False)),
                use_sessions_spawn=bool(bridge.get("use_sessions_spawn", False)),
                use_web_fetch=bool(bridge.get("use_web_fetch", False)),
                use_browser=bool(bridge.get("use_browser", False)),
            ),
            llm=_parse_llm_config(llm),
            security=SecurityConfig(
                hitl_required_stages=tuple(
                    int(s) for s in security.get("hitl_required_stages", (5, 9, 20))
                ),
                allow_publish_without_approval=bool(
                    security.get("allow_publish_without_approval", False)
                ),
                redact_sensitive_logs=bool(security.get("redact_sensitive_logs", True)),
            ),
            experiment=_parse_experiment_config(experiment),
            export=ExportConfig(
                target_conference=export.get("target_conference", "neurips_2025"),
                authors=export.get("authors", "Anonymous"),
                bib_file=export.get("bib_file", "references"),
            ),
            prompts=PromptsConfig(
                custom_file=prompts.get("custom_file", ""),
            ),
            web_search=WebSearchConfig(
                enabled=bool(web_search.get("enabled", True)),
                tavily_api_key=str(web_search.get("tavily_api_key", "")),
                tavily_api_key_env=str(
                    web_search.get("tavily_api_key_env", "TAVILY_API_KEY")
                ),
                enable_scholar=bool(web_search.get("enable_scholar", True)),
                enable_crawling=bool(web_search.get("enable_crawling", True)),
                enable_pdf_extraction=bool(
                    web_search.get("enable_pdf_extraction", True)
                ),
                max_web_results=int(web_search.get("max_web_results", 10)),
                max_scholar_results=int(web_search.get("max_scholar_results", 10)),
                max_crawl_urls=int(web_search.get("max_crawl_urls", 5)),
            ),
            metaclaw_bridge=_parse_metaclaw_bridge_config(metaclaw),
            memory=_parse_memory_config(memory_data),
            skills=_parse_skills_config(skills_data),
            knowledge_graph=_parse_knowledge_graph_config(knowledge_graph_data),
            multi_project=_parse_multi_project_config(multi_project),
            compute_servers=_parse_servers_config(compute_servers),
            mcp=_parse_mcp_config(mcp_data),
            overleaf=_parse_overleaf_config(overleaf),
            server=_parse_server_config(server),
            dashboard=_parse_dashboard_config(dashboard_data),
            trends=_parse_trends_config(trends_data),
            copilot=_parse_copilot_config(copilot_data),
            quality_assessor=_parse_quality_assessor_config(quality_assessor_data),
            calendar=_parse_calendar_config(calendar_data),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
        check_paths: bool = True,
    ) -> RCConfig:
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Config root must be a mapping, got {type(data).__name__}. "
                f"Check that {config_path} is valid YAML."
            )
        resolved_root = (
            Path(project_root).expanduser().resolve()
            if project_root
            else config_path.parent
        )
        return cls.from_dict(data, project_root=resolved_root, check_paths=check_paths)


def validate_config(
    data: dict[str, Any],
    *,
    project_root: Path | None = None,
    check_paths: bool = True,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    llm_provider = _get_by_path(data, "llm.provider")
    for key in REQUIRED_FIELDS:
        # ACP provider doesn't need base_url or api_key_env
        if llm_provider == "acp" and key in ("llm.base_url", "llm.api_key_env"):
            continue
        value = _get_by_path(data, key)
        if _is_blank(value):
            errors.append(f"Missing required field: {key}")

    project_mode = _get_by_path(data, "project.mode")
    if not _is_blank(project_mode) and project_mode not in PROJECT_MODES:
        errors.append(f"Invalid project.mode: {project_mode}")

    kb_backend = _get_by_path(data, "knowledge_base.backend")
    if not _is_blank(kb_backend) and kb_backend not in KB_BACKENDS:
        errors.append(f"Invalid knowledge_base.backend: {kb_backend}")

    llm_wire_api = _get_by_path(data, "llm.wire_api")
    if not _is_blank(llm_wire_api) and llm_wire_api not in (
        "chat_completions",
        "responses",
    ):
        errors.append(f"Invalid llm.wire_api: {llm_wire_api}")

    hitl_required_stages = _get_by_path(data, "security.hitl_required_stages")
    if hitl_required_stages is not None:
        if not isinstance(hitl_required_stages, list):
            errors.append("security.hitl_required_stages must be a list")
        else:
            for stage in hitl_required_stages:
                if not isinstance(stage, int) or not 1 <= stage <= 23:
                    errors.append(
                        f"Invalid security.hitl_required_stages entry: {stage}"
                    )

    exp_mode = _get_by_path(data, "experiment.mode")
    if not _is_blank(exp_mode) and exp_mode not in EXPERIMENT_MODES:
        errors.append(f"Invalid experiment.mode: {exp_mode}")

    exp_direction = _get_by_path(data, "experiment.metric_direction")
    if not _is_blank(exp_direction) and exp_direction not in ("minimize", "maximize"):
        errors.append(f"Invalid experiment.metric_direction: {exp_direction}")

    cli_agent_provider = _get_by_path(data, "experiment.cli_agent.provider")
    if (
        not _is_blank(cli_agent_provider)
        and cli_agent_provider not in CLI_AGENT_PROVIDERS
    ):
        errors.append(f"Invalid experiment.cli_agent.provider: {cli_agent_provider}")

    kb_root_raw = _get_by_path(data, "knowledge_base.root")
    if check_paths and not _is_blank(kb_root_raw) and project_root is not None:
        kb_root = project_root / str(kb_root_raw)
        if not kb_root.exists():
            errors.append(f"Missing path: {kb_root}")
        else:
            for subdir in KB_SUBDIRS:
                candidate = kb_root / subdir
                if not candidate.exists():
                    warnings.append(f"Missing recommended kb subdir: {candidate}")

    return ValidationResult(
        ok=not errors, errors=tuple(errors), warnings=tuple(warnings)
    )


def _parse_llm_config(data: dict[str, Any]) -> LlmConfig:
    acp_data = data.get("acp") or {}
    return LlmConfig(
        provider=data.get("provider", "openai-compatible"),
        base_url=data.get("base_url", ""),
        wire_api=data.get("wire_api", "chat_completions"),
        api_key_env=data.get("api_key_env", ""),
        api_key=data.get("api_key", ""),
        primary_model=data.get("primary_model", ""),
        fallback_models=tuple(data.get("fallback_models") or ()),
        s2_api_key=data.get("s2_api_key", ""),
        notes=data.get("notes", ""),
        acp=AcpConfig(
            agent=acp_data.get("agent", "claude"),
            cwd=acp_data.get("cwd", "."),
            acpx_command=acp_data.get("acpx_command", ""),
            session_name=acp_data.get("session_name", "researchclaw"),
            timeout_sec=int(acp_data.get("timeout_sec", 1800)),
        ),
    )


def _parse_agentic_config(data: dict[str, Any]) -> AgenticConfig:
    if not data:
        return AgenticConfig()
    return AgenticConfig(
        image=data.get("image", "researchclaw/experiment:latest"),
        agent_cli=data.get("agent_cli", "claude"),
        agent_install_cmd=data.get(
            "agent_install_cmd", "npm install -g @anthropic-ai/claude-code"
        ),
        network_policy=data.get("network_policy", "full"),
        timeout_sec=int(data.get("timeout_sec", 1800)),
        memory_limit_mb=int(data.get("memory_limit_mb", 8192)),
        gpu_enabled=bool(data.get("gpu_enabled", False)),
        mount_skills=bool(data.get("mount_skills", True)),
        allow_shell_commands=bool(data.get("allow_shell_commands", True)),
        max_turns=int(data.get("max_turns", 50)),
    )


def _parse_experiment_config(data: dict[str, Any]) -> ExperimentConfig:
    sandbox_data = data.get("sandbox") or {}
    docker_data = data.get("docker") or {}
    ssh_data = data.get("ssh_remote") or {}
    colab_data = data.get("colab_drive") or {}
    return ExperimentConfig(
        mode=data.get("mode", "simulated"),
        time_budget_sec=_safe_int(data.get("time_budget_sec"), 300),
        max_iterations=_safe_int(data.get("max_iterations"), 10),
        max_refine_duration_sec=_safe_int(data.get("max_refine_duration_sec"), 0),
        metric_key=data.get("metric_key", "primary_metric"),
        metric_direction=data.get("metric_direction", "minimize"),
        keep_threshold=_safe_float(data.get("keep_threshold"), 0.0),
        sandbox=SandboxConfig(
            python_path=sandbox_data.get("python_path", DEFAULT_PYTHON_PATH),
            gpu_required=bool(sandbox_data.get("gpu_required", False)),
            allowed_imports=tuple(
                sandbox_data.get("allowed_imports", SandboxConfig.allowed_imports)
            ),
            max_memory_mb=_safe_int(sandbox_data.get("max_memory_mb"), 4096),
        ),
        docker=DockerSandboxConfig(
            image=docker_data.get("image", "researchclaw/experiment:latest"),
            gpu_enabled=bool(docker_data.get("gpu_enabled", True)),
            gpu_device_ids=tuple(int(g) for g in docker_data.get("gpu_device_ids", ())),
            memory_limit_mb=_safe_int(docker_data.get("memory_limit_mb"), 8192),
            network_policy=_validate_network_policy(
                docker_data.get("network_policy", "setup_only"),
            ),
            pip_pre_install=tuple(docker_data.get("pip_pre_install", ())),
            auto_install_deps=bool(docker_data.get("auto_install_deps", True)),
            shm_size_mb=_safe_int(docker_data.get("shm_size_mb"), 2048),
            container_python=docker_data.get("container_python", "/usr/bin/python3"),
            keep_containers=bool(docker_data.get("keep_containers", False)),
        ),
        ssh_remote=SshRemoteConfig(
            host=ssh_data.get("host", ""),
            user=ssh_data.get("user", ""),
            port=_safe_int(ssh_data.get("port"), 22),
            key_path=ssh_data.get("key_path", ""),
            gpu_ids=tuple(int(g) for g in ssh_data.get("gpu_ids", ())),
            remote_workdir=ssh_data.get(
                "remote_workdir", "/tmp/researchclaw_experiments"
            ),
            remote_python=ssh_data.get("remote_python", "python3"),
            setup_commands=tuple(ssh_data.get("setup_commands") or ()),
            use_docker=bool(ssh_data.get("use_docker", False)),
            docker_image=ssh_data.get("docker_image", "researchclaw/experiment:latest"),
            docker_network_policy=_validate_network_policy(
                ssh_data.get("docker_network_policy", "none"),
            ),
            docker_memory_limit_mb=_safe_int(
                ssh_data.get("docker_memory_limit_mb"), 8192
            ),
            docker_shm_size_mb=_safe_int(ssh_data.get("docker_shm_size_mb"), 2048),
            timeout_sec=_safe_int(ssh_data.get("timeout_sec"), 600),
            scp_timeout_sec=_safe_int(ssh_data.get("scp_timeout_sec"), 300),
            setup_timeout_sec=_safe_int(ssh_data.get("setup_timeout_sec"), 300),
        ),
        colab_drive=ColabDriveConfig(
            drive_root=colab_data.get("drive_root", ""),
            poll_interval_sec=_safe_int(colab_data.get("poll_interval_sec"), 30),
            timeout_sec=_safe_int(colab_data.get("timeout_sec"), 3600),
            setup_script=colab_data.get("setup_script", ""),
        ),
        agentic=_parse_agentic_config(data.get("agentic") or {}),
        code_agent=_parse_code_agent_config(data.get("code_agent") or {}),
        opencode=_parse_opencode_config(data.get("opencode") or {}),
        benchmark_agent=_parse_benchmark_agent_config(
            data.get("benchmark_agent") or {}
        ),
        figure_agent=_parse_figure_agent_config(data.get("figure_agent") or {}),
        repair=_parse_experiment_repair_config(data.get("repair") or {}),
        cli_agent=_parse_cli_agent_config(data.get("cli_agent") or {}),
    )


def _parse_benchmark_agent_config(data: dict[str, Any]) -> BenchmarkAgentConfig:
    if not data:
        return BenchmarkAgentConfig()
    return BenchmarkAgentConfig(
        enabled=bool(data.get("enabled", True)),
        enable_hf_search=bool(data.get("enable_hf_search", True)),
        max_hf_results=_safe_int(data.get("max_hf_results"), 10),
        enable_web_search=bool(data.get("enable_web_search", True)),
        max_web_results=_safe_int(data.get("max_web_results"), 5),
        web_search_min_local=_safe_int(data.get("web_search_min_local"), 3),
        tier_limit=_safe_int(data.get("tier_limit"), 2),
        min_benchmarks=_safe_int(data.get("min_benchmarks"), 1),
        min_baselines=_safe_int(data.get("min_baselines"), 2),
        prefer_cached=bool(data.get("prefer_cached", True)),
        max_iterations=_safe_int(data.get("max_iterations"), 2),
    )


def _parse_figure_agent_config(data: dict[str, Any]) -> FigureAgentConfig:
    if not data:
        return FigureAgentConfig()
    use_docker_raw = data.get("use_docker", None)
    return FigureAgentConfig(
        enabled=bool(data.get("enabled", True)),
        min_figures=_safe_int(data.get("min_figures"), 3),
        max_figures=_safe_int(data.get("max_figures"), 8),
        max_iterations=_safe_int(data.get("max_iterations"), 3),
        render_timeout_sec=_safe_int(data.get("render_timeout_sec"), 30),
        use_docker=(None if use_docker_raw is None else bool(use_docker_raw)),
        docker_image=data.get("docker_image", "researchclaw/experiment:latest"),
        output_format=data.get("output_format", "python"),
        gemini_api_key=data.get("gemini_api_key", ""),
        gemini_model=data.get("gemini_model", "gemini-2.5-flash-image"),
        nano_banana_enabled=bool(data.get("nano_banana_enabled", True)),
        strict_mode=bool(data.get("strict_mode", False)),
        dpi=_safe_int(data.get("dpi"), 300),
    )


def _parse_experiment_repair_config(data: dict[str, Any]) -> ExperimentRepairConfig:
    if not data:
        return ExperimentRepairConfig()
    return ExperimentRepairConfig(
        enabled=bool(data.get("enabled", True)),
        max_cycles=_safe_int(data.get("max_cycles"), 3),
        min_completion_rate=_safe_float(data.get("min_completion_rate"), 0.5),
        min_conditions=_safe_int(data.get("min_conditions"), 2),
        use_opencode=bool(data.get("use_opencode", True)),
        timeout_sec_per_cycle=_safe_int(data.get("timeout_sec_per_cycle"), 600),
    )


def _parse_cli_agent_config(data: dict[str, Any]) -> CliAgentConfig:
    if not data:
        return CliAgentConfig()
    return CliAgentConfig(
        provider=data.get("provider", "llm"),
        binary_path=data.get("binary_path", ""),
        model=data.get("model", ""),
        max_budget_usd=_safe_float(data.get("max_budget_usd"), 5.0),
        timeout_sec=_safe_int(data.get("timeout_sec"), 600),
        extra_args=tuple(data.get("extra_args") or ()),
    )


def _parse_code_agent_config(data: dict[str, Any]) -> CodeAgentConfig:
    if not data:
        return CodeAgentConfig()
    return CodeAgentConfig(
        enabled=bool(data.get("enabled", True)),
        architecture_planning=bool(data.get("architecture_planning", True)),
        sequential_generation=bool(data.get("sequential_generation", True)),
        hard_validation=bool(data.get("hard_validation", True)),
        hard_validation_max_repairs=_safe_int(
            data.get("hard_validation_max_repairs"), 4
        ),
        exec_fix_max_iterations=_safe_int(data.get("exec_fix_max_iterations"), 3),
        exec_fix_timeout_sec=_safe_int(data.get("exec_fix_timeout_sec"), 60),
        tree_search_enabled=bool(data.get("tree_search_enabled", False)),
        tree_search_candidates=_safe_int(data.get("tree_search_candidates"), 3),
        tree_search_max_depth=_safe_int(data.get("tree_search_max_depth"), 2),
        tree_search_eval_timeout_sec=_safe_int(
            data.get("tree_search_eval_timeout_sec"), 120
        ),
        review_max_rounds=_safe_int(data.get("review_max_rounds"), 2),
    )


def _parse_opencode_config(data: dict[str, Any]) -> OpenCodeConfig:
    if not data:
        return OpenCodeConfig()
    return OpenCodeConfig(
        enabled=bool(data.get("enabled", True)),
        auto=bool(data.get("auto", True)),
        complexity_threshold=_safe_float(data.get("complexity_threshold"), 0.2),
        model=str(data.get("model", "")),
        timeout_sec=_safe_int(data.get("timeout_sec"), 600),
        max_retries=_safe_int(data.get("max_retries"), 1),
        workspace_cleanup=bool(data.get("workspace_cleanup", True)),
    )


def _parse_metaclaw_bridge_config(data: dict[str, Any]) -> MetaClawBridgeConfig:
    prm_data = data.get("prm") or {}
    l2s_data = data.get("lesson_to_skill") or {}
    return MetaClawBridgeConfig(
        enabled=bool(data.get("enabled", False)),
        proxy_url=data.get("proxy_url", "http://localhost:30000"),
        skills_dir=data.get("skills_dir", "~/.metaclaw/skills"),
        fallback_url=data.get("fallback_url", ""),
        fallback_api_key=data.get("fallback_api_key", ""),
        prm=MetaClawPRMConfig(
            enabled=bool(prm_data.get("enabled", False)),
            api_base=prm_data.get("api_base", ""),
            api_key_env=prm_data.get("api_key_env", ""),
            api_key=prm_data.get("api_key", ""),
            model=prm_data.get("model", "gpt-5.4"),
            votes=_safe_int(prm_data.get("votes"), 3),
            temperature=_safe_float(prm_data.get("temperature"), 0.6),
            gate_stages=tuple(
                int(s) for s in prm_data.get("gate_stages", (5, 9, 15, 20))
            ),
        ),
        lesson_to_skill=MetaClawLessonToSkillConfig(
            enabled=bool(l2s_data.get("enabled", True)),
            min_severity=l2s_data.get("min_severity", "warning"),
            max_skills_per_run=_safe_int(l2s_data.get("max_skills_per_run"), 3),
        ),
    )


def _parse_memory_config(data: dict[str, Any]) -> MemoryConfig:
    if not data:
        return MemoryConfig()
    stages = data.get("inject_at_stages", (1, 9, 10, 17))
    return MemoryConfig(
        enabled=bool(data.get("enabled", True)),
        store_dir=str(data.get("store_dir", ".researchclaw/memory")),
        embedding_model=str(data.get("embedding_model", "text-embedding-3-small")),
        max_entries_per_category=int(data.get("max_entries_per_category", 500)),
        decay_half_life_days=int(data.get("decay_half_life_days", 90)),
        confidence_threshold=float(data.get("confidence_threshold", 0.3)),
        inject_at_stages=tuple(int(s) for s in stages),
    )


def _parse_skills_config(data: dict[str, Any]) -> SkillsConfig:
    if not data:
        return SkillsConfig()
    return SkillsConfig(
        enabled=bool(data.get("enabled", True)),
        builtin_dir=str(data.get("builtin_dir", "")),
        custom_dirs=tuple(str(d) for d in (data.get("custom_dirs") or ())),
        external_dirs=tuple(str(d) for d in (data.get("external_dirs") or ())),
        auto_match=bool(data.get("auto_match", True)),
        max_skills_per_stage=int(data.get("max_skills_per_stage", 3)),
        fallback_matching=bool(data.get("fallback_matching", True)),
    )


def _parse_knowledge_graph_config(data: dict[str, Any]) -> KnowledgeGraphConfig:
    if not data:
        return KnowledgeGraphConfig()
    return KnowledgeGraphConfig(
        enabled=bool(data.get("enabled", False)),
        store_path=str(data.get("store_path", ".researchclaw/knowledge_graph")),
        max_entities=int(data.get("max_entities", 10000)),
        auto_update=bool(data.get("auto_update", True)),
    )


def _parse_multi_project_config(data: dict[str, Any]) -> MultiProjectConfig:
    if not data:
        return MultiProjectConfig()
    return MultiProjectConfig(
        enabled=bool(data.get("enabled", False)),
        projects_dir=data.get("projects_dir", ".researchclaw/projects"),
        max_concurrent=int(data.get("max_concurrent", 2)),
        shared_knowledge=bool(data.get("shared_knowledge", True)),
    )


def _parse_servers_config(data: dict[str, Any]) -> ServersConfig:
    if not data:
        return ServersConfig()
    raw_servers = data.get("servers") or ()
    servers = tuple(
        ServerEntryConfig(
            name=s.get("name", ""),
            host=s.get("host", ""),
            server_type=s.get("server_type", "ssh"),
            gpu=s.get("gpu", ""),
            vram_gb=int(s.get("vram_gb", 0)),
            priority=int(s.get("priority", 1)),
            cost_per_hour=float(s.get("cost_per_hour", 0.0)),
            scheduler=s.get("scheduler", ""),
            cloud_provider=s.get("cloud_provider", ""),
        )
        for s in raw_servers
    )
    return ServersConfig(
        enabled=bool(data.get("enabled", False)),
        servers=servers,
        prefer_free=bool(data.get("prefer_free", True)),
        failover=bool(data.get("failover", True)),
        monitor_interval_sec=int(data.get("monitor_interval_sec", 60)),
    )


def _parse_mcp_config(data: dict[str, Any]) -> MCPIntegrationConfig:
    if not data:
        return MCPIntegrationConfig()
    return MCPIntegrationConfig(
        server_enabled=bool(data.get("server_enabled", False)),
        server_port=int(data.get("server_port", 3000)),
        server_transport=data.get("server_transport", "stdio"),
        external_servers=tuple(data.get("external_servers") or ()),
    )


def _parse_overleaf_config(data: dict[str, Any]) -> OverleafConfig:
    if not data:
        return OverleafConfig()
    return OverleafConfig(
        enabled=bool(data.get("enabled", False)),
        git_url=data.get("git_url", ""),
        branch=data.get("branch", "main"),
        auto_push=bool(data.get("auto_push", True)),
        auto_pull=bool(data.get("auto_pull", False)),
        poll_interval_sec=int(data.get("poll_interval_sec", 300)),
    )


def _parse_server_config(data: dict[str, Any]) -> ServerConfig:
    if not data:
        return ServerConfig()
    cors = data.get("cors_origins")
    if isinstance(cors, list):
        cors = tuple(cors)
    elif cors is None:
        cors = ("*",)
    else:
        cors = (str(cors),)
    return ServerConfig(
        enabled=bool(data.get("enabled", False)),
        host=data.get("host", "0.0.0.0"),
        port=int(data.get("port", 8080)),
        cors_origins=cors,
        auth_token=data.get("auth_token", ""),
        voice_enabled=bool(data.get("voice_enabled", False)),
        whisper_model=data.get("whisper_model", "whisper-1"),
        whisper_api_url=data.get("whisper_api_url", ""),
    )


def _parse_dashboard_config(data: dict[str, Any]) -> DashboardConfig:
    if not data:
        return DashboardConfig()
    return DashboardConfig(
        enabled=bool(data.get("enabled", True)),
        refresh_interval_sec=int(data.get("refresh_interval_sec", 5)),
        max_log_lines=int(data.get("max_log_lines", 1000)),
        browser_notifications=bool(data.get("browser_notifications", True)),
    )


def _parse_trends_config(data: dict[str, Any]) -> TrendsConfig:
    if not data:
        return TrendsConfig()
    sources = data.get("sources", ("arxiv", "semantic_scholar"))
    if isinstance(sources, list):
        sources = tuple(sources)
    domains = data.get("domains", ())
    if isinstance(domains, list):
        domains = tuple(domains)
    return TrendsConfig(
        enabled=bool(data.get("enabled", False)),
        domains=domains,
        daily_digest=bool(data.get("daily_digest", True)),
        digest_time=data.get("digest_time", "08:00"),
        max_papers_per_day=int(data.get("max_papers_per_day", 20)),
        trend_window_days=int(data.get("trend_window_days", 30)),
        sources=sources,
    )


def _parse_copilot_config(data: dict[str, Any]) -> CoPilotConfig:
    if not data:
        return CoPilotConfig()
    return CoPilotConfig(
        mode=data.get("mode", "auto-pilot"),
        pause_at_gates=bool(data.get("pause_at_gates", True)),
        pause_at_every_stage=bool(data.get("pause_at_every_stage", False)),
        feedback_timeout_sec=int(data.get("feedback_timeout_sec", 3600)),
        allow_branching=bool(data.get("allow_branching", True)),
        max_branches=int(data.get("max_branches", 3)),
    )


def _parse_quality_assessor_config(data: dict[str, Any]) -> QualityAssessorConfig:
    if not data:
        return QualityAssessorConfig()
    dimensions = data.get(
        "dimensions", ("novelty", "rigor", "clarity", "impact", "experiments")
    )
    if isinstance(dimensions, list):
        dimensions = tuple(dimensions)
    return QualityAssessorConfig(
        enabled=bool(data.get("enabled", True)),
        dimensions=dimensions,
        venue_recommendation=bool(data.get("venue_recommendation", True)),
        score_history=bool(data.get("score_history", True)),
    )


def _parse_calendar_config(data: dict[str, Any]) -> CalendarConfig:
    if not data:
        return CalendarConfig()
    venues = data.get("target_venues", ())
    if isinstance(venues, list):
        venues = tuple(venues)
    reminder = data.get("reminder_days_before", (30, 14, 7, 3, 1))
    if isinstance(reminder, list):
        reminder = tuple(reminder)
    return CalendarConfig(
        enabled=bool(data.get("enabled", False)),
        target_venues=venues,
        reminder_days_before=reminder,
        auto_plan=bool(data.get("auto_plan", True)),
    )


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_paths: bool = True,
) -> RCConfig:
    return RCConfig.load(path, project_root=project_root, check_paths=check_paths)
