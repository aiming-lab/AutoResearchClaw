"""ResearchClaw config loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
EXPERIMENT_MODES = {"simulated", "sandbox", "docker", "ssh_remote"}


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
    timeout_sec: int = 600


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    base_url: str = ""
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
    python_path: str = ".venv/bin/python3"
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
    gpu_ids: tuple[int, ...] = ()
    remote_workdir: str = "/tmp/researchclaw_experiments"


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
class CodeAgentConfig:
    """Configuration for the advanced multi-phase code generation agent."""

    enabled: bool = True
    # Phase 1: Architecture planning before code generation
    architecture_planning: bool = True
    # Phase 2: Execution-in-the-loop (run → parse error → fix)
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60
    # Phase 3: Solution tree search (off by default — higher cost)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2
    tree_search_eval_timeout_sec: int = 120
    # Phase 4: Multi-agent review dialog
    review_max_rounds: int = 2


@dataclass(frozen=True)
class BenchmarkAgentConfig:
    """Configuration for the BenchmarkAgent multi-agent system."""

    enabled: bool = True
    # Surveyor
    enable_hf_search: bool = True
    max_hf_results: int = 10
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
class ExperimentConfig:
    mode: str = "simulated"
    time_budget_sec: int = 300
    max_iterations: int = 10
    metric_key: str = "primary_metric"
    metric_direction: str = "minimize"
    keep_threshold: float = 0.0
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    docker: DockerSandboxConfig = field(default_factory=DockerSandboxConfig)
    ssh_remote: SshRemoteConfig = field(default_factory=SshRemoteConfig)
    code_agent: CodeAgentConfig = field(default_factory=CodeAgentConfig)
    benchmark_agent: BenchmarkAgentConfig = field(default_factory=BenchmarkAgentConfig)
    figure_agent: FigureAgentConfig = field(default_factory=FigureAgentConfig)


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

        return cls(
            project=ProjectConfig(
                name=project["name"], mode=project.get("mode", "docs-first")
            ),
            research=ResearchConfig(
                topic=research["topic"],
                domains=tuple(research.get("domains") or ()),
                daily_paper_count=int(research.get("daily_paper_count", 0)),
                quality_threshold=float(research.get("quality_threshold", 0.0)),
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
            timeout_sec=int(acp_data.get("timeout_sec", 600)),
        ),
    )


def _parse_experiment_config(data: dict[str, Any]) -> ExperimentConfig:
    sandbox_data = data.get("sandbox") or {}
    docker_data = data.get("docker") or {}
    ssh_data = data.get("ssh_remote") or {}
    return ExperimentConfig(
        mode=data.get("mode", "simulated"),
        time_budget_sec=int(data.get("time_budget_sec", 300)),
        max_iterations=int(data.get("max_iterations", 10)),
        metric_key=data.get("metric_key", "primary_metric"),
        metric_direction=data.get("metric_direction", "minimize"),
        keep_threshold=float(data.get("keep_threshold", 0.0)),
        sandbox=SandboxConfig(
            python_path=sandbox_data.get("python_path", ".venv/bin/python3"),
            gpu_required=bool(sandbox_data.get("gpu_required", False)),
            allowed_imports=tuple(
                sandbox_data.get("allowed_imports", SandboxConfig.allowed_imports)
            ),
            max_memory_mb=int(sandbox_data.get("max_memory_mb", 4096)),
        ),
        docker=DockerSandboxConfig(
            image=docker_data.get("image", "researchclaw/experiment:latest"),
            gpu_enabled=bool(docker_data.get("gpu_enabled", True)),
            gpu_device_ids=tuple(
                int(g) for g in docker_data.get("gpu_device_ids", ())
            ),
            memory_limit_mb=int(docker_data.get("memory_limit_mb", 8192)),
            network_policy=docker_data.get("network_policy", "setup_only"),
            pip_pre_install=tuple(docker_data.get("pip_pre_install", ())),
            auto_install_deps=bool(docker_data.get("auto_install_deps", True)),
            shm_size_mb=int(docker_data.get("shm_size_mb", 2048)),
            container_python=docker_data.get("container_python", "/usr/bin/python3"),
            keep_containers=bool(docker_data.get("keep_containers", False)),
        ),
        ssh_remote=SshRemoteConfig(
            host=ssh_data.get("host", ""),
            gpu_ids=tuple(int(g) for g in ssh_data.get("gpu_ids", ())),
            remote_workdir=ssh_data.get(
                "remote_workdir", "/tmp/researchclaw_experiments"
            ),
        ),
        code_agent=_parse_code_agent_config(data.get("code_agent") or {}),
        benchmark_agent=_parse_benchmark_agent_config(
            data.get("benchmark_agent") or {}
        ),
        figure_agent=_parse_figure_agent_config(data.get("figure_agent") or {}),
    )


def _parse_benchmark_agent_config(data: dict[str, Any]) -> BenchmarkAgentConfig:
    if not data:
        return BenchmarkAgentConfig()
    return BenchmarkAgentConfig(
        enabled=bool(data.get("enabled", True)),
        enable_hf_search=bool(data.get("enable_hf_search", True)),
        max_hf_results=int(data.get("max_hf_results", 10)),
        tier_limit=int(data.get("tier_limit", 2)),
        min_benchmarks=int(data.get("min_benchmarks", 1)),
        min_baselines=int(data.get("min_baselines", 2)),
        prefer_cached=bool(data.get("prefer_cached", True)),
        max_iterations=int(data.get("max_iterations", 2)),
    )


def _parse_figure_agent_config(data: dict[str, Any]) -> FigureAgentConfig:
    if not data:
        return FigureAgentConfig()

    # Parse use_docker: None (auto), True, or False
    _use_docker_raw = data.get("use_docker")
    _use_docker: bool | None = None
    if _use_docker_raw is not None:
        _use_docker = bool(_use_docker_raw)

    return FigureAgentConfig(
        enabled=bool(data.get("enabled", True)),
        min_figures=int(data.get("min_figures", 3)),
        max_figures=int(data.get("max_figures", 8)),
        max_iterations=int(data.get("max_iterations", 3)),
        render_timeout_sec=int(data.get("render_timeout_sec", 30)),
        use_docker=_use_docker,
        docker_image=data.get("docker_image", "researchclaw/experiment:latest"),
        output_format=data.get("output_format", "python"),
        gemini_api_key=data.get("gemini_api_key", ""),
        gemini_model=data.get("gemini_model", "gemini-2.5-flash-image"),
        nano_banana_enabled=bool(data.get("nano_banana_enabled", True)),
        strict_mode=bool(data.get("strict_mode", False)),
        dpi=int(data.get("dpi", 300)),
    )


def _parse_code_agent_config(data: dict[str, Any]) -> CodeAgentConfig:
    if not data:
        return CodeAgentConfig()
    return CodeAgentConfig(
        enabled=bool(data.get("enabled", True)),
        architecture_planning=bool(data.get("architecture_planning", True)),
        exec_fix_max_iterations=int(data.get("exec_fix_max_iterations", 3)),
        exec_fix_timeout_sec=int(data.get("exec_fix_timeout_sec", 60)),
        tree_search_enabled=bool(data.get("tree_search_enabled", False)),
        tree_search_candidates=int(data.get("tree_search_candidates", 3)),
        tree_search_max_depth=int(data.get("tree_search_max_depth", 2)),
        tree_search_eval_timeout_sec=int(
            data.get("tree_search_eval_timeout_sec", 120)
        ),
        review_max_rounds=int(data.get("review_max_rounds", 2)),
    )


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_paths: bool = True,
) -> RCConfig:
    return RCConfig.load(path, project_root=project_root, check_paths=check_paths)
