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
class LlmConfig:
    base_url: str
    api_key_env: str
    provider: str = "openai"  # "openai" | "anthropic" | "openrouter"
    api_key: str = ""
    primary_model: str = ""
    fallback_models: tuple[str, ...] = ()
    s2_api_key: str = ""
    notes: str = ""


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
    network_policy: str = "none"  # none | pip_only | full
    pip_pre_install: tuple[str, ...] = ()
    auto_install_deps: bool = True
    shm_size_mb: int = 2048
    container_python: str = "/usr/bin/python3"
    keep_containers: bool = False


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
            llm=LlmConfig(
                provider=llm.get("provider", "openai") or "openai",
                base_url=llm["base_url"],
                api_key_env=llm["api_key_env"],
                api_key=llm.get("api_key", ""),
                primary_model=llm.get("primary_model", ""),
                fallback_models=tuple(llm.get("fallback_models") or ()),
                s2_api_key=llm.get("s2_api_key", ""),
                notes=llm.get("notes", ""),
            ),
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

    for key in REQUIRED_FIELDS:
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
            network_policy=docker_data.get("network_policy", "none"),
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
    )


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_paths: bool = True,
) -> RCConfig:
    return RCConfig.load(path, project_root=project_root, check_paths=check_paths)
