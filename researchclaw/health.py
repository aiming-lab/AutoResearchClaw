from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
from collections.abc import Callable as AbcCallable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ContextManager, cast

import yaml

from researchclaw.config import RCConfig, validate_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class DoctorReport:
    timestamp: str
    checks: list[CheckResult]
    overall: str

    @property
    def actionable_fixes(self) -> list[str]:
        return [check.fix for check in self.checks if check.fix]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "overall": self.overall,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "detail": check.detail,
                    "fix": check.fix,
                }
                for check in self.checks
            ],
            "actionable_fixes": self.actionable_fixes,
        }


def check_python_version() -> CheckResult:
    version_tuple = (
        int(sys.version_info.major),
        int(sys.version_info.minor),
        int(sys.version_info.micro),
    )
    if version_tuple >= (3, 11, 0):
        return CheckResult(
            name="python_version",
            status="pass",
            detail=(
                f"Python {sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
        )
    return CheckResult(
        name="python_version",
        status="fail",
        detail=(
            f"Python {sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro} is unsupported"
        ),
        fix="Install Python 3.11 or newer",
    )


def check_yaml_import() -> CheckResult:
    try:
        _ = importlib.import_module("yaml")
    except ImportError:
        return CheckResult(
            name="yaml_import",
            status="fail",
            detail="PyYAML is not importable",
            fix="pip install pyyaml",
        )
    return CheckResult(name="yaml_import", status="pass", detail="PyYAML import ok")


def check_config_valid(config_path: str | Path) -> CheckResult:
    path = Path(config_path)
    if not path.exists():
        return CheckResult(
            name="config_valid",
            status="fail",
            detail=f"Config file not found: {path}",
            fix="Provide --config path to an existing YAML config file",
        )

    try:
        with path.open(encoding="utf-8") as handle:
            data_obj = _load_yaml_object(handle.read())
    except yaml.YAMLError as exc:
        return CheckResult(
            name="config_valid",
            status="fail",
            detail=f"Config YAML parse error: {exc}",
            fix="Fix YAML syntax errors in the config file",
        )
    except OSError as exc:
        return CheckResult(
            name="config_valid",
            status="fail",
            detail=f"Could not read config file: {exc}",
            fix="Verify file permissions and path",
        )

    data: object = {} if data_obj is None else data_obj
    if not isinstance(data, dict):
        return CheckResult(
            name="config_valid",
            status="fail",
            detail="Config root must be a mapping",
            fix="Ensure the config file starts with key-value mappings",
        )
    data_map = cast(Mapping[object, object], data)
    typed_data = {str(key): value for key, value in data_map.items()}
    result = validate_config(typed_data)
    if result.ok:
        return CheckResult(
            name="config_valid", status="pass", detail="Config validation ok"
        )
    return CheckResult(
        name="config_valid",
        status="fail",
        detail="; ".join(result.errors),
        fix="Fix validation errors in config file",
    )


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


def check_llm_connectivity(base_url: str) -> CheckResult:
    if not base_url.strip():
        return CheckResult(
            name="llm_connectivity",
            status="fail",
            detail="LLM base URL is empty",
            fix="Set llm.base_url in config",
        )

    url = _models_url(base_url)
    req = urllib.request.Request(url, method="HEAD")

    try:
        with urllib.request.urlopen(req, timeout=5):
            return CheckResult(
                name="llm_connectivity",
                status="pass",
                detail=f"Reachable: {url}",
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            try:
                with urllib.request.urlopen(url, timeout=5):
                    return CheckResult(
                        name="llm_connectivity",
                        status="pass",
                        detail=f"Reachable: {url}",
                    )
            except urllib.error.HTTPError as get_exc:
                return CheckResult(
                    name="llm_connectivity",
                    status="fail",
                    detail=f"LLM endpoint HTTP {get_exc.code}",
                    fix="Check llm.base_url and provider status",
                )
            except urllib.error.URLError as get_exc:
                if _is_timeout(get_exc):
                    return CheckResult(
                        name="llm_connectivity",
                        status="fail",
                        detail="LLM endpoint unreachable",
                        fix="Verify endpoint URL and network connectivity",
                    )
                return CheckResult(
                    name="llm_connectivity",
                    status="fail",
                    detail=f"LLM connectivity error: {get_exc.reason}",
                    fix="Verify endpoint URL and network connectivity",
                )
            except TimeoutError:
                return CheckResult(
                    name="llm_connectivity",
                    status="fail",
                    detail="LLM endpoint unreachable",
                    fix="Verify endpoint URL and network connectivity",
                )

        return CheckResult(
            name="llm_connectivity",
            status="fail",
            detail=f"LLM endpoint HTTP {exc.code}",
            fix="Check llm.base_url and provider status",
        )
    except urllib.error.URLError as exc:
        if _is_timeout(exc):
            return CheckResult(
                name="llm_connectivity",
                status="fail",
                detail="LLM endpoint unreachable",
                fix="Verify endpoint URL and network connectivity",
            )
        return CheckResult(
            name="llm_connectivity",
            status="fail",
            detail=f"LLM connectivity error: {exc.reason}",
            fix="Verify endpoint URL and network connectivity",
        )
    except TimeoutError:
        return CheckResult(
            name="llm_connectivity",
            status="fail",
            detail="LLM endpoint unreachable",
            fix="Verify endpoint URL and network connectivity",
        )


def _fetch_models(base_url: str, api_key: str = "") -> tuple[int, dict[str, object]]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(_models_url(base_url), headers=headers)
    with _urlopen(request, timeout=5) as response:
        raw_bytes = _read_response_bytes(response)
        payload_map = _load_json_mapping(raw_bytes.decode("utf-8") or "{}")
        payload: dict[str, object] = {
            str(key): value for key, value in payload_map.items()
        }
        return 200, payload


def _read_response_bytes(response: object) -> bytes:
    if not hasattr(response, "read"):
        raise ValueError("Response object has no read method")
    reader_obj = getattr(response, "read", None)
    if reader_obj is None or not isinstance(reader_obj, AbcCallable):
        raise ValueError("Response read attribute is not callable")
    reader = cast(AbcCallable[[], object], reader_obj)
    raw = reader()
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("Response body is not bytes")
    return bytes(raw)


def _urlopen(req: str | urllib.request.Request, timeout: int) -> ContextManager[object]:
    return cast(ContextManager[object], urllib.request.urlopen(req, timeout=timeout))


def _load_yaml_object(content: str) -> object:
    return cast(object, yaml.safe_load(content))


def _load_json_mapping(content: str) -> Mapping[object, object]:
    payload_obj = cast(object, json.loads(content))
    if not isinstance(payload_obj, dict):
        raise ValueError("models response must be a JSON object")
    return cast(Mapping[object, object], payload_obj)


def check_api_key_valid(base_url: str, api_key: str) -> CheckResult:
    if not api_key.strip():
        return CheckResult(
            name="api_key_valid",
            status="fail",
            detail="API key is empty",
            fix="Set llm.api_key or environment variable defined by llm.api_key_env",
        )

    try:
        status, _ = _fetch_models(base_url, api_key)
        if status == 200:
            return CheckResult(
                name="api_key_valid",
                status="pass",
                detail="API key accepted",
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return CheckResult(
                name="api_key_valid",
                status="fail",
                detail="Invalid API key",
                fix="Set a valid API key for the configured endpoint",
            )
        return CheckResult(
            name="api_key_valid",
            status="warn",
            detail=f"API key check returned HTTP {exc.code}",
            fix="Verify endpoint health and API key permissions",
        )
    except urllib.error.URLError as exc:
        return CheckResult(
            name="api_key_valid",
            status="warn",
            detail=f"Could not verify API key: {exc.reason}",
            fix="Retry when endpoint/network is available",
        )
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return CheckResult(
            name="api_key_valid",
            status="warn",
            detail=f"Could not verify API key: {exc}",
            fix="Retry when endpoint/network is available",
        )

    return CheckResult(
        name="api_key_valid",
        status="warn",
        detail="Could not verify API key",
        fix="Retry when endpoint/network is available",
    )


def check_model_available(base_url: str, api_key: str, model: str) -> CheckResult:
    """Check if a single model is available (kept for backward compat)."""
    results = _check_models_against_endpoint(base_url, api_key, [model])
    if results is None:
        return CheckResult(
            name="model_available",
            status="warn",
            detail="Could not verify model availability",
            fix="Retry when endpoint/network is available",
        )
    available, _missing = results
    if model in available:
        return CheckResult(
            name="model_available",
            status="pass",
            detail=f"Model available: {model}",
        )
    return CheckResult(
        name="model_available",
        status="fail",
        detail=f"Model {model} not available",
        fix="Update llm.primary_model or endpoint model access",
    )


def check_model_chain(
    base_url: str,
    api_key: str,
    primary_model: str,
    fallback_models: tuple[str, ...] | list[str] = (),
) -> CheckResult:
    """Check the full model fallback chain — pass if ANY model works."""
    all_models = [m for m in [primary_model] + list(fallback_models) if m.strip()]
    if not all_models:
        return CheckResult(
            name="model_chain",
            status="warn",
            detail="No models configured",
            fix="Set llm.primary_model in config",
        )

    results = _check_models_against_endpoint(base_url, api_key, all_models)
    if results is None:
        return CheckResult(
            name="model_chain",
            status="warn",
            detail="Could not verify model availability",
            fix="Retry when endpoint/network is available",
        )

    available, missing = results

    if not available:
        return CheckResult(
            name="model_chain",
            status="fail",
            detail=f"No models available (tested: {', '.join(all_models)})",
            fix="Update llm.primary_model/fallback_models or endpoint model access",
        )

    if missing:
        return CheckResult(
            name="model_chain",
            status="pass",
            detail=(
                f"Fallback chain OK — available: {', '.join(sorted(available))}; "
                f"unavailable: {', '.join(sorted(missing))}"
            ),
        )

    return CheckResult(
        name="model_chain",
        status="pass",
        detail=f"All models available: {', '.join(sorted(available))}",
    )


def _check_models_against_endpoint(
    base_url: str, api_key: str, models: list[str]
) -> tuple[set[str], set[str]] | None:
    """Return (available, missing) sets, or None if endpoint unreachable."""
    if not models or not all(m.strip() for m in models):
        models = [m for m in models if m.strip()]
    if not models:
        return set(), set()

    try:
        _, payload = _fetch_models(base_url, api_key)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
        OSError,
    ):
        return None

    models_obj = payload.get("data")
    endpoint_models = cast(
        list[object] | None, models_obj if isinstance(models_obj, list) else None
    )
    if not isinstance(endpoint_models, list):
        return None

    available_ids: set[str] = set()
    for item in endpoint_models:
        if not isinstance(item, dict):
            continue
        item_map = cast(Mapping[object, object], item)
        model_id_obj = item_map.get("id")
        if isinstance(model_id_obj, str):
            available_ids.add(model_id_obj)

    requested = set(models)
    available = requested & available_ids
    missing = requested - available_ids
    return available, missing


def check_sandbox_python(python_path: str) -> CheckResult:
    if not python_path.strip():
        return CheckResult(
            name="sandbox_python",
            status="warn",
            detail="Sandbox python path is empty",
            fix="Set experiment.sandbox.python_path in config",
        )

    path = Path(python_path)
    if path.exists() and os.access(path, os.X_OK):
        return CheckResult(
            name="sandbox_python",
            status="pass",
            detail=f"Sandbox python found: {path}",
        )
    return CheckResult(
        name="sandbox_python",
        status="warn",
        detail=f"Sandbox python missing or not executable: {path}",
        fix="Install sandbox interpreter or update experiment.sandbox.python_path",
    )


def check_matplotlib() -> CheckResult:
    try:
        _ = importlib.import_module("matplotlib")
    except ImportError:
        return CheckResult(
            name="matplotlib",
            status="warn",
            detail="Not installed; charts will be skipped",
            fix="pip install matplotlib",
        )
    return CheckResult(name="matplotlib", status="pass", detail="matplotlib import ok")


def check_experiment_mode(mode: str) -> CheckResult:
    if mode == "simulated":
        return CheckResult(
            name="experiment_mode",
            status="warn",
            detail="Experiment mode is simulated — results will be synthetic",
            fix="Use sandbox or docker mode for real execution",
        )
    return CheckResult(
        name="experiment_mode",
        status="pass",
        detail=f"Experiment mode: {mode}",
    )


def check_acp_agent(agent_command: str) -> CheckResult:
    """Check that the ACP agent CLI is available on PATH."""
    resolved = shutil.which(agent_command)
    if resolved:
        return CheckResult(
            name="acp_agent",
            status="pass",
            detail=f"ACP agent found: {resolved}",
        )
    return CheckResult(
        name="acp_agent",
        status="fail",
        detail=f"ACP agent '{agent_command}' not found on PATH",
        fix=f"Install {agent_command} or update llm.acp.agent in config",
    )


def check_docker_runtime(config: RCConfig) -> CheckResult:
    """Check Docker daemon, image availability, and optional NVIDIA runtime."""
    from researchclaw.experiment.docker_sandbox import DockerSandbox

    if not DockerSandbox.check_docker_available():
        return CheckResult(
            name="docker_runtime",
            status="fail",
            detail="Docker daemon is not reachable",
            fix="Install and start Docker, or switch to mode: sandbox",
        )

    docker_cfg = config.experiment.docker
    if not DockerSandbox.ensure_image(docker_cfg.image):
        return CheckResult(
            name="docker_runtime",
            status="fail",
            detail=f"Docker image '{docker_cfg.image}' not found locally",
            fix=f"docker build -t {docker_cfg.image} researchclaw/docker/",
        )

    detail = f"Docker OK, image={docker_cfg.image}"
    if docker_cfg.gpu_enabled:
        detail += ", GPU passthrough enabled"

    return CheckResult(name="docker_runtime", status="pass", detail=detail)


def run_doctor(config_path: str | Path) -> DoctorReport:
    """Run all health checks and return report."""
    checks: list[CheckResult] = []
    path = Path(config_path)

    checks.append(check_python_version())
    checks.append(check_yaml_import())
    checks.append(check_config_valid(path))

    base_url = ""
    api_key = ""
    model = ""
    fallback_models: tuple[str, ...] = ()
    sandbox_python_path = ""
    experiment_mode = ""
    provider = ""
    acp_agent_command = "claude"

    try:
        config = RCConfig.load(path, check_paths=False)
        provider = config.llm.provider
        base_url = config.llm.base_url
        api_key = config.llm.api_key or os.environ.get(config.llm.api_key_env, "")
        model = config.llm.primary_model
        fallback_models = config.llm.fallback_models
        sandbox_python_path = config.experiment.sandbox.python_path
        experiment_mode = config.experiment.mode
        acp_agent_command = config.llm.acp.agent
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError) as exc:
        logger.debug("Could not fully load config for doctor checks: %s", exc)

    if provider == "acp":
        checks.append(check_acp_agent(acp_agent_command))
    else:
        checks.append(check_llm_connectivity(base_url))
        checks.append(check_api_key_valid(base_url, api_key))
        checks.append(check_model_chain(base_url, api_key, model, fallback_models))
    checks.append(check_sandbox_python(sandbox_python_path))
    checks.append(check_matplotlib())
    checks.append(check_experiment_mode(experiment_mode))

    if experiment_mode == "docker":
        try:
            checks.append(check_docker_runtime(config))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Docker health check failed: %s", exc)
            checks.append(
                CheckResult(
                    name="docker_runtime",
                    status="fail",
                    detail=f"Docker health check error: {exc}",
                    fix="Ensure Docker is installed and the daemon is running",
                )
            )

    overall = "fail" if any(c.status == "fail" for c in checks) else "pass"
    return DoctorReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checks=checks,
        overall=overall,
    )


def print_doctor_report(report: DoctorReport) -> None:
    """Pretty-print doctor report to stdout."""
    icon_by_status = {"pass": "✅", "fail": "❌", "warn": "⚠️"}
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        for icon in icon_by_status.values():
            icon.encode(encoding)
    except UnicodeEncodeError:
        icon_by_status = {"pass": "[OK]", "fail": "[FAIL]", "warn": "[WARN]"}
    print(f"ResearchClaw Doctor Report ({report.timestamp})")
    for check in report.checks:
        icon = icon_by_status.get(check.status, "-")
        print(f"{icon} {check.name}: {check.detail}")
        if check.fix:
            print(f"   Fix: {check.fix}")

    fail_count = sum(1 for check in report.checks if check.status == "fail")
    warn_count = sum(1 for check in report.checks if check.status == "warn")
    if report.overall == "pass":
        print("Result: PASS")
    else:
        print(f"Result: FAIL ({fail_count} errors, {warn_count} warnings)")


def write_doctor_report(report: DoctorReport, path: Path) -> None:
    """Write report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
