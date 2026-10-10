"""Execution profiles must reach design/code prompts without local guessing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig, SshRemoteConfig
from researchclaw.hardware import (
    _detect_nvidia_remote,
    detect_hardware,
    format_hardware_prompt,
)
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._code_generation import _execute_code_generation
from researchclaw.pipeline.stage_impls._experiment_design import _execute_experiment_design
from researchclaw.pipeline.stage_impls._topic import _execute_topic_init
from researchclaw.pipeline.stages import StageStatus


REMOTE_PROFILE = {
    "has_gpu": True,
    "gpu_type": "cuda",
    "gpu_name": "Tesla V100-PCIE-32GB (remote: compute.example)",
    "vram_mb": 32510,
    "tier": "high",
    "warning": "",
    "execution_mode": "ssh_remote",
    "detection_status": "detected",
}
MPS_PROFILE = {
    "has_gpu": True,
    "gpu_type": "mps",
    "gpu_name": "Apple test chip",
    "vram_mb": None,
    "tier": "limited",
    "warning": "MPS performance is limited",
    "execution_mode": "sandbox",
    "detection_status": "detected",
}
CPU_PROFILE = {
    "has_gpu": False,
    "gpu_type": "cpu",
    "gpu_name": "CPU only",
    "vram_mb": None,
    "tier": "cpu_only",
    "warning": "No GPU detected",
    "execution_mode": "sandbox",
    "detection_status": "detected",
}


class CapturingLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return LLMResponse(content=self.response, model="fake-model")


def _config(tmp_path: Path, mode: str, **ssh_overrides) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "remote-prompts", "mode": "docs-first"},
            "research": {
                "topic": "image classification optimizer comparison",
                "domains": ["ml"],
                "daily_paper_count": 1,
                "quality_threshold": 8.0,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "openclaw_bridge": {"use_memory": False, "use_message": False},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "REMOTE_PROMPTS_TEST_KEY",
                "api_key": "test-key",
                "primary_model": "fake-model",
            },
            "security": {"hitl_required_stages": []},
            "experiment": {
                "mode": mode,
                "time_budget_sec": 60,
                "metric_key": "best_loss",
                "code_agent": {"enabled": False},
                "benchmark_agent": {"enabled": False},
                "ssh_remote": {"host": "compute.example", **ssh_overrides},
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _stage_prompt(tmp_path, stage, mode, profile, **ssh_overrides):
    run_dir = tmp_path / "run"
    stage_1 = run_dir / "stage-01"
    stage_1.mkdir(parents=True)
    if profile is not None:
        (stage_1 / "hardware_profile.json").write_text(json.dumps(profile))
    cfg = _config(tmp_path, mode, **ssh_overrides)
    if stage == "design":
        stage_dir = run_dir / "stage-09"
        stage_dir.mkdir()
        llm = CapturingLLM("baselines: [sgd]\nproposed_methods: [adam]\nmetrics: [best_loss]")
        result = _execute_experiment_design(stage_dir, run_dir, cfg, AdapterBundle(), llm=llm)
    else:
        stage_dir = run_dir / "stage-10"
        stage_dir.mkdir()
        llm = CapturingLLM("```filename:main.py\nimport numpy as np\nprint('best_loss: 0.1')\n```")
        result = _execute_code_generation(stage_dir, run_dir, cfg, AdapterBundle(), llm=llm)
    assert result.status == StageStatus.DONE
    return llm.calls[0][-1]["content"]


@pytest.mark.parametrize("stage", ["design", "code"])
def test_remote_profile_reaches_design_and_code_prompts(tmp_path, stage):
    prompt = _stage_prompt(tmp_path, stage, "ssh_remote", REMOTE_PROFILE)
    assert "Tesla V100-PCIE-32GB" in prompt
    assert "32510 MB" in prompt
    assert "torch.device('cuda')" in prompt
    assert "do not use bf16=True" in prompt
    assert "do not request attn_implementation='flash_attention_2'" in prompt
    assert "RTX 6000 Ada" not in prompt
    assert "49140" not in prompt
    assert "torch.device('mps')" not in prompt
    assert "datasets are already in the Docker image" not in prompt
    assert "exactly ONE GPU" not in prompt
    if stage == "code":
        assert "not been verified" in prompt
        assert "not automatically executed" in prompt
        assert "setup_commands" in prompt
        assert "Hyperparameter Reporting" in prompt
        assert "Multi-Seed" in prompt or "multi-seed" in prompt.lower()


@pytest.mark.parametrize("stage", ["design", "code"])
@pytest.mark.parametrize("profile", [None, MPS_PROFILE, {**REMOTE_PROFILE, "execution_mode": "sandbox"}])
def test_remote_mode_does_not_reuse_missing_or_local_profiles(tmp_path, stage, profile):
    prompt = _stage_prompt(tmp_path, stage, "ssh_remote", profile)
    assert "Hardware: unknown" in prompt
    assert "torch.device('mps')" not in prompt
    assert "torch.device('cuda')" not in prompt
    assert "Apple test chip" not in prompt
    assert "32510 MB" not in prompt


@pytest.mark.parametrize("stage", ["design", "code"])
@pytest.mark.parametrize("profile", [MPS_PROFILE, CPU_PROFILE])
def test_local_profiles_keep_local_device_behavior(tmp_path, stage, profile):
    prompt = _stage_prompt(tmp_path, stage, "sandbox", profile)
    if profile["has_gpu"]:
        assert "Apple test chip" in prompt
        assert "torch.device('mps')" in prompt
        assert "lightweight" in prompt.lower()
    else:
        assert "GPU: none reported" in prompt
        assert "Use CPU execution" in prompt
        assert "torch.device('cuda')" not in prompt


def test_unmarked_old_remote_profile_stays_unknown():
    old_profile = {k: v for k, v in REMOTE_PROFILE.items() if k not in {"execution_mode", "detection_status"}}
    assert "Hardware: unknown" in format_hardware_prompt(old_profile, remote=True)


@pytest.mark.parametrize("failure", ["no_host", "failed_probe"])
def test_failed_remote_detection_never_probes_coordinator(monkeypatch, failure):
    local_probe = MagicMock(side_effect=AssertionError("local probe must not run"))
    remote_probe = MagicMock(return_value=None)
    monkeypatch.setattr("researchclaw.hardware._detect_nvidia", local_probe)
    monkeypatch.setattr("researchclaw.hardware._detect_mps", local_probe)
    monkeypatch.setattr("researchclaw.hardware._detect_nvidia_remote", remote_probe)
    cfg = SshRemoteConfig(host="" if failure == "no_host" else "compute.example")
    profile = detect_hardware(ssh_config=cfg)
    assert profile.gpu_type == "unknown"
    assert profile.detection_status == "failed"
    assert profile.execution_mode == "ssh_remote"
    assert "not evidence" in profile.warning
    local_probe.assert_not_called()


@pytest.mark.parametrize("stdout", ["", "not a gpu line"])
def test_empty_or_malformed_remote_probe_is_unknown(monkeypatch, stdout):
    monkeypatch.setattr("researchclaw.hardware.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout))
    assert _detect_nvidia_remote(SshRemoteConfig(host="compute.example")) is None


def test_remote_probe_respects_selected_gpu_and_persists_source(monkeypatch, tmp_path):
    probe = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="Tesla V100-PCIE-32GB, 32510\n"))
    monkeypatch.setattr("researchclaw.hardware.subprocess.run", probe)
    stage_dir = tmp_path / "run" / "stage-01"
    stage_dir.mkdir(parents=True)
    result = _execute_topic_init(
        stage_dir, stage_dir.parent, _config(tmp_path, "ssh_remote", gpu_ids=[1]), AdapterBundle()
    )
    assert result.status == StageStatus.DONE
    saved = json.loads((stage_dir / "hardware_profile.json").read_text())
    assert saved["execution_mode"] == "ssh_remote"
    assert saved["detection_status"] == "detected"
    assert saved["vram_mb"] == 32510
    command = probe.call_args.args[0]
    assert "nvidia-smi -i 1" in command[-1]
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=no" not in command


@pytest.mark.parametrize(
    ("ssh_settings", "expected"),
    [
        ({"network_isolation": "required"}, "requires network isolation"),
        ({"network_isolation": "disabled"}, "Network isolation is explicitly disabled"),
        ({"use_docker": True, "docker_network_policy": "none"}, "main experiment has no network access"),
        ({"use_docker": True, "docker_network_policy": "full"}, "Remote Docker network policy: full"),
    ],
)
def test_remote_data_guidance_matches_execution_network_policy(tmp_path, ssh_settings, expected):
    prompt = _stage_prompt(tmp_path, "code", "ssh_remote", REMOTE_PROFILE, **ssh_settings)
    assert expected in prompt
    assert "No datasets or pretrained models are known to be cached" in prompt


def test_remote_profile_also_reaches_code_agent(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    stage_1 = run_dir / "stage-01"
    stage_1.mkdir(parents=True)
    (stage_1 / "hardware_profile.json").write_text(json.dumps(REMOTE_PROFILE))
    stage_dir = run_dir / "stage-10"
    stage_dir.mkdir()
    cfg = _config(tmp_path, "ssh_remote")
    cfg = replace(cfg, experiment=replace(
        cfg.experiment, code_agent=replace(cfg.experiment.code_agent, enabled=True)
    ))
    generated = SimpleNamespace(
        files={"main.py": "import numpy as np\nprint('best_loss: 0.1')\n"},
        validation_log=[], total_llm_calls=0, total_sandbox_runs=0,
        best_score=1.0, tree_nodes_explored=0, review_rounds=0, architecture_spec="",
    )
    agent = MagicMock()
    agent.generate.return_value = generated
    factory = MagicMock(return_value=agent)
    monkeypatch.setattr("researchclaw.pipeline.code_agent.CodeAgent", factory)
    result = _execute_code_generation(
        stage_dir, run_dir, cfg, AdapterBundle(), llm=CapturingLLM("{}")
    )
    assert result.status == StageStatus.DONE
    context = agent.generate.call_args.kwargs["pkg_hint"]
    assert "Tesla V100-PCIE-32GB" in context
    assert "32510 MB" in context
    assert "torch.device('cuda')" in context
    assert "The Stage 1 execution hardware constraints take precedence" in context
    assert factory.call_args.kwargs["sandbox_factory"] is None
    assert factory.call_args.kwargs["experiment_config"] is cfg.experiment


@pytest.mark.parametrize("vram_mb", [32510, None])
def test_benchmark_agent_uses_profile_memory_instead_of_49gb(monkeypatch, tmp_path, vram_mb):
    run_dir = tmp_path / "run"
    stage_1 = run_dir / "stage-01"
    stage_1.mkdir(parents=True)
    profile = {**REMOTE_PROFILE, "execution_mode": "sandbox", "vram_mb": vram_mb}
    (stage_1 / "hardware_profile.json").write_text(json.dumps(profile))
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir()
    cfg = _config(tmp_path, "sandbox")
    cfg = replace(cfg, experiment=replace(
        cfg.experiment, benchmark_agent=replace(cfg.experiment.benchmark_agent, enabled=True)
    ))
    plan = SimpleNamespace(
        selected_benchmarks=[], selected_baselines=[], total_llm_calls=0,
        elapsed_sec=0, to_dict=lambda: {},
    )
    benchmark = MagicMock()
    benchmark.orchestrate.return_value = plan
    factory = MagicMock(return_value=benchmark)
    monkeypatch.setattr("researchclaw.agents.benchmark_agent.BenchmarkOrchestrator", factory)
    result = _execute_experiment_design(
        stage_dir, run_dir, cfg, AdapterBundle(),
        llm=CapturingLLM("baselines: [sgd]\nproposed_methods: [adam]"),
    )
    assert result.status == StageStatus.DONE
    assert factory.call_args.kwargs["gpu_memory_mb"] == (vram_mb or 0)

