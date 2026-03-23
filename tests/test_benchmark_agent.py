"""Tests for the BenchmarkAgent multi-agent system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fake LLM client (same pattern as test_code_agent.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMResponse:
    content: str = ""
    model: str = "fake"
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30
    finish_reason: str = "stop"
    truncated: bool = False
    raw: dict = field(default_factory=dict)


class FakeLLM:
    """Fake LLM that returns preconfigured responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, **kwargs) -> FakeLLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            content = '{"benchmarks": [], "baselines": []}'
        return FakeLLMResponse(content=content)


# ---------------------------------------------------------------------------
# Knowledge base tests
# ---------------------------------------------------------------------------


class TestBenchmarkKnowledge:
    """Test the benchmark_knowledge.yaml file."""

    def test_knowledge_file_exists(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import _KNOWLEDGE_PATH
        assert _KNOWLEDGE_PATH.exists(), f"Knowledge file missing: {_KNOWLEDGE_PATH}"

    def test_knowledge_loads(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import _KNOWLEDGE_PATH
        data = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "domains" in data

    def test_knowledge_has_domains(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import _KNOWLEDGE_PATH
        data = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        domains = data["domains"]
        assert len(domains) >= 10, f"Expected 10+ domains, got {len(domains)}"

    def test_each_domain_has_benchmarks_and_baselines(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import _KNOWLEDGE_PATH
        data = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        for did, info in data["domains"].items():
            assert "keywords" in info, f"Domain {did} missing keywords"
            assert "standard_benchmarks" in info, f"Domain {did} missing benchmarks"
            assert "common_baselines" in info, f"Domain {did} missing baselines"
            assert len(info["standard_benchmarks"]) > 0, f"Domain {did} has 0 benchmarks"
            assert len(info["common_baselines"]) > 0, f"Domain {did} has 0 baselines"

    def test_benchmark_entries_have_required_fields(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import _KNOWLEDGE_PATH
        data = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        for did, info in data["domains"].items():
            for b in info["standard_benchmarks"]:
                assert "name" in b, f"Benchmark in {did} missing name"
                assert "tier" in b, f"Benchmark {b.get('name')} in {did} missing tier"
                assert b["tier"] in (1, 2, 3), f"Invalid tier for {b.get('name')}"

    def test_baseline_entries_have_required_fields(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import _KNOWLEDGE_PATH
        data = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        for did, info in data["domains"].items():
            for bl in info["common_baselines"]:
                assert "name" in bl, f"Baseline in {did} missing name"
                assert "source" in bl, f"Baseline {bl.get('name')} in {did} missing source"
                assert "paper" in bl, f"Baseline {bl.get('name')} in {did} missing paper"


# ---------------------------------------------------------------------------
# Surveyor tests
# ---------------------------------------------------------------------------


class TestSurveyor:
    """Test SurveyorAgent domain matching and local search."""

    def test_domain_matching_image_classification(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        domains = agent._match_domains(
            "Image Classification with Contrastive Learning"
        )
        assert "image_classification" in domains

    def test_domain_matching_rl(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        domains = agent._match_domains(
            "Reinforcement Learning for Continuous Control"
        )
        assert "reinforcement_learning" in domains

    def test_domain_matching_knowledge_distillation(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        domains = agent._match_domains(
            "Knowledge Distillation with Feature Alignment"
        )
        assert "knowledge_distillation" in domains

    def test_domain_matching_multiple(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        domains = agent._match_domains(
            "Self-Supervised Contrastive Learning for Image Classification"
        )
        assert len(domains) >= 2

    def test_local_candidates_returns_benchmarks(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        result = agent._get_local_candidates(["image_classification"])
        assert len(result["benchmarks"]) > 0
        assert len(result["baselines"]) > 0

    def test_execute_returns_benchmarks(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        result = agent.execute({
            "topic": "Image Classification with Data Augmentation",
            "hypothesis": "Novel augmentation improves accuracy",
        })
        assert result.success
        assert len(result.data["benchmarks"]) > 0

    def test_execute_with_unknown_topic_uses_llm_fallback(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        llm = FakeLLM([json.dumps({
            "benchmarks": [{"name": "CustomDS", "tier": 2}],
            "baselines": [{"name": "CustomBL", "source": "custom", "paper": "X"}],
            "rationale": "test",
        })])
        agent = SurveyorAgent(llm, enable_hf_search=False)
        result = agent.execute({
            "topic": "Completely Novel Alien Technology Classification",
            "hypothesis": "",
        })
        assert result.success
        assert result.data["llm_fallback_used"]

    def test_extract_search_keywords(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        kws = SurveyorAgent._extract_search_keywords(
            "Novel Approach for Image Classification using Contrastive Learning"
        )
        assert len(kws) >= 1
        for kw in kws:
            assert "novel" not in kw.lower()
            assert "using" not in kw.lower()

    def test_execute_empty_topic_fails(self) -> None:
        from researchclaw.agents.benchmark_agent.surveyor import SurveyorAgent
        agent = SurveyorAgent(FakeLLM(), enable_hf_search=False)
        result = agent.execute({"topic": ""})
        assert not result.success


# ---------------------------------------------------------------------------
# Selector tests
# ---------------------------------------------------------------------------


class TestSelector:
    """Test SelectorAgent filtering and ranking logic."""

    @pytest.fixture()
    def benchmarks(self) -> list[dict]:
        return [
            {"name": "CIFAR-10", "tier": 1, "size_mb": 170, "origin": "knowledge_base",
             "metrics": ["accuracy"]},
            {"name": "CIFAR-100", "tier": 1, "size_mb": 170, "origin": "knowledge_base",
             "metrics": ["accuracy"]},
            {"name": "Tiny-ImageNet", "tier": 2, "size_mb": 237, "origin": "knowledge_base",
             "metrics": ["top1_accuracy"]},
            {"name": "ImageNet-1K", "tier": 3, "size_mb": 168000, "origin": "knowledge_base",
             "metrics": ["top1_accuracy"]},
            {"name": "hf/custom-ds", "tier": 2, "size_mb": 500, "origin": "huggingface_hub",
             "downloads": 1000},
        ]

    @pytest.fixture()
    def baselines(self) -> list[dict]:
        return [
            {"name": "ResNet-18", "origin": "knowledge_base", "pip": [],
             "paper": "He et al."},
            {"name": "ViT-B/16", "origin": "knowledge_base", "pip": ["timm"],
             "paper": "Dosovitskiy et al."},
        ]

    def test_filter_excludes_tier3(self, benchmarks: list[dict]) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent
        agent = SelectorAgent(FakeLLM(), tier_limit=2)
        filtered = agent._filter_benchmarks(benchmarks)
        names = [b["name"] for b in filtered]
        assert "ImageNet-1K" not in names
        assert "CIFAR-10" in names

    def test_filter_network_none_only_tier1(self, benchmarks: list[dict]) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent
        agent = SelectorAgent(FakeLLM(), network_policy="none")
        filtered = agent._filter_benchmarks(benchmarks)
        for b in filtered:
            assert b["tier"] == 1

    def test_ranking_prefers_tier1(self, benchmarks: list[dict]) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent
        agent = SelectorAgent(FakeLLM())
        filtered = agent._filter_benchmarks(benchmarks)
        ranked = agent._rank_benchmarks(filtered)
        # Tier 1 should come first
        assert ranked[0]["tier"] == 1

    def test_ranking_prefers_knowledge_base(self, benchmarks: list[dict]) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent
        agent = SelectorAgent(FakeLLM())
        filtered = agent._filter_benchmarks(benchmarks)
        ranked = agent._rank_benchmarks(filtered)
        # Knowledge base entries should precede HF entries of same tier
        kb_indices = [i for i, b in enumerate(ranked) if b["origin"] == "knowledge_base"]
        hf_indices = [i for i, b in enumerate(ranked) if b["origin"] == "huggingface_hub"]
        if kb_indices and hf_indices:
            assert min(kb_indices) < min(hf_indices)

    def test_execute_selects_minimum(self, benchmarks: list[dict],
                                     baselines: list[dict]) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent
        llm = FakeLLM([json.dumps({
            "primary_benchmark": "CIFAR-10",
            "secondary_benchmarks": ["CIFAR-100"],
            "selected_baselines": ["ResNet-18", "ViT-B/16"],
            "rationale": "Standard benchmarks",
            "experiment_notes": "",
        })])
        agent = SelectorAgent(llm, min_benchmarks=1, min_baselines=2)
        result = agent.execute({
            "topic": "Image Classification",
            "survey": {"benchmarks": benchmarks, "baselines": baselines},
        })
        assert result.success
        assert len(result.data["selected_benchmarks"]) >= 1
        assert len(result.data["selected_baselines"]) >= 2


# ---------------------------------------------------------------------------
# Acquirer tests
# ---------------------------------------------------------------------------


class TestAcquirer:
    """Test AcquirerAgent code generation."""

    def test_generate_setup_script_tier1_only(self) -> None:
        from researchclaw.agents.benchmark_agent.acquirer import AcquirerAgent
        agent = AcquirerAgent(FakeLLM())
        script = agent._generate_setup_script(
            [{"name": "CIFAR-10", "tier": 1, "api": "torchvision..."}], []
        )
        # Tier 1 datasets don't need setup scripts
        assert script == ""

    def test_generate_setup_script_tier2(self) -> None:
        from researchclaw.agents.benchmark_agent.acquirer import AcquirerAgent
        agent = AcquirerAgent(FakeLLM())
        script = agent._generate_setup_script(
            [{"name": "IMDB", "tier": 2,
              "api": "datasets.load_dataset('imdb', cache_dir='/workspace/data/hf')"}],
            [],
        )
        assert "download_datasets" in script
        assert "load_dataset" in script

    def test_generate_requirements_filters_builtin(self) -> None:
        from researchclaw.agents.benchmark_agent.acquirer import AcquirerAgent
        agent = AcquirerAgent(FakeLLM())
        reqs = agent._generate_requirements(["torch", "numpy", "xgboost", "timm"])
        assert "torch" not in reqs
        assert "numpy" not in reqs
        assert "timm" not in reqs
        assert "xgboost" in reqs

    def test_strip_fences(self) -> None:
        from researchclaw.agents.benchmark_agent.acquirer import AcquirerAgent
        code = "```python\nimport torch\n```"
        assert AcquirerAgent._strip_fences(code) == "import torch"

    def test_execute_generates_code(self) -> None:
        from researchclaw.agents.benchmark_agent.acquirer import AcquirerAgent
        llm = FakeLLM([
            "import torchvision\ndef get_datasets(): pass",
            "import torch.nn as nn\ndef get_baselines(): pass",
        ])
        agent = AcquirerAgent(llm)
        result = agent.execute({
            "topic": "Image Classification",
            "selection": {
                "selected_benchmarks": [
                    {"name": "CIFAR-10", "tier": 1, "role": "primary",
                     "api": "torchvision.datasets.CIFAR10(...)"},
                ],
                "selected_baselines": [
                    {"name": "ResNet-18", "source": "torchvision.models.resnet18()",
                     "paper": "He et al.", "pip": []},
                ],
                "required_pip": [],
            },
        })
        assert result.success
        assert result.data["data_loader_code"]


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidator:
    """Test ValidatorAgent code validation."""

    def test_syntax_check_valid(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        agent = ValidatorAgent(FakeLLM())
        errors = agent._check_syntax("import torch\nx = 1 + 2", "test")
        assert errors == []

    def test_syntax_check_invalid(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        agent = ValidatorAgent(FakeLLM())
        errors = agent._check_syntax("def foo(\n  x = ", "test")
        assert len(errors) > 0
        assert "SyntaxError" in errors[0]

    def test_import_check_builtin_ok(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        agent = ValidatorAgent(FakeLLM())
        warnings = agent._check_imports("import torch\nimport numpy", "test", [])
        assert warnings == []

    def test_import_check_unknown(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        agent = ValidatorAgent(FakeLLM())
        warnings = agent._check_imports("import some_obscure_lib", "test", [])
        assert len(warnings) > 0

    def test_import_check_with_requirements(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        agent = ValidatorAgent(FakeLLM())
        warnings = agent._check_imports(
            "import xgboost", "test", ["xgboost"],
        )
        assert warnings == []

    def test_execute_passes_valid_code(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        llm = FakeLLM([json.dumps({
            "passed": True,
            "issues": [],
            "suggestions": [],
            "severity": "none",
        })])
        agent = ValidatorAgent(llm)
        result = agent.execute({
            "acquisition": {
                "data_loader_code": "import torch\ndef get_datasets(): pass",
                "baseline_code": "import torch.nn as nn\ndef get_baselines(): pass",
                "setup_code": "",
                "requirements": "",
                "benchmark_names": ["CIFAR-10"],
                "baseline_names": ["ResNet-18"],
            },
        })
        assert result.success
        assert result.data["passed"]

    def test_execute_fails_syntax_error(self) -> None:
        from researchclaw.agents.benchmark_agent.validator import ValidatorAgent
        agent = ValidatorAgent(FakeLLM())
        result = agent.execute({
            "acquisition": {
                "data_loader_code": "def foo(\n  x = ",
                "baseline_code": "",
                "setup_code": "",
                "requirements": "",
                "benchmark_names": [],
                "baseline_names": [],
            },
        })
        assert not result.data["passed"]
        assert len(result.data["errors"]) > 0


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    """Test BenchmarkOrchestrator end-to-end."""

    def test_orchestrate_produces_plan(self, tmp_path: Path) -> None:
        from researchclaw.agents.benchmark_agent.orchestrator import (
            BenchmarkAgentConfig,
            BenchmarkOrchestrator,
        )

        responses = [
            # Selector LLM response
            json.dumps({
                "primary_benchmark": "CIFAR-10",
                "secondary_benchmarks": ["CIFAR-100"],
                "selected_baselines": ["ResNet-18", "ViT-B/16"],
                "rationale": "Standard CV benchmarks",
                "experiment_notes": "Use standard augmentation",
            }),
            # Acquirer: data_loader_code
            "import torchvision\ndef get_datasets(data_root='/workspace/data'):\n    return {}",
            # Acquirer: baseline_code
            "import torch.nn as nn\ndef get_baselines(num_classes=10):\n    return {}",
            # Validator: LLM review
            json.dumps({
                "passed": True,
                "issues": [],
                "suggestions": ["Add transforms"],
                "severity": "none",
            }),
        ]

        cfg = BenchmarkAgentConfig(enable_hf_search=False)
        orchestrator = BenchmarkOrchestrator(
            FakeLLM(responses),
            config=cfg,
            stage_dir=tmp_path / "benchmark_agent",
        )
        plan = orchestrator.orchestrate({
            "topic": "Image Classification with Data Augmentation",
            "hypothesis": "Novel augmentation improves accuracy",
        })

        assert len(plan.selected_benchmarks) >= 1
        assert len(plan.selected_baselines) >= 1
        assert plan.validation_passed
        assert plan.total_llm_calls > 0
        assert plan.elapsed_sec > 0

    def test_orchestrate_saves_artifacts(self, tmp_path: Path) -> None:
        from researchclaw.agents.benchmark_agent.orchestrator import (
            BenchmarkAgentConfig,
            BenchmarkOrchestrator,
        )

        responses = [
            json.dumps({
                "primary_benchmark": "CIFAR-10",
                "secondary_benchmarks": [],
                "selected_baselines": ["ResNet-18"],
                "rationale": "test",
                "experiment_notes": "",
            }),
            "def get_datasets(): pass",
            "def get_baselines(): pass",
            json.dumps({"passed": True, "issues": [], "suggestions": [], "severity": "none"}),
        ]

        stage_dir = tmp_path / "benchmark_agent"
        cfg = BenchmarkAgentConfig(enable_hf_search=False)
        orchestrator = BenchmarkOrchestrator(
            FakeLLM(responses),
            config=cfg,
            stage_dir=stage_dir,
        )
        orchestrator.orchestrate({
            "topic": "Image Classification",
            "hypothesis": "",
        })

        assert (stage_dir / "survey_results.json").exists()
        assert (stage_dir / "selection_results.json").exists()
        assert (stage_dir / "benchmark_plan.json").exists()

    def test_plan_to_prompt_block(self) -> None:
        from researchclaw.agents.benchmark_agent.orchestrator import BenchmarkPlan
        plan = BenchmarkPlan(
            selected_benchmarks=[
                {"name": "CIFAR-10", "role": "primary", "metrics": ["accuracy"],
                 "api": "torchvision.datasets.CIFAR10(...)"},
            ],
            selected_baselines=[
                {"name": "ResNet-18", "source": "torchvision.models.resnet18()",
                 "paper": "He et al."},
            ],
            data_loader_code="def get_datasets(): pass",
            baseline_code="def get_baselines(): pass",
        )
        block = plan.to_prompt_block()
        assert "CIFAR-10" in block
        assert "ResNet-18" in block
        assert "get_datasets" in block
        assert "get_baselines" in block

    def test_plan_to_dict_serializable(self) -> None:
        from researchclaw.agents.benchmark_agent.orchestrator import BenchmarkPlan
        plan = BenchmarkPlan(
            selected_benchmarks=[{"name": "test"}],
            data_loader_code="code",
        )
        d = plan.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert "test" in json_str


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Test BenchmarkAgentConfig in config.py."""

    def test_default_config_has_benchmark_agent(self) -> None:
        from researchclaw.config import ExperimentConfig
        cfg = ExperimentConfig()
        assert hasattr(cfg, "benchmark_agent")
        assert cfg.benchmark_agent.enabled is True

    def test_parse_benchmark_agent_config(self) -> None:
        from researchclaw.config import _parse_benchmark_agent_config
        cfg = _parse_benchmark_agent_config({
            "enabled": False,
            "tier_limit": 1,
            "min_baselines": 3,
        })
        assert cfg.enabled is False
        assert cfg.tier_limit == 1
        assert cfg.min_baselines == 3

    def test_parse_benchmark_agent_config_empty(self) -> None:
        from researchclaw.config import _parse_benchmark_agent_config
        cfg = _parse_benchmark_agent_config({})
        assert cfg.enabled is True
        assert cfg.tier_limit == 2


# ---------------------------------------------------------------------------
# Base agent tests
# ---------------------------------------------------------------------------


class TestBaseAgent:
    """Test the base agent class."""

    def test_parse_json_direct(self) -> None:
        from researchclaw.agents.base import BaseAgent
        result = BaseAgent._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_fenced(self) -> None:
        from researchclaw.agents.base import BaseAgent
        result = BaseAgent._parse_json('Some text\n```json\n{"key": 1}\n```\nMore text')
        assert result == {"key": 1}

    def test_parse_json_embedded(self) -> None:
        from researchclaw.agents.base import BaseAgent
        result = BaseAgent._parse_json('Here is the result: {"a": 2} end')
        assert result == {"a": 2}

    def test_parse_json_invalid(self) -> None:
        from researchclaw.agents.base import BaseAgent
        result = BaseAgent._parse_json("no json here at all")
        assert result is None


# ---------------------------------------------------------------------------
# Required baselines injection (Improvement E)
# ---------------------------------------------------------------------------


class TestRequiredBaselines:
    """Test that required baselines are injected from knowledge base."""

    def test_inject_required_baselines_image_classification(self) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent

        llm = FakeLLM()
        agent = SelectorAgent(llm, min_baselines=1)
        selected: list[dict[str, Any]] = [
            {"name": "EfficientNet-B0", "origin": "knowledge_base"},
        ]
        injected = agent._inject_required_baselines(
            "image classification on CIFAR-10",
            selected,
            [],
        )
        # Should inject ResNet-50 and ViT-B/16 (required for image_classification)
        injected_names = {b["name"] for b in injected}
        assert "ResNet-50" in injected_names
        assert "ViT-B/16" in injected_names
        # Already-present baselines should not be duplicated
        assert sum(1 for b in selected if b["name"] == "EfficientNet-B0") == 1

    def test_inject_required_baselines_no_duplicates(self) -> None:
        from researchclaw.agents.benchmark_agent.selector import SelectorAgent

        llm = FakeLLM()
        agent = SelectorAgent(llm, min_baselines=1)
        selected: list[dict[str, Any]] = [
            {"name": "ResNet-50", "origin": "knowledge_base"},
            {"name": "ViT-B/16", "origin": "llm_suggestion"},
        ]
        injected = agent._inject_required_baselines(
            "image classification on CIFAR-10",
            selected,
            [],
        )
        # Both are already present → nothing should be injected
        assert len(injected) == 0
