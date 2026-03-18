"""Tests for the FigureAgent multi-agent chart generation system."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLMResponse:
    content: str = ""
    model: str = "gpt-4.1"
    prompt_tokens: int = 100
    completion_tokens: int = 200
    total_tokens: int = 300


class _FakeLLM:
    """Minimal mock LLM client conforming to _LLMClientLike."""

    def __init__(self, response: str = "{}"):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, *, system=None, max_tokens=None,
             temperature=None, json_mode=False):
        self.calls.append({
            "messages": messages,
            "system": system,
            "json_mode": json_mode,
        })
        return _FakeLLMResponse(content=self._response)


# Sample experiment data for tests
_SAMPLE_CONDITIONS = {
    "proposed_method": {
        "metrics": {
            "primary_metric": 0.85,
            "primary_metric_mean": 0.85,
            "primary_metric_std": 0.02,
            "secondary_metric": 0.72,
        },
        "ci95_low": 0.83,
        "ci95_high": 0.87,
        "n_seeds": 3,
    },
    "baseline_resnet": {
        "metrics": {
            "primary_metric": 0.78,
            "primary_metric_mean": 0.78,
            "primary_metric_std": 0.03,
            "secondary_metric": 0.65,
        },
        "ci95_low": 0.75,
        "ci95_high": 0.81,
        "n_seeds": 3,
    },
    "ablation_no_attention": {
        "metrics": {
            "primary_metric": 0.80,
            "primary_metric_mean": 0.80,
            "primary_metric_std": 0.02,
            "secondary_metric": 0.68,
        },
        "ci95_low": 0.78,
        "ci95_high": 0.82,
        "n_seeds": 3,
    },
}

_SAMPLE_METRICS_SUMMARY = {
    "primary_metric": {"mean": 0.81, "min": 0.78, "max": 0.85, "count": 3},
    "secondary_metric": {"mean": 0.68, "min": 0.65, "max": 0.72, "count": 3},
}


# =========================================================================
# Style Config tests
# =========================================================================

class TestStyleConfig:
    def test_constants_exist(self):
        from researchclaw.agents.figure_agent.style_config import (
            COLORS_BRIGHT, DPI_PUBLICATION, FIGURE_WIDTH,
            MATPLOTLIB_STYLES, OUTPUT_FORMAT_PRIMARY,
        )
        assert len(COLORS_BRIGHT) >= 7
        assert DPI_PUBLICATION >= 300
        assert "single_column" in FIGURE_WIDTH
        assert "double_column" in FIGURE_WIDTH
        assert len(MATPLOTLIB_STYLES) >= 1
        assert OUTPUT_FORMAT_PRIMARY in ("pdf", "png")

    def test_get_style_preamble(self):
        from researchclaw.agents.figure_agent.style_config import get_style_preamble
        preamble = get_style_preamble()
        assert "matplotlib" in preamble
        assert "plt" in preamble
        assert "COLORS" in preamble
        assert "300" in preamble

    def test_custom_dpi(self):
        from researchclaw.agents.figure_agent.style_config import get_style_preamble
        preamble = get_style_preamble(dpi=150)
        assert "150" in preamble


# =========================================================================
# Planner Agent tests
# =========================================================================

class TestPlannerAgent:
    def test_domain_detection_classification(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        agent = PlannerAgent(_FakeLLM())
        assert agent._detect_domain("Image classification with CIFAR-10") == "classification"

    def test_domain_detection_rl(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        agent = PlannerAgent(_FakeLLM())
        assert agent._detect_domain("Reinforcement learning with reward shaping") == "reinforcement_learning"

    def test_domain_detection_default(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        agent = PlannerAgent(_FakeLLM())
        assert agent._detect_domain("Quantum computing analysis") == "default"

    def test_analyze_data_basic(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        agent = PlannerAgent(_FakeLLM())
        analysis = agent._analyze_data(
            results={},
            conditions=["proposed", "baseline", "ablation_no_x"],
            metrics_summary=_SAMPLE_METRICS_SUMMARY,
            condition_summaries=_SAMPLE_CONDITIONS,
            metric_key="primary_metric",
        )
        assert analysis["num_conditions"] == 3
        assert analysis["has_ablation"] is True
        assert analysis["has_per_condition_data"] is True
        assert analysis["has_multiple_seeds"] is True

    def test_analyze_data_training_history(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        agent = PlannerAgent(_FakeLLM())
        analysis = agent._analyze_data(
            results={"training_history": [1.0, 0.5, 0.3]},
            conditions=["a"],
            metrics_summary={},
            condition_summaries={},
            metric_key="loss",
        )
        assert analysis["has_training_history"] is True

    def test_fallback_plan(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        agent = PlannerAgent(_FakeLLM())
        analysis = {
            "num_conditions": 3,
            "num_metrics": 2,
            "metric_names": ["primary_metric", "secondary_metric"],
            "has_training_history": False,
            "has_ablation": True,
            "has_multiple_seeds": True,
            "has_per_condition_data": True,
            "condition_values": {"proposed": 0.85, "baseline": 0.78},
        }
        figures = agent._fallback_plan("classification", analysis, "primary_metric", ["proposed", "baseline"])
        assert len(figures) >= 2
        types = {f["chart_type"] for f in figures}
        assert "bar_comparison" in types
        assert "ablation_grouped" in types

    def test_execute_with_llm_response(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        llm = _FakeLLM(json.dumps({
            "figures": [
                {
                    "figure_id": "fig_main",
                    "chart_type": "bar_comparison",
                    "title": "Main Results",
                    "caption": "Comparison of methods.",
                    "data_source": {"type": "condition_comparison", "metric": "primary_metric"},
                    "x_label": "Method",
                    "y_label": "Accuracy",
                    "width": "single_column",
                    "priority": 1,
                    "section": "results",
                },
                {
                    "figure_id": "fig_ablation",
                    "chart_type": "ablation_grouped",
                    "title": "Ablation",
                    "caption": "Component analysis.",
                    "data_source": {"type": "ablation_comparison", "metric": "primary_metric"},
                    "x_label": "Variant",
                    "y_label": "Accuracy",
                    "width": "single_column",
                    "priority": 1,
                    "section": "results",
                },
                {
                    "figure_id": "fig_heatmap",
                    "chart_type": "heatmap",
                    "title": "Metric Heatmap",
                    "caption": "Cross-metric analysis.",
                    "data_source": {"type": "multi_metric"},
                    "x_label": "Metric",
                    "y_label": "Method",
                    "width": "double_column",
                    "priority": 2,
                    "section": "analysis",
                },
            ]
        }))
        agent = PlannerAgent(llm, min_figures=3)
        result = agent.execute({
            "experiment_results": {},
            "topic": "Image classification with knowledge distillation",
            "metric_key": "primary_metric",
            "conditions": list(_SAMPLE_CONDITIONS.keys()),
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "condition_summaries": _SAMPLE_CONDITIONS,
        })
        assert result.success
        assert len(result.data["figures"]) == 3

    def test_execute_fallback_on_empty_llm(self):
        from researchclaw.agents.figure_agent.planner import PlannerAgent
        llm = _FakeLLM("{}")  # Empty response
        agent = PlannerAgent(llm, min_figures=2)
        result = agent.execute({
            "experiment_results": {},
            "topic": "Image classification",
            "metric_key": "primary_metric",
            "conditions": list(_SAMPLE_CONDITIONS.keys()),
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "condition_summaries": _SAMPLE_CONDITIONS,
        })
        assert result.success
        assert len(result.data["figures"]) >= 2


# =========================================================================
# CodeGen Agent tests
# =========================================================================

class TestCodeGenAgent:
    def test_template_bar_comparison(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        agent = CodeGenAgent(_FakeLLM())
        result = agent.execute({
            "figures": [{
                "figure_id": "fig_main",
                "chart_type": "bar_comparison",
                "title": "Results",
                "caption": "Main results.",
                "data_source": {"type": "condition_comparison", "metric": "primary_metric"},
                "x_label": "Method",
                "y_label": "Accuracy",
                "width": "single_column",
                "section": "results",
            }],
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
            "output_dir": "charts",
        })
        assert result.success
        scripts = result.data["scripts"]
        assert len(scripts) == 1
        script = scripts[0]["script"]
        assert "0.85" in script  # proposed_method value
        assert "0.78" in script  # baseline value
        assert "savefig" in script

    def test_template_grouped_bar(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        agent = CodeGenAgent(_FakeLLM())
        result = agent.execute({
            "figures": [{
                "figure_id": "fig_multi",
                "chart_type": "grouped_bar",
                "title": "Multi-metric",
                "caption": "Multi-metric comparison.",
                "data_source": {
                    "type": "multi_metric",
                    "metrics": ["primary_metric", "secondary_metric"],
                },
                "x_label": "Method",
                "y_label": "Score",
                "width": "double_column",
                "section": "analysis",
            }],
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
            "output_dir": "charts",
        })
        assert result.success
        scripts = result.data["scripts"]
        assert len(scripts) == 1
        assert "secondary_metric" in scripts[0]["script"]

    def test_template_heatmap(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        agent = CodeGenAgent(_FakeLLM())
        result = agent.execute({
            "figures": [{
                "figure_id": "fig_heat",
                "chart_type": "heatmap",
                "title": "Heatmap",
                "caption": "Analysis.",
                "data_source": {"type": "heatmap"},
                "x_label": "Metric",
                "y_label": "Method",
                "width": "double_column",
                "section": "analysis",
            }],
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
            "output_dir": "charts",
        })
        assert result.success
        scripts = result.data["scripts"]
        assert len(scripts) == 1
        assert "imshow" in scripts[0]["script"]

    def test_llm_fallback_for_unknown_type(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        llm = _FakeLLM("```python\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1,2,3])\nfig.savefig('charts/fig_custom.png')\nplt.close(fig)\n```")
        agent = CodeGenAgent(llm)
        result = agent.execute({
            "figures": [{
                "figure_id": "fig_custom",
                "chart_type": "radar_chart",
                "title": "Radar",
                "caption": "Custom chart.",
                "data_source": {},
                "x_label": "X",
                "y_label": "Y",
                "width": "single_column",
                "section": "analysis",
            }],
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
            "output_dir": "charts",
        })
        assert result.success
        assert "matplotlib" in result.data["scripts"][0]["script"]

    def test_strip_fences(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        code = "```python\nprint('hello')\n```"
        assert CodeGenAgent._strip_fences(code) == "print('hello')"

    def test_strip_fences_no_fences(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        code = "print('hello')"
        assert CodeGenAgent._strip_fences(code) == "print('hello')"

    def test_multiple_figures(self):
        from researchclaw.agents.figure_agent.codegen import CodeGenAgent
        agent = CodeGenAgent(_FakeLLM())
        figures = [
            {
                "figure_id": f"fig_{i}",
                "chart_type": "bar_comparison",
                "title": f"Figure {i}",
                "caption": f"Caption {i}.",
                "data_source": {"type": "condition_comparison", "metric": "primary_metric"},
                "x_label": "X",
                "y_label": "Y",
                "width": "single_column",
                "section": "results",
            }
            for i in range(3)
        ]
        result = agent.execute({
            "figures": figures,
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
            "output_dir": "charts",
        })
        assert result.success
        assert len(result.data["scripts"]) == 3


# =========================================================================
# Renderer Agent tests
# =========================================================================

class TestRendererAgent:
    def test_render_simple_script(self, tmp_path):
        from researchclaw.agents.figure_agent.renderer import RendererAgent
        agent = RendererAgent(_FakeLLM(), timeout_sec=10, use_docker=False)
        output_dir = tmp_path / "charts"

        # Use a script that creates a valid PNG without matplotlib
        # (creates a minimal 1x1 PNG file directly)
        script = textwrap.dedent("""\
            import struct, zlib
            output_path = "{output_dir}/fig_test.png"
            # Minimal valid PNG: 1x1 white pixel
            def write_png(path):
                sig = b'\\x89PNG\\r\\n\\x1a\\n'
                def chunk(ctype, data):
                    c = ctype + data
                    return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
                ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
                raw = zlib.compress(b'\\x00\\xff\\xff\\xff')
                with open(path, 'wb') as f:
                    f.write(sig)
                    f.write(chunk(b'IHDR', ihdr))
                    f.write(chunk(b'IDAT', raw))
                    f.write(chunk(b'IEND', b''))
            write_png(output_path)
            # Pad file to meet minimum size requirement
            with open(output_path, 'ab') as f:
                f.write(b'\\x00' * 2048)
            print(f"Saved: {{output_path}}")
        """).format(output_dir=output_dir)

        result = agent.execute({
            "scripts": [{
                "figure_id": "fig_test",
                "script": script,
                "output_filename": "fig_test.png",
                "title": "Test",
                "caption": "Test chart",
                "section": "results",
            }],
            "output_dir": str(output_dir),
        })
        assert result.success
        rendered = result.data["rendered"]
        assert len(rendered) == 1
        assert rendered[0]["success"] is True
        assert Path(rendered[0]["output_path"]).exists()

    def test_render_syntax_error(self, tmp_path):
        from researchclaw.agents.figure_agent.renderer import RendererAgent
        agent = RendererAgent(_FakeLLM(), timeout_sec=5)
        result = agent.execute({
            "scripts": [{
                "figure_id": "fig_bad",
                "script": "this is not valid python!!!",
                "output_filename": "fig_bad.png",
            }],
            "output_dir": str(tmp_path / "charts"),
        })
        # The renderer itself succeeds (returns results), but individual
        # figures have success=False
        rendered = result.data["rendered"]
        assert len(rendered) == 1
        assert rendered[0]["success"] is False
        assert rendered[0]["error"]

    def test_render_empty_script(self, tmp_path):
        from researchclaw.agents.figure_agent.renderer import RendererAgent
        agent = RendererAgent(_FakeLLM(), timeout_sec=5)
        result = agent.execute({
            "scripts": [{
                "figure_id": "fig_empty",
                "script": "",
                "output_filename": "fig_empty.png",
            }],
            "output_dir": str(tmp_path / "charts"),
        })
        rendered = result.data["rendered"]
        assert rendered[0]["success"] is False
        assert "Empty" in rendered[0]["error"]

    def test_script_saved_for_reproducibility(self, tmp_path):
        from researchclaw.agents.figure_agent.renderer import RendererAgent
        agent = RendererAgent(_FakeLLM(), timeout_sec=5)
        output_dir = tmp_path / "charts"
        result = agent.execute({
            "scripts": [{
                "figure_id": "fig_save",
                "script": "print('hello')",
                "output_filename": "fig_save.png",
            }],
            "output_dir": str(output_dir),
        })
        # Script should be saved even if rendering fails
        script_path = output_dir / "scripts" / "fig_save.py"
        assert script_path.exists()
        assert script_path.read_text() == "print('hello')"


# =========================================================================
# Critic Agent tests
# =========================================================================

class TestCriticAgent:
    def test_numerical_accuracy_pass(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        llm = _FakeLLM(json.dumps({
            "quality_score": 8,
            "issues": [],
        }))
        agent = CriticAgent(llm)
        script = "values = [0.85, 0.78, 0.80]\nax.bar(x, values)\nfig.savefig('out.png')\nplt.close(fig)"
        issues = agent._check_numerical_accuracy(script, _SAMPLE_CONDITIONS, "primary_metric")
        # Values 0.85 and 0.78 are in script → should pass
        assert not any(i["severity"] == "critical" for i in issues)

    def test_numerical_accuracy_fail(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        agent = CriticAgent(_FakeLLM())
        script = "values = [0.99, 0.98, 0.97]"  # Wrong values
        issues = agent._check_numerical_accuracy(script, _SAMPLE_CONDITIONS, "primary_metric")
        assert any(i["severity"] == "critical" for i in issues)

    def test_text_correctness_missing_labels(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        agent = CriticAgent(_FakeLLM())
        script = "fig, ax = plt.subplots()\nax.bar([0], [1])"  # Missing labels + savefig
        issues = agent._check_text_correctness(script, {})
        types = {i["message"] for i in issues}
        assert any("x-axis" in t for t in types)
        assert any("savefig" in t for t in types)

    def test_text_correctness_all_present(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        agent = CriticAgent(_FakeLLM())
        script = (
            "ax.set_xlabel('X')\n"
            "ax.set_ylabel('Y')\n"
            "ax.set_title('T')\n"
            "fig.savefig('out.png')\n"
            "plt.close(fig)"
        )
        issues = agent._check_text_correctness(script, {})
        assert len(issues) == 0

    def test_visual_quality_llm_review(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        llm = _FakeLLM(json.dumps({
            "quality_score": 9,
            "issues": [],
        }))
        agent = CriticAgent(llm)
        issues = agent._check_visual_quality(
            "import matplotlib\nplt.figure()\nplt.savefig('x.png')",
            {"title": "Test"},
        )
        assert not any(i["severity"] == "critical" for i in issues)

    def test_visual_quality_low_score(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        llm = _FakeLLM(json.dumps({
            "quality_score": 3,
            "issues": [{"severity": "critical", "message": "Bad colors"}],
        }))
        agent = CriticAgent(llm)
        issues = agent._check_visual_quality("plt.plot([1,2])", {"title": "Bad"})
        assert any(i["severity"] == "critical" for i in issues)

    def test_execute_full_review(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        llm = _FakeLLM(json.dumps({
            "quality_score": 8,
            "issues": [],
        }))
        agent = CriticAgent(llm)
        result = agent.execute({
            "rendered": [
                {
                    "figure_id": "fig_1",
                    "success": True,
                    "output_path": "/tmp/fig.png",
                    "title": "Test",
                    "caption": "Test fig",
                },
            ],
            "scripts": [
                {
                    "figure_id": "fig_1",
                    "script": (
                        "values = [0.85, 0.78]\n"
                        "ax.set_xlabel('X')\nax.set_ylabel('Y')\n"
                        "ax.set_title('T')\nfig.savefig('x.png')\nplt.close(fig)"
                    ),
                },
            ],
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
        })
        assert result.success
        assert result.data["passed_count"] >= 0

    def test_review_failed_render(self):
        from researchclaw.agents.figure_agent.critic import CriticAgent
        agent = CriticAgent(_FakeLLM())
        result = agent.execute({
            "rendered": [
                {"figure_id": "fig_1", "success": False, "error": "Crash"},
            ],
            "scripts": [],
            "condition_summaries": {},
            "metrics_summary": {},
            "metric_key": "primary_metric",
        })
        assert result.success
        assert result.data["reviews"][0]["passed"] is False


# =========================================================================
# Integrator Agent tests
# =========================================================================

class TestIntegratorAgent:
    def test_build_manifest(self):
        from researchclaw.agents.figure_agent.integrator import IntegratorAgent
        agent = IntegratorAgent(_FakeLLM())
        rendered = [
            {
                "figure_id": "fig_main",
                "success": True,
                "output_path": "/tmp/charts/fig_main.png",
                "title": "Main Results",
                "caption": "Comparison.",
                "section": "results",
                "width": "single_column",
            },
            {
                "figure_id": "fig_ablation",
                "success": True,
                "output_path": "/tmp/charts/fig_ablation.png",
                "title": "Ablation",
                "caption": "Analysis.",
                "section": "results",
                "width": "single_column",
            },
        ]
        manifest = agent._build_manifest(rendered, Path("/tmp/charts"))
        assert len(manifest) == 2
        assert manifest[0]["figure_number"] == 1
        assert manifest[0]["paper_section"] == "Results"
        assert "charts/" in manifest[0]["file_path"]

    def test_generate_markdown_refs(self):
        from researchclaw.agents.figure_agent.integrator import IntegratorAgent
        agent = IntegratorAgent(_FakeLLM())
        manifest = [
            {
                "figure_number": 1,
                "file_path": "charts/fig_1.png",
                "caption": "Main results comparison",
            },
        ]
        refs = agent._generate_markdown_refs(manifest)
        assert "![Figure 1:" in refs
        assert "charts/fig_1.png" in refs

    def test_generate_descriptions(self):
        from researchclaw.agents.figure_agent.integrator import IntegratorAgent
        agent = IntegratorAgent(_FakeLLM())
        manifest = [
            {
                "figure_number": 1,
                "file_path": "charts/fig_1.png",
                "title": "Main Results",
                "caption": "Comparison",
                "paper_section": "Results",
            },
        ]
        desc = agent._generate_descriptions(manifest)
        assert "AVAILABLE FIGURES" in desc
        assert "Main Results" in desc
        assert "Results" in desc

    def test_execute_empty(self):
        from researchclaw.agents.figure_agent.integrator import IntegratorAgent
        agent = IntegratorAgent(_FakeLLM())
        result = agent.execute({
            "rendered": [],
            "topic": "Test",
            "output_dir": "/tmp/charts",
        })
        assert result.success
        assert result.data["figure_count"] == 0

    def test_execute_with_figures(self, tmp_path):
        from researchclaw.agents.figure_agent.integrator import IntegratorAgent
        agent = IntegratorAgent(_FakeLLM())
        output_dir = tmp_path / "charts"
        output_dir.mkdir()

        result = agent.execute({
            "rendered": [
                {
                    "figure_id": "fig_main",
                    "success": True,
                    "output_path": str(output_dir / "fig_main.png"),
                    "title": "Main",
                    "caption": "Main comparison.",
                    "section": "results",
                },
            ],
            "topic": "Test",
            "output_dir": str(output_dir),
        })
        assert result.success
        assert result.data["figure_count"] == 1
        assert (output_dir / "figure_manifest.json").exists()

    def test_section_ordering(self):
        from researchclaw.agents.figure_agent.integrator import IntegratorAgent
        assert IntegratorAgent._section_order("method") < IntegratorAgent._section_order("results")
        assert IntegratorAgent._section_order("results") < IntegratorAgent._section_order("analysis")


# =========================================================================
# Orchestrator tests
# =========================================================================

class TestOrchestrator:
    def test_orchestrate_basic(self, tmp_path):
        from researchclaw.agents.figure_agent.orchestrator import (
            FigureAgentConfig, FigureOrchestrator,
        )

        # LLM returns plan, then quality review
        responses = iter([
            json.dumps({
                "figures": [{
                    "figure_id": "fig_main",
                    "chart_type": "bar_comparison",
                    "title": "Main",
                    "caption": "Main comparison.",
                    "data_source": {"type": "condition_comparison", "metric": "primary_metric"},
                    "x_label": "Method",
                    "y_label": "Accuracy",
                    "width": "single_column",
                    "priority": 1,
                    "section": "results",
                }, {
                    "figure_id": "fig_ablation",
                    "chart_type": "ablation_grouped",
                    "title": "Ablation",
                    "caption": "Ablation study.",
                    "data_source": {"type": "ablation_comparison", "metric": "primary_metric"},
                    "x_label": "Variant",
                    "y_label": "Accuracy",
                    "width": "single_column",
                    "priority": 1,
                    "section": "results",
                }, {
                    "figure_id": "fig_heatmap",
                    "chart_type": "heatmap",
                    "title": "Heatmap",
                    "caption": "Metric heatmap.",
                    "data_source": {"type": "heatmap"},
                    "x_label": "Metric",
                    "y_label": "Method",
                    "width": "double_column",
                    "priority": 2,
                    "section": "analysis",
                }],
            }),
            # Critic review (called multiple times)
            json.dumps({"quality_score": 8, "issues": []}),
            json.dumps({"quality_score": 8, "issues": []}),
            json.dumps({"quality_score": 8, "issues": []}),
        ])

        class _MultiLLM:
            def __init__(self):
                self.calls = []
            def chat(self, messages, **kwargs):
                self.calls.append(messages)
                try:
                    resp = next(responses)
                except StopIteration:
                    resp = json.dumps({"quality_score": 8, "issues": []})
                return _FakeLLMResponse(content=resp)

        cfg = FigureAgentConfig(
            min_figures=3,
            max_figures=5,
            max_iterations=1,
            render_timeout_sec=10,
        )
        orch = FigureOrchestrator(_MultiLLM(), cfg, stage_dir=tmp_path)
        plan = orch.orchestrate({
            "experiment_results": {},
            "condition_summaries": _SAMPLE_CONDITIONS,
            "metrics_summary": _SAMPLE_METRICS_SUMMARY,
            "metric_key": "primary_metric",
            "conditions": list(_SAMPLE_CONDITIONS.keys()),
            "topic": "Image classification",
            "output_dir": str(tmp_path / "charts"),
        })

        assert plan.total_llm_calls > 0
        assert plan.elapsed_sec > 0
        # Plan should have chart files (some may fail rendering, that's OK)
        assert isinstance(plan.manifest, list)

    def test_figure_plan_serialization(self):
        from researchclaw.agents.figure_agent.orchestrator import FigurePlan
        plan = FigurePlan(
            manifest=[{"figure_number": 1, "file_path": "charts/fig.png"}],
            figure_count=1,
            passed_count=1,
        )
        d = plan.to_dict()
        assert d["figure_count"] == 1
        assert len(d["manifest"]) == 1

    def test_get_chart_files(self):
        from researchclaw.agents.figure_agent.orchestrator import FigurePlan
        plan = FigurePlan(
            manifest=[
                {"figure_number": 1, "file_path": "charts/fig_main.png"},
                {"figure_number": 2, "file_path": "charts/fig_ablation.png"},
            ],
        )
        files = plan.get_chart_files()
        assert files == ["fig_main.png", "fig_ablation.png"]


# =========================================================================
# Config tests
# =========================================================================

class TestFigureAgentConfig:
    def test_default_config(self):
        from researchclaw.config import FigureAgentConfig
        cfg = FigureAgentConfig()
        assert cfg.enabled is True
        assert cfg.min_figures == 3
        assert cfg.max_figures == 8
        assert cfg.max_iterations == 3
        assert cfg.dpi == 300
        assert cfg.strict_mode is False

    def test_parse_from_dict(self):
        from researchclaw.config import _parse_figure_agent_config
        cfg = _parse_figure_agent_config({
            "enabled": False,
            "min_figures": 2,
            "max_figures": 6,
            "dpi": 150,
        })
        assert cfg.enabled is False
        assert cfg.min_figures == 2
        assert cfg.max_figures == 6
        assert cfg.dpi == 150

    def test_parse_from_dict_extended_fields(self):
        from researchclaw.config import _parse_figure_agent_config
        cfg = _parse_figure_agent_config({
            "use_docker": False,
            "docker_image": "custom/figure:latest",
            "output_format": "latex",
            "gemini_api_key": "test-key",
            "gemini_model": "gemini-test",
            "nano_banana_enabled": False,
        })
        assert cfg.use_docker is False
        assert cfg.docker_image == "custom/figure:latest"
        assert cfg.output_format == "latex"
        assert cfg.gemini_api_key == "test-key"
        assert cfg.gemini_model == "gemini-test"
        assert cfg.nano_banana_enabled is False

    def test_parse_empty(self):
        from researchclaw.config import _parse_figure_agent_config
        cfg = _parse_figure_agent_config({})
        assert cfg.enabled is True
        assert cfg.min_figures == 3

    def test_experiment_config_has_figure_agent(self):
        from researchclaw.config import ExperimentConfig
        ec = ExperimentConfig()
        assert hasattr(ec, "figure_agent")
        assert ec.figure_agent.enabled is True


# =========================================================================
# Backward compatibility test
# =========================================================================

class TestBackwardCompatibility:
    def test_visualize_still_importable(self):
        """Old visualize.py functions should still be importable."""
        from researchclaw.experiment.visualize import (
            generate_all_charts,
            plot_condition_comparison,
            plot_experiment_comparison,
            plot_metric_trajectory,
        )
        assert callable(generate_all_charts)
        assert callable(plot_condition_comparison)
        assert callable(plot_experiment_comparison)
        assert callable(plot_metric_trajectory)

    def test_figure_agent_importable(self):
        from researchclaw.agents.figure_agent import FigureOrchestrator, FigurePlan
        assert FigureOrchestrator is not None
        assert FigurePlan is not None
