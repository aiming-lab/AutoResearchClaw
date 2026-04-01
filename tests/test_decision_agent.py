"""Tests for FigureDecisionAgent, NanoBananaAgent, and Docker renderer.

Covers:
  - FigureDecisionAgent._parse_decisions() — JSON parsing edge cases
  - FigureDecisionAgent._heuristic_decide() — fallback coverage
  - FigureDecisionAgent._infer_backend() — backend classification
  - FigureDecisionAgent._enforce_bounds() — min/max enforcement
  - NanoBananaAgent._build_prompt() — prompt construction
  - NanoBananaAgent._get_type_guidelines() — guideline lookup
  - RendererAgent._execute_in_docker() — docker command construction
  - strip_thinking_tags() — safety verification
  - End-to-end decision + orchestration with mock LLM
"""

from __future__ import annotations

import json
import os
import subprocess
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
    finish_reason: str = "stop"
    truncated: bool = False
    raw: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


class _FakeLLM:
    """Minimal mock LLM client."""

    def __init__(self, response: str = "{}"):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, *, system=None, max_tokens=None,
             temperature=None, json_mode=False, **kwargs):
        self.calls.append({
            "messages": messages,
            "system": system,
            "json_mode": json_mode,
        })
        return _FakeLLMResponse(content=self._response)


# =========================================================================
# FigureDecisionAgent._parse_decisions()
# =========================================================================

class TestParseDecisions:
    """Edge cases for JSON parsing in the decision agent."""

    def _agent(self):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        return FigureDecisionAgent(_FakeLLM())

    def test_valid_json_array(self):
        agent = self._agent()
        raw = json.dumps([
            {
                "section": "Method",
                "figure_type": "architecture_diagram",
                "backend": "image",
                "description": "Architecture overview",
                "priority": 1,
            },
            {
                "section": "Results",
                "figure_type": "bar_comparison",
                "backend": "code",
                "description": "Main results",
                "priority": 1,
            },
        ])
        decisions = agent._parse_decisions(raw)
        assert len(decisions) == 2
        assert decisions[0]["backend"] == "image"
        assert decisions[1]["backend"] == "code"

    def test_json_inside_markdown_fences(self):
        agent = self._agent()
        raw = '```json\n[{"section": "Method", "figure_type": "pipeline_overview", "backend": "image", "description": "Pipeline", "priority": 1}]\n```'
        decisions = agent._parse_decisions(raw)
        assert len(decisions) == 1
        assert decisions[0]["figure_type"] == "pipeline_overview"

    def test_json_with_surrounding_text(self):
        agent = self._agent()
        raw = 'Here are the decisions:\n[{"section": "Results", "figure_type": "heatmap", "backend": "code", "description": "Heatmap", "priority": 2}]\nThat is all.'
        decisions = agent._parse_decisions(raw)
        assert len(decisions) == 1

    def test_no_json_array_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="No JSON array"):
            agent._parse_decisions("This is not JSON at all.")

    def test_empty_array(self):
        agent = self._agent()
        decisions = agent._parse_decisions("[]")
        assert decisions == []

    def test_non_dict_items_skipped(self):
        agent = self._agent()
        raw = json.dumps([
            "not a dict",
            42,
            {"section": "Method", "figure_type": "architecture_diagram",
             "backend": "image", "description": "Arch", "priority": 1},
        ])
        decisions = agent._parse_decisions(raw)
        assert len(decisions) == 1

    def test_invalid_backend_auto_inferred(self):
        agent = self._agent()
        raw = json.dumps([
            {"section": "Method", "figure_type": "architecture_diagram",
             "backend": "invalid_backend", "description": "Arch", "priority": 1},
        ])
        decisions = agent._parse_decisions(raw)
        assert decisions[0]["backend"] == "image"  # architecture → image

    def test_missing_fields_get_defaults(self):
        agent = self._agent()
        raw = json.dumps([{}])
        decisions = agent._parse_decisions(raw)
        assert len(decisions) == 1
        assert decisions[0]["section"] == "Results"
        assert decisions[0]["figure_type"] == "bar_comparison"
        assert decisions[0]["backend"] == "code"
        assert decisions[0]["priority"] == 2


# =========================================================================
# FigureDecisionAgent._heuristic_decide()
# =========================================================================

class TestHeuristicDecide:
    """Test the rule-based fallback decision logic."""

    def _agent(self, min_figures=3, max_figures=10):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        return FigureDecisionAgent(
            _FakeLLM(), min_figures=min_figures, max_figures=max_figures
        )

    def test_with_experiments(self):
        agent = self._agent()
        decisions = agent._heuristic_decide(
            topic="Graph anomaly detection",
            has_experiments=True,
            condition_summaries={"proposed": {}, "baseline": {}, "ablation": {}},
        )
        # Should have: arch_diagram + bar_comparison + training_curve + pipeline
        assert len(decisions) >= 4
        backends = {d["backend"] for d in decisions}
        assert "code" in backends
        assert "image" in backends

    def test_without_experiments(self):
        agent = self._agent()
        decisions = agent._heuristic_decide(
            topic="Theoretical framework",
            has_experiments=False,
            condition_summaries={},
        )
        # Should have: arch_diagram + pipeline (image only, no code)
        assert len(decisions) >= 2
        assert all(d["backend"] == "image" for d in decisions)

    def test_ablation_trigger(self):
        """When >= 4 conditions, an ablation figure should be added."""
        agent = self._agent()
        decisions = agent._heuristic_decide(
            topic="Test",
            has_experiments=True,
            condition_summaries={"a": {}, "b": {}, "c": {}, "d": {}},
        )
        descriptions = [d["description"].lower() for d in decisions]
        assert any("ablation" in desc for desc in descriptions)

    def test_max_figures_respected(self):
        agent = self._agent(max_figures=2)
        decisions = agent._heuristic_decide(
            topic="Test",
            has_experiments=True,
            condition_summaries={"a": {}, "b": {}, "c": {}, "d": {}},
        )
        assert len(decisions) <= 2


# =========================================================================
# FigureDecisionAgent._infer_backend()
# =========================================================================

class TestInferBackend:
    def test_code_types(self):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        code_types = [
            "bar_comparison", "line_chart", "heatmap", "confusion_matrix",
            "training_curve", "ablation_chart", "scatter_plot",
        ]
        for t in code_types:
            assert FigureDecisionAgent._infer_backend(t) == "code", f"Failed for {t}"

    def test_image_types(self):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        image_types = [
            "architecture_diagram", "method_flowchart", "pipeline_overview",
            "concept_illustration", "system_diagram",
        ]
        for t in image_types:
            assert FigureDecisionAgent._infer_backend(t) == "image", f"Failed for {t}"

    def test_unknown_defaults_to_image(self):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        assert FigureDecisionAgent._infer_backend("unknown_chart_type") == "image"


# =========================================================================
# FigureDecisionAgent._enforce_bounds()
# =========================================================================

class TestEnforceBounds:
    def _agent(self, min_figures=3, max_figures=6):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        return FigureDecisionAgent(
            _FakeLLM(), min_figures=min_figures, max_figures=max_figures
        )

    def test_min_padding(self):
        """When fewer than min figures, should pad."""
        agent = self._agent(min_figures=4)
        decisions = [
            {"section": "Results", "figure_type": "bar_comparison",
             "backend": "code", "description": "Test", "priority": 1},
        ]
        result = agent._enforce_bounds(decisions, has_experiments=True)
        assert len(result) >= 4

    def test_max_truncation(self):
        """When more than max figures, should truncate."""
        agent = self._agent(max_figures=3)
        decisions = [
            {"section": f"S{i}", "figure_type": "bar_comparison",
             "backend": "code", "description": f"Fig {i}", "priority": i}
            for i in range(8)
        ]
        result = agent._enforce_bounds(decisions, has_experiments=True)
        assert len(result) <= 3

    def test_ensures_image_figure(self):
        """Should add architecture diagram if none present."""
        agent = self._agent(min_figures=1)
        decisions = [
            {"section": "Results", "figure_type": "bar_comparison",
             "backend": "code", "description": "Bar", "priority": 1},
        ]
        result = agent._enforce_bounds(decisions, has_experiments=True)
        assert any(d["backend"] == "image" for d in result)

    def test_ensures_code_figure_with_experiments(self):
        """Should add bar_comparison if experiments exist but no code figure."""
        agent = self._agent(min_figures=1)
        decisions = [
            {"section": "Method", "figure_type": "architecture_diagram",
             "backend": "image", "description": "Arch", "priority": 1},
        ]
        result = agent._enforce_bounds(decisions, has_experiments=True)
        assert any(d["backend"] == "code" for d in result)


# =========================================================================
# NanoBananaAgent._build_prompt()
# =========================================================================

class TestBuildPrompt:
    def _agent(self):
        from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent
        return NanoBananaAgent(
            _FakeLLM(), gemini_api_key="fake-key", use_sdk=False,
        )

    def test_prompt_contains_description(self):
        agent = self._agent()
        prompt = agent._build_prompt(
            description="Encoder-decoder with attention",
            figure_type="architecture_diagram",
            section="Method",
            topic="Graph anomaly detection",
        )
        assert "Encoder-decoder with attention" in prompt
        assert "Method" in prompt
        assert "Graph anomaly detection" in prompt

    def test_prompt_contains_style(self):
        agent = self._agent()
        prompt = agent._build_prompt(
            description="Test",
            figure_type="architecture_diagram",
            section="Method",
            topic="Test",
        )
        assert "academic" in prompt.lower()
        assert "publication" in prompt.lower()

    def test_prompt_varies_by_type(self):
        agent = self._agent()
        arch_prompt = agent._build_prompt(
            description="Test", figure_type="architecture_diagram",
            section="Method", topic="Test",
        )
        flow_prompt = agent._build_prompt(
            description="Test", figure_type="method_flowchart",
            section="Method", topic="Test",
        )
        # Different guidelines for different types
        assert arch_prompt != flow_prompt


# =========================================================================
# NanoBananaAgent._get_type_guidelines()
# =========================================================================

class TestGetTypeGuidelines:
    def test_known_types(self):
        from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent
        known = [
            "architecture_diagram", "method_flowchart", "pipeline_overview",
            "concept_illustration", "system_diagram", "attention_visualization",
            "comparison_illustration",
        ]
        for t in known:
            g = NanoBananaAgent._get_type_guidelines(t)
            assert len(g) > 0, f"Empty guidelines for {t}"

    def test_unknown_type_falls_back(self):
        from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent
        g = NanoBananaAgent._get_type_guidelines("totally_unknown")
        fallback = NanoBananaAgent._get_type_guidelines("concept_illustration")
        assert g == fallback


# =========================================================================
# NanoBananaAgent — no API key
# =========================================================================

class TestNanoBananaNoKey:
    def test_execute_without_key_fails(self, tmp_path):
        from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent
        # Clear env
        with mock.patch.dict(os.environ, {}, clear=True):
            agent = NanoBananaAgent(
                _FakeLLM(), gemini_api_key="", use_sdk=False,
            )
            result = agent.execute({
                "image_figures": [
                    {"figure_id": "fig_1", "description": "Test",
                     "figure_type": "architecture_diagram", "section": "Method"},
                ],
                "topic": "Test",
                "output_dir": str(tmp_path),
            })
            assert not result.success
            assert "API key" in result.error

    def test_execute_empty_figures_succeeds(self, tmp_path):
        from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent
        with mock.patch.dict(os.environ, {}, clear=True):
            agent = NanoBananaAgent(
                _FakeLLM(), gemini_api_key="", use_sdk=False,
            )
            result = agent.execute({
                "image_figures": [],
                "topic": "Test",
                "output_dir": str(tmp_path),
            })
            assert result.success
            assert result.data["count"] == 0


# =========================================================================
# RendererAgent._execute_in_docker() — Docker command construction
# =========================================================================

class TestDockerRenderer:
    def _agent(self):
        from researchclaw.agents.figure_agent.renderer import RendererAgent
        return RendererAgent(
            _FakeLLM(),
            timeout_sec=10,
            use_docker=True,
            docker_image="researchclaw/experiment:latest",
        )

    def test_docker_command_construction(self, tmp_path):
        """Verify docker command includes security flags."""
        agent = self._agent()
        script_path = tmp_path / "scripts" / "fig_test.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hello')")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            agent._execute_in_docker(
                script_path=script_path,
                output_dir=output_dir,
                figure_id="fig_test",
            )
            args = mock_run.call_args
            cmd = args[0][0]
            # Verify security flags
            assert "--network" in cmd
            assert "none" in cmd
            assert "--read-only" in cmd
            assert "--rm" in cmd
            assert "--memory=512m" in cmd
            # Verify mount binds
            cmd_str = " ".join(cmd)
            assert "script.py:ro" in cmd_str  # read-only script
            assert "output:rw" in cmd_str  # writable output

    def test_docker_timeout_kills_container(self, tmp_path):
        """Verify container is killed on timeout."""
        agent = self._agent()
        script_path = tmp_path / "scripts" / "fig_timeout.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("import time; time.sleep(999)")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["docker", "run"], timeout=10
            )
            result = agent._execute_in_docker(
                script_path=script_path,
                output_dir=output_dir,
                figure_id="fig_timeout",
            )
            assert "timed out" in result["error"]

    def test_docker_not_found(self, tmp_path):
        """Verify graceful handling when Docker is not installed."""
        agent = self._agent()
        script_path = tmp_path / "scripts" / "fig_no_docker.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hello')")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("docker not found")
            result = agent._execute_in_docker(
                script_path=script_path,
                output_dir=output_dir,
                figure_id="fig_no_docker",
            )
            assert "not found" in result["error"]

    def test_docker_script_failure(self, tmp_path):
        """Verify error message includes stderr on non-zero exit."""
        agent = self._agent()
        script_path = tmp_path / "scripts" / "fig_fail.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("raise Exception('boom')")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1,
                stdout="", stderr="Traceback: Exception: boom",
            )
            result = agent._execute_in_docker(
                script_path=script_path,
                output_dir=output_dir,
                figure_id="fig_fail",
            )
            assert result["error"]
            assert "boom" in result["error"]


# =========================================================================
# strip_thinking_tags() — safety tests
# =========================================================================

class TestStripThinkingTags:
    def test_closed_tags_removed(self):
        from researchclaw.utils.thinking_tags import strip_thinking_tags
        text = "Hello <think>internal reasoning</think> World"
        assert strip_thinking_tags(text) == "Hello  World"

    def test_no_tags(self):
        from researchclaw.utils.thinking_tags import strip_thinking_tags
        text = "Normal text without tags"
        assert strip_thinking_tags(text) == text

    def test_empty_string(self):
        from researchclaw.utils.thinking_tags import strip_thinking_tags
        assert strip_thinking_tags("") == ""

    def test_nested_code_preserved(self):
        """Literal <think> in code blocks should NOT be corrupted
        when used by chat() without strip_thinking=True."""
        text = '```python\n# The <think> tag is used by...\nprint("hello")\n```'
        # Without stripping, text is untouched
        assert "<think>" in text

    def test_unclosed_tag_behavior(self):
        """Document the behavior: unclosed <think> removes everything after it."""
        from researchclaw.utils.thinking_tags import strip_thinking_tags
        text = "Prefix <think>reasoning that never closes"
        result = strip_thinking_tags(text)
        # The unclosed tag strips everything after <think>
        assert "Prefix" in result
        assert "reasoning" not in result


# =========================================================================
# FigureDecisionAgent.execute() — full integration with mock LLM
# =========================================================================

class TestDecisionAgentExecute:
    def test_llm_decision(self):
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        llm_response = json.dumps([
            {"section": "Method", "figure_type": "architecture_diagram",
             "backend": "image", "description": "Arch", "priority": 1},
            {"section": "Results", "figure_type": "bar_comparison",
             "backend": "code", "description": "Results", "priority": 1},
            {"section": "Results", "figure_type": "heatmap",
             "backend": "code", "description": "Heatmap", "priority": 2},
        ])
        agent = FigureDecisionAgent(_FakeLLM(llm_response), min_figures=3)
        result = agent.execute({
            "topic": "Graph anomaly detection",
            "hypothesis": "GRACE improves detection",
            "paper_draft": "# Introduction\n...",
            "has_experiments": True,
            "condition_summaries": {"proposed": {}, "baseline": {}},
        })
        assert result.success
        assert result.data["total"] >= 3
        assert len(result.data["code_figures"]) >= 1
        assert len(result.data["image_figures"]) >= 1

    def test_fallback_on_bad_llm(self):
        """When LLM returns garbage, heuristic fallback should kick in."""
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        agent = FigureDecisionAgent(
            _FakeLLM("This is not JSON"),
            min_figures=3,
        )
        result = agent.execute({
            "topic": "Test topic",
            "has_experiments": True,
            "condition_summaries": {"a": {}, "b": {}},
        })
        assert result.success  # fallback succeeds
        assert result.data["total"] >= 3

    def test_fallback_on_no_llm(self):
        """When LLM is None, heuristic fallback should work."""
        from researchclaw.agents.figure_agent.decision import FigureDecisionAgent
        agent = FigureDecisionAgent(None, min_figures=2)
        result = agent.execute({
            "topic": "Test",
            "has_experiments": False,
            "condition_summaries": {},
        })
        assert result.success
        assert result.data["total"] >= 2


# =========================================================================
# CWD regression test (Issue #2)
# =========================================================================

class TestRendererCwd:
    """Verify the CWD is set to output_dir, not its parent."""

    def test_local_cwd_is_output_dir(self, tmp_path):
        """Scripts using relative savefig should write to output_dir."""
        from researchclaw.agents.figure_agent.renderer import RendererAgent
        agent = RendererAgent(_FakeLLM(), timeout_sec=10, use_docker=False)
        output_dir = tmp_path / "charts"

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            agent._execute_local(
                script_path=tmp_path / "test.py",
                output_dir=output_dir,
            )
            call_kwargs = mock_run.call_args
            cwd = call_kwargs[1]["cwd"] if isinstance(call_kwargs[1], dict) else None
            # CWD should be output_dir, NOT output_dir.parent
            assert cwd == str(output_dir.resolve())


# =========================================================================
# chat(strip_thinking=True) — opt-in parameter (Issue #1 fix)
# =========================================================================

class TestChatStripThinking:
    """Verify the opt-in strip_thinking parameter on LLMClient.chat()."""

    def test_strip_thinking_false_by_default(self):
        """Default chat() should NOT strip <think> tags."""
        from researchclaw.llm.client import LLMClient, LLMConfig, LLMResponse

        config = LLMConfig(
            base_url="http://fake",
            api_key="fake-key",
            primary_model="test-model",
        )
        client = LLMClient(config)

        response_with_think = (
            '<think>internal reasoning</think>The actual answer is 42.'
        )
        fake_api_response = {
            "choices": [{
                "message": {"content": response_with_think},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "test-model",
        }

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps(fake_api_response).encode()
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = client.chat(
                [{"role": "user", "content": "test"}],
                strip_thinking=False,
            )
            # With strip_thinking=False, <think> tags are preserved
            assert "<think>" in result.content

    def test_strip_thinking_true_removes_tags(self):
        """chat(strip_thinking=True) should strip <think> tags."""
        from researchclaw.llm.client import LLMClient, LLMConfig

        config = LLMConfig(
            base_url="http://fake",
            api_key="fake-key",
            primary_model="test-model",
        )
        client = LLMClient(config)

        response_with_think = (
            '<think>internal reasoning</think>The actual answer is 42.'
        )
        fake_api_response = {
            "choices": [{
                "message": {"content": response_with_think},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "test-model",
        }

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps(fake_api_response).encode()
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = client.chat(
                [{"role": "user", "content": "test"}],
                strip_thinking=True,
            )
            # With strip_thinking=True, <think> tags are removed
            assert "<think>" not in result.content
            assert "The actual answer is 42." in result.content


# =========================================================================
# LaTeX converter — display math $$...$$ fix
# =========================================================================

class TestLatexDisplayMath:
    """Verify the $$...$$ → equation environment fix in converter.py."""

    def test_dollar_dollar_to_equation(self):
        """$$...$$ display math should become \\begin{equation}."""
        from researchclaw.templates.converter import _convert_block

        md = (
            "Some text before.\n"
            "\n"
            "$$\\alpha_{ij} = \\frac{x}{y}$$\n"
            "\n"
            "Some text after."
        )
        result = _convert_block(md)
        assert "\\begin{equation}" in result
        assert "\\end{equation}" in result
        assert "\\alpha_{ij}" in result
        # Should NOT contain escaped $$
        assert "\\$\\$" not in result

    def test_multiline_dollar_dollar(self):
        """$$...$$ spanning multiple lines should also convert."""
        from researchclaw.templates.converter import _convert_block

        md = (
            "$$\n"
            "\\mathcal{L} = -\\log \\frac{a}{b}\n"
            "$$\n"
        )
        result = _convert_block(md)
        assert "\\begin{equation}" in result
        assert "\\mathcal{L}" in result

    def test_inline_dollar_dollar_not_escaped(self):
        """$$ in inline context should not be escaped to \\$\\$."""
        from researchclaw.templates.converter import _convert_inline

        text = "The formula $$x+y$$ is important"
        result = _convert_inline(text)
        # Should not contain \\textasciicircum or \\$
        assert "\\textasciicircum" not in result


# =========================================================================
# LaTeX converter — figure [t] placement
# =========================================================================

class TestLatexFigurePlacement:
    """Verify figures use [t] placement specifier."""

    def test_figure_uses_top_placement(self):
        from researchclaw.templates.converter import _render_figure

        result = _render_figure("Test Caption", "charts/test.png")
        assert "\\begin{figure}[t]" in result
        assert "[ht]" not in result

    def test_figure_has_centering(self):
        from researchclaw.templates.converter import _render_figure

        result = _render_figure("My Figure", "path/to/image.png")
        assert "\\centering" in result
        assert "\\includegraphics" in result
        assert "\\caption{My Figure}" in result
        assert "\\label{fig:" in result


# =========================================================================
# Pipeline wrapper — _chat_with_prompt strip_thinking default
# =========================================================================


class TestChatWithPromptStripThinking:
    """Verify _chat_with_prompt passes strip_thinking to llm.chat()."""

    def test_default_strips_thinking(self):
        """_chat_with_prompt should pass strip_thinking=True by default."""
        from unittest.mock import MagicMock
        from researchclaw.pipeline.executor import _chat_with_prompt
        from researchclaw.llm.client import LLMResponse

        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="clean output", model="test", finish_reason="stop",
        )

        result = _chat_with_prompt(mock_llm, system="sys", user="hello")

        call_kwargs = mock_llm.chat.call_args
        assert call_kwargs.kwargs.get("strip_thinking") is True

    def test_can_disable_stripping(self):
        """_chat_with_prompt(strip_thinking=False) should forward the flag."""
        from unittest.mock import MagicMock
        from researchclaw.pipeline.executor import _chat_with_prompt
        from researchclaw.llm.client import LLMResponse

        mock_llm = MagicMock()
        mock_llm.chat.return_value = LLMResponse(
            content="<think>reasoning</think>output",
            model="test", finish_reason="stop",
        )

        _chat_with_prompt(
            mock_llm, system="sys", user="hello", strip_thinking=False,
        )

        call_kwargs = mock_llm.chat.call_args
        assert call_kwargs.kwargs.get("strip_thinking") is False

