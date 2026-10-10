"""Verify figure code generation uses the optional execution model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest

from researchclaw.agents.base import AgentStepResult
from researchclaw.agents.figure_agent.orchestrator import (
    FigureAgentConfig,
    FigureOrchestrator,
)


@dataclass
class _Response:
    content: str
    total_tokens: int = 3


class _QueuedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return _Response(next(self.responses))


@pytest.mark.parametrize("separate_execution_model", [False, True])
def test_only_codegen_receives_execution_model(separate_execution_model):
    research_llm = _QueuedLLM([])
    execution_llm = _QueuedLLM([]) if separate_execution_model else None
    orchestrator = FigureOrchestrator(
        research_llm,
        FigureAgentConfig(use_docker=False, gemini_api_key="test-key"),
        execution_llm=execution_llm,
    )

    assert orchestrator._llm is research_llm
    assert orchestrator._codegen._llm is (execution_llm or research_llm)
    for agent in (
        orchestrator._decision,
        orchestrator._planner,
        orchestrator._renderer,
        orchestrator._critic,
        orchestrator._integrator,
        orchestrator._nano_banana,
    ):
        assert agent is not None
        assert agent._llm is research_llm
    assert research_llm.calls == []
    if execution_llm is not None:
        assert execution_llm.calls == []


@pytest.mark.parametrize("separate_execution_model", [False, True])
def test_codegen_and_review_retry_route_to_expected_models(
    tmp_path, separate_execution_model,
):
    decision = json.dumps([{
        "figure_type": "radar_chart",
        "backend": "code",
        "description": "Compare metrics across conditions.",
    }])
    figure = {
        "figure_id": "fig_metrics",
        # No built-in template: initial generation must call the model too.
        "chart_type": "radar_chart",
        "title": "Metrics",
        "caption": "Comparison of experimental metrics.",
        "section": "results",
    }
    planning = json.dumps({"figures": [figure]})
    failed_review = json.dumps({
        "quality_score": 5,
        "issues": [{
            "severity": "critical",
            "message": "Move legend outside the plotting area.",
        }],
    })
    passed_review = json.dumps({"quality_score": 9, "issues": []})
    initial_script = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.set_xlabel('Method')\n"
        "ax.set_ylabel('Metric')\n"
        "ax.set_title('Metrics')\n"
        "fig.savefig('fig_metrics.png')\n"
        "plt.close(fig)\n"
        "# initial generation\n"
    )
    revised_script = initial_script.replace(
        "# initial generation", "# revision: legend moved",
    )

    if separate_execution_model:
        research_llm = _QueuedLLM([
            decision, planning, failed_review, passed_review,
        ])
        execution_llm = _QueuedLLM([initial_script, revised_script])
        orchestrator = FigureOrchestrator(
            research_llm,
            FigureAgentConfig(
                min_figures=1, max_figures=2, max_iterations=2,
                use_docker=False, nano_banana_enabled=False,
            ),
            stage_dir=tmp_path,
            execution_llm=execution_llm,
        )
    else:
        research_llm = _QueuedLLM([
            decision, planning, initial_script, failed_review,
            revised_script, passed_review,
        ])
        execution_llm = research_llm
        # Preserve the existing constructor without the optional keyword.
        orchestrator = FigureOrchestrator(
            research_llm,
            FigureAgentConfig(
                min_figures=1, max_figures=2, max_iterations=2,
                use_docker=False, nano_banana_enabled=False,
            ),
            stage_dir=tmp_path,
        )

    def render(context):
        return AgentStepResult(True, data={"rendered": [{
            "figure_id": script["figure_id"],
            "success": True,
            "title": script["title"],
            "caption": script["caption"],
            "section": script["section"],
        } for script in context["scripts"]]})

    orchestrator._renderer.execute = Mock(side_effect=render)
    plan = orchestrator.orchestrate({
        "topic": "Metric comparison",
        "experiment_results": {"metric": 0.8},
        "output_dir": tmp_path / "charts",
    })

    research_calls = research_llm.calls
    programming_calls = [
        call for call in execution_llm.calls
        if "visualization programmer" in call["system"]
    ]
    assert len(research_calls) == (4 if separate_execution_model else 6)
    assert "visualization advisor" in research_calls[1]["system"]
    assert sum("expert reviewer" in c["system"] for c in research_calls) == 2
    assert len(programming_calls) == 2
    if separate_execution_model:
        assert len(execution_llm.calls) == 2
        assert all("visualization programmer" not in c["system"]
                   for c in research_calls)
    assert "PREVIOUS ATTEMPT FAILED REVIEW" not in programming_calls[0]["messages"][0]["content"]
    assert "PREVIOUS ATTEMPT FAILED REVIEW" in programming_calls[1]["messages"][0]["content"]
    assert "Move legend outside" in programming_calls[1]["messages"][0]["content"]
    assert orchestrator._renderer.execute.call_count == 2
    rendered_calls = orchestrator._renderer.execute.call_args_list
    assert "# initial generation" in rendered_calls[0].args[0]["scripts"][0]["script"]
    assert "# revision: legend moved" in rendered_calls[1].args[0]["scripts"][0]["script"]
    assert plan.figure_count == 1
    assert plan.passed_count == 1
    assert plan.manifest[0]["figure_id"] == "fig_metrics"
    assert plan.total_llm_calls == 6
    assert plan.total_tokens == 18
    assert (tmp_path / "charts" / "figure_manifest.json").is_file()
