from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from researchclaw.pipeline.stage_impls._paper_writing import (
    _validate_stage17_manuscript_structure,
    _write_paper_sections,
)
from researchclaw.pipeline.stage_impls._review_publish import _execute_peer_review
from researchclaw.pipeline.stages import StageStatus


class _PromptManagerStub:
    def block(self, _name: str, **_kwargs: object) -> str:
        return ""

    def for_stage(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            system="system",
            user="review the paper",
            json_mode=False,
            max_tokens=8192,
        )


class _SequentialLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], **_kwargs: object) -> SimpleNamespace:
        self.calls.append(messages)
        return SimpleNamespace(content=self.responses.pop(0))


def _write_draft(run_dir: Path) -> None:
    stage_dir = run_dir / "stage-17"
    stage_dir.mkdir(parents=True)
    (stage_dir / "paper_draft.md").write_text(
        "## Title\n\nExample\n\n## Method\n\nBody.\n",
        encoding="utf-8",
    )


def _config() -> Any:
    return SimpleNamespace(research=SimpleNamespace(topic="test topic"))


def test_stage17_structure_report_rejects_checked_in_duplicate_fixture(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "stage17_duplicate_headings.md"
    report = _validate_stage17_manuscript_structure(
        fixture.read_text(encoding="utf-8"),
        stage_dir=tmp_path,
    )

    assert report["valid"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "duplicate_heading_path"
    }
    assert json.loads(
        (tmp_path / "paper_structure_report.json").read_text(encoding="utf-8")
    ) == report


def test_stage17_structure_report_accepts_unique_heading_paths(tmp_path: Path) -> None:
    report = _validate_stage17_manuscript_structure(
        "## Title\n\nExample\n\n## Method\n\nBody.\n",
        stage_dir=tmp_path,
    )

    assert report["valid"] is True
    assert report["section_count"] == 2
    assert report["issues"] == []


def test_all_three_stage17_calls_receive_the_section_output_contract() -> None:
    llm = _SequentialLLM(
        [
            "## Title\n\nExample\n\n## Abstract\n\nAbstract.",
            "## Method\n\nMethod.\n\n## Experiments\n\nExperiments.",
            "## Results\n\nResults.\n\n## Conclusion\n\nConclusion.",
        ]
    )

    _write_paper_sections(
        llm=cast(Any, llm),
        pm=cast(Any, _PromptManagerStub()),
        preamble="",
        topic_constraint="",
        exp_metrics_instruction="",
        citation_instruction="",
        outline="",
    )

    assert len(llm.calls) == 3
    for call in llm.calls:
        prompt = "\n".join(message["content"] for message in call)
        assert "SECTION OUTPUT CONTRACT" in prompt
        assert "Output only the sections requested in this call" in prompt


def test_stage18_prompt_contract_is_last_and_valid_output_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-18"
    stage_dir.mkdir(parents=True)
    _write_draft(run_dir)
    reviews = """## Reviewer A

### Strengths
Clear scope.

### Weaknesses
Limited evidence.

### Actionable Revisions
1. Add uncertainty estimates.
"""
    llm = _SequentialLLM([reviews])

    result = _execute_peer_review(
        stage_dir,
        run_dir,
        cast(Any, _config()),
        cast(Any, None),
        llm=cast(Any, llm),
        prompts=cast(Any, _PromptManagerStub()),
    )

    assert result.status == StageStatus.DONE
    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert prompt.rstrip().endswith("Do not emit any other markdown heading at any level.")
    report = json.loads(
        (stage_dir / "review_structure_report.json").read_text(encoding="utf-8")
    )
    assert report["valid"] is True
    assert report["comment_count"] == 1


def test_stage18_unknown_subsection_fixture_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-18"
    stage_dir.mkdir(parents=True)
    _write_draft(run_dir)
    fixture = Path(__file__).parent / "fixtures" / "stage18_reviews_unknown_subsection.md"
    llm = _SequentialLLM([fixture.read_text(encoding="utf-8")])

    result = _execute_peer_review(
        stage_dir,
        run_dir,
        cast(Any, _config()),
        cast(Any, None),
        llm=cast(Any, llm),
        prompts=cast(Any, _PromptManagerStub()),
    )

    assert result.status == StageStatus.FAILED
    report = json.loads(
        (stage_dir / "review_structure_report.json").read_text(encoding="utf-8")
    )
    assert report["valid"] is False
    assert report["source_reviews_sha256"] == hashlib.sha256(
        fixture.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert "unknown_review_subsection" in {
        issue["code"] for issue in report["issues"]
    }


def test_stage18_no_llm_fallback_obeys_the_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-18"
    stage_dir.mkdir(parents=True)
    _write_draft(run_dir)

    result = _execute_peer_review(
        stage_dir,
        run_dir,
        cast(Any, _config()),
        cast(Any, None),
    )

    assert result.status == StageStatus.DONE
    report = json.loads(
        (stage_dir / "review_structure_report.json").read_text(encoding="utf-8")
    )
    assert report["valid"] is True
    assert report["comment_count"] == 4
