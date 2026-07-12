from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from researchclaw.pipeline.stage_impls import _review_publish
from researchclaw.pipeline.stage_impls._paper_writing import (
    PaperSectionContractError,
    _validate_paper_part_sections,
    _validate_stage17_manuscript_structure,
    _write_paper_sections,
)
from researchclaw.pipeline.stage_impls._review_publish import (
    _citation_count_policy_violations,
    _execute_peer_review,
)
from researchclaw.pipeline.stages import StageStatus


@pytest.fixture(autouse=True)
def _stub_effective_citation_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _review_publish,
        "load_effective_citation_policy",
        lambda *_args: {
            "effective_min_unique_sources": 1,
            "effective_target_unique_sources": 15,
        },
    )
    monkeypatch.setattr(
        _review_publish, "validate_experiment_fact_closure_report", lambda *_args: {}
    )
    monkeypatch.setattr(
        _review_publish, "validate_citation_closure_report", lambda *_args: {}
    )


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
            "## Title\n\nExample\n\n## Abstract\n\nAbstract.\n\n"
            "## Introduction\n\nIntroduction.\n\n## Related Work\n\nPrior work.",
            "## Method\n\nMethod.\n\n## Experiments\n\nExperiments.",
            "## Results\n\nResults.\n\n## Discussion\n\nDiscussion.\n\n"
            "## Limitations\n\nLimitations.\n\n## Conclusion\n\nConclusion.",
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


def test_stage17_part_contract_rejects_extra_major_section() -> None:
    text = (
        "## A Paper Title\n\nTitle body.\n\n## Abstract\n\nAbstract.\n\n"
        "## Introduction\n\nIntro.\n\n## Related Work\n\nPrior.\n\n"
        "## Method\n\nNot owned by part 1.\n"
    )
    violations = _validate_paper_part_sections(
        text,
        expected_major_sections=("Abstract", "Introduction", "Related Work"),
        title_slot=True,
    )
    assert violations == ("section_part_major_sequence_mismatch",)


def test_stage17_part_contract_rejects_reserved_title_slot() -> None:
    text = (
        "## Abstract\n\nNot a title.\n\n## Abstract\n\nA.\n\n"
        "## Introduction\n\nI.\n\n## Related Work\n\nR.\n"
    )
    assert "section_part_title_invalid" in _validate_paper_part_sections(
        text,
        expected_major_sections=("Abstract", "Introduction", "Related Work"),
        title_slot=True,
    )


def test_stage17_part_contract_rejects_out_of_order_sections() -> None:
    text = "## Experiments\n\nE.\n\n## Method\n\nM.\n"
    assert _validate_paper_part_sections(
        text,
        expected_major_sections=("Method", "Experiments"),
        title_slot=False,
    ) == ("section_part_major_sequence_mismatch",)


def test_stage17_part_contract_uses_commonmark_for_fenced_heading() -> None:
    text = (
        "## Method\n\n```markdown\n## Discussion\n```\n\n"
        "## Experiments\n\nSetup.\n"
    )
    assert _validate_paper_part_sections(
        text,
        expected_major_sections=("Method", "Experiments"),
        title_slot=False,
    ) == ()


def test_stage17_hep_conclusions_contract_is_distinct() -> None:
    text = "## Results\n\nR.\n\n## Discussion\n\nD.\n\n## Conclusions\n\nC.\n"
    assert _validate_paper_part_sections(
        text,
        expected_major_sections=("Results", "Discussion", "Conclusions"),
        title_slot=False,
    ) == ()
    assert "section_part_major_sequence_mismatch" in _validate_paper_part_sections(
        text,
        expected_major_sections=("Results", "Discussion", "Limitations", "Conclusion"),
        title_slot=False,
    )


def test_stage17_part_contract_regenerates_once_and_records_attempts(
    tmp_path: Path,
) -> None:
    llm = _SequentialLLM(
        [
            "## Method\n\nWrong part.",
            "## Title\n\nPaper.\n\n## Abstract\n\nA.\n\n## Introduction\n\nI.\n\n## Related Work\n\nR.",
            "## Method\n\nM.\n\n## Experiments\n\nE.",
            "## Results\n\nR.\n\n## Discussion\n\nD.\n\n## Limitations\n\nL.\n\n## Conclusion\n\nC.",
        ]
    )
    draft = _write_paper_sections(
        llm=cast(Any, llm),
        pm=cast(Any, _PromptManagerStub()),
        preamble="",
        topic_constraint="",
        exp_metrics_instruction="",
        citation_instruction="",
        outline="",
        stage_dir=tmp_path,
    )
    assert "## Conclusion" in draft
    assert len(llm.calls) == 4
    report = json.loads(
        (tmp_path / "section_generation_report.json").read_text(encoding="utf-8")
    )
    assert len(report["parts"][0]["attempts"]) == 2
    assert report["parts"][0]["attempts"][0]["valid"] is False
    assert report["parts"][0]["attempts"][1]["valid"] is True


def test_stage17_part_contract_fails_after_one_regeneration(
    tmp_path: Path,
) -> None:
    llm = _SequentialLLM(
        ["## Method\n\nWrong part.", "## Experiments\n\nStill wrong."]
    )
    with pytest.raises(PaperSectionContractError, match="part-1"):
        _write_paper_sections(
            llm=cast(Any, llm),
            pm=cast(Any, _PromptManagerStub()),
            preamble="",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="",
            stage_dir=tmp_path,
        )
    assert len(llm.calls) == 2
    report = json.loads(
        (tmp_path / "section_generation_report.json").read_text(encoding="utf-8")
    )
    assert len(report["parts"]) == 1
    assert report["parts"][0]["attempts"][-1]["valid"] is False


def test_stage17_initial_transport_failure_stops_before_later_parts(
    tmp_path: Path,
) -> None:
    class FailingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, _messages: object, **_kwargs: object) -> SimpleNamespace:
            self.calls += 1
            raise RuntimeError("transport unavailable")

    llm = FailingLLM()
    with pytest.raises(PaperSectionContractError, match="part-1"):
        _write_paper_sections(
            llm=cast(Any, llm),
            pm=cast(Any, _PromptManagerStub()),
            preamble="",
            topic_constraint="",
            exp_metrics_instruction="",
            citation_instruction="",
            outline="",
            stage_dir=tmp_path,
        )
    assert llm.calls == 2
    report = json.loads(
        (tmp_path / "section_generation_report.json").read_text(encoding="utf-8")
    )
    assert len(report["parts"]) == 1
    assert report["parts"][0]["part"] == "part-1"
    assert report["parts"][0]["attempts"][0]["violations"] == [
        "section_part_transport_error:RuntimeError"
    ]


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


def test_stage18_repairs_citation_requirement_above_effective_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-18"
    stage_dir.mkdir(parents=True)
    _write_draft(run_dir)
    excessive = """## Reviewer A

### Strengths
Clear scope.

### Weaknesses
Limited evidence.

### Actionable Revisions
1. Ensure at least 20 relevant references.
"""
    repaired = excessive.replace("20", "15")
    llm = _SequentialLLM([excessive, repaired])
    result = _execute_peer_review(
        stage_dir,
        run_dir,
        cast(Any, _config()),
        cast(Any, None),
        llm=cast(Any, llm),
        prompts=cast(Any, _PromptManagerStub()),
    )
    assert result.status == StageStatus.DONE
    assert len(llm.calls) == 2
    assert "effective target of 15" in llm.calls[1][0]["content"]


def test_stage18_fails_when_citation_requirement_remains_above_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-18"
    stage_dir.mkdir(parents=True)
    _write_draft(run_dir)
    excessive = """## Reviewer A

### Strengths
Clear scope.

### Weaknesses
Limited evidence.

### Actionable Revisions
1. Include at least twenty five unique citations.
"""
    llm = _SequentialLLM([excessive, excessive])
    result = _execute_peer_review(
        stage_dir,
        run_dir,
        cast(Any, _config()),
        cast(Any, None),
        llm=cast(Any, llm),
        prompts=cast(Any, _PromptManagerStub()),
    )
    assert result.status == StageStatus.FAILED
    report = json.loads((stage_dir / "review_structure_report.json").read_text())
    assert report["issues"][0]["code"] == "citation_count_policy_exceeded"


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("At least fifteen unique citations are required.", 14),
        ("Ensure at least twenty five relevant references.", 20),
        ("Increase references to 30.", 15),
        ("Raise the citation count above twenty-five.", 20),
        ("30 references are required.", 15),
        ("Twenty five or more sources should be included.", 20),
    ],
)
def test_citation_count_detector_does_not_swallow_qualifiers(
    text: str, target: int
) -> None:
    ledger = SimpleNamespace(
        comments=(
            SimpleNamespace(
                category="actionable_revision",
                exact_text=text,
                comment_id="comment-001",
            ),
        )
    )
    assert _citation_count_policy_violations(ledger, target) == ["comment-001"]
