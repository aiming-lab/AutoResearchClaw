# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.executor import _sanitize_fabricated_data


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def _write_experiment_summary(run_dir: Path, data: dict) -> None:
    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True, exist_ok=True)
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def test_sanitize_replaces_unverified_numbers(run_dir: Path) -> None:
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85, "f1": 0.82},
        "best_run": {"metrics": {"accuracy": 0.85}},
    })
    paper = (
        "## Results\n\n"
        "| Method | Accuracy | F1 | Precision |\n"
        "| --- | --- | --- | --- |\n"
        "| Ours | 0.85 | 0.82 | 0.91 |\n"
        "| Baseline | 0.70 | 0.65 | 0.78 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)

    # 0.85 and 0.82 should be kept (verified), 0.91, 0.70, 0.65, 0.78 replaced
    assert "0.85" in sanitized
    assert "0.82" in sanitized
    assert "0.91" not in sanitized
    assert "0.70" not in sanitized
    assert "---" in sanitized
    assert report["sanitized"] is True
    assert report["numbers_replaced"] == 4
    assert report["numbers_kept"] == 2


def test_sanitize_preserves_table_structure(run_dir: Path) -> None:
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"loss": 0.12},
    })
    paper = (
        "| Model | Loss |\n"
        "| --- | --- |\n"
        "| A | 0.12 |\n"
        "| B | 0.99 |\n"
    )
    sanitized, _ = _sanitize_fabricated_data(paper, run_dir)
    # Table pipes should still be intact
    assert sanitized.count("|") == paper.count("|")
    assert "0.12" in sanitized
    assert "0.99" not in sanitized


def test_sanitize_no_experiment_summary(run_dir: Path) -> None:
    paper = "| A | 0.5 |\n| --- | --- |\n| B | 0.6 |\n"
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    assert report["sanitized"] is False
    assert sanitized == paper  # unchanged


def test_sanitize_tolerance_within_1_percent(run_dir: Path) -> None:
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 100.0},
    })
    paper = (
        "| Method | Acc |\n"
        "| --- | --- |\n"
        "| Ours | 100.5 |\n"  # within 1% of 100.0
        "| Other | 110.0 |\n"  # outside 1%
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    assert "100.5" in sanitized  # kept (within tolerance)
    assert "110.0" not in sanitized  # replaced


def test_sanitize_header_row_preserved(run_dir: Path) -> None:
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"val": 5.0},
    })
    paper = (
        "| Col1 | Col2 |\n"
        "| --- | --- |\n"
        "| data | 99.9 |\n"
    )
    sanitized, _ = _sanitize_fabricated_data(paper, run_dir)
    # Header row should be untouched
    assert "| Col1 | Col2 |" in sanitized
