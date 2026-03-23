# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.pipeline.executor import _sanitize_fabricated_data
from researchclaw.pipeline.stage_impls._code_generation import _check_rl_compatibility


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
        "| Baseline | 0.73 | 0.65 | 0.78 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)

    # 0.85 and 0.82 should be kept (verified), 0.91, 0.73, 0.65, 0.78 replaced
    assert "0.85" in sanitized
    assert "0.82" in sanitized
    assert "0.91" not in sanitized
    assert "0.73" not in sanitized
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
        "| B | 0.8765 |\n"
    )
    sanitized, _ = _sanitize_fabricated_data(paper, run_dir)
    # Table pipes should still be intact
    assert sanitized.count("|") == paper.count("|")
    assert "0.12" in sanitized
    assert "0.8765" not in sanitized


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


def test_sanitize_hp_columns_preserved_in_mixed_table(run_dir: Path) -> None:
    """BUG-184: HP columns in mixed tables should not be sanitized."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
        "best_run": {"metrics": {"accuracy": 0.85}},
    })
    paper = (
        "## Results\n\n"
        "| Method | LR | Batch Size | Accuracy | F1 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Ours | 0.0007 | 48 | 0.85 | 0.91 |\n"
        "| Baseline | 0.0001 | 24 | 0.73 | 0.78 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # HP columns (LR, Batch Size) should be preserved regardless of verification
    assert "0.0007" in sanitized, "HP column 'LR' value should not be sanitized"
    assert "0.0001" in sanitized, "HP column 'LR' value should not be sanitized"
    # Result columns: 0.85 verified → kept; 0.91, 0.73, 0.78 → replaced
    assert "0.85" in sanitized
    assert "0.91" not in sanitized
    assert "0.73" not in sanitized


def test_sanitize_pure_hp_table_skipped(run_dir: Path) -> None:
    """BUG-192: Pure HP tables (header keywords) should be fully skipped."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
    })
    paper = (
        "| Hyperparameter | Value |\n"
        "| --- | --- |\n"
        "| Learning Rate | 0.0007 |\n"
        "| Batch Size | 48 |\n"
        "| Weight Decay | 0.0005 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # Entire table should be skipped — no sanitization at all
    assert "0.0007" in sanitized
    assert "0.0005" in sanitized
    assert report["tables_processed"] == 0


def test_prose_sanitization_replaces_unverified(run_dir: Path) -> None:
    """Prose numbers in Results section should be sanitized."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
        "best_run": {"metrics": {"accuracy": 0.85}},
    })
    paper = (
        "# Introduction\n"
        "Prior work achieved 92.3% accuracy on this task.\n\n"
        "# Results\n"
        "Our method achieved 85.0% accuracy, which is significantly better.\n"
        "The baseline obtained 72.4% accuracy on the same benchmark.\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # 85.0 is verified (matches 0.85 × 100), should be kept
    assert "85.0" in sanitized
    # 72.4 is unverified in Results → replaced
    assert "72.4" not in sanitized
    assert "[value removed]" in sanitized
    # 92.3 is in Introduction (not Results) → should be preserved
    assert "92.3" in sanitized
    assert report["prose_numbers_replaced"] >= 1


def test_sanitize_model_name_numbers_preserved(run_dir: Path) -> None:
    """BUG-206: Numbers in model names (ResNet-34) must not be replaced."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
        "best_run": {"metrics": {"accuracy": 0.85}},
    })
    # Table with model variant numbers in the first column (ci=1, skipped)
    paper = (
        "## Results\n\n"
        "| Method | Accuracy |\n"
        "| --- | --- |\n"
        "| ResNet-34 (baseline) | 0.85 |\n"
        "| ResNet-50 (teacher) | 0.91 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # First column is method names — must be preserved (includes "34", "50")
    assert "ResNet-34" in sanitized, "Model name 'ResNet-34' should not be sanitized"
    assert "ResNet-50" in sanitized, "Model name 'ResNet-50' should not be sanitized"


def test_sanitize_unicode_hyphen_model_names_preserved(run_dir: Path) -> None:
    """BUG-206: Unicode non-breaking hyphen in model names must not be replaced."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
        "best_run": {"metrics": {"accuracy": 0.85}},
    })
    # U+2011 non-breaking hyphen (common LLM output)
    paper = (
        "## Results\n\n"
        "| Method | Accuracy |\n"
        "| --- | --- |\n"
        "| ResNet\u201134 (baseline) | 0.85 |\n"
        "| ResNet\u201150 (teacher) | 0.91 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    assert "ResNet\u201134" in sanitized, "Model name with U+2011 hyphen should not be sanitized"
    assert "ResNet\u201150" in sanitized, "Model name with U+2011 hyphen should not be sanitized"


def test_prose_sanitization_preserves_introduction(run_dir: Path) -> None:
    """Numbers outside Results/Experiments should NOT be touched."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"val": 0.50},
    })
    paper = (
        "# Introduction\n"
        "Previous methods achieved 94.2% accuracy.\n\n"
        "# Related Work\n"
        "Smith et al. reported 88.7% on the benchmark.\n\n"
        "# Conclusion\n"
        "We demonstrated 50.0% accuracy.\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # None of these sections are Results/Experiments → all preserved
    assert "94.2" in sanitized
    assert "88.7" in sanitized
    assert report["prose_numbers_replaced"] == 0


# ---------------------------------------------------------------------------
# RL compatibility check (Improvement G)
# ---------------------------------------------------------------------------


def test_rl_compatibility_dqn_continuous_detected() -> None:
    """DQN + continuous env should produce errors."""
    code = """
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make('Pendulum-v1')
model = DQN('MlpPolicy', env)
model.learn(total_timesteps=10000)
"""
    errors = _check_rl_compatibility(code)
    assert len(errors) >= 1
    assert "DQN" in errors[0]
    assert "pendulum" in errors[0].lower()


def test_rl_compatibility_ppo_continuous_ok() -> None:
    """PPO + continuous env should be fine."""
    code = """
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('HalfCheetah-v5')
model = PPO('MlpPolicy', env)
model.learn(total_timesteps=100000)
"""
    errors = _check_rl_compatibility(code)
    assert len(errors) == 0


def test_sanitize_reads_promoted_best_data(run_dir: Path) -> None:
    """BUG-222: Sanitizer uses experiment_summary_best.json (promoted best).

    After REFINE, the pipeline promotes the best iteration's data to
    experiment_summary_best.json.  The sanitizer should validate against
    that file, not scan all refinement logs.
    """
    # Stale stage-14 data (from a regressed iteration)
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"primary_metric": {"min": 8.42, "max": 8.91, "mean": 8.6467, "count": 3}},
        "best_run": {"metrics": {"primary_metric": 8.65}},
    })
    # Promoted best data (from the winning iteration)
    (run_dir / "experiment_summary_best.json").write_text(
        json.dumps({
            "metrics_summary": {"primary_metric": {"min": 73.07, "max": 78.93, "mean": 75.56, "count": 3}},
            "best_run": {"metrics": {"primary_metric": 78.93}},
            "condition_summaries": {
                "Ours": {"metrics": {"primary_metric": 78.93}},
                "SGD": {"metrics": {"primary_metric": 73.07}},
                "AdamW": {"metrics": {"primary_metric": 68.67}},
            },
        }, indent=2), encoding="utf-8"
    )
    # Paper uses values from promoted best
    paper = (
        "## Results\n\n"
        "| Method | Accuracy |\n"
        "| --- | --- |\n"
        "| Ours | 78.93 |\n"
        "| SGD | 73.07 |\n"
        "| AdamW | 68.67 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    assert "78.93" in sanitized
    assert "73.07" in sanitized
    assert "68.67" in sanitized
    assert report["numbers_kept"] == 3
    assert report["numbers_replaced"] == 0


def test_sanitize_rejects_regressed_refine_data(run_dir: Path) -> None:
    """BUG-222: Regressed REFINE iteration data must NOT pass sanitizer.

    Reproduces the Run 75 fabrication bypass: v1 had 74.52%, v3 regressed
    to 69.30%.  Paper cited v3 numbers.  The sanitizer should reject them.
    """
    # v1 (best) promoted to experiment_summary_best.json
    (run_dir / "experiment_summary_best.json").write_text(
        json.dumps({
            "best_run": {"metrics": {"FeatureKD/0/metric": 0.7452}},
            "condition_summaries": {
                "FeatureKD": {"metrics": {"metric": 0.7452}},
                "Teacher": {"metrics": {"metric": 0.7431}},
            },
            "metrics_summary": {"metric": {"mean": 0.7442, "min": 0.7431, "max": 0.7452}},
        }, indent=2), encoding="utf-8"
    )
    # v3 (regressed) in stage-14 (stale)
    _write_experiment_summary(run_dir, {
        "best_run": {"metrics": {"FeatureKD/0/metric": 0.6930}},
        "condition_summaries": {
            "FeatureKD": {"metrics": {"metric": 0.6930}},
            "Teacher": {"metrics": {"metric": 0.7292}},
        },
        "metrics_summary": {"metric": {"mean": 0.7111, "min": 0.6930, "max": 0.7292}},
    })
    # v3 sandbox data in refinement_log
    stage13 = run_dir / "stage-13_v2"
    stage13.mkdir(parents=True, exist_ok=True)
    (stage13 / "refinement_log.json").write_text(json.dumps({
        "iterations": [{"sandbox": {"metrics": {"primary_metric": 0.6930}}}]
    }), encoding="utf-8")
    # Paper fabricates v3 numbers
    paper = (
        "## Results\n\n"
        "| Method | Accuracy |\n"
        "| --- | --- |\n"
        "| FeatureKD | 69.30 |\n"
        "| Teacher | 72.92 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # 69.30 should be REPLACED — it's from regressed v3, not promoted v1
    assert "69.30" not in sanitized
    assert report["numbers_replaced"] >= 1
    # But 74.52 or 74.31 (v1 best) would pass if cited
    paper_v1 = (
        "## Results\n\n"
        "| Method | Accuracy |\n"
        "| --- | --- |\n"
        "| FeatureKD | 74.52 |\n"
        "| Teacher | 74.31 |\n"
    )
    sanitized_v1, report_v1 = _sanitize_fabricated_data(paper_v1, run_dir)
    assert "74.52" in sanitized_v1
    assert "74.31" in sanitized_v1
    assert report_v1["numbers_replaced"] == 0


def test_sanitize_condition_names_with_decimals_preserved(run_dir: Path) -> None:
    """BUG-210: Condition names with decimal params (ema_decay_0.9) must not be damaged."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 73.07},
        "best_run": {"metrics": {"accuracy": 73.07}},
    })
    paper = (
        "## Results\n\n"
        "| Condition | Accuracy |\n"
        "| --- | --- |\n"
        "| ema_decay_0.9 | 73.07 |\n"
        "| ema_decay_0.99 | 69.33 |\n"
        "| swa_start_0.75 | 68.67 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # First column (condition names) must be completely preserved
    assert "ema_decay_0.9 " in sanitized, "Condition name 'ema_decay_0.9' damaged"
    assert "ema_decay_0.99" in sanitized, "Condition name 'ema_decay_0.99' damaged"
    assert "swa_start_0.75" in sanitized, "Condition name 'swa_start_0.75' damaged"
    # 73.07 is verified → kept
    assert "73.07" in sanitized


def test_rl_compatibility_dqn_discrete_ok() -> None:
    """DQN + discrete env (CartPole) should be fine."""
    code = """
import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env)
"""
    errors = _check_rl_compatibility(code)
    assert len(errors) == 0


# ---------------------------------------------------------------------------
# BUG-211: LaTeX tabular sanitization
# ---------------------------------------------------------------------------


def test_sanitize_latex_tabular_replaces_unverified(run_dir: Path) -> None:
    """BUG-211: Numbers inside \\begin{tabular} must be sanitized."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.4816},
        "best_run": {"metrics": {"accuracy": 0.4816}},
    })
    paper = (
        "## Results\n\n"
        "```latex\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Test accuracy for all configurations.}\n"
        "\\begin{tabular}{l c}\n"
        "\\toprule\n"
        "Method & Accuracy \\\\\n"
        "\\midrule\n"
        "baseline\\_resnet18 & \\textbf{0.4816} \\\\\n"
        "baseline\\_resnet50 & 0.4451 \\\\\n"
        "dropout\\_standard & 0.3243 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "```\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # 0.4816 is verified → kept
    assert "0.4816" in sanitized
    # 0.4451 and 0.3243 are unverified → replaced with ---
    assert "0.4451" not in sanitized
    assert "0.3243" not in sanitized
    assert "---" in sanitized
    assert report["tables_processed"] >= 1
    assert report["numbers_replaced"] >= 2


def test_sanitize_latex_tabular_hp_table_skipped(run_dir: Path) -> None:
    """BUG-211: LaTeX HP tables should be skipped just like markdown ones."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
    })
    paper = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Training hyperparameters.}\n"
        "\\begin{tabular}{l c}\n"
        "\\toprule\n"
        "Hyperparameter & Value \\\\\n"
        "\\midrule\n"
        "Learning Rate & 0.001 \\\\\n"
        "Batch Size & 128 \\\\\n"
        "Weight Decay & 0.0005 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # HP table — all values preserved, table NOT processed
    assert "0.001" in sanitized
    assert "0.0005" in sanitized


def test_sanitize_latex_tabular_with_pm(run_dir: Path) -> None:
    """BUG-211: Numbers with ± in LaTeX cells must be individually checked."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 48.16, "accuracy_std": 0.35},
        "best_run": {"metrics": {"accuracy": 48.16}},
        "condition_summaries": {
            "method_a": {"primary_metric_mean": 48.16, "primary_metric_std": 0.35},
        },
    })
    paper = (
        "\\begin{tabular}{l c}\n"
        "\\toprule\n"
        "Method & Accuracy (mean $\\pm$ std) \\\\\n"
        "\\midrule\n"
        "method\\_a & 48.16 $\\pm$ 0.35 \\\\\n"
        "method\\_b & 32.43 $\\pm$ 0.45 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # 48.16 and 0.35 are verified → kept
    assert "48.16" in sanitized
    assert "0.35" in sanitized
    # 32.43 and 0.45 are unverified → replaced
    assert "32.43" not in sanitized
    assert "0.45" not in sanitized


def test_sanitize_latex_tabular_preserves_first_column(run_dir: Path) -> None:
    """BUG-211: First column (method names) must be preserved."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": 0.85},
        "best_run": {"metrics": {"accuracy": 0.85}},
    })
    paper = (
        "\\begin{tabular}{l r r r r}\n"
        "\\toprule\n"
        "Method & Seed 0 & Seed 1 & Seed 2 & Mean \\\\\n"
        "\\midrule\n"
        "resnet\\_18 & 0.4861 & 0.4809 & 0.4777 & 0.4816 \\\\\n"
        "resnet\\_50 & 0.4455 & 0.4459 & 0.4438 & 0.4451 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # Method names in first column must be preserved
    assert "resnet\\_18" in sanitized
    assert "resnet\\_50" in sanitized


# ---------------------------------------------------------------------------
# BUG-224: Statistical analysis tables should NOT be sanitized
# ---------------------------------------------------------------------------

def test_sanitize_skips_statistical_analysis_table(run_dir: Path) -> None:
    """BUG-224: Tables with t-statistics, p-values, and effect sizes are
    derived from experiment data and should not be sanitized."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": {"mean": 64.26}},
        "condition_summaries": {"ce": {"metrics": {"accuracy": 64.26}}},
    })
    paper = (
        "## Results\n\n"
        "| Method | Accuracy |\n"
        "|--------|----------|\n"
        "| CE | 64.26 |\n"
        "| SCE | 56.93 |\n\n"
        "## Statistical Analysis\n\n"
        "| Comparison | t-statistic | p-value |\n"
        "|-----------|------------|--------|\n"
        "| CE vs SCE | 7.3267 | 0.0123 |\n"
        "| CE vs GCE | 1.7100 | 0.0569 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # Results table: 64.26 is verified, 56.93 is NOT → gets replaced
    assert "56.93" not in sanitized or "---" in sanitized
    # Statistical table: 7.3267 and 0.0123 are derived → MUST be preserved
    assert "7.3267" in sanitized, "BUG-224: t-statistic was sanitized"
    assert "0.0123" in sanitized, "BUG-224: p-value was sanitized"
    assert "1.7100" in sanitized, "BUG-224: t-statistic was sanitized"
    assert "0.0569" in sanitized, "BUG-224: p-value was sanitized"


def test_sanitize_preserves_common_hp_values(run_dir: Path) -> None:
    """BUG-224: Common HP values like 0.7 should be in the always-allowed set."""
    _write_experiment_summary(run_dir, {
        "metrics_summary": {"accuracy": {"mean": 64.26}},
        "condition_summaries": {"ce": {"metrics": {"accuracy": 64.26}}},
    })
    paper = (
        "| Method | q | Accuracy |\n"
        "|--------|---|----------|\n"
        "| GCE | 0.7 | 64.26 |\n"
        "| GCE-05 | 0.5 | 66.77 |\n"
    )
    sanitized, report = _sanitize_fabricated_data(paper, run_dir)
    # 0.7 should be preserved (always-allowed HP value)
    assert "0.7" in sanitized, "BUG-224: q=0.7 was incorrectly sanitized"
    # 0.5 should also be preserved
    assert "0.5" in sanitized
