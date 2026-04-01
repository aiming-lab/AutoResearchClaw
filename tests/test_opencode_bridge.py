"""Tests for OpenCode Beast Mode bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import OpenCodeConfig, _parse_opencode_config
from researchclaw.pipeline.opencode_bridge import (
    ComplexityScore,
    OpenCodeBridge,
    OpenCodeResult,
    count_historical_failures,
    score_complexity,
)


# ============================================================
# TestComplexityScorer
# ============================================================


class TestComplexityScorer:
    """Tests for complexity scoring logic."""

    def test_low_complexity_simple_classification(self):
        plan = (
            "Train a ResNet-18 on CIFAR-10 with SGD optimizer.\n"
            "Report test accuracy as the primary metric.\n"
            "condition_0: baseline (lr=0.1)\n"
            "condition_1: ablation (lr=0.01)\n"
        )
        result = score_complexity(plan, topic="Image classification on CIFAR-10")
        assert result.score < 0.4
        assert result.recommendation == "code_agent"

    def test_high_complexity_multimodal_gan(self):
        plan = (
            "Implement a vision-language GAN with the following components:\n"
            "- Encoder: ViT-based image encoder\n"
            "- Decoder: Transformer text decoder\n"
            "- Generator: produces synthetic image-text pairs\n"
            "- Discriminator: classifies real vs fake\n"
            "- Critic: provides auxiliary reward signal\n"
            "Multiple files needed: model.py, trainer.py, dataset.py\n"
            "condition_0: baseline\n"
            "condition_1: ablation without critic\n"
            "condition_2: ablation without encoder pretraining\n"
            "condition_3: ablation with reduced generator\n"
            "Custom loss function and custom layer for cross-modal attention.\n"
        )
        result = score_complexity(
            plan, topic="Multi-modal GAN for vision-language synthesis"
        )
        assert result.score > 0.6
        assert result.recommendation == "beast_mode"

    def test_historical_failures_boost_score(self):
        plan = (
            "Train a simple model with encoder and decoder.\n"
            "condition_0: baseline\n"
        )
        score_without = score_complexity(plan, topic="test", historical_failures=0)
        score_with = score_complexity(plan, topic="test", historical_failures=3)
        assert score_with.score > score_without.score
        assert score_with.signals["historical_failure"] > 0

    def test_empty_plan_returns_zero(self):
        result = score_complexity("", topic="")
        assert result.score == 0.0
        assert result.recommendation == "legacy"
        assert result.reason == "Empty plan"

    def test_threshold_boundary(self):
        """A plan scoring exactly at threshold should recommend beast_mode."""
        plan = (
            "Multi-modal diffusion model with encoder, decoder, discriminator.\n"
            "Custom loss, custom layer, wrapper pattern.\n"
            "model.py, trainer.py needed.\n"
        )
        # Use a low threshold to ensure it triggers
        result = score_complexity(plan, topic="Diffusion model", threshold=0.2)
        assert result.recommendation == "beast_mode"

        # Use a very high threshold to ensure it doesn't trigger
        result2 = score_complexity(plan, topic="Diffusion model", threshold=0.99)
        assert result2.recommendation == "code_agent"

    def test_signals_all_present(self):
        result = score_complexity("some plan", topic="some topic")
        expected_keys = {
            "component_count",
            "file_count_hint",
            "domain_complexity",
            "condition_count",
            "historical_failure",
            "dependency_depth",
        }
        assert set(result.signals.keys()) == expected_keys

    def test_score_clamped_to_unit_interval(self):
        """Score should never exceed 1.0 even with extreme inputs."""
        plan = " ".join(
            ["encoder decoder discriminator generator critic actor teacher student"] * 10
            + ["model.py trainer.py dataset.py multiple files modular"] * 10
            + ["multi-modal distributed GAN diffusion NeRF MoE meta-learning"] * 10
            + ["condition_1 condition_2 condition_3 ablation_4 variant_5 baseline"] * 10
            + ["custom layer custom loss wrapper registry hook callback"] * 10
        )
        result = score_complexity(plan, topic="everything", historical_failures=100)
        assert 0.0 <= result.score <= 1.0

    def test_domain_complexity_keywords(self):
        plan = "Implement a physics-informed neural network (PINN) with neural ODE solver."
        result = score_complexity(plan, topic="PINN for fluid dynamics")
        assert result.signals["domain_complexity"] > 0


# ============================================================
# TestOpenCodeBridge
# ============================================================


class TestOpenCodeBridge:
    """Tests for the OpenCode bridge class."""

    def test_check_available_returns_false_when_not_installed(self):
        with patch(
            "researchclaw.pipeline.opencode_bridge.shutil.which",
            return_value=None,
        ):
            assert OpenCodeBridge.check_available() is False

    def test_check_available_returns_false_on_timeout(self):
        with patch(
            "researchclaw.pipeline.opencode_bridge.shutil.which",
            return_value=r"C:\Users\tester\AppData\Roaming\npm\opencode.cmd",
        ), patch(
            "researchclaw.pipeline.opencode_bridge.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="opencode", timeout=15),
        ):
            assert OpenCodeBridge.check_available() is False

    def test_check_available_returns_true(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch(
            "researchclaw.pipeline.opencode_bridge.shutil.which",
            return_value=r"C:\Users\tester\AppData\Roaming\npm\opencode.cmd",
        ), patch(
            "researchclaw.pipeline.opencode_bridge.subprocess.run",
            return_value=mock_result,
        ) as run_mock:
            assert OpenCodeBridge.check_available() is True
        assert run_mock.call_args.args[0][0].endswith("opencode.cmd")

    def test_workspace_creates_correct_files(self, tmp_path):
        bridge = OpenCodeBridge(
            model="gpt-5.2",
            llm_base_url="https://example.com",
            api_key_env="TEST_KEY",
        )
        ws = bridge._prepare_workspace(
            stage_dir=tmp_path,
            topic="Test topic",
            exp_plan="plan: test",
            metric="accuracy",
            pkg_hint="torch available",
            extra_guidance="Be careful",
            time_budget_sec=300,
        )
        assert (ws / "EXPERIMENT_PLAN.yaml").exists()
        assert (ws / "GUIDANCE.md").exists()
        assert (ws / "opencode.json").exists()

        guidance = (ws / "GUIDANCE.md").read_text()
        assert "Test topic" in guidance
        assert "accuracy" in guidance

    def test_opencode_config_azure_format(self, tmp_path):
        bridge = OpenCodeBridge(
            model="gpt-5.2",
            llm_base_url="https://huaxi.openai.azure.com/openai/v1",
            api_key_env="AZURE_OPENAI_API_KEY",
            llm_provider="azure",
        )
        ws = bridge._prepare_workspace(
            stage_dir=tmp_path,
            topic="t",
            exp_plan="p",
            metric="m",
            pkg_hint="",
            extra_guidance="",
            time_budget_sec=300,
        )
        cfg = json.loads((ws / "opencode.json").read_text())
        # Azure now uses the unified "openai" provider (Bearer token auth
        # works on Azure endpoints and Responses API is supported)
        assert cfg["model"] == "openai/gpt-5.2"
        assert "provider" in cfg
        assert "openai" in cfg["provider"]
        assert cfg["provider"]["openai"]["options"]["baseURL"] == "https://huaxi.openai.azure.com/openai/v1"
        assert "{env:AZURE_OPENAI_API_KEY}" in cfg["provider"]["openai"]["options"]["apiKey"]

    def test_opencode_config_openai_format(self, tmp_path):
        bridge = OpenCodeBridge(
            model="gpt-4o",
            llm_base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
        ws = bridge._prepare_workspace(
            stage_dir=tmp_path,
            topic="t",
            exp_plan="p",
            metric="m",
            pkg_hint="",
            extra_guidance="",
            time_budget_sec=300,
        )
        cfg = json.loads((ws / "opencode.json").read_text())
        assert cfg["model"] == "openai/gpt-4o"
        assert "openai" in cfg["provider"]

    def test_opencode_config_preserves_prefixed_model(self, tmp_path):
        """Model with '/' prefix (e.g. anthropic/...) should NOT get double-prefixed (BUG-C fix)."""
        bridge = OpenCodeBridge(
            model="anthropic/claude-sonnet-4-6",
            llm_base_url="https://huaxi.openai.azure.com/openai/v1",
            api_key_env="AZURE_API_KEY",
            llm_provider="azure",
        )
        ws = bridge._prepare_workspace(
            stage_dir=tmp_path,
            topic="t",
            exp_plan="p",
            metric="m",
            pkg_hint="",
            extra_guidance="",
            time_budget_sec=300,
        )
        cfg = json.loads((ws / "opencode.json").read_text())
        # Should be "anthropic/claude-sonnet-4-6", NOT "azure/anthropic/claude-sonnet-4-6"
        assert cfg["model"] == "anthropic/claude-sonnet-4-6"

    def test_resolve_model_azure_uses_openai_prefix(self):
        """Azure endpoint → uses openai/ prefix (Azure supports Responses API now)."""
        bridge = OpenCodeBridge(
            model="gpt-5.2",
            llm_base_url="https://huaxi.openai.azure.com/openai/v1",
            llm_provider="azure",
        )
        resolved = bridge._resolve_opencode_model()
        assert resolved == "openai/gpt-5.2"

    def test_resolve_model_preserves_explicit_prefix(self):
        """Model with '/' prefix should be used as-is regardless of provider."""
        bridge = OpenCodeBridge(
            model="anthropic/claude-sonnet-4-6",
            llm_base_url="https://huaxi.openai.azure.com/openai/v1",
            llm_provider="azure",
        )
        resolved = bridge._resolve_opencode_model()
        assert resolved == "anthropic/claude-sonnet-4-6"

    def test_resolve_model_no_model_default(self):
        """Empty model string → default Anthropic model."""
        bridge = OpenCodeBridge()
        assert bridge._resolve_opencode_model() == "anthropic/claude-sonnet-4-6"

    def test_collect_files_ignores_pycache(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_text("bytecode")
        # Also write a .py in pycache to test filtering
        (pycache / "cached.py").write_text("cached")

        files = OpenCodeBridge._collect_files(tmp_path)
        assert "main.py" in files
        assert not any("__pycache__" in k for k in files)

    def test_collect_files_includes_requirements(self, tmp_path):
        (tmp_path / "main.py").write_text("import torch")
        (tmp_path / "requirements.txt").write_text("torch>=2.0")
        files = OpenCodeBridge._collect_files(tmp_path)
        assert "requirements.txt" in files
        assert "main.py" in files

    def test_collect_files_flattens_subdirectories(self, tmp_path):
        """Files in subdirs should be flattened to basenames (BUG-D fix)."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "model.py").write_text("class Model: pass")
        (src / "utils.py").write_text("def helper(): pass")
        (tmp_path / "main.py").write_text("from model import Model")
        files = OpenCodeBridge._collect_files(tmp_path)
        # Keys should be flat basenames, not paths like "src/model.py"
        assert "model.py" in files
        assert "utils.py" in files
        assert "main.py" in files
        assert not any("/" in k for k in files)

    def test_collect_files_root_takes_priority_over_subdir(self, tmp_path):
        """Root-level file wins when basename collides with subdir file."""
        (tmp_path / "main.py").write_text("root version")
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("subdir version")
        files = OpenCodeBridge._collect_files(tmp_path)
        assert files["main.py"] == "root version"

    def test_generate_returns_error_on_not_installed(self, tmp_path):
        bridge = OpenCodeBridge()
        with patch.object(OpenCodeBridge, "check_available", return_value=False):
            result = bridge.generate(
                stage_dir=tmp_path,
                topic="test",
                exp_plan="plan",
                metric="acc",
            )
        assert not result.success
        assert "not installed" in result.error

    def test_generate_returns_error_on_cli_failure(self, tmp_path):
        bridge = OpenCodeBridge(max_retries=0, workspace_cleanup=True)

        with patch.object(OpenCodeBridge, "check_available", return_value=True), \
             patch.object(
                 bridge,
                 "_invoke_opencode",
                 return_value=(False, "CLI error", 1.5),
             ):
            result = bridge.generate(
                stage_dir=tmp_path,
                topic="test",
                exp_plan="plan",
                metric="acc",
            )
        assert not result.success
        assert "failed" in result.error.lower()

    def test_generate_success(self, tmp_path):
        bridge = OpenCodeBridge(max_retries=0, workspace_cleanup=False)

        def fake_invoke(workspace, prompt):
            # Write main.py into the workspace to simulate OpenCode output
            (workspace / "main.py").write_text("print('acc: 0.95')")
            (workspace / "requirements.txt").write_text("torch")
            return True, "success", 5.0

        with patch.object(OpenCodeBridge, "check_available", return_value=True), \
             patch.object(bridge, "_invoke_opencode", side_effect=fake_invoke):
            result = bridge.generate(
                stage_dir=tmp_path,
                topic="test",
                exp_plan="plan",
                metric="acc",
            )
        assert result.success
        assert "main.py" in result.files
        assert result.elapsed_sec == 5.0

    def test_invoke_opencode_uses_resolved_path(self, tmp_path):
        bridge = OpenCodeBridge(model="gpt-5.2", timeout_sec=10)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""

        with patch(
            "researchclaw.pipeline.opencode_bridge.shutil.which",
            return_value=r"C:\Users\tester\AppData\Roaming\npm\opencode.cmd",
        ), patch(
            "researchclaw.pipeline.opencode_bridge.subprocess.run",
            return_value=mock_result,
        ) as run_mock:
            success, _log, _elapsed = bridge._invoke_opencode(tmp_path, "test prompt")

        assert success is True
        assert run_mock.call_args.args[0][0].endswith("opencode.cmd")


# ============================================================
# TestEnsureMainEntryPoint (BUG-R52-01)
# ============================================================


class TestHasMainGuard:
    """Tests for _has_main_guard static method."""

    def test_with_guard(self):
        code = 'def main():\n    pass\n\nif __name__ == "__main__":\n    main()\n'
        assert OpenCodeBridge._has_main_guard(code) is True

    def test_without_guard(self):
        code = "def main():\n    pass\n"
        assert OpenCodeBridge._has_main_guard(code) is False

    def test_syntax_error(self):
        assert OpenCodeBridge._has_main_guard("def broken(") is False

    def test_empty(self):
        assert OpenCodeBridge._has_main_guard("") is False

    def test_single_quote_guard(self):
        code = "if __name__ == '__main__':\n    print('hi')\n"
        assert OpenCodeBridge._has_main_guard(code) is True


class TestEnsureMainEntryPoint:
    """Tests for _ensure_main_entry_point — BUG-R52-01 fix."""

    def test_already_has_guard_unchanged(self):
        files = {
            "main.py": 'def run():\n    pass\n\nif __name__ == "__main__":\n    run()\n',
            "utils.py": "def helper(): pass\n",
        }
        result = OpenCodeBridge._ensure_main_entry_point(files)
        assert result is files  # Same object, unchanged

    def test_no_main_py_unchanged(self):
        files = {"utils.py": "def helper(): pass\n"}
        result = OpenCodeBridge._ensure_main_entry_point(files)
        assert result is files

    def test_swap_entry_point_from_other_file(self):
        """When main.py is library-only and another file has __main__, swap."""
        lib_code = "class Model:\n    pass\n\ndef train(model):\n    pass\n"
        entry_code = (
            'from main import Model, train\n\n'
            'if __name__ == "__main__":\n'
            '    m = Model()\n'
            '    train(m)\n'
        )
        files = {
            "main.py": lib_code,
            "run_experiment.py": entry_code,
        }
        result = OpenCodeBridge._ensure_main_entry_point(files)
        # main.py should now contain the entry point code
        assert '__main__' in result["main.py"]
        # The old main.py content should be in run_experiment.py
        assert result["run_experiment.py"] == lib_code

    def test_inject_entry_for_main_function(self):
        """When main.py defines main() but no guard, inject one."""
        code = "import torch\n\ndef main():\n    print('training')\n"
        files = {"main.py": code}
        result = OpenCodeBridge._ensure_main_entry_point(files)
        assert '__main__' in result["main.py"]
        assert "main()" in result["main.py"]

    def test_inject_entry_for_run_function(self):
        """Should also detect run(), train(), etc."""
        code = "def run_experiment():\n    print('running')\n"
        files = {"main.py": code}
        result = OpenCodeBridge._ensure_main_entry_point(files)
        assert '__main__' in result["main.py"]
        assert "run_experiment()" in result["main.py"]

    def test_no_known_entry_function_warns(self):
        """When no known entry function exists, return unchanged with warning."""
        code = "class Config:\n    x = 1\n\nclass Trainer:\n    pass\n"
        files = {"main.py": code}
        result = OpenCodeBridge._ensure_main_entry_point(files)
        # Should return unchanged since no entry function found
        assert result["main.py"] == code

    def test_non_py_files_not_checked(self):
        """requirements.txt and setup.py should not be checked for __main__."""
        lib_code = "class Model:\n    pass\n"
        files = {
            "main.py": lib_code,
            "requirements.txt": "torch>=2.0\n",
            "setup.py": "# setup\n",
        }
        result = OpenCodeBridge._ensure_main_entry_point(files)
        # No swap should occur — only .py files are checked
        assert result["main.py"] == lib_code

    def test_swap_preserves_other_files(self):
        """Swapping should not lose any files from the dict."""
        files = {
            "main.py": "class Lib: pass\n",
            "run.py": 'if __name__ == "__main__":\n    print("go")\n',
            "utils.py": "def helper(): pass\n",
            "requirements.txt": "numpy\n",
        }
        result = OpenCodeBridge._ensure_main_entry_point(files)
        assert len(result) == len(files)
        assert "utils.py" in result
        assert "requirements.txt" in result


# ============================================================
# TestOpenCodeConfig
# ============================================================


class TestOpenCodeConfig:
    """Tests for OpenCodeConfig dataclass and parser."""

    def test_default_values(self):
        cfg = OpenCodeConfig()
        assert cfg.enabled is True
        assert cfg.auto is True
        assert cfg.complexity_threshold == 0.2
        assert cfg.model == ""
        assert cfg.timeout_sec == 600
        assert cfg.max_retries == 1
        assert cfg.workspace_cleanup is True

    def test_parse_from_dict(self):
        data = {
            "enabled": True,
            "auto": True,
            "complexity_threshold": 0.5,
            "model": "gpt-5.2",
            "timeout_sec": 900,
            "max_retries": 2,
            "workspace_cleanup": False,
        }
        cfg = _parse_opencode_config(data)
        assert cfg.enabled is True
        assert cfg.auto is True
        assert cfg.complexity_threshold == 0.5
        assert cfg.model == "gpt-5.2"
        assert cfg.timeout_sec == 900
        assert cfg.max_retries == 2
        assert cfg.workspace_cleanup is False

    def test_empty_dict_returns_default(self):
        cfg = _parse_opencode_config({})
        assert cfg == OpenCodeConfig()


# ============================================================
# TestCountHistoricalFailures
# ============================================================


class TestCountHistoricalFailures:
    def test_no_failures(self, tmp_path):
        assert count_historical_failures(tmp_path) == 0

    def test_counts_beast_mode_failures(self, tmp_path):
        d = tmp_path / "stage-10_001"
        d.mkdir()
        (d / "beast_mode_log.json").write_text(json.dumps({"success": False}))
        assert count_historical_failures(tmp_path) >= 1

    def test_counts_validation_failures(self, tmp_path):
        d = tmp_path / "stage-10_002"
        d.mkdir()
        (d / "validation_report.md").write_text("**Status**: FAILED after 5 repairs")
        assert count_historical_failures(tmp_path) >= 1

    def test_deduplicates_multiple_failure_indicators(self, tmp_path):
        """Same dir with beast_mode_log + stage_health + validation_report = 1 failure (BUG-E fix)."""
        d = tmp_path / "stage-10_003"
        d.mkdir()
        (d / "beast_mode_log.json").write_text(json.dumps({"success": False}))
        (d / "stage_health.json").write_text(json.dumps({"status": "FAILED"}))
        (d / "validation_report.md").write_text("FAILED after 3 repairs")
        assert count_historical_failures(tmp_path) == 1
