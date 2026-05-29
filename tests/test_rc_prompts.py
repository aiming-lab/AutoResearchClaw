"""Tests for researchclaw.prompts — PromptManager and template rendering."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from researchclaw.prompts import (
    PromptManager,
    RenderedPrompt,
    _render,
)


# ---------------------------------------------------------------------------
# _render() — template variable substitution
# ---------------------------------------------------------------------------


class TestRender:
    def test_simple_substitution(self) -> None:
        assert _render("Hello {name}!", {"name": "World"}) == "Hello World!"

    def test_multiple_variables(self) -> None:
        result = _render(
            "Topic: {topic}, Domain: {domain}", {"topic": "RL", "domain": "ML"}
        )
        assert result == "Topic: RL, Domain: ML"

    def test_missing_variable_left_untouched(self) -> None:
        assert _render("Value: {unknown}", {}) == "Value: {unknown}"

    def test_json_schema_not_substituted(self) -> None:
        template = "Return JSON: {candidates:[...]} with >=8 rows."
        assert _render(template, {"candidates": "SHOULD_NOT_APPEAR"}) == template

    def test_json_schema_complex_not_substituted(self) -> None:
        template = "Schema: {score_1_to_10:number, verdict:string}"
        assert _render(template, {}) == template

    def test_curly_braces_in_code_not_substituted(self) -> None:
        template = "def foo(): { return 1; }"
        assert _render(template, {}) == template

    def test_underscore_variable(self) -> None:
        assert _render("{my_var}", {"my_var": "ok"}) == "ok"

    def test_numeric_suffix(self) -> None:
        assert _render("{score_1}", {"score_1": "9"}) == "9"

    def test_empty_template(self) -> None:
        assert _render("", {"x": "y"}) == ""

    def test_no_placeholders(self) -> None:
        assert _render("No variables here", {"x": "y"}) == "No variables here"


# ---------------------------------------------------------------------------
# PromptManager — defaults
# ---------------------------------------------------------------------------


class TestPromptManagerDefaults:
    def test_all_stages_present(self) -> None:
        """20 stages have for_stage() prompts; iterative_refine uses sub_prompts only."""
        pm = PromptManager()
        names = pm.stage_names()
        assert len(names) >= 20
        for required in [
            "topic_init",
            "problem_decompose",
            "search_strategy",
            "literature_collect",
            "literature_screen",
            "knowledge_extract",
            "synthesis",
            "hypothesis_gen",
            "experiment_design",
            "code_generation",
            "resource_planning",
            "result_analysis",
            "research_decision",
            "paper_outline",
            "paper_draft",
            "peer_review",
            "paper_revision",
            "quality_gate",
            "knowledge_archive",
            "export_publish",
        ]:
            assert pm.has_stage(required), f"Missing stage: {required}"

    def test_system_prompt_nonempty(self) -> None:
        pm = PromptManager()
        for name in pm.stage_names():
            assert pm.system(name), f"Empty system prompt for {name}"

    def test_for_stage_returns_rendered_prompt(self) -> None:
        pm = PromptManager()
        sp = pm.for_stage(
            "topic_init",
            topic="RL",
            domains="ml",
            project_name="test",
            quality_threshold="4.0",
        )
        assert isinstance(sp, RenderedPrompt)
        assert "RL" in sp.user
        assert "ml" in sp.user
        assert sp.system

    def test_json_mode_stages(self) -> None:
        pm = PromptManager()
        json_stages = [
            "search_strategy",
            "literature_collect",
            "literature_screen",
            "knowledge_extract",
            "resource_planning",
            "quality_gate",
        ]
        for stage in json_stages:
            assert pm.json_mode(stage), f"{stage} should have json_mode=True"

    def test_non_json_stages(self) -> None:
        pm = PromptManager()
        assert not pm.json_mode("topic_init")
        assert not pm.json_mode("synthesis")

    def test_max_tokens(self) -> None:
        pm = PromptManager()
        assert pm.max_tokens("code_generation") == 8192
        assert pm.max_tokens("paper_draft") == 16384
        assert pm.max_tokens("topic_init") is None

    def test_block_topic_constraint(self) -> None:
        pm = PromptManager()
        block = pm.block("topic_constraint", topic="Neural Architecture Search")
        assert "Neural Architecture Search" in block
        assert "HARD TOPIC CONSTRAINT" in block

    def test_block_pkg_hint(self) -> None:
        pm = PromptManager()
        block = pm.block("pkg_hint_sandbox")
        assert "numpy" in block
        assert "torch" in block  # mentioned as prohibited

    def test_block_network_disabled_guidance_uses_default_root(self) -> None:
        # When the caller passes the SandboxConfig default the rendered block
        # still names /opt/datasets, matching the prior hardcoded value.
        pm = PromptManager()
        block = pm.block(
            "network_disabled_guidance", dataset_cache_root="/opt/datasets"
        )
        # All six dataset examples should reference the configured root.
        assert block.count("/opt/datasets") >= 6
        # Critical invariants preserved.
        assert "network_policy='none'" in block
        assert "download=False" in block
        assert "Do NOT use `download=True`" in block
        # The unrendered placeholder must not leak through.
        assert "{dataset_cache_root}" not in block

    def test_block_network_disabled_guidance_uses_custom_root(self) -> None:
        # A custom dataset cache root flows through every dataset example.
        pm = PromptManager()
        block = pm.block(
            "network_disabled_guidance",
            dataset_cache_root="/tmp/arc_sandbox_trial/datasets",
        )
        assert block.count("/tmp/arc_sandbox_trial/datasets") >= 6
        # The previous hardcoded path must no longer appear.
        assert "/opt/datasets" not in block
        # Invariants still hold.
        assert "download=False" in block
        assert "Do NOT use `download=True`" in block
        assert "{dataset_cache_root}" not in block

    def test_block_network_disabled_guidance_forbids_synthetic_fallback(
        self,
    ) -> None:
        # The fail-loud bullet is an intentional behavior change shipped with
        # the dataset_cache_root parameterisation: missing pre-cached data must
        # raise FileNotFoundError rather than silently substituting synthetic
        # tensors. Lock it in for both default and custom cache roots.
        pm = PromptManager()
        for root in ("/opt/datasets", "/tmp/arc_sandbox_trial/datasets"):
            block = pm.block("network_disabled_guidance", dataset_cache_root=root)
            lowered = block.lower()
            assert "fall back to synthetic data" in lowered
            assert "filenotfounderror" in lowered
            assert "exit non-zero" in lowered

    def test_block_network_disabled_guidance_requires_dataset_used_stamp(
        self,
    ) -> None:
        # Downstream metric capture relies on a single-line stdout stamp of
        # the form `DATASET_USED: <name>` as a provenance signal independent
        # of whatever JSON result schema the generated code chooses.
        pm = PromptManager()
        for root in ("/opt/datasets", "/tmp/arc_sandbox_trial/datasets"):
            block = pm.block("network_disabled_guidance", dataset_cache_root=root)
            assert "DATASET_USED:" in block
            assert "dataset provenance stamp" in block.lower()
            assert "exactly once" in block.lower()

    def test_sub_prompt_code_repair(self) -> None:
        pm = PromptManager()
        rp = pm.sub_prompt(
            "code_repair",
            fname="model.py",
            issues_text="SyntaxError",
            all_files_ctx="...",
        )
        assert "model.py" in rp.user
        assert "SyntaxError" in rp.user
        assert rp.system

    def test_sub_prompt_iterative_improve(self) -> None:
        pm = PromptManager()
        ip = pm.sub_prompt(
            "iterative_improve",
            metric_key="val_loss",
            metric_direction="minimize",
            files_context="...",
            run_summaries="...",
        )
        assert "val_loss" in ip.user
        assert "minimize" in ip.user

    def test_sub_prompt_iterative_repair(self) -> None:
        pm = PromptManager()
        irp = pm.sub_prompt(
            "iterative_repair", issue_text="import error", all_files_ctx="..."
        )
        assert "import error" in irp.user


# ---------------------------------------------------------------------------
# PromptManager — YAML override
# ---------------------------------------------------------------------------


class TestPromptManagerOverrides:
    def test_override_system_prompt(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            stages:
              topic_init:
                system: "You are a custom planner."
        """)
        override_file = tmp_path / "custom.yaml"
        override_file.write_text(yaml_content, encoding="utf-8")
        pm = PromptManager(override_file)
        assert pm.system("topic_init") == "You are a custom planner."
        # Other stages should keep defaults
        assert pm.system("problem_decompose") == "You are a senior research strategist."

    def test_override_user_template(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            stages:
              topic_init:
                user: "Custom prompt for {topic}."
        """)
        override_file = tmp_path / "custom.yaml"
        override_file.write_text(yaml_content, encoding="utf-8")
        pm = PromptManager(override_file)
        result = pm.user("topic_init", topic="GAN")
        assert result == "Custom prompt for GAN."

    def test_override_block(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            blocks:
              topic_constraint: "Stay focused on {topic}."
        """)
        override_file = tmp_path / "custom.yaml"
        override_file.write_text(yaml_content, encoding="utf-8")
        pm = PromptManager(override_file)
        assert pm.block("topic_constraint", topic="NAS") == "Stay focused on NAS."

    def test_override_json_mode(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            stages:
              topic_init:
                json_mode: true
        """)
        override_file = tmp_path / "custom.yaml"
        override_file.write_text(yaml_content, encoding="utf-8")
        pm = PromptManager(override_file)
        assert pm.json_mode("topic_init") is True

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        pm = PromptManager(tmp_path / "nonexistent.yaml")
        assert pm.has_stage("topic_init")
        assert pm.system("topic_init")

    def test_invalid_yaml_uses_defaults(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": invalid: yaml: [", encoding="utf-8")
        pm = PromptManager(bad_file)
        assert pm.has_stage("topic_init")

    def test_unknown_stage_in_override_ignored(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            stages:
              nonexistent_stage:
                system: "Should be ignored."
        """)
        override_file = tmp_path / "custom.yaml"
        override_file.write_text(yaml_content, encoding="utf-8")
        # Should not raise
        pm = PromptManager(override_file)
        assert not pm.has_stage("nonexistent_stage")


# ---------------------------------------------------------------------------
# PromptManager — export_yaml
# ---------------------------------------------------------------------------


class TestExportYaml:
    def test_export_roundtrip(self, tmp_path: Path) -> None:
        pm1 = PromptManager()
        export_path = tmp_path / "exported.yaml"
        pm1.export_yaml(export_path)
        assert export_path.exists()

        # Load it back — should parse cleanly
        data = yaml.safe_load(export_path.read_text(encoding="utf-8"))
        assert "stages" in data
        assert "blocks" in data
        assert "version" in data

    def test_export_contains_all_stages(self, tmp_path: Path) -> None:
        pm = PromptManager()
        export_path = tmp_path / "exported.yaml"
        pm.export_yaml(export_path)
        data = yaml.safe_load(export_path.read_text(encoding="utf-8"))
        for stage in pm.stage_names():
            assert stage in data["stages"], f"Missing {stage} in export"

    def test_export_with_overrides(self, tmp_path: Path) -> None:
        override_file = tmp_path / "custom.yaml"
        override_file.write_text(
            "stages:\n  topic_init:\n    system: CUSTOM\n",
            encoding="utf-8",
        )
        pm = PromptManager(override_file)
        export_path = tmp_path / "exported.yaml"
        pm.export_yaml(export_path)
        data = yaml.safe_load(export_path.read_text(encoding="utf-8"))
        assert data["stages"]["topic_init"]["system"] == "CUSTOM"


# ---------------------------------------------------------------------------
# RenderedPrompt dataclass
# ---------------------------------------------------------------------------


class TestRenderedPrompt:
    def test_defaults(self) -> None:
        rp = RenderedPrompt(system="sys", user="usr")
        assert rp.json_mode is False
        assert rp.max_tokens is None

    def test_with_options(self) -> None:
        rp = RenderedPrompt(system="s", user="u", json_mode=True, max_tokens=4096)
        assert rp.json_mode is True
        assert rp.max_tokens == 4096

    def test_frozen(self) -> None:
        rp = RenderedPrompt(system="s", user="u")
        with pytest.raises(AttributeError):
            rp.system = "modified"  # type: ignore[misc]
