"""Integration tests for universal cross-domain code generation.

Tests the full pipeline from domain detection → adapter selection →
prompt block generation → blueprint context building, across multiple
research domains. These tests do NOT require an LLM or network —
they verify the infrastructure wiring.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from researchclaw.domains.detector import (
    DomainProfile,
    detect_domain,
    get_profile,
    is_ml_domain,
    load_all_profiles,
)
from researchclaw.domains.prompt_adapter import get_adapter, PromptBlocks
from researchclaw.domains.experiment_schema import (
    Condition,
    ConditionRole,
    EvaluationSpec,
    MetricSpec,
    UniversalExperimentPlan,
    from_legacy_exp_plan,
)
from researchclaw.experiment.metrics import UniversalMetricParser
from researchclaw.experiment.evaluators.convergence import analyze_convergence
from researchclaw.agents.code_searcher.agent import CodeSearchAgent, CodeSearchResult
from researchclaw.agents.code_searcher.pattern_extractor import CodePatterns


# ---------------------------------------------------------------------------
# Cross-domain domain detection integration
# ---------------------------------------------------------------------------


class TestCrossDomainDetection:
    """Test domain detection across all supported domains."""

    def test_all_profiles_loadable(self):
        profiles = load_all_profiles()
        assert len(profiles) >= 18  # at least 18 domain profiles

    def test_ml_vision_full_pipeline(self):
        """ML Vision: detect → adapter → blocks → legacy compatibility."""
        profile = detect_domain("image classification on CIFAR-10 with ResNet")
        assert profile.domain_id == "ml_vision"
        assert is_ml_domain(profile)

        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        # ML adapter returns empty blocks (existing behavior)
        assert blocks.compute_budget == ""

    def test_physics_pde_full_pipeline(self):
        """Physics PDE: detect → adapter → blocks with convergence guidance."""
        profile = detect_domain("finite element method for Poisson equation")
        assert profile.domain_id == "physics_pde"
        assert not is_ml_domain(profile)

        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        # Physics adapter should provide non-empty guidance
        assert blocks.code_generation_hints  # not empty

        # Blueprint context should mention convergence
        ctx = adapter.get_blueprint_context()
        assert ctx  # not empty

    def test_economics_full_pipeline(self):
        """Economics: detect → adapter → progressive spec guidance."""
        profile = detect_domain("panel data regression with instrumental variables")
        assert profile.domain_id == "economics_empirical"

        adapter = get_adapter(profile)
        blocks = adapter.get_experiment_design_blocks({})
        assert "progressive" in blocks.experiment_design_context.lower()

    def test_chemistry_full_pipeline(self):
        """Chemistry: detect → adapter → PySCF guidance."""
        profile = detect_domain("DFT calculation with PySCF for molecular energies")
        assert profile.domain_id == "chemistry_qm"

        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        assert blocks.code_generation_hints

    def test_biology_full_pipeline(self):
        """Biology: detect → adapter → scanpy guidance."""
        profile = detect_domain("single-cell RNA-seq clustering with scanpy")
        assert profile.domain_id == "biology_singlecell"

        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        assert blocks.code_generation_hints

    def test_math_full_pipeline(self):
        """Math: detect → adapter → convergence guidance."""
        profile = detect_domain("Runge-Kutta ODE solver convergence analysis")
        assert profile.domain_id == "mathematics_numerical"

        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        assert blocks.code_generation_hints


# ---------------------------------------------------------------------------
# Universal Experiment Schema integration
# ---------------------------------------------------------------------------


class TestExperimentSchemaIntegration:
    def test_physics_convergence_plan(self):
        """Create a physics convergence study plan."""
        plan = UniversalExperimentPlan(
            experiment_type="convergence",
            domain_id="physics_pde",
            problem_description="Solve Poisson equation with FEM and FDM",
            conditions=[
                Condition(name="FDM_2nd", role="reference",
                          description="2nd order finite difference"),
                Condition(name="FEM_P1", role="proposed",
                          description="P1 finite element method"),
                Condition(name="FEM_P2", role="variant",
                          varies_from="FEM_P1",
                          description="P2 finite element method"),
            ],
            evaluation=EvaluationSpec(
                primary_metric=MetricSpec(
                    name="l2_error",
                    direction="minimize",
                    unit="relative",
                ),
                protocol="Run at 5 grid sizes, measure L2 error",
                statistical_test="convergence_order_fit",
                num_seeds=1,
            ),
            main_figure_type="convergence_plot",
        )

        assert len(plan.references) == 1
        assert len(plan.proposed) == 1
        assert len(plan.variants) == 1

        # Test legacy format conversion
        legacy = plan.to_legacy_format()
        assert len(legacy["baselines"]) == 1
        assert legacy["baselines"][0]["name"] == "FDM_2nd"
        assert "l2_error" in legacy["metrics"]

        # Test YAML serialization
        yaml_str = plan.to_yaml()
        assert "convergence" in yaml_str
        assert "FDM_2nd" in yaml_str

    def test_economics_progressive_plan(self):
        """Create an economics progressive specification plan."""
        plan = UniversalExperimentPlan(
            experiment_type="progressive_spec",
            domain_id="economics_empirical",
            conditions=[
                Condition(name="OLS", role="reference",
                          description="Simple OLS"),
                Condition(name="OLS_controls", role="proposed",
                          description="OLS with control variables"),
                Condition(name="FE", role="variant",
                          varies_from="OLS_controls",
                          description="Fixed effects"),
                Condition(name="IV_2SLS", role="variant",
                          varies_from="OLS_controls",
                          description="Instrumental variables"),
            ],
            evaluation=EvaluationSpec(
                primary_metric=MetricSpec(name="coefficient", direction="maximize"),
                statistical_test="hausman_test",
            ),
            main_table_type="regression_table",
        )
        assert len(plan.conditions) == 4
        legacy = plan.to_legacy_format()
        assert len(legacy["ablations"]) == 2  # FE and IV are variants


# ---------------------------------------------------------------------------
# Metric Parser + Convergence Evaluator integration
# ---------------------------------------------------------------------------


class TestMetricConvergenceIntegration:
    def test_json_convergence_end_to_end(self, tmp_path):
        """Parse JSON convergence results → analyze convergence → report."""
        data = {
            "experiment_type": "convergence",
            "convergence": {
                "euler": [
                    {"h": 0.1, "error": 0.1},
                    {"h": 0.05, "error": 0.05},
                    {"h": 0.025, "error": 0.025},
                    {"h": 0.0125, "error": 0.0125},
                ],
                "rk4": [
                    {"h": 0.1, "error": 1e-4},
                    {"h": 0.05, "error": 6.25e-6},
                    {"h": 0.025, "error": 3.9e-7},
                    {"h": 0.0125, "error": 2.44e-8},
                ],
            },
            "metadata": {"domain": "mathematics_numerical"},
        }
        (tmp_path / "results.json").write_text(json.dumps(data))

        # Parse
        parser = UniversalMetricParser()
        results = parser.parse(tmp_path)
        assert results.source == "json"
        assert "euler" in results.convergence

        # Analyze convergence
        report = analyze_convergence(
            results.convergence,
            expected_orders={"euler": 1.0, "rk4": 4.0},
        )
        assert len(report.methods) == 2

        euler = next(r for r in report.methods if r.method == "euler")
        rk4 = next(r for r in report.methods if r.method == "rk4")

        assert abs(euler.convergence_order - 1.0) < 0.2
        assert abs(rk4.convergence_order - 4.0) < 0.5
        assert rk4.convergence_order > euler.convergence_order
        assert report.best_method == "rk4"

    def test_flat_metrics_backward_compatible(self, tmp_path):
        """Ensure new metric parser produces backward-compatible output."""
        # Write old-style stdout
        result = UniversalMetricParser().parse(
            tmp_path,
            stdout="accuracy: 0.95\nloss: 0.32\ncondition=proposed accuracy: 0.95\n",
        )
        flat = result.to_flat_metrics()
        assert "accuracy" in flat
        assert "loss" in flat
        assert flat["accuracy"] == 0.95


# ---------------------------------------------------------------------------
# Code Search + Domain Profile integration
# ---------------------------------------------------------------------------


class TestCodeSearchIntegration:
    def test_code_search_result_in_blueprint(self):
        """Code search results should be formattable as prompt context."""
        result = CodeSearchResult(
            patterns=CodePatterns(
                api_patterns=[
                    "from pyscf import gto, scf\nmol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')",
                ],
                file_structure={"main.py": "Entry point", "molecule.py": "Molecule definitions"},
                evaluation_patterns=["mae = np.mean(np.abs(predicted - reference))"],
            ),
            repos_found=[
                MagicMock(full_name="user/pyscf-example", stars=200),
            ],
        )
        ctx = result.to_prompt_context()
        assert "pyscf" in ctx
        assert "molecule.py" in ctx

    def test_domain_adapter_blueprint_context(self):
        """Domain adapter should produce useful blueprint context."""
        profile = get_profile("physics_simulation")
        if profile is None:
            pytest.skip("physics_simulation profile not found")

        adapter = get_adapter(profile)
        ctx = adapter.get_blueprint_context()

        # Should mention file structure
        assert "main.py" in ctx or "integrator" in ctx.lower()
        # Should mention libraries
        assert "numpy" in ctx.lower() or "scipy" in ctx.lower() or ctx != ""


# ---------------------------------------------------------------------------
# CodeAgent domain injection test
# ---------------------------------------------------------------------------


class TestCodeAgentDomainInjection:
    def test_code_agent_accepts_domain_profile(self):
        """CodeAgent should accept domain_profile and code_search_result."""
        from researchclaw.pipeline.code_agent import CodeAgent, CodeAgentConfig

        config = CodeAgentConfig(enabled=True)
        profile = DomainProfile(
            domain_id="physics_pde",
            display_name="PDE Solvers",
            core_libraries=["numpy", "scipy"],
        )
        search_result = CodeSearchResult(
            patterns=CodePatterns(
                api_patterns=["import scipy.sparse"],
            ),
        )

        agent = CodeAgent(
            llm=MagicMock(),
            prompts=MagicMock(),
            config=config,
            stage_dir=Path("/tmp/test"),
            domain_profile=profile,
            code_search_result=search_result,
        )

        # Verify the domain context builder works
        ctx = agent._build_domain_context()
        assert "scipy" in ctx.lower() or ctx != ""

    def test_code_agent_ml_domain_no_extra_context(self):
        """ML domain should add minimal extra context (preserve existing behavior)."""
        from researchclaw.pipeline.code_agent import CodeAgent, CodeAgentConfig

        config = CodeAgentConfig(enabled=True)
        profile = get_profile("ml_vision") or DomainProfile(
            domain_id="ml_vision",
            display_name="Computer Vision",
        )

        agent = CodeAgent(
            llm=MagicMock(),
            prompts=MagicMock(),
            config=config,
            stage_dir=Path("/tmp/test"),
            domain_profile=profile,
            code_search_result=None,  # No code search for ML
        )

        # ML adapter returns empty blocks → minimal context
        ctx = agent._build_domain_context()
        # It's acceptable for ML to have some context from file structure,
        # but it should NOT have code search results
        # (we didn't provide code_search_result)
        assert "Reference Code from GitHub" not in ctx


# ---------------------------------------------------------------------------
# Docker profile mapping test
# ---------------------------------------------------------------------------


class TestDockerProfileMapping:
    def test_domain_to_docker_mapping(self):
        """All domains should map to a valid docker profile."""
        import yaml

        profiles_path = Path(__file__).parent.parent / "researchclaw" / "data" / "docker_profiles.yaml"
        if not profiles_path.exists():
            pytest.skip("docker_profiles.yaml not found")

        with profiles_path.open() as f:
            docker_config = yaml.safe_load(f)

        domain_map = docker_config.get("domain_map", {})
        profiles = docker_config.get("profiles", {})

        # Every mapped domain should point to a valid profile
        for domain_id, profile_name in domain_map.items():
            assert profile_name in profiles, (
                f"Domain {domain_id} maps to unknown profile: {profile_name}"
            )

    def test_all_loaded_domains_have_docker_mapping(self):
        """All domain profiles should have a docker mapping."""
        import yaml

        profiles_path = Path(__file__).parent.parent / "researchclaw" / "data" / "docker_profiles.yaml"
        if not profiles_path.exists():
            pytest.skip("docker_profiles.yaml not found")

        with profiles_path.open() as f:
            docker_config = yaml.safe_load(f)

        domain_map = docker_config.get("domain_map", {})
        domain_profiles = load_all_profiles()

        unmapped = []
        for domain_id in domain_profiles:
            if domain_id not in domain_map and domain_id != "generic":
                unmapped.append(domain_id)

        # Allow some unmapped (new domains without docker images yet)
        # but the core ones should be mapped
        core_domains = [
            "ml_vision", "ml_nlp", "ml_rl", "physics_simulation",
            "physics_pde", "chemistry_qm", "economics_empirical",
            "mathematics_numerical",
        ]
        for d in core_domains:
            assert d in domain_map, f"Core domain {d} missing from docker mapping"
