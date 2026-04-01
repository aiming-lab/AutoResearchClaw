"""Tests for the universal experiment schema."""

from __future__ import annotations

import pytest
import yaml

from researchclaw.domains.experiment_schema import (
    Condition,
    ConditionRole,
    EvaluationSpec,
    ExperimentType,
    MetricSpec,
    UniversalExperimentPlan,
    from_legacy_exp_plan,
)


# ---------------------------------------------------------------------------
# Condition tests
# ---------------------------------------------------------------------------


class TestCondition:
    def test_default_role(self):
        c = Condition(name="test")
        assert c.role == ConditionRole.PROPOSED.value

    def test_custom_role(self):
        c = Condition(name="baseline_method", role=ConditionRole.REFERENCE.value)
        assert c.role == "reference"

    def test_variant_with_parent(self):
        c = Condition(
            name="ablation_no_attn",
            role=ConditionRole.VARIANT.value,
            varies_from="proposed_method",
            variation="remove_attention",
        )
        assert c.varies_from == "proposed_method"


# ---------------------------------------------------------------------------
# UniversalExperimentPlan tests
# ---------------------------------------------------------------------------


class TestUniversalExperimentPlan:
    def test_empty_plan(self):
        plan = UniversalExperimentPlan()
        assert plan.conditions == []
        assert plan.experiment_type == "comparison"

    def test_plan_with_conditions(self):
        plan = UniversalExperimentPlan(
            experiment_type="comparison",
            conditions=[
                Condition(name="baseline", role="reference"),
                Condition(name="proposed", role="proposed"),
                Condition(name="ablation", role="variant", varies_from="proposed"),
            ],
        )
        assert len(plan.references) == 1
        assert len(plan.proposed) == 1
        assert len(plan.variants) == 1

    def test_to_legacy_format(self):
        plan = UniversalExperimentPlan(
            conditions=[
                Condition(name="ResNet-18", role="reference", description="Standard baseline"),
                Condition(name="OurMethod", role="proposed", description="Our new method"),
                Condition(name="OurMethod-NoAttn", role="variant", varies_from="OurMethod"),
            ],
            evaluation=EvaluationSpec(
                primary_metric=MetricSpec(name="accuracy", direction="maximize"),
            ),
        )
        legacy = plan.to_legacy_format()
        assert len(legacy["baselines"]) == 1
        assert legacy["baselines"][0]["name"] == "ResNet-18"
        assert len(legacy["proposed_methods"]) == 1
        assert len(legacy["ablations"]) == 1
        assert "accuracy" in legacy["metrics"]

    def test_to_yaml(self):
        plan = UniversalExperimentPlan(
            experiment_type="convergence",
            domain_id="physics_pde",
            conditions=[
                Condition(name="FD2", role="reference"),
                Condition(name="FD4", role="proposed"),
            ],
        )
        yaml_str = plan.to_yaml()
        data = yaml.safe_load(yaml_str)
        assert data["experiment"]["type"] == "convergence"
        assert data["experiment"]["domain"] == "physics_pde"
        assert len(data["experiment"]["conditions"]) == 2


# ---------------------------------------------------------------------------
# from_legacy_exp_plan tests
# ---------------------------------------------------------------------------


class TestFromLegacy:
    def test_basic_legacy_plan(self):
        legacy = {
            "baselines": [
                {"name": "ResNet-18", "description": "Standard CNN"},
            ],
            "proposed_methods": [
                {"name": "OurNet", "description": "Our new architecture"},
            ],
            "ablations": [
                {"name": "OurNet-NoSkip", "description": "Without skip connections"},
            ],
            "metrics": {
                "accuracy": {"direction": "maximize"},
            },
        }
        plan = from_legacy_exp_plan(legacy, domain_id="ml_vision")
        assert plan.domain_id == "ml_vision"
        assert len(plan.references) == 1
        assert plan.references[0].name == "ResNet-18"
        assert len(plan.proposed) == 1
        assert len(plan.variants) == 1
        assert plan.evaluation.primary_metric.name == "accuracy"
        assert plan.evaluation.primary_metric.direction == "maximize"

    def test_legacy_string_names(self):
        legacy = {
            "baselines": ["baseline_1", "baseline_2"],
            "proposed_methods": ["our_method"],
            "ablations": [],
        }
        plan = from_legacy_exp_plan(legacy)
        assert len(plan.references) == 2
        assert plan.references[0].name == "baseline_1"

    def test_legacy_yaml_string(self):
        yaml_str = """
baselines:
  - name: Euler
    description: Basic Euler method
proposed_methods:
  - name: RK4
    description: Runge-Kutta 4th order
metrics:
  convergence_order:
    direction: maximize
"""
        plan = from_legacy_exp_plan(yaml_str, domain_id="mathematics_numerical")
        assert plan.domain_id == "mathematics_numerical"
        assert len(plan.references) == 1
        assert plan.evaluation.primary_metric.name == "convergence_order"

    def test_roundtrip_legacy(self):
        """Test that converting to legacy and back preserves structure."""
        plan = UniversalExperimentPlan(
            conditions=[
                Condition(name="A", role="reference"),
                Condition(name="B", role="proposed"),
            ],
            evaluation=EvaluationSpec(
                primary_metric=MetricSpec(name="error", direction="minimize"),
            ),
        )
        legacy = plan.to_legacy_format()
        plan2 = from_legacy_exp_plan(legacy)
        assert len(plan2.references) == 1
        assert len(plan2.proposed) == 1
        assert plan2.evaluation.primary_metric.direction == "minimize"

    def test_empty_legacy(self):
        plan = from_legacy_exp_plan({})
        assert plan.conditions == []

    def test_metrics_as_list(self):
        legacy = {"metrics": ["accuracy", "f1"]}
        plan = from_legacy_exp_plan(legacy)
        assert plan.evaluation.primary_metric.name == "accuracy"


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_condition_role_values(self):
        assert ConditionRole.REFERENCE.value == "reference"
        assert ConditionRole.PROPOSED.value == "proposed"
        assert ConditionRole.VARIANT.value == "variant"

    def test_experiment_type_values(self):
        assert ExperimentType.COMPARISON.value == "comparison"
        assert ExperimentType.CONVERGENCE.value == "convergence"
        assert ExperimentType.PROGRESSIVE_SPEC.value == "progressive_spec"
