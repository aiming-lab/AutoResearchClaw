"""Tests for the universal experiment schema."""

from __future__ import annotations

import pytest
import yaml

import researchclaw.domains.experiment_schema as experiment_schema
from researchclaw.domains.experiment_schema import (
    Condition,
    ConditionRole,
    EvaluationSpec,
    ExperimentType,
    MetricSpec,
    UniversalExperimentPlan,
    from_legacy_exp_plan,
)


def _full_v1_plan() -> UniversalExperimentPlan:
    return UniversalExperimentPlan(
        experiment_type=ExperimentType.COMPARISON.value,
        domain_id="ml_vision",
        problem_description="Compare classifier variants",
        conditions=[
            Condition(
                name="baseline",
                role=ConditionRole.REFERENCE.value,
                description="Reference classifier",
                parameters={"depth": 18},
            ),
            Condition(
                name="proposed",
                role=ConditionRole.PROPOSED.value,
                description="New classifier",
                parameters={"depth": 34},
            ),
            Condition(
                name="proposed_no_aug",
                role=ConditionRole.VARIANT.value,
                description="No augmentation variant",
                varies_from="proposed",
                variation="disable augmentation",
                parameters={"augmentation": False},
            ),
        ],
        input_type="benchmark_dataset",
        input_description="CIFAR-like benchmark",
        evaluation=EvaluationSpec(
            primary_metric=MetricSpec(
                name="accuracy",
                direction="maximize",
                unit="fraction",
                description="Held-out accuracy",
            ),
            secondary_metrics=[
                MetricSpec(name="latency", direction="minimize", unit="ms"),
            ],
            protocol="train on train split, evaluate on holdout",
            statistical_test="bootstrap_ci",
            num_seeds=5,
        ),
        main_figure_type="line_chart",
        main_table_type="metric_table",
        raw_yaml="raw: yaml\n",
        mode="falsify",
        budget=experiment_schema.ExperimentBudget(
            wall_clock_seconds=7200,
            max_cost_usd=12.5,
        ),
        seeds=[1, 2, 3],
        prediction=experiment_schema.PreregisteredPrediction(
            statement="The proposed method improves accuracy",
            metric="accuracy",
            condition="proposed",
            baseline="baseline",
            comparison="greater_than",
            min_effect_size=0.02,
        ),
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

    def test_legacy_positional_constructor_order_is_preserved(self):
        plan = UniversalExperimentPlan("convergence", "physics_pde", "Track error")

        assert plan.experiment_type == "convergence"
        assert plan.domain_id == "physics_pde"
        assert plan.problem_description == "Track error"
        assert plan.schema_version == experiment_schema.SCHEMA_VERSION

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

    def test_v1_yaml_file_roundtrip_preserves_every_field(self, tmp_path):
        plan = _full_v1_plan()
        path = tmp_path / "experiment_spec.yaml"

        path.write_text(plan.to_yaml_v1(), encoding="utf-8")
        plan2 = UniversalExperimentPlan.from_yaml_v1(
            path.read_text(encoding="utf-8")
        )

        assert plan2 == plan
        assert plan2.to_dict() == plan.to_dict()

    def test_from_dict_rejects_unknown_top_level_keys(self):
        data = _full_v1_plan().to_dict()
        data["silently_lost"] = True
        error_type = experiment_schema.SpecValidationError

        with pytest.raises(error_type) as excinfo:
            UniversalExperimentPlan.from_dict(data)

        assert "Unknown top-level key: silently_lost" in excinfo.value.violations

    def test_validate_strict_requires_prediction_seeds_and_schema_version(self):
        plan = _full_v1_plan()
        plan.prediction = None
        plan.seeds = []
        plan.schema_version = "0.9"

        violations = plan.validate(strict=True)

        assert "prediction is required for strict v1 validation" in violations
        assert "seeds must be non-empty for strict v1 validation" in violations
        assert "schema_version must be 1.0" in violations

    def test_validate_structural_checks(self):
        plan = _full_v1_plan()
        plan.experiment_type = "unsupported"
        plan.mode = "explore"
        plan.budget.wall_clock_seconds = 0
        plan.conditions[0].role = "control"
        assert plan.prediction is not None
        plan.prediction.metric = "missing_metric"
        plan.prediction.condition = "missing_condition"
        plan.prediction.baseline = "missing_baseline"
        plan.prediction.comparison = "equal_to"

        violations = plan.validate(strict=False)

        assert "experiment_type is invalid: unsupported" in violations
        assert "mode must be 'falsify' or 'optimize'" in violations
        assert "budget.wall_clock_seconds must be > 0" in violations
        assert "condition role for baseline is invalid: control" in violations
        assert "prediction.metric is not defined in evaluation metrics: missing_metric" in violations
        assert "prediction.condition is not defined in conditions: missing_condition" in violations
        assert "prediction.baseline is not defined in conditions: missing_baseline" in violations
        assert "prediction.comparison must be 'greater_than' or 'less_than'" in violations


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

    def test_legacy_converted_plan_passes_structural_not_strict_validation(self):
        legacy = {
            "baselines": [{"name": "baseline"}],
            "proposed_methods": [{"name": "proposed"}],
            "metrics": {"accuracy": {"direction": "maximize"}},
        }
        plan = from_legacy_exp_plan(legacy)

        assert plan.validate(strict=False) == []
        assert "prediction is required for strict v1 validation" in plan.validate(
            strict=True
        )


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
