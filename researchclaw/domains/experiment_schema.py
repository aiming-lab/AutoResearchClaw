"""Universal experiment schema — domain-agnostic experiment plan structure.

Replaces the fixed ``baselines/proposed_methods/ablations`` keys with a
generic ``conditions`` list that uses role-based terminology, adaptable
to any research domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

SCHEMA_VERSION = "1.0"


class ConditionRole(str, Enum):
    """Role of an experimental condition."""
    REFERENCE = "reference"  # baseline / reference solver / standard pipeline
    PROPOSED = "proposed"  # the method being investigated
    VARIANT = "variant"  # ablation / parameter variation / robustness check


class ExperimentType(str, Enum):
    COMPARISON = "comparison"
    CONVERGENCE = "convergence"
    PROGRESSIVE_SPEC = "progressive_spec"
    SIMULATION = "simulation"
    ABLATION_STUDY = "ablation_study"


@dataclass
class Condition:
    """A single experimental condition (method, configuration, etc.)."""
    name: str
    role: str = ConditionRole.PROPOSED.value
    description: str = ""
    varies_from: str = ""  # parent condition for variants
    variation: str = ""  # what is varied
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSpec:
    """Specification of a metric to evaluate."""
    name: str
    direction: str = "minimize"  # "minimize" | "maximize"
    unit: str = ""
    description: str = ""


@dataclass
class EvaluationSpec:
    """Evaluation protocol for the experiment."""
    primary_metric: MetricSpec = field(default_factory=lambda: MetricSpec(name="primary_metric"))
    secondary_metrics: list[MetricSpec] = field(default_factory=list)
    protocol: str = ""
    statistical_test: str = "paired_t_test"
    num_seeds: int = 3


@dataclass
class ExperimentBudget:
    """Execution budget for an experiment."""
    wall_clock_seconds: int = 3600
    max_cost_usd: float | None = None


@dataclass
class PreregisteredPrediction:
    """Machine-checkable preregistered prediction."""
    statement: str
    metric: str
    condition: str
    baseline: str
    comparison: str
    min_effect_size: float = 0.0


class SpecValidationError(ValueError):
    """Raised when an experiment spec cannot be parsed as v1."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


@dataclass
class UniversalExperimentPlan:
    """Domain-agnostic experiment plan.

    This can represent ML train-eval, physics convergence studies,
    economics regression tables, and any other paradigm.
    """

    experiment_type: str = ExperimentType.COMPARISON.value
    domain_id: str = ""
    problem_description: str = ""

    # Conditions (replaces baselines / proposed_methods / ablations)
    conditions: list[Condition] = field(default_factory=list)

    # Inputs
    input_type: str = "generated"  # "benchmark_dataset" | "generated" | "loaded"
    input_description: str = ""

    # Evaluation
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)

    # Presentation
    main_figure_type: str = "bar_chart"
    main_table_type: str = "comparison_table"

    # Raw YAML (for backward compatibility with existing pipeline)
    raw_yaml: str = ""

    # ExperimentSpec v1 contract fields
    schema_version: str = SCHEMA_VERSION
    mode: str = "falsify"
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)
    seeds: list[int] = field(default_factory=list)
    prediction: PreregisteredPrediction | None = None

    @property
    def references(self) -> list[Condition]:
        """Get conditions with 'reference' role (baselines)."""
        return [c for c in self.conditions if c.role == ConditionRole.REFERENCE.value]

    @property
    def proposed(self) -> list[Condition]:
        """Get conditions with 'proposed' role."""
        return [c for c in self.conditions if c.role == ConditionRole.PROPOSED.value]

    @property
    def variants(self) -> list[Condition]:
        """Get conditions with 'variant' role (ablations)."""
        return [c for c in self.conditions if c.role == ConditionRole.VARIANT.value]

    def to_legacy_format(self) -> dict[str, Any]:
        """Convert to legacy baselines/proposed_methods/ablations format.

        This allows the universal plan to be consumed by existing pipeline
        code that expects the old key names.
        """
        baselines = [
            {"name": c.name, "description": c.description}
            for c in self.references
        ]
        proposed = [
            {"name": c.name, "description": c.description}
            for c in self.proposed
        ]
        ablations = [
            {
                "name": c.name,
                "description": c.description,
                "varies_from": c.varies_from,
                "variation": c.variation,
            }
            for c in self.variants
        ]

        return {
            "baselines": baselines,
            "proposed_methods": proposed,
            "ablations": ablations,
            "metrics": {
                self.evaluation.primary_metric.name: {
                    "direction": self.evaluation.primary_metric.direction,
                }
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the canonical ExperimentSpec v1 dict shape."""
        return {
            "schema_version": self.schema_version,
            "experiment_type": self.experiment_type,
            "domain_id": self.domain_id,
            "problem_description": self.problem_description,
            "conditions": [
                {
                    "name": c.name,
                    "role": c.role,
                    "description": c.description,
                    "varies_from": c.varies_from,
                    "variation": c.variation,
                    "parameters": dict(c.parameters),
                }
                for c in self.conditions
            ],
            "input_type": self.input_type,
            "input_description": self.input_description,
            "evaluation": {
                "primary_metric": {
                    "name": self.evaluation.primary_metric.name,
                    "direction": self.evaluation.primary_metric.direction,
                    "unit": self.evaluation.primary_metric.unit,
                    "description": self.evaluation.primary_metric.description,
                },
                "secondary_metrics": [
                    {
                        "name": m.name,
                        "direction": m.direction,
                        "unit": m.unit,
                        "description": m.description,
                    }
                    for m in self.evaluation.secondary_metrics
                ],
                "protocol": self.evaluation.protocol,
                "statistical_test": self.evaluation.statistical_test,
                "num_seeds": self.evaluation.num_seeds,
            },
            "main_figure_type": self.main_figure_type,
            "main_table_type": self.main_table_type,
            "raw_yaml": self.raw_yaml,
            "mode": self.mode,
            "budget": {
                "wall_clock_seconds": self.budget.wall_clock_seconds,
                "max_cost_usd": self.budget.max_cost_usd,
            },
            "seeds": list(self.seeds),
            "prediction": (
                {
                    "statement": self.prediction.statement,
                    "metric": self.prediction.metric,
                    "condition": self.prediction.condition,
                    "baseline": self.prediction.baseline,
                    "comparison": self.prediction.comparison,
                    "min_effect_size": self.prediction.min_effect_size,
                }
                if self.prediction is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniversalExperimentPlan":
        """Load a canonical ExperimentSpec v1 dict."""
        if not isinstance(data, dict):
            raise SpecValidationError(["spec must be a mapping"])

        allowed_keys = {
            "schema_version",
            "experiment_type",
            "domain_id",
            "problem_description",
            "conditions",
            "input_type",
            "input_description",
            "evaluation",
            "main_figure_type",
            "main_table_type",
            "raw_yaml",
            "mode",
            "budget",
            "seeds",
            "prediction",
        }
        unknown_keys = sorted(set(data) - allowed_keys)
        if unknown_keys:
            raise SpecValidationError(
                [f"Unknown top-level key: {key}" for key in unknown_keys]
            )

        violations: list[str] = []

        conditions = _conditions_from_dict(data.get("conditions", []), violations)
        evaluation = _evaluation_from_dict(data.get("evaluation", {}), violations)
        budget = _budget_from_dict(data.get("budget", {}), violations)
        prediction = _prediction_from_dict(data.get("prediction"), violations)
        seeds = data.get("seeds", [])
        if not isinstance(seeds, list):
            violations.append("seeds must be a list")
            seeds = []
        converted_seeds: list[int] = []
        for idx, seed in enumerate(seeds):
            try:
                converted_seeds.append(int(seed))
            except (TypeError, ValueError):
                violations.append(f"seeds[{idx}] must be an integer")

        if violations:
            raise SpecValidationError(violations)

        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            experiment_type=str(
                data.get("experiment_type", ExperimentType.COMPARISON.value)
            ),
            domain_id=str(data.get("domain_id", "")),
            problem_description=str(data.get("problem_description", "")),
            conditions=conditions,
            input_type=str(data.get("input_type", "generated")),
            input_description=str(data.get("input_description", "")),
            evaluation=evaluation,
            main_figure_type=str(data.get("main_figure_type", "bar_chart")),
            main_table_type=str(data.get("main_table_type", "comparison_table")),
            raw_yaml=str(data.get("raw_yaml", "")),
            mode=str(data.get("mode", "falsify")),
            budget=budget,
            seeds=converted_seeds,
            prediction=prediction,
        )

    def to_yaml_v1(self) -> str:
        """Serialize to canonical ExperimentSpec v1 YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml_v1(cls, text: str) -> "UniversalExperimentPlan":
        """Load canonical ExperimentSpec v1 YAML."""
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise SpecValidationError([f"invalid YAML: {exc}"]) from exc
        return cls.from_dict(data)

    def validate(self, strict: bool = True) -> list[str]:
        """Return validation violations for this ExperimentSpec."""
        violations: list[str] = []

        if not self.conditions:
            violations.append("at least one condition is required")

        valid_roles = {role.value for role in ConditionRole}
        for condition in self.conditions:
            role = (
                condition.role.value
                if isinstance(condition.role, ConditionRole)
                else condition.role
            )
            if role not in valid_roles:
                violations.append(
                    f"condition role for {condition.name} is invalid: {role}"
                )

        valid_experiment_types = {item.value for item in ExperimentType}
        experiment_type = (
            self.experiment_type.value
            if isinstance(self.experiment_type, ExperimentType)
            else self.experiment_type
        )
        if experiment_type not in valid_experiment_types:
            violations.append(f"experiment_type is invalid: {self.experiment_type}")

        if self.mode not in {"falsify", "optimize"}:
            violations.append("mode must be 'falsify' or 'optimize'")

        if self.budget.wall_clock_seconds <= 0:
            violations.append("budget.wall_clock_seconds must be > 0")

        if self.prediction is not None:
            metric_names = {
                self.evaluation.primary_metric.name,
                *(metric.name for metric in self.evaluation.secondary_metrics),
            }
            condition_names = {condition.name for condition in self.conditions}
            if self.prediction.metric not in metric_names:
                violations.append(
                    "prediction.metric is not defined in evaluation metrics: "
                    f"{self.prediction.metric}"
                )
            if self.prediction.condition not in condition_names:
                violations.append(
                    "prediction.condition is not defined in conditions: "
                    f"{self.prediction.condition}"
                )
            if self.prediction.baseline not in condition_names:
                violations.append(
                    "prediction.baseline is not defined in conditions: "
                    f"{self.prediction.baseline}"
                )
            if self.prediction.comparison not in {"greater_than", "less_than"}:
                violations.append(
                    "prediction.comparison must be 'greater_than' or 'less_than'"
                )

        if strict:
            if self.prediction is None:
                violations.append("prediction is required for strict v1 validation")
            if not self.seeds:
                violations.append("seeds must be non-empty for strict v1 validation")
            if self.schema_version != SCHEMA_VERSION:
                violations.append(f"schema_version must be {SCHEMA_VERSION}")

        return violations

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data: dict[str, Any] = {
            "experiment": {
                "type": self.experiment_type,
                "domain": self.domain_id,
                "problem": {"description": self.problem_description},
                "conditions": [
                    {
                        "name": c.name,
                        "role": c.role,
                        "description": c.description,
                        **({"varies_from": c.varies_from} if c.varies_from else {}),
                        **({"variation": c.variation} if c.variation else {}),
                    }
                    for c in self.conditions
                ],
                "inputs": {
                    "type": self.input_type,
                    "description": self.input_description,
                },
                "evaluation": {
                    "primary_metric": {
                        "name": self.evaluation.primary_metric.name,
                        "direction": self.evaluation.primary_metric.direction,
                    },
                    "protocol": self.evaluation.protocol,
                    "statistical_test": self.evaluation.statistical_test,
                },
                "presentation": {
                    "main_figure": self.main_figure_type,
                    "main_table": self.main_table_type,
                },
            }
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _reject_unknown_keys(
    name: str,
    data: dict[str, Any],
    allowed_keys: set[str],
    violations: list[str],
) -> None:
    for key in sorted(set(data) - allowed_keys):
        violations.append(f"Unknown {name} key: {key}")


def _metric_from_dict(data: Any, name: str, violations: list[str]) -> MetricSpec:
    if not isinstance(data, dict):
        violations.append(f"{name} must be a mapping")
        return MetricSpec(name="")

    _reject_unknown_keys(
        name,
        data,
        {"name", "direction", "unit", "description"},
        violations,
    )
    return MetricSpec(
        name=str(data.get("name", "")),
        direction=str(data.get("direction", "minimize")),
        unit=str(data.get("unit", "")),
        description=str(data.get("description", "")),
    )


def _conditions_from_dict(data: Any, violations: list[str]) -> list[Condition]:
    if not isinstance(data, list):
        violations.append("conditions must be a list")
        return []

    conditions: list[Condition] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            violations.append(f"conditions[{idx}] must be a mapping")
            continue
        _reject_unknown_keys(
            f"conditions[{idx}]",
            item,
            {"name", "role", "description", "varies_from", "variation", "parameters"},
            violations,
        )
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            violations.append(f"conditions[{idx}].parameters must be a mapping")
            parameters = {}
        conditions.append(
            Condition(
                name=str(item.get("name", "")),
                role=str(item.get("role", ConditionRole.PROPOSED.value)),
                description=str(item.get("description", "")),
                varies_from=str(item.get("varies_from", "")),
                variation=str(item.get("variation", "")),
                parameters=dict(parameters),
            )
        )
    return conditions


def _evaluation_from_dict(data: Any, violations: list[str]) -> EvaluationSpec:
    if not isinstance(data, dict):
        violations.append("evaluation must be a mapping")
        return EvaluationSpec()

    _reject_unknown_keys(
        "evaluation",
        data,
        {
            "primary_metric",
            "secondary_metrics",
            "protocol",
            "statistical_test",
            "num_seeds",
        },
        violations,
    )
    primary_metric = _metric_from_dict(
        data.get("primary_metric", {"name": "primary_metric"}),
        "evaluation.primary_metric",
        violations,
    )
    secondary_data = data.get("secondary_metrics", [])
    if not isinstance(secondary_data, list):
        violations.append("evaluation.secondary_metrics must be a list")
        secondary_data = []
    secondary_metrics = [
        _metric_from_dict(metric, f"evaluation.secondary_metrics[{idx}]", violations)
        for idx, metric in enumerate(secondary_data)
    ]

    try:
        num_seeds = int(data.get("num_seeds", 3))
    except (TypeError, ValueError):
        violations.append("evaluation.num_seeds must be an integer")
        num_seeds = 3

    return EvaluationSpec(
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        protocol=str(data.get("protocol", "")),
        statistical_test=str(data.get("statistical_test", "paired_t_test")),
        num_seeds=num_seeds,
    )


def _budget_from_dict(data: Any, violations: list[str]) -> ExperimentBudget:
    if not isinstance(data, dict):
        violations.append("budget must be a mapping")
        return ExperimentBudget()

    _reject_unknown_keys(
        "budget",
        data,
        {"wall_clock_seconds", "max_cost_usd"},
        violations,
    )
    max_cost_usd = data.get("max_cost_usd")
    try:
        wall_clock_seconds = int(data.get("wall_clock_seconds", 3600))
    except (TypeError, ValueError):
        violations.append("budget.wall_clock_seconds must be an integer")
        wall_clock_seconds = 3600
    try:
        converted_max_cost = (
            float(max_cost_usd) if max_cost_usd is not None else None
        )
    except (TypeError, ValueError):
        violations.append("budget.max_cost_usd must be numeric or null")
        converted_max_cost = None
    return ExperimentBudget(
        wall_clock_seconds=wall_clock_seconds,
        max_cost_usd=converted_max_cost,
    )


def _prediction_from_dict(
    data: Any,
    violations: list[str],
) -> PreregisteredPrediction | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        violations.append("prediction must be a mapping or null")
        return None

    _reject_unknown_keys(
        "prediction",
        data,
        {
            "statement",
            "metric",
            "condition",
            "baseline",
            "comparison",
            "min_effect_size",
        },
        violations,
    )
    try:
        min_effect_size = float(data.get("min_effect_size", 0.0))
    except (TypeError, ValueError):
        violations.append("prediction.min_effect_size must be numeric")
        min_effect_size = 0.0
    return PreregisteredPrediction(
        statement=str(data.get("statement", "")),
        metric=str(data.get("metric", "")),
        condition=str(data.get("condition", "")),
        baseline=str(data.get("baseline", "")),
        comparison=str(data.get("comparison", "")),
        min_effect_size=min_effect_size,
    )


def from_legacy_exp_plan(
    plan_yaml: str | dict[str, Any],
    domain_id: str = "",
) -> UniversalExperimentPlan:
    """Convert a legacy exp_plan.yaml (baselines/proposed/ablations) to
    the universal format.

    This allows existing ML experiment plans to work with the new system.
    """
    if isinstance(plan_yaml, str):
        try:
            data = yaml.safe_load(plan_yaml) or {}
        except yaml.YAMLError as exc:
            raise SpecValidationError([f"invalid legacy YAML: {exc}"]) from exc
    else:
        data = plan_yaml
    if not isinstance(data, dict):
        raise SpecValidationError(["legacy exp_plan must be a mapping"])

    conditions: list[Condition] = []

    # Parse baselines → reference
    for b in data.get("baselines", []):
        if isinstance(b, str):
            conditions.append(Condition(name=b, role=ConditionRole.REFERENCE.value))
        elif isinstance(b, dict):
            conditions.append(Condition(
                name=b.get("name", "baseline"),
                role=ConditionRole.REFERENCE.value,
                description=b.get("description", ""),
            ))

    # Parse proposed_methods → proposed
    for p in data.get("proposed_methods", []):
        if isinstance(p, str):
            conditions.append(Condition(name=p, role=ConditionRole.PROPOSED.value))
        elif isinstance(p, dict):
            conditions.append(Condition(
                name=p.get("name", "proposed"),
                role=ConditionRole.PROPOSED.value,
                description=p.get("description", ""),
            ))

    # Parse ablations → variant
    for a in data.get("ablations", []):
        if isinstance(a, str):
            conditions.append(Condition(name=a, role=ConditionRole.VARIANT.value))
        elif isinstance(a, dict):
            conditions.append(Condition(
                name=a.get("name", "ablation"),
                role=ConditionRole.VARIANT.value,
                description=a.get("description", ""),
                varies_from=a.get("varies_from", ""),
                variation=a.get("variation", ""),
            ))

    # Parse metrics
    metrics = data.get("metrics", {})
    primary_name = "primary_metric"
    primary_direction = "minimize"
    if isinstance(metrics, dict):
        for name, spec in metrics.items():
            primary_name = name
            if isinstance(spec, dict):
                primary_direction = spec.get("direction", "minimize")
            break
    elif isinstance(metrics, list) and metrics:
        primary_name = metrics[0] if isinstance(metrics[0], str) else "primary_metric"

    return UniversalExperimentPlan(
        experiment_type=data.get("experiment_type", "comparison"),
        domain_id=domain_id,
        problem_description=data.get("objective", ""),
        conditions=conditions,
        evaluation=EvaluationSpec(
            primary_metric=MetricSpec(name=primary_name, direction=primary_direction),
        ),
        raw_yaml=yaml.dump(data, default_flow_style=False) if isinstance(data, dict) else str(plan_yaml),
    )
