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


def from_legacy_exp_plan(
    plan_yaml: str | dict[str, Any],
    domain_id: str = "",
) -> UniversalExperimentPlan:
    """Convert a legacy exp_plan.yaml (baselines/proposed/ablations) to
    the universal format.

    This allows existing ML experiment plans to work with the new system.
    """
    if isinstance(plan_yaml, str):
        data = yaml.safe_load(plan_yaml) or {}
    else:
        data = plan_yaml

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
