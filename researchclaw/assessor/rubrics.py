"""Paper quality assessment rubrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rubric:
    """A single evaluation dimension rubric."""

    name: str
    criteria: str
    scale: str
    weight: float = 1.0


RUBRICS: dict[str, Rubric] = {
    "novelty": Rubric(
        name="Novelty",
        criteria="Originality of the idea. Does it propose something genuinely new?",
        scale="1=rehash, 3=incremental, 5=solid contribution, 7=novel, 10=breakthrough",
    ),
    "rigor": Rubric(
        name="Rigor",
        criteria=(
            "Scientific rigor. Are experiments well-designed? "
            "Statistical significance reported?"
        ),
        scale="1=no experiments, 3=basic, 5=adequate, 7=thorough, 10=exemplary",
    ),
    "clarity": Rubric(
        name="Clarity",
        criteria="Writing quality. Is the paper well-organized and easy to follow?",
        scale="1=incomprehensible, 3=poor, 5=adequate, 7=clear, 10=excellent",
    ),
    "impact": Rubric(
        name="Impact",
        criteria="Potential impact on the field. Will others cite/use this work?",
        scale="1=none, 3=limited, 5=moderate, 7=significant, 10=transformative",
    ),
    "experiments": Rubric(
        name="Experiments",
        criteria="Experimental sufficiency. Are baselines fair? Ablations complete?",
        scale="1=none, 3=minimal, 5=adequate, 7=comprehensive, 10=exceptional",
    ),
}
