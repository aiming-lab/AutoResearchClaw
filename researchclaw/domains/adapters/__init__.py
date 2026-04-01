"""Domain-specific prompt adapters.

Each adapter customizes prompt blocks for a specific research domain
while the ML adapter preserves existing behavior unchanged.
"""

from researchclaw.domains.adapters.ml import MLPromptAdapter
from researchclaw.domains.adapters.generic import GenericPromptAdapter
from researchclaw.domains.adapters.physics import PhysicsPromptAdapter
from researchclaw.domains.adapters.economics import EconomicsPromptAdapter
from researchclaw.domains.adapters.biology import BiologyPromptAdapter
from researchclaw.domains.adapters.chemistry import ChemistryPromptAdapter
from researchclaw.domains.adapters.neuroscience import NeurosciencePromptAdapter
from researchclaw.domains.adapters.robotics import RoboticsPromptAdapter

__all__ = [
    "MLPromptAdapter",
    "GenericPromptAdapter",
    "PhysicsPromptAdapter",
    "EconomicsPromptAdapter",
    "BiologyPromptAdapter",
    "ChemistryPromptAdapter",
    "NeurosciencePromptAdapter",
    "RoboticsPromptAdapter",
]
