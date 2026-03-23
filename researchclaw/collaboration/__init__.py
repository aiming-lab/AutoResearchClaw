"""Agent collaboration and knowledge sharing system.

Enables multiple AutoResearchClaw instances to share research artifacts
(literature summaries, experiment results, code templates, review feedback)
through a file-system-based shared repository.
"""

from researchclaw.collaboration.repository import ResearchRepository
from researchclaw.collaboration.publisher import ArtifactPublisher
from researchclaw.collaboration.subscriber import ArtifactSubscriber
from researchclaw.collaboration.dedup import deduplicate_artifacts

__all__ = [
    "ResearchRepository",
    "ArtifactPublisher",
    "ArtifactSubscriber",
    "deduplicate_artifacts",
]
