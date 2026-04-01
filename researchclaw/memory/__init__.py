"""Persistent evolutionary memory system for AutoResearchClaw.

Provides three categories of memory:
- **Ideation**: Research topics, hypotheses, and their outcomes.
- **Experiment**: Hyperparameters, architectures, and training tricks.
- **Writing**: Review feedback, paper structure patterns.

Each category supports semantic retrieval via embeddings, time-decay
weighting, and confidence scoring.
"""

from researchclaw.memory.store import MemoryEntry, MemoryStore
from researchclaw.memory.retriever import MemoryRetriever
from researchclaw.memory.decay import time_decay_weight, confidence_update

__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "MemoryRetriever",
    "time_decay_weight",
    "confidence_update",
]
