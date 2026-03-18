"""Universal cross-domain research code generation framework.

This package provides domain detection, prompt adaptation, and experiment
schema generalization so the pipeline can generate code for any
computational research domain — not just ML/AI.
"""

from researchclaw.domains.detector import DomainProfile, detect_domain

__all__ = ["DomainProfile", "detect_domain"]
