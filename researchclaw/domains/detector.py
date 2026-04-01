"""Domain detection and profile loading.

Provides :class:`DomainProfile` (the canonical representation of a research
domain's experiment conventions) and :func:`detect_domain` which maps a
research topic + context to the most appropriate profile.

Detection strategy (three-level):
  1. **Keyword matching** — fast, deterministic, hits known domains.
  2. **LLM classification** — for ambiguous topics.
  3. **Hybrid resolution** — e.g. "physics-informed neural networks"
     matches both physics and ML; we pick the primary and tag secondaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).parent / "profiles"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExperimentParadigm(str, Enum):
    """High-level experiment structure used by different domains."""

    COMPARISON = "comparison"  # A vs B (ML, security)
    CONVERGENCE = "convergence"  # error vs refinement (math, physics)
    PROGRESSIVE_SPEC = "progressive_spec"  # OLS → +FE → +IV (economics)
    SIMULATION = "simulation"  # run → observe → analyze (physics)
    ABLATION_STUDY = "ablation_study"  # systematic component removal


class MetricType(str, Enum):
    SCALAR = "scalar"
    TABLE = "table"
    CONVERGENCE = "convergence"
    LEARNING_CURVE = "learning_curve"
    CONFUSION_MATRIX = "confusion"
    STRUCTURED = "structured"
    PARETO = "pareto"


# ---------------------------------------------------------------------------
# DomainProfile
# ---------------------------------------------------------------------------


@dataclass
class DomainProfile:
    """Complete description of a research domain's experiment conventions.

    Loaded from YAML files in ``researchclaw/domains/profiles/``.
    """

    # Identity
    domain_id: str  # e.g. "computational_physics"
    display_name: str  # e.g. "Computational Physics"
    parent_domain: str = ""  # e.g. "physics"

    # Experiment paradigm
    experiment_paradigm: str = ExperimentParadigm.COMPARISON.value
    condition_terminology: dict[str, str] = field(default_factory=lambda: {
        "baseline": "baseline",
        "proposed": "proposed method",
        "variant": "ablation",
        "input": "dataset",
        "metric": "accuracy/loss",
    })

    # Code structure
    typical_file_structure: dict[str, str] = field(default_factory=dict)
    entry_point: str = "main.py"

    # Dependencies & environment
    core_libraries: list[str] = field(default_factory=list)
    docker_image: str = "researchclaw/sandbox-generic:latest"
    gpu_required: bool = False
    pip_packages: list[str] = field(default_factory=list)

    # Metrics & evaluation
    metric_types: list[str] = field(default_factory=lambda: ["scalar"])
    standard_baselines: list[str] = field(default_factory=list)
    evaluation_protocol: str = ""
    statistical_tests: list[str] = field(default_factory=lambda: ["paired_t_test"])

    # Output & presentation
    output_formats: list[str] = field(default_factory=lambda: ["latex_table"])
    figure_types: list[str] = field(default_factory=lambda: ["bar_chart", "line_plot"])

    # Search keywords (for Code Searcher and literature)
    github_search_terms: list[str] = field(default_factory=list)
    paper_keywords: list[str] = field(default_factory=list)

    # Prompt guidance blocks (domain-specific instruction text)
    compute_budget_guidance: str = ""
    dataset_guidance: str = ""
    hp_reporting_guidance: str = ""
    code_generation_hints: str = ""
    result_analysis_hints: str = ""


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

_profile_cache: dict[str, DomainProfile] = {}


def _load_profile(path: Path) -> DomainProfile:
    """Load a single YAML profile into a DomainProfile."""
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    return DomainProfile(
        domain_id=data.get("domain_id", path.stem),
        display_name=data.get("display_name", path.stem.replace("_", " ").title()),
        parent_domain=data.get("parent_domain", ""),
        experiment_paradigm=data.get("experiment_paradigm", "comparison"),
        condition_terminology=data.get("condition_terminology", {}),
        typical_file_structure=data.get("typical_file_structure", {}),
        entry_point=data.get("entry_point", "main.py"),
        core_libraries=data.get("core_libraries", []),
        docker_image=data.get("docker_image", "researchclaw/sandbox-generic:latest"),
        gpu_required=data.get("gpu_required", False),
        pip_packages=data.get("pip_packages", []),
        metric_types=data.get("metric_types", ["scalar"]),
        standard_baselines=data.get("standard_baselines", []),
        evaluation_protocol=data.get("evaluation_protocol", ""),
        statistical_tests=data.get("statistical_tests", ["paired_t_test"]),
        output_formats=data.get("output_formats", ["latex_table"]),
        figure_types=data.get("figure_types", ["bar_chart", "line_plot"]),
        github_search_terms=data.get("github_search_terms", []),
        paper_keywords=data.get("paper_keywords", []),
        compute_budget_guidance=data.get("compute_budget_guidance", ""),
        dataset_guidance=data.get("dataset_guidance", ""),
        hp_reporting_guidance=data.get("hp_reporting_guidance", ""),
        code_generation_hints=data.get("code_generation_hints", ""),
        result_analysis_hints=data.get("result_analysis_hints", ""),
    )


def load_all_profiles() -> dict[str, DomainProfile]:
    """Load all YAML profiles from the profiles directory."""
    global _profile_cache
    if _profile_cache:
        return _profile_cache

    if not _PROFILES_DIR.is_dir():
        logger.warning("Profiles directory not found: %s", _PROFILES_DIR)
        return {}

    for yaml_path in sorted(_PROFILES_DIR.glob("*.yaml")):
        try:
            profile = _load_profile(yaml_path)
            _profile_cache[profile.domain_id] = profile
        except Exception:
            logger.warning("Failed to load profile %s", yaml_path, exc_info=True)

    logger.info("Loaded %d domain profiles", len(_profile_cache))
    return _profile_cache


def get_profile(domain_id: str) -> DomainProfile | None:
    """Get a specific domain profile by ID."""
    profiles = load_all_profiles()
    return profiles.get(domain_id)


def get_generic_profile() -> DomainProfile:
    """Return the generic fallback profile."""
    profile = get_profile("generic")
    if profile is not None:
        return profile
    # Hardcoded fallback if YAML not found
    return DomainProfile(
        domain_id="generic",
        display_name="Generic Computational Research",
        experiment_paradigm="comparison",
        core_libraries=["numpy", "scipy", "matplotlib", "pandas"],
        docker_image="researchclaw/sandbox-generic:latest",
    )


# ---------------------------------------------------------------------------
# Keyword-based detection rules
# ---------------------------------------------------------------------------

# Ordered list: first match wins (more specific patterns first).
_KEYWORD_RULES: list[tuple[list[str], str]] = [
    # ML sub-domains (most specific first)
    (["reinforcement learning", "rl agent", "policy gradient", "q-learning",
      "actor-critic", "reward shaping", "gymnasium", "stable-baselines"],
     "ml_rl"),
    (["knowledge distillation", "teacher-student", "model compression",
      "pruning", "quantization"], "ml_compression"),
    (["natural language", "nlp", "text classification", "sentiment",
      "language model", "transformer", "bert", "gpt", "llm", "tokeniz"],
     "ml_nlp"),
    (["object detection", "image segmentation", "image classification",
      "convolutional", "cnn", "resnet", "vit", "vision transformer",
      "computer vision", "visual"], "ml_vision"),
    (["graph neural", "gnn", "node classification", "link prediction",
      "graph convolution", "message passing"], "ml_graph"),
    (["tabular", "xgboost", "lightgbm", "catboost", "feature engineering"],
     "ml_tabular"),
    (["generative adversarial", "gan", "diffusion model", "vae",
      "variational autoencoder", "image generation"], "ml_generative"),
    # Neuroscience (before ML catch-all so "spiking neural" is not swallowed
    # by the "neural network" pattern in ml_generic)
    (["spiking neural", "spike train", "brian2", "hodgkin-huxley",
      "integrate-and-fire", "lif model", "izhikevich",
      "membrane potential", "action potential", "neural circuit",
      "neural dynamics", "population coding", "neural decoding",
      "raster plot", "firing rate", "synaptic", "connectome"],
     "neuroscience_computational"),
    (["fmri", "eeg", "meg", "neuroimaging", "brain imaging",
      "nilearn", "mne-python", "bold signal", "brain network",
      "functional connectivity"], "neuroscience_imaging"),
    (["neuroscience", "neuron model", "brain simulation",
      "neural computation", "neural encoding"], "neuroscience_computational"),

    # Catch-all ML
    (["neural network", "deep learning", "machine learning", "training loop",
      "backpropagation", "gradient descent", "pytorch", "tensorflow",
      "torch", "sklearn"], "ml_generic"),

    # Physics
    (["molecular dynamics", "n-body", "lennard-jones", "force field",
      "jax-md", "ase", "openmm"], "physics_simulation"),
    (["partial differential", "pde", "finite element", "finite difference",
      "fenics", "navier-stokes", "heat equation", "wave equation",
      "poisson", "laplace"], "physics_pde"),
    (["quantum mechanics", "schrodinger", "hamiltonian", "wavefunction",
      "density functional"], "physics_quantum"),
    (["physics", "simulation", "integrator", "conservation",
      "energy drift", "symplectic"], "physics_simulation"),

    # Chemistry
    (["quantum chemistry", "dft", "hartree-fock", "pyscf", "ccsd",
      "molecular orbital", "basis set"], "chemistry_qm"),
    (["molecular property", "smiles", "rdkit", "fingerprint", "drug",
      "binding affinity", "admet"], "chemistry_molprop"),
    (["chemistry", "molecule", "reaction", "catalyst"], "chemistry_general"),

    # Biology
    (["single-cell", "scrna", "scanpy", "anndata", "leiden",
      "differential expression", "pseudotime"], "biology_singlecell"),
    (["genomics", "genome", "variant calling", "sequencing",
      "biopython", "alignment"], "biology_genomics"),
    (["protein", "alphafold", "protein folding", "amino acid",
      "esm"], "biology_protein"),
    (["biology", "bioinformatics", "omics"], "biology_general"),

    # Economics
    (["econometrics", "regression", "instrumental variable",
      "fixed effect", "panel data", "difference-in-difference",
      "causal inference", "statsmodels", "linearmodels"],
     "economics_empirical"),
    (["economics", "economic", "market", "equilibrium",
      "utility", "welfare"], "economics_general"),

    # Mathematics
    (["numerical method", "numerical analysis", "convergence order",
      "finite difference", "quadrature", "interpolation",
      "ode solver", "runge-kutta", "sympy"], "mathematics_numerical"),
    (["optimization", "convex", "linear programming",
      "gradient-free", "evolutionary algorithm"], "mathematics_optimization"),
    (["mathematics", "mathematical", "theorem", "proof",
      "algebra", "topology"], "mathematics_general"),

    # Security
    (["intrusion detection", "malware", "anomaly detection",
      "network traffic", "cybersecurity", "vulnerability",
      "threat detection", "scapy"], "security_detection"),

    # Robotics / Control
    (["robot", "robotic", "control", "manipulation",
      "mujoco", "pybullet", "locomotion", "navigation"],
     "robotics_control"),
]


def _keyword_detect(text: str) -> str | None:
    """Match text against keyword rules. Returns domain_id or None."""
    lower = text.lower()
    for keywords, domain_id in _KEYWORD_RULES:
        for kw in keywords:
            if kw in lower:
                return domain_id
    return None


# ---------------------------------------------------------------------------
# LLM-based detection
# ---------------------------------------------------------------------------

_LLM_CLASSIFY_PROMPT = """\
You are a domain classifier for computational research topics.
Given the research topic and context, classify it into EXACTLY ONE domain.

Available domains:
- ml_vision: Computer vision (image classification, detection, segmentation)
- ml_nlp: Natural language processing (text, language models, transformers)
- ml_rl: Reinforcement learning (agents, environments, rewards)
- ml_graph: Graph neural networks (node/edge/graph tasks)
- ml_tabular: Tabular ML (XGBoost, feature engineering)
- ml_generative: Generative models (GANs, diffusion, VAE)
- ml_compression: Model compression (distillation, pruning, quantization)
- ml_generic: Other ML/AI research
- physics_simulation: Molecular dynamics, N-body, classical simulations
- physics_pde: PDE solvers (FEM, FDM, spectral methods)
- physics_quantum: Quantum mechanics, quantum chemistry
- chemistry_qm: Quantum chemistry (DFT, Hartree-Fock, PySCF)
- chemistry_molprop: Molecular property prediction (SMILES, RDKit)
- biology_singlecell: Single-cell analysis (scRNA-seq, scanpy)
- biology_genomics: Genomics (sequencing, variant calling)
- biology_protein: Protein science (folding, property prediction)
- economics_empirical: Empirical economics (regression, causal inference)
- mathematics_numerical: Numerical methods (ODE/PDE solvers, convergence)
- mathematics_optimization: Optimization (convex, evolutionary)
- security_detection: Security/intrusion detection
- neuroscience_computational: Computational neuroscience (spiking networks, neural dynamics, population coding)
- neuroscience_imaging: Brain imaging analysis (fMRI, EEG, MEG, functional connectivity)
- robotics_control: Robotics and control
- generic: Cannot classify / cross-domain

Topic: {topic}
Context: {context}

Respond with ONLY the domain_id (e.g., "ml_vision"). Nothing else."""


def _llm_detect(
    topic: str, context: str, llm: Any,
) -> str | None:
    """Use LLM to classify a research topic into a domain.

    Synchronous — ``llm.chat()`` is a blocking call.
    """
    try:
        prompt = _LLM_CLASSIFY_PROMPT.format(topic=topic, context=context)
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            system="You are a precise domain classifier.",
            max_tokens=50,
        )
        content = getattr(response, "content", None)
        if not content or not content.strip():
            logger.warning("LLM domain detection returned empty response")
            return None
        domain_id = content.strip().strip('"').strip("'").lower()
        # Validate it's a known domain
        profiles = load_all_profiles()
        if domain_id in profiles or domain_id == "generic":
            return domain_id
        # Try fuzzy match (require at least 4 chars to avoid over-matching)
        if len(domain_id) >= 4:
            for pid in profiles:
                if pid in domain_id or domain_id in pid:
                    return pid
        logger.warning("LLM returned unknown domain: %s", domain_id)
        return None
    except Exception:
        logger.warning("LLM domain detection failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_domain(
    topic: str,
    hypotheses: str = "",
    literature: str = "",
    llm: Any | None = None,
) -> DomainProfile:
    """Detect the research domain from topic and context.

    Three-level detection:
    1. Keyword matching (fast, deterministic)
    2. LLM classification (if llm is provided, for ambiguous topics)
    3. Fallback to generic profile

    Parameters
    ----------
    topic : str
        Research topic description.
    hypotheses : str
        Hypotheses text for additional context.
    literature : str
        Literature review text for additional context.
    llm : LLMClient, optional
        LLM client for classification fallback.

    Returns
    -------
    DomainProfile
        The detected domain profile.
    """
    combined_text = f"{topic} {hypotheses} {literature}"

    # Level 1: Keyword matching
    domain_id = _keyword_detect(combined_text)
    if domain_id:
        profile = get_profile(domain_id)
        if profile:
            logger.info(
                "Domain detected via keywords: %s (%s)",
                profile.display_name, domain_id,
            )
            return profile
        logger.warning(
            "Keyword matched domain_id=%s but no profile found, falling back",
            domain_id,
        )

    # Level 2: LLM classification
    if llm is not None:
        domain_id = _llm_detect(combined_text, f"hypotheses: {hypotheses}", llm)
        if domain_id:
            profile = get_profile(domain_id)
            if profile:
                logger.info(
                    "Domain detected via LLM: %s (%s)",
                    profile.display_name, domain_id,
                )
                return profile

    # Level 3: Fallback to generic
    logger.info("Using generic domain profile for topic: %.80s", topic)
    return get_generic_profile()


async def detect_domain_async(
    topic: str,
    hypotheses: str = "",
    literature: str = "",
    llm: Any | None = None,
) -> DomainProfile:
    """Async version of detect_domain with LLM classification support."""
    combined_text = f"{topic} {hypotheses} {literature}"

    # Level 1: Keyword matching
    domain_id = _keyword_detect(combined_text)
    if domain_id:
        profile = get_profile(domain_id)
        if profile:
            logger.info(
                "Domain detected via keywords: %s (%s)",
                profile.display_name, domain_id,
            )
            return profile

    # Level 2: LLM classification
    if llm is not None:
        domain_id = _llm_detect(topic, combined_text, llm)
        if domain_id:
            profile = get_profile(domain_id)
            if profile:
                logger.info(
                    "Domain detected via LLM: %s (%s)",
                    profile.display_name, domain_id,
                )
                return profile

    # Level 3: Fallback
    logger.info("Using generic domain profile for topic: %.80s", topic)
    return get_generic_profile()


def detect_domain_id(topic: str, hypotheses: str = "", literature: str = "") -> str:
    """Quick keyword-only detection that returns a domain_id string.

    Useful for lightweight checks where a full profile isn't needed.
    """
    combined = f"{topic} {hypotheses} {literature}"
    return _keyword_detect(combined) or "generic"


def is_ml_domain(domain: DomainProfile) -> bool:
    """Check if a domain profile represents an ML/AI domain."""
    return domain.domain_id.startswith("ml_") or domain.domain_id in (
        "ml_generic", "ml_vision", "ml_nlp", "ml_rl", "ml_graph",
        "ml_tabular", "ml_generative", "ml_compression",
    )
