"""Tests for domain detection and profile loading."""

from __future__ import annotations

import pytest
from pathlib import Path

from researchclaw.domains.detector import (
    DomainProfile,
    ExperimentParadigm,
    MetricType,
    detect_domain,
    detect_domain_id,
    get_generic_profile,
    get_profile,
    is_ml_domain,
    load_all_profiles,
    _keyword_detect,
    _profile_cache,
)


# ---------------------------------------------------------------------------
# Profile loading tests
# ---------------------------------------------------------------------------


class TestProfileLoading:
    def setup_method(self):
        _profile_cache.clear()

    def test_load_all_profiles_returns_dict(self):
        profiles = load_all_profiles()
        assert isinstance(profiles, dict)
        assert len(profiles) >= 10  # currently 24 profiles

    def test_profiles_have_required_fields(self):
        profiles = load_all_profiles()
        for domain_id, profile in profiles.items():
            assert profile.domain_id == domain_id
            assert profile.display_name
            assert profile.experiment_paradigm
            assert profile.entry_point

    def test_get_profile_existing(self):
        profile = get_profile("ml_vision")
        assert profile is not None
        assert profile.domain_id == "ml_vision"
        assert profile.display_name == "Computer Vision (ML)"
        assert profile.gpu_required is True

    def test_get_profile_nonexistent(self):
        profile = get_profile("nonexistent_domain_xyz")
        assert profile is None

    def test_get_generic_profile(self):
        profile = get_generic_profile()
        assert profile.domain_id == "generic"
        assert "numpy" in profile.core_libraries

    def test_ml_profiles_exist(self):
        for domain_id in ["ml_vision", "ml_nlp", "ml_rl", "ml_generic"]:
            profile = get_profile(domain_id)
            assert profile is not None, f"Missing profile: {domain_id}"

    def test_physics_profiles_exist(self):
        for domain_id in ["physics_simulation", "physics_pde"]:
            profile = get_profile(domain_id)
            assert profile is not None, f"Missing profile: {domain_id}"

    def test_other_domain_profiles_exist(self):
        for domain_id in [
            "mathematics_numerical",
            "chemistry_qm",
            "chemistry_molprop",
            "biology_singlecell",
            "economics_empirical",
            "security_detection",
            "robotics_control",
        ]:
            profile = get_profile(domain_id)
            assert profile is not None, f"Missing profile: {domain_id}"

    def test_physics_profile_paradigm(self):
        profile = get_profile("physics_pde")
        assert profile is not None
        assert profile.experiment_paradigm == "convergence"
        assert "convergence_order_fit" in profile.statistical_tests

    def test_economics_profile_paradigm(self):
        profile = get_profile("economics_empirical")
        assert profile is not None
        assert profile.experiment_paradigm == "progressive_spec"
        assert "hausman_test" in profile.statistical_tests


# ---------------------------------------------------------------------------
# Keyword detection tests
# ---------------------------------------------------------------------------


class TestKeywordDetection:
    def test_ml_vision_keywords(self):
        assert _keyword_detect("image classification with ResNet") == "ml_vision"
        assert _keyword_detect("convolutional neural network for object detection") == "ml_vision"

    def test_ml_nlp_keywords(self):
        assert _keyword_detect("text classification using BERT") == "ml_nlp"
        assert _keyword_detect("natural language processing transformer") == "ml_nlp"

    def test_ml_rl_keywords(self):
        assert _keyword_detect("reinforcement learning policy gradient") == "ml_rl"
        assert _keyword_detect("actor-critic algorithm for robot control") == "ml_rl"

    def test_physics_keywords(self):
        assert _keyword_detect("molecular dynamics simulation with Lennard-Jones") == "physics_simulation"
        assert _keyword_detect("finite element method for Navier-Stokes equation") == "physics_pde"

    def test_chemistry_keywords(self):
        assert _keyword_detect("DFT calculation with PySCF") == "chemistry_qm"
        assert _keyword_detect("molecular property prediction using RDKit fingerprints") == "chemistry_molprop"

    def test_biology_keywords(self):
        assert _keyword_detect("single-cell RNA-seq analysis with scanpy") == "biology_singlecell"

    def test_economics_keywords(self):
        assert _keyword_detect("panel data regression with fixed effects") == "economics_empirical"
        assert _keyword_detect("instrumental variable causal inference") == "economics_empirical"

    def test_math_keywords(self):
        assert _keyword_detect("Runge-Kutta ODE solver convergence") == "mathematics_numerical"
        assert _keyword_detect("numerical analysis of quadrature methods") == "mathematics_numerical"

    def test_security_keywords(self):
        assert _keyword_detect("intrusion detection system for network traffic") == "security_detection"

    def test_robotics_keywords(self):
        assert _keyword_detect("robot manipulation with MuJoCo") == "robotics_control"

    def test_generic_ml_fallback(self):
        assert _keyword_detect("neural network training with pytorch") == "ml_generic"
        assert _keyword_detect("deep learning for regression") == "ml_generic"

    def test_unknown_topic(self):
        assert _keyword_detect("cooking recipes for italian food") is None

    def test_case_insensitive(self):
        assert _keyword_detect("IMAGE CLASSIFICATION WITH RESNET") == "ml_vision"
        assert _keyword_detect("DFT Calculation") == "chemistry_qm"


# ---------------------------------------------------------------------------
# detect_domain tests
# ---------------------------------------------------------------------------


class TestDetectDomain:
    def test_detect_ml_vision(self):
        profile = detect_domain("image classification on CIFAR-10")
        assert is_ml_domain(profile)
        assert profile.domain_id == "ml_vision"

    def test_detect_physics(self):
        profile = detect_domain("molecular dynamics simulation of Lennard-Jones fluid")
        assert profile.domain_id == "physics_simulation"
        assert not is_ml_domain(profile)

    def test_detect_with_hypotheses(self):
        profile = detect_domain(
            topic="novel numerical scheme",
            hypotheses="We propose a 4th order finite difference scheme for the Poisson equation",
        )
        assert profile.domain_id == "physics_pde"

    def test_detect_generic_fallback(self):
        profile = detect_domain("studying the behavior of abstract systems")
        assert profile.domain_id == "generic"

    def test_detect_domain_id_shortcut(self):
        domain_id = detect_domain_id("image classification")
        assert domain_id == "ml_vision"

        domain_id = detect_domain_id("cooking recipes")
        assert domain_id == "generic"


# ---------------------------------------------------------------------------
# is_ml_domain tests
# ---------------------------------------------------------------------------


class TestIsMLDomain:
    def test_ml_domains(self):
        for domain_id in ["ml_vision", "ml_nlp", "ml_rl", "ml_generic"]:
            profile = get_profile(domain_id)
            assert profile is not None
            assert is_ml_domain(profile)

    def test_non_ml_domains(self):
        for domain_id in ["physics_simulation", "chemistry_qm", "economics_empirical"]:
            profile = get_profile(domain_id)
            assert profile is not None
            assert not is_ml_domain(profile)

    def test_generic_not_ml(self):
        profile = get_generic_profile()
        assert not is_ml_domain(profile)


# ---------------------------------------------------------------------------
# DomainProfile dataclass tests
# ---------------------------------------------------------------------------


class TestDomainProfile:
    def test_default_values(self):
        profile = DomainProfile(domain_id="test", display_name="Test")
        assert profile.experiment_paradigm == ExperimentParadigm.COMPARISON.value
        assert profile.entry_point == "main.py"
        assert profile.gpu_required is False
        assert "paired_t_test" in profile.statistical_tests

    def test_custom_values(self):
        profile = DomainProfile(
            domain_id="custom",
            display_name="Custom Domain",
            experiment_paradigm="convergence",
            gpu_required=True,
            core_libraries=["numpy", "custom_lib"],
        )
        assert profile.experiment_paradigm == "convergence"
        assert profile.gpu_required is True
        assert "custom_lib" in profile.core_libraries


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_experiment_paradigm_values(self):
        assert ExperimentParadigm.COMPARISON.value == "comparison"
        assert ExperimentParadigm.CONVERGENCE.value == "convergence"
        assert ExperimentParadigm.PROGRESSIVE_SPEC.value == "progressive_spec"
        assert ExperimentParadigm.SIMULATION.value == "simulation"

    def test_metric_type_values(self):
        assert MetricType.SCALAR.value == "scalar"
        assert MetricType.TABLE.value == "table"
        assert MetricType.CONVERGENCE.value == "convergence"


# ---------------------------------------------------------------------------
# Domain detection accuracy test (50-topic benchmark)
# ---------------------------------------------------------------------------


class TestDetectionAccuracy:
    """Test domain detection accuracy on a diverse set of topics."""

    TOPIC_EXPECTATIONS = [
        # ML topics
        ("Image classification with ResNet on CIFAR-10", "ml_vision"),
        ("Object detection using YOLO", "ml_vision"),
        ("Text sentiment analysis with BERT", "ml_nlp"),
        ("Language model fine-tuning", "ml_nlp"),
        ("Reinforcement learning for Atari games", "ml_rl"),
        ("Policy gradient optimization in continuous control", "ml_rl"),
        ("Graph neural network for node classification", "ml_graph"),
        ("Knowledge distillation from large teacher models", "ml_compression"),
        ("GAN for image synthesis", "ml_generative"),
        ("Tabular data prediction with XGBoost", "ml_tabular"),
        ("Deep learning regression model", "ml_generic"),
        ("Neural network for time series forecasting", "ml_generic"),
        # Physics topics
        ("Molecular dynamics of Lennard-Jones particles", "physics_simulation"),
        ("N-body gravitational simulation", "physics_simulation"),
        ("Symplectic integrator for Hamiltonian systems", "physics_simulation"),
        ("Finite element solution of Poisson equation", "physics_pde"),
        ("Heat equation solver comparison", "physics_pde"),
        ("Navier-Stokes finite difference scheme", "physics_pde"),
        # Chemistry topics
        ("Hartree-Fock calculation for small molecules", "chemistry_qm"),
        ("DFT energy with PySCF", "chemistry_qm"),
        ("Molecular property prediction from SMILES", "chemistry_molprop"),
        ("Drug binding affinity with RDKit fingerprints", "chemistry_molprop"),
        # Biology topics
        ("Single-cell clustering with scanpy", "biology_singlecell"),
        ("scRNA-seq differential expression analysis", "biology_singlecell"),
        ("Genome variant calling pipeline", "biology_genomics"),
        ("Protein folding prediction", "biology_protein"),
        # Economics topics
        ("Panel data regression with fixed effects", "economics_empirical"),
        ("Instrumental variable estimation", "economics_empirical"),
        ("Causal inference with difference-in-differences", "economics_empirical"),
        # Math topics
        ("Runge-Kutta ODE solver convergence analysis", "mathematics_numerical"),
        ("Numerical quadrature comparison", "mathematics_numerical"),
        ("Convex optimization benchmark", "mathematics_optimization"),
        # Security topics
        ("Network intrusion detection system", "security_detection"),
        ("Malware classification using random forest", "security_detection"),
        # Robotics topics
        ("Robot manipulation policy learning", "robotics_control"),
        ("Locomotion control with MuJoCo", "robotics_control"),
    ]

    def test_keyword_detection_accuracy(self):
        """Test that keyword detection achieves > 90% accuracy."""
        correct = 0
        total = len(self.TOPIC_EXPECTATIONS)

        for topic, expected_domain in self.TOPIC_EXPECTATIONS:
            detected = _keyword_detect(topic)
            if detected == expected_domain:
                correct += 1

        accuracy = correct / total
        assert accuracy > 0.90, (
            f"Keyword detection accuracy: {accuracy:.1%} ({correct}/{total}). "
            f"Expected > 90%."
        )

    def test_full_detection_accuracy(self):
        """Test that full detect_domain achieves > 90% accuracy."""
        correct = 0
        total = len(self.TOPIC_EXPECTATIONS)

        for topic, expected_domain in self.TOPIC_EXPECTATIONS:
            profile = detect_domain(topic)
            if profile.domain_id == expected_domain:
                correct += 1

        accuracy = correct / total
        assert accuracy > 0.90, (
            f"Full detection accuracy: {accuracy:.1%} ({correct}/{total}). "
            f"Expected > 90%."
        )
