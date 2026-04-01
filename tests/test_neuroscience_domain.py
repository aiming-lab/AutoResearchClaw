"""Tests for computational neuroscience domain support.

Covers profile loading, keyword detection, adapter dispatch, and
prompt block generation for neuroscience_computational and
neuroscience_imaging domains.
"""

from __future__ import annotations

import pytest

from researchclaw.domains.detector import (
    DomainProfile,
    detect_domain,
    detect_domain_id,
    get_profile,
    _keyword_detect,
    _profile_cache,
)
from researchclaw.domains.prompt_adapter import (
    MLPromptAdapter,
    PromptBlocks,
    get_adapter,
)


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


class TestNeuroscienceProfiles:
    def setup_method(self):
        _profile_cache.clear()

    def test_computational_profile_exists(self):
        profile = get_profile("neuroscience_computational")
        assert profile is not None
        assert profile.domain_id == "neuroscience_computational"
        assert profile.display_name == "Computational Neuroscience"

    def test_computational_profile_fields(self):
        profile = get_profile("neuroscience_computational")
        assert profile is not None
        assert profile.experiment_paradigm == "simulation"
        assert "brian2" in profile.core_libraries
        assert "numpy" in profile.core_libraries
        assert profile.gpu_required is False

    def test_computational_profile_baselines(self):
        profile = get_profile("neuroscience_computational")
        assert profile is not None
        assert len(profile.standard_baselines) >= 2
        assert any("LIF" in b or "Integrate-and-Fire" in b
                    for b in profile.standard_baselines)

    def test_imaging_profile_exists(self):
        profile = get_profile("neuroscience_imaging")
        assert profile is not None
        assert profile.domain_id == "neuroscience_imaging"
        assert profile.display_name == "Brain Imaging Analysis"

    def test_imaging_profile_fields(self):
        profile = get_profile("neuroscience_imaging")
        assert profile is not None
        assert profile.experiment_paradigm == "comparison"
        assert "nilearn" in profile.core_libraries
        assert "mne" in profile.core_libraries


# ---------------------------------------------------------------------------
# Keyword detection
# ---------------------------------------------------------------------------


class TestNeuroscienceKeywordDetection:
    def test_spiking_network(self):
        assert _keyword_detect("spiking neural model of cortical columns") == "neuroscience_computational"

    def test_brian2(self):
        assert _keyword_detect("network model implemented in brian2") == "neuroscience_computational"

    def test_hodgkin_huxley(self):
        assert _keyword_detect("Hodgkin-Huxley neuron model") == "neuroscience_computational"

    def test_integrate_and_fire(self):
        assert _keyword_detect("leaky integrate-and-fire model") == "neuroscience_computational"

    def test_izhikevich(self):
        assert _keyword_detect("Izhikevich neuron dynamics") == "neuroscience_computational"

    def test_neural_decoding(self):
        assert _keyword_detect("neural decoding of population coding in cortex") == "neuroscience_computational"

    def test_firing_rate(self):
        assert _keyword_detect("firing rate analysis of cortical neurons") == "neuroscience_computational"

    def test_fmri(self):
        assert _keyword_detect("fmri resting state analysis") == "neuroscience_imaging"

    def test_eeg(self):
        assert _keyword_detect("EEG classification for BCI") == "neuroscience_imaging"

    def test_nilearn(self):
        assert _keyword_detect("brain parcellation with nilearn") == "neuroscience_imaging"

    def test_mne_python(self):
        assert _keyword_detect("ERP analysis using mne-python") == "neuroscience_imaging"

    def test_generic_neuroscience(self):
        result = _keyword_detect("neuroscience of learning and memory")
        assert result == "neuroscience_computational"

    def test_detect_domain_integration(self):
        profile = detect_domain("brian2 spiking neural model of cortical microcircuits")
        assert profile.domain_id == "neuroscience_computational"

    def test_detect_domain_id_shortcut(self):
        domain_id = detect_domain_id("brian2 leaky integrate-and-fire cortical model")
        assert domain_id == "neuroscience_computational"


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------


class TestNeuroscienceAdapter:
    def test_computational_gets_neuroscience_adapter(self):
        profile = get_profile("neuroscience_computational")
        if profile is None:
            pytest.skip("neuroscience_computational profile not found")
        adapter = get_adapter(profile)
        assert not isinstance(adapter, MLPromptAdapter)
        from researchclaw.domains.adapters.neuroscience import (
            NeurosciencePromptAdapter,
        )
        assert isinstance(adapter, NeurosciencePromptAdapter)

    def test_imaging_gets_neuroscience_adapter(self):
        profile = get_profile("neuroscience_imaging")
        if profile is None:
            pytest.skip("neuroscience_imaging profile not found")
        adapter = get_adapter(profile)
        assert not isinstance(adapter, MLPromptAdapter)

    def test_code_generation_blocks_nonempty(self):
        profile = get_profile("neuroscience_computational")
        if profile is None:
            pytest.skip("neuroscience_computational profile not found")
        adapter = get_adapter(profile)
        blocks = adapter.get_code_generation_blocks({})
        assert blocks.code_generation_hints
        assert blocks.dataset_guidance
        assert blocks.output_format_guidance

    def test_experiment_design_blocks(self):
        profile = get_profile("neuroscience_computational")
        if profile is None:
            pytest.skip("neuroscience_computational profile not found")
        adapter = get_adapter(profile)
        blocks = adapter.get_experiment_design_blocks({})
        assert "neuroscience" in blocks.experiment_design_context.lower() or \
               "Computational Neuroscience" in blocks.experiment_design_context
        assert blocks.statistical_test_guidance

    def test_result_analysis_blocks(self):
        profile = get_profile("neuroscience_computational")
        if profile is None:
            pytest.skip("neuroscience_computational profile not found")
        adapter = get_adapter(profile)
        blocks = adapter.get_result_analysis_blocks({})
        assert "firing rate" in blocks.result_analysis_hints.lower()

    def test_blueprint_context(self):
        profile = get_profile("neuroscience_computational")
        if profile is None:
            pytest.skip("neuroscience_computational profile not found")
        adapter = get_adapter(profile)
        ctx = adapter.get_blueprint_context()
        # Should include file structure and libraries from the profile
        if profile.typical_file_structure:
            assert "network.py" in ctx or "neuron.py" in ctx
        if profile.core_libraries:
            assert "brian2" in ctx or "numpy" in ctx
