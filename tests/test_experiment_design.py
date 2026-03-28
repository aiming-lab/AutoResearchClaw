"""Tests for experiment design stage helpers."""

from __future__ import annotations

from researchclaw.pipeline.stage_impls._experiment_design import (
    _hardware_profile_context,
)


def test_hardware_profile_context_uses_generic_remote_cuda_language() -> None:
    context = _hardware_profile_context()
    assert "RTX 6000 Ada" not in context
    assert "49 GB VRAM" not in context
    assert "single-GPU" in context
    assert "remote CUDA environment" in context
