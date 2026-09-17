"""Live check against OrcaRouter, driven through the shipped provider path.

Requires ``ORCAROUTER_API_KEY``; without it the test skips rather than fails,
so the ordinary suite stays green on machines that have no credential. The
work itself lives in ``scripts/orcarouter_live_check.py`` (same code a
maintainer can run by hand) so nothing about the wiring is duplicated here.

The assertion is deliberately about *the implementation*: the catalogue the
model selector is given must come from the live ``/v1/models`` response, the
multimodal selector must be a declared-modality subset of it, and a real chat
completion must succeed through ``create_llm_client``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "orcarouter_live_check.py"


@pytest.fixture(scope="module")
def report() -> dict:
    if not os.environ.get("ORCAROUTER_API_KEY"):
        pytest.skip("ORCAROUTER_API_KEY is not set — no live OrcaRouter run")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"live check failed (exit {proc.returncode}):\n"
        f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    return json.loads(proc.stdout)


def test_origins_are_the_documented_ones(report: dict) -> None:
    assert report["auth_origin"] == "https://www.orcarouter.ai"
    assert report["inference_origin"] == "https://api.orcarouter.ai/v1"
    assert report["inference_url"] == "https://api.orcarouter.ai/v1/chat/completions"


def test_catalogue_is_live_and_builds_the_selector(report: dict) -> None:
    assert report["catalog_backend"] == "live"
    assert report["catalog_source"] == "https://api.orcarouter.ai/v1/models?capability=chat"
    assert report["chat_model_count"] > 0
    assert report["selector_matches_catalog"] is True
    assert report["selector_option_count"] == report["chat_model_count"]
    # Every option is namespaced `vendor/model`, never a hand-written example.
    assert all("/" in model_id for model_id in report["chat_models"])


def test_multimodal_options_are_declared_modality_only(report: dict) -> None:
    assert report["multimodal_is_subset"] is True
    assert report["multimodal_declared_only"] is True
    assert set(report["image_capable_chat_models"]) <= set(report["chat_models"])


def test_a_real_completion_succeeds_through_the_provider(report: dict) -> None:
    assert report["primary_is_from_catalog"] is True
    assert report["inference_ok"] is True
    assert report["inference_reply"].strip(), "the model returned no visible text"


def test_no_credential_leaks_into_the_report(report: dict) -> None:
    rendered = json.dumps(report)
    key = os.environ.get("ORCAROUTER_API_KEY", "")
    assert key and key not in rendered
    assert "code_verifier" not in rendered
