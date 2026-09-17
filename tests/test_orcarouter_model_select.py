"""Model selector options: capability filtering, invalidation, degradation.

The requirement these tests exist for: when the provider, the entry point,
or the attachment type changes, the *options handed to the selector* change
— and an option that is no longer compatible is cleared rather than kept.
A pre-send guard is not a substitute.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.llm import orcarouter as orca
from researchclaw.llm.model_select import (
    MODALITY_IMAGE,
    apply_selection,
    build_model_selector,
    capability_for_entry_point,
    is_orcarouter,
    model_options_for,
    required_modalities_for_attachments,
    resolve_primary_model,
)

FAKE_KEY = "sk-orca-fake-0000000000000000000001"

LIVE = {
    "data": [
        {
            "id": "deepseek/deepseek-v4-pro",
            "supported_endpoint_types": ["openai"],
            "architecture": {"input_modalities": ["text"]},
            "context_length": 1048576,
        },
        {
            "id": "deepseek/deepseek-v4.1-flash",
            "supported_endpoint_types": ["openai", "anthropic"],
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {
            "id": "vendor/nano-banana",
            "supported_endpoint_types": ["image-generation"],
        },
        {
            "id": "vendor/text-embed-3",
            "supported_endpoint_types": ["embeddings"],
        },
    ]
}


def _config(provider: str = "orcarouter", **llm_overrides: Any):
    llm = dict(
        provider=provider,
        api_key=FAKE_KEY,
        api_key_env="ORCAROUTER_API_KEY",
        base_url="",
        primary_model="",
        fallback_models=(),
    )
    llm.update(llm_overrides)
    return SimpleNamespace(llm=SimpleNamespace(**llm))


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    for var in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _fetcher(payload: Any):
    def _fetch(url: str, api_key: str, timeout: float, max_bytes: int):
        return payload

    return _fetch


# --------------------------------------------------------------------------
# Provider awareness — other providers are untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "openrouter", "anthropic", "ollama", "acp"])
def test_non_orcarouter_providers_get_no_override(provider: str) -> None:
    assert is_orcarouter(_config(provider)) is False
    assert build_model_selector(_config(provider)) is None
    assert model_options_for(_config(provider)) is None


def test_both_orcarouter_entries_are_recognised() -> None:
    for provider in ("orcarouter", "orcarouter-oauth"):
        assert is_orcarouter(_config(provider)) is True
        state = build_model_selector(_config(provider), fetcher=_fetcher(LIVE))
        assert state is not None
        assert state.provider == provider


# --------------------------------------------------------------------------
# Entry-point mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry_point", "capability"),
    [
        ("chat", "chat"),
        ("agent", "chat"),
        ("code", "chat"),
        ("review", "chat"),
        ("debate", "chat"),
        ("embedding", "embedding"),
        ("rag", "embedding"),
        ("image", "image"),
        ("figure", "image"),
        ("video", "video"),
        ("rerank", "rerank"),
    ],
)
def test_entry_points_map_to_capabilities(entry_point: str, capability: str) -> None:
    assert capability_for_entry_point(entry_point) == capability


def test_unknown_entry_point_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError):
        capability_for_entry_point("telepathy")


# --------------------------------------------------------------------------
# Attachment-driven filtering
# --------------------------------------------------------------------------


def test_attachment_modalities_are_hard_requirements() -> None:
    assert required_modalities_for_attachments([]) == ()
    assert required_modalities_for_attachments(["text"]) == ()
    assert required_modalities_for_attachments(["image"]) == (MODALITY_IMAGE,)
    assert required_modalities_for_attachments(["image", "image"]) == (MODALITY_IMAGE,)
    assert required_modalities_for_attachments(["image", "audio"]) == ("image", "audio")
    # An unknown attachment kind cannot be satisfied by anything.
    assert required_modalities_for_attachments(["hologram"]) == ("hologram",)


def test_options_come_from_the_api_not_a_handwritten_list() -> None:
    state = build_model_selector(_config(), fetcher=_fetcher(LIVE))
    assert state.source == "live"
    assert state.ids == [
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4.1-flash",
    ]
    assert state.catalog_source_url.endswith("/v1/models?capability=chat")
    # A catalogue model only exists here because the API returned it.
    assert "vendor/nano-banana" not in state.ids
    assert state.is_free_text is False


def test_adding_an_image_attachment_removes_undeclared_models() -> None:
    before = build_model_selector(_config(), fetcher=_fetcher(LIVE))
    assert "deepseek/deepseek-v4-pro" in before.ids

    after = build_model_selector(
        _config(),
        attachments=["image"],
        current_model="deepseek/deepseek-v4-pro",
        fetcher=_fetcher(LIVE),
    )
    assert after.ids == ["deepseek/deepseek-v4.1-flash"]
    # The text model that was selected is cleared, with a reason.
    assert after.selection_cleared is True
    assert after.selected == ""
    assert "deepseek/deepseek-v4-pro" in after.clear_reason
    assert "image" in after.clear_reason


def test_switching_entry_point_recomputes_the_options() -> None:
    chat = build_model_selector(_config(), entry_point="chat", fetcher=_fetcher(LIVE))
    embedding = build_model_selector(
        _config(), entry_point="embedding", fetcher=_fetcher(LIVE)
    )
    image = build_model_selector(_config(), entry_point="image", fetcher=_fetcher(LIVE))

    assert chat.ids == ["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4.1-flash"]
    assert embedding.ids == ["vendor/text-embed-3"]
    assert image.ids == ["vendor/nano-banana"]


def test_switching_provider_recomputes_the_options() -> None:
    api = build_model_selector(_config("orcarouter"), fetcher=_fetcher(LIVE))
    oauth = build_model_selector(_config("orcarouter-oauth"), fetcher=_fetcher(LIVE))
    assert api.ids == oauth.ids  # one catalogue, two credential entries
    assert api.provider != oauth.provider

    other = build_model_selector(_config("openrouter"), fetcher=_fetcher(LIVE))
    assert other is None, "another provider's control must not be replaced"


def test_a_still_compatible_selection_is_kept() -> None:
    state = build_model_selector(
        _config(),
        current_model="deepseek/deepseek-v4.1-flash",
        fetcher=_fetcher(LIVE),
    )
    assert state.selected == "deepseek/deepseek-v4.1-flash"
    assert state.selection_cleared is False


def test_a_model_that_vanished_from_the_catalogue_is_cleared() -> None:
    state = build_model_selector(
        _config(),
        current_model="vendor/retired-model",
        fetcher=_fetcher(LIVE),
    )
    assert state.selected == ""
    assert state.selection_cleared is True
    assert "not available" in state.clear_reason


def test_apply_selection_is_idempotent() -> None:
    state = build_model_selector(_config(), fetcher=_fetcher(LIVE))
    apply_selection(state, "")
    assert state.selected == ""
    assert state.selection_cleared is False


# --------------------------------------------------------------------------
# Degradation: still a real list, never free text, never an unverified example
# --------------------------------------------------------------------------


def test_catalogue_failure_offers_only_the_labelled_verified_fallback() -> None:
    def _explode(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.URLError("down")

    state = build_model_selector(_config(), fetcher=_explode)
    assert state.degraded is True
    assert state.source == "seed"
    assert state.options, "an outage must not leave the selector empty"
    assert all(option.verified for option in state.options)
    assert "orcarouter/auto" in state.ids
    # And it is still a list, not a text box.
    assert state.is_free_text is False


def test_the_fallback_survives_attachment_filtering() -> None:
    def _explode(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.URLError("down")

    state = build_model_selector(_config(), attachments=["image"], fetcher=_explode)
    assert state.degraded is True
    assert state.ids, "documented image-capable seed entries must survive"
    assert all(
        "image" in option.input_modalities for option in state.options
    ), "fail closed: nothing without a declared image input may appear"


def test_no_credential_degrades_without_leaking_anything() -> None:
    state = build_model_selector(_config(api_key="", api_key_env=""), fetcher=_fetcher(LIVE))
    assert state.degraded is True
    assert state.source == "seed"
    assert FAKE_KEY not in json.dumps(state.as_dict())


def test_degraded_state_is_reported_so_the_ui_can_say_so() -> None:
    def _explode(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.URLError("down")

    payload = build_model_selector(_config(), fetcher=_explode).as_dict()
    assert payload["degraded"] is True
    assert payload["source"] == "seed"
    assert payload["error"]
    assert payload["options"]


def test_options_payload_carries_no_credential() -> None:
    payload = json.dumps(build_model_selector(_config(), fetcher=_fetcher(LIVE)).as_dict())
    assert FAKE_KEY not in payload
    assert "authorization" not in payload.lower()
    assert "api_key" not in payload.lower()


# --------------------------------------------------------------------------
# Default model resolution
# --------------------------------------------------------------------------


def test_default_model_prefers_the_routing_alias_then_the_configured_value() -> None:
    from researchclaw.llm.orcarouter_catalog import parse_models

    models = parse_models(LIVE)
    assert resolve_primary_model(_config(), models) == "deepseek/deepseek-v4-pro"
    configured = _config(primary_model="deepseek/deepseek-v4.1-flash")
    assert resolve_primary_model(configured, models) == "deepseek/deepseek-v4.1-flash"


def test_default_model_never_invents_an_id() -> None:
    from researchclaw.llm.orcarouter_catalog import parse_models

    models = parse_models({"data": [{"id": "orcarouter/auto"}]})
    assert resolve_primary_model(_config(primary_model="ghost/model"), models) == "orcarouter/auto"
    assert resolve_primary_model(_config(), []) == ""


def test_client_without_a_configured_model_uses_the_catalogue() -> None:
    """An empty primary_model resolves from the live catalogue, not a guess."""
    from researchclaw.llm import create_llm_client

    import researchclaw.llm.orcarouter_catalog as catalog_mod

    original = catalog_mod._default_fetcher
    catalog_mod._default_fetcher = _fetcher(LIVE)
    try:
        client = create_llm_client(_config(primary_model=""))
    finally:
        catalog_mod._default_fetcher = original

    assert client.config.base_url == "https://api.orcarouter.ai/v1"
    assert client.config.primary_model in {m["id"] for m in LIVE["data"]}
