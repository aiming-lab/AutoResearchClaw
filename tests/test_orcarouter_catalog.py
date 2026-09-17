"""OrcaRouter model catalogue: parsing, capability filtering, degradation.

Every fixture here is a real shape returned by
``GET https://api.orcarouter.ai/v1/models`` on 2026-09-16 (trimmed), so the
filtering rules are tested against the actual payload rather than a guess.
No network access, no credentials.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from researchclaw.llm.orcarouter_catalog import (
    CAPABILITY_CHAT,
    CAPABILITY_EMBEDDING,
    CAPABILITY_IMAGE,
    CAPABILITY_RERANK,
    CAPABILITY_VIDEO,
    VERIFIED_SEED,
    CatalogModel,
    catalog_url,
    discover_models,
    filter_models,
    is_model_available,
    matches_capability,
    merge_verified_metadata,
    parse_models,
    read_cache,
    seed_for_capability,
    write_cache,
)

# --------------------------------------------------------------------------
# Fixtures shaped like the live catalogue
# --------------------------------------------------------------------------

TEXT_ONLY_CHAT = {
    "id": "deepseek/deepseek-v4-pro",
    "object": "model",
    "owned_by": "deepseek",
    "supported_endpoint_types": ["openai", "openai-response"],
    "context_length": 1048576,
    "max_completion_tokens": 384000,
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
}

IMAGE_INPUT_CHAT = {
    "id": "deepseek/deepseek-v4.1-flash",
    "object": "model",
    "owned_by": "deepseek",
    "supported_endpoint_types": ["openai", "openai-response", "anthropic"],
    "context_length": 1048576,
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
}

AUDIO_INPUT_CHAT = {
    "id": "vendor/audio-understanding",
    "object": "model",
    "supported_endpoint_types": ["openai"],
    "architecture": {"input_modalities": ["text", "audio"], "output_modalities": ["text"]},
}

ROUTING_ALIAS = {
    "id": "orcarouter/auto",
    "object": "model",
    "owned_by": "orcarouter",
    "supported_endpoint_types": ["openai", "openai-response", "anthropic", "gemini"],
}

EMBEDDING = {
    "id": "vendor/text-embed-3",
    "object": "model",
    "supported_endpoint_types": ["embeddings"],
    "architecture": {"input_modalities": ["text"], "output_modalities": ["embedding"]},
}

IMAGE_GEN = {
    "id": "vendor/nano-banana",
    "object": "model",
    "supported_endpoint_types": ["image-generation"],
    "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
}

VIDEO_GEN = {
    "id": "vendor/veo-3",
    "object": "model",
    "supported_endpoint_types": ["openai-video"],
    "architecture": {"input_modalities": ["text"], "output_modalities": ["video"]},
}

RERANK = {
    "id": "jina/jina-rerank-v3",
    "object": "model",
    "supported_endpoint_types": ["jina-rerank"],
}

LIVE_PAYLOAD = {
    "object": "list",
    "success": True,
    "data": [
        ROUTING_ALIAS,
        TEXT_ONLY_CHAT,
        IMAGE_INPUT_CHAT,
        AUDIO_INPUT_CHAT,
        EMBEDDING,
        IMAGE_GEN,
        VIDEO_GEN,
        RERANK,
    ],
}


def _fetcher_returning(payload):
    def _fetch(url: str, api_key: str, timeout: float, max_bytes: int):
        return payload

    return _fetch


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_vendor_namespace_verbatim() -> None:
    models = parse_models(LIVE_PAYLOAD)
    assert "deepseek/deepseek-v4-pro" in [m.id for m in models]
    assert "orcarouter/auto" in [m.id for m in models]


def test_parses_declared_capability_metadata() -> None:
    models = {m.id: m for m in parse_models(LIVE_PAYLOAD)}
    text_only = models["deepseek/deepseek-v4-pro"]
    assert text_only.input_modalities == ("text",)
    assert text_only.context_length == 1048576
    assert text_only.max_completion_tokens == 384000
    assert text_only.supports_chat is True
    assert models["deepseek/deepseek-v4.1-flash"].input_modalities == ("text", "image")
    assert models["vendor/nano-banana"].input_modalities == ("text",)


def test_unknown_records_are_skipped_not_trusted() -> None:
    payload = {
        "data": [
            {"id": "ok/model", "supported_endpoint_types": ["openai"]},
            {"object": "model"},                       # no id
            {"id": "", "supported_endpoint_types": []},  # empty id
            "not-a-dict",
            {"id": "ok/second", "architecture": "nonsense"},
        ]
    }
    ids = [m.id for m in parse_models(payload)]
    assert ids == ["ok/model", "ok/second"]


def test_parsing_is_bounded() -> None:
    payload = {"data": [{"id": f"m/{i}"} for i in range(50)]}
    assert len(parse_models(payload, max_items=10)) == 10


def test_non_list_payload_yields_nothing() -> None:
    assert parse_models({"data": "nope"}) == []
    assert parse_models(None) == []


# --------------------------------------------------------------------------
# Capability filtering
# --------------------------------------------------------------------------


def test_chat_capability_uses_text_wire_endpoints() -> None:
    models = parse_models(LIVE_PAYLOAD)
    chat = {m.id for m in filter_models(models, CAPABILITY_CHAT)}

    assert "deepseek/deepseek-v4-pro" in chat
    assert "deepseek/deepseek-v4.1-flash" in chat
    assert "orcarouter/auto" in chat
    # Non-text-only entries never appear in a text dropdown.
    assert "vendor/nano-banana" not in chat
    assert "vendor/veo-3" not in chat
    assert "jina/jina-rerank-v3" not in chat
    assert "vendor/text-embed-3" not in chat


def test_chat_capability_hits_the_documented_query() -> None:
    assert (
        catalog_url("https://api.orcarouter.ai/v1", CAPABILITY_CHAT)
        == "https://api.orcarouter.ai/v1/models?capability=chat"
    )
    assert "capability=embedding" in catalog_url("https://api.orcarouter.ai/v1", CAPABILITY_EMBEDDING)
    assert "capability=image" in catalog_url("https://api.orcarouter.ai/v1", CAPABILITY_IMAGE)
    # Video and rerank have no server-side filter param; they filter locally.
    assert "?" not in catalog_url("https://api.orcarouter.ai/v1", CAPABILITY_VIDEO)
    assert "?" not in catalog_url("https://api.orcarouter.ai/v1", CAPABILITY_RERANK)


def test_image_attachment_fails_closed_on_undeclared_modality() -> None:
    models = parse_models(LIVE_PAYLOAD)
    multimodal = {
        m.id
        for m in filter_models(models, CAPABILITY_CHAT, required_input_modalities=("image",))
    }
    assert multimodal == {"deepseek/deepseek-v4.1-flash"}
    # The text-only model and the routing alias declare nothing, so they are
    # excluded rather than assumed compatible.
    assert "deepseek/deepseek-v4-pro" not in multimodal
    assert "orcarouter/auto" not in multimodal


def test_audio_attachment_only_matches_declared_audio_input() -> None:
    models = parse_models(LIVE_PAYLOAD)
    audio = {
        m.id
        for m in filter_models(models, CAPABILITY_CHAT, required_input_modalities=("audio",))
    }
    assert audio == {"vendor/audio-understanding"}


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (CAPABILITY_EMBEDDING, {"vendor/text-embed-3"}),
        (CAPABILITY_IMAGE, {"vendor/nano-banana"}),
        (CAPABILITY_VIDEO, {"vendor/veo-3"}),
        (CAPABILITY_RERANK, {"jina/jina-rerank-v3"}),
    ],
)
def test_each_entry_point_gets_its_own_models(capability: str, expected: set[str]) -> None:
    models = parse_models(LIVE_PAYLOAD)
    assert {m.id for m in filter_models(models, capability)} == expected


def test_capability_is_never_guessed_from_the_model_name() -> None:
    """A name that says 'image' still needs the declared endpoint/capability."""
    liar = CatalogModel(
        id="vendor/super-image-embed-rerank-model",
        supported_endpoint_types=("openai",),
        input_modalities=("text",),
    )
    assert matches_capability(liar, CAPABILITY_CHAT) is True
    assert matches_capability(liar, CAPABILITY_IMAGE) is False
    assert matches_capability(liar, CAPABILITY_EMBEDDING) is False
    assert matches_capability(liar, CAPABILITY_RERANK) is False
    assert (
        matches_capability(liar, CAPABILITY_CHAT, required_input_modalities=("image",))
        is False
    )


def test_unknown_capability_is_an_error_not_a_silent_pass() -> None:
    with pytest.raises(ValueError):
        matches_capability(CatalogModel(id="x"), "telepathy")


# --------------------------------------------------------------------------
# Live discovery, caching, degradation
# --------------------------------------------------------------------------


def test_live_discovery_is_authoritative_and_never_mixes_in_the_seed(tmp_path: Path) -> None:
    result = discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_fetcher_returning(LIVE_PAYLOAD),
        cache_dir=tmp_path,
    )
    assert result.source == "live"
    assert result.degraded is False
    assert result.live_model_count == len(LIVE_PAYLOAD["data"])
    # A model that exists only in the seed is NOT offered.
    assert "openai/gpt-5.5" not in result.ids
    assert result.ids == [m.id for m in result.models]


def test_live_discovery_sends_the_bearer_key_to_the_configured_origin(
    tmp_path: Path,
) -> None:
    seen: dict[str, str] = {}

    def _fetch(url: str, api_key: str, timeout: float, max_bytes: int):
        seen["url"] = url
        seen["key"] = api_key
        assert max_bytes > 0
        return LIVE_PAYLOAD

    discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_fetch,
        cache_dir=tmp_path,
    )
    assert seen["url"] == "https://api.orcarouter.ai/v1/models?capability=chat"
    assert seen["key"] == "sk-orca-fake"


def test_outage_falls_back_to_the_verified_seed_only(tmp_path: Path) -> None:
    def _explode(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.URLError("down")

    result = discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_explode,
        cache_dir=tmp_path,
    )
    assert result.source == "seed"
    assert result.degraded is True
    assert result.models == seed_for_capability(CAPABILITY_CHAT)
    assert "could not reach" in result.error
    assert "sk-orca-fake" not in result.error


def test_seed_carries_verified_metadata_and_reasoning_ladder() -> None:
    seed = {m.id: m for m in VERIFIED_SEED}
    gpt = seed["openai/gpt-5.5"]
    assert gpt.reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert "image" in gpt.input_modalities
    assert gpt.verified is True
    assert seed["deepseek/deepseek-v4-pro"].context_length == 1048576
    assert all("orcarouter" in m.provenance or m.provenance for m in VERIFIED_SEED)


def test_seed_is_capability_filtered_too() -> None:
    # The seed has no embedding/image/video/rerank entries, so those
    # dropdowns stay empty rather than offering a chat model.
    assert seed_for_capability(CAPABILITY_EMBEDDING) == ()
    assert seed_for_capability(CAPABILITY_IMAGE) == ()
    chat_seed = seed_for_capability(CAPABILITY_CHAT, required_input_modalities=("image",))
    assert all("image" in m.input_modalities for m in chat_seed)


def test_no_credential_means_no_catalogue_call(tmp_path: Path) -> None:
    called = {"n": 0}

    def _fetch(url: str, api_key: str, timeout: float, max_bytes: int):
        called["n"] += 1
        return LIVE_PAYLOAD

    result = discover_models(
        "https://api.orcarouter.ai/v1", "", fetcher=_fetch, cache_dir=tmp_path
    )
    assert called["n"] == 0
    assert result.source == "seed"
    assert result.degraded is True
    assert "credential" in result.error


def test_auth_error_is_reported_without_the_key(tmp_path: Path) -> None:
    def _fetch(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)

    result = discover_models(
        "https://api.orcarouter.ai/v1", "sk-orca-fake", fetcher=_fetch, cache_dir=tmp_path
    )
    assert result.degraded is True
    assert "401" in result.error
    assert "sk-orca-fake" not in result.error


def test_last_known_good_cache_is_preferred_over_the_seed_on_outage(tmp_path: Path) -> None:
    discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_fetcher_returning(LIVE_PAYLOAD),
        cache_dir=tmp_path,
    )
    assert read_cache(CAPABILITY_CHAT, cache_dir=tmp_path, ttl=60)

    def _explode(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.URLError("down")

    result = discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_explode,
        cache_dir=tmp_path,
        cache_ttl=60,
    )
    assert result.source == "cache"
    assert result.degraded is True
    assert "deepseek/deepseek-v4-pro" in result.ids


def test_cache_expires(tmp_path: Path) -> None:
    write_cache(CAPABILITY_CHAT, LIVE_PAYLOAD, cache_dir=tmp_path)
    assert read_cache(CAPABILITY_CHAT, cache_dir=tmp_path, ttl=3600) is not None
    assert read_cache(CAPABILITY_CHAT, cache_dir=tmp_path, ttl=0) is None


def test_corrupt_cache_is_ignored_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "catalog_chat.json").write_text("{oops", encoding="utf-8")
    assert read_cache(CAPABILITY_CHAT, cache_dir=tmp_path) is None


def test_cached_catalogue_never_stores_a_credential(tmp_path: Path) -> None:
    discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_fetcher_returning(LIVE_PAYLOAD),
        cache_dir=tmp_path,
    )
    blob = (tmp_path / "catalog_chat.json").read_text(encoding="utf-8")
    assert "sk-orca-fake" not in blob
    assert "Authorization" not in blob


# --------------------------------------------------------------------------
# Metadata preservation and stale selections
# --------------------------------------------------------------------------


def test_live_discovery_does_not_erase_verified_metadata() -> None:
    """Live rows replace capability claims only when live actually has them."""
    live = parse_models({"data": [{"id": "openai/gpt-5.5", "context_length": 0}]})
    merged = {m.id: m for m in merge_verified_metadata(live)}
    gpt = merged["openai/gpt-5.5"]
    assert gpt.reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert gpt.context_length == 400000
    assert gpt.input_modalities == ("text", "image")
    assert gpt.verified is True


def test_live_discovery_does_not_add_models_it_did_not_return() -> None:
    live = parse_models({"data": [{"id": "only/this-one"}]})
    merged = merge_verified_metadata(live)
    assert [m.id for m in merged] == ["only/this-one"]


def test_live_modalities_win_when_present() -> None:
    live = parse_models(
        {
            "data": [
                {
                    "id": "openai/gpt-5.5",
                    "architecture": {"input_modalities": ["text"]},
                }
            ]
        }
    )
    merged = {m.id: m for m in merge_verified_metadata(live)}
    assert merged["openai/gpt-5.5"].input_modalities == ("text",)
    # ...and the reasoning ladder is still preserved.
    assert merged["openai/gpt-5.5"].reasoning_efforts == ("low", "medium", "high", "xhigh")


def test_a_remembered_selection_is_revalidated_before_restoring() -> None:
    models = parse_models(LIVE_PAYLOAD)
    assert is_model_available("deepseek/deepseek-v4-pro", models) is True
    # The user's attachment change invalidated it: not in the filtered list.
    multimodal = filter_models(models, CAPABILITY_CHAT, required_input_modalities=("image",))
    assert is_model_available("deepseek/deepseek-v4-pro", multimodal) is False
    assert is_model_available("deepseek/deepseek-v4.1-flash", multimodal) is True


def test_result_payload_is_serializable_and_carries_no_secret(tmp_path: Path) -> None:
    result = discover_models(
        "https://api.orcarouter.ai/v1",
        "sk-orca-fake",
        fetcher=_fetcher_returning(LIVE_PAYLOAD),
        cache_dir=tmp_path,
    )
    blob = json.dumps(result.as_dict())
    assert "sk-orca-fake" not in blob
    payload = json.loads(blob)
    assert payload["source"] == "live"
    assert payload["models"][0]["id"]
    assert "authorization" not in blob.lower()
