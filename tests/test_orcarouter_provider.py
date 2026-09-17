"""OrcaRouter provider seam: origins, credential lifecycle, dual auth.

The load-bearing assertions here are that the two credential adapters
(``orcarouter`` for a pasted key, ``orcarouter-oauth`` for the PKCE grant)
produce the *same* :class:`OrcaCredential` and that everything downstream —
the OpenAI-compatible client and model discovery — is indifferent to which
one produced it.
"""

from __future__ import annotations

import json
import os
import stat
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import pytest

from researchclaw.llm import PROVIDER_PRESETS, create_llm_client
from researchclaw.llm.client import LLMClient
from researchclaw.llm.orcarouter import (
    ApiKeySource,
    CredentialStore,
    OrcaAuthRequired,
    OrcaConfigError,
    OrcaCredential,
    PkceSource,
    build_credential_sources,
    build_orcarouter_client,
    complete_connect,
    handle_unauthorized,
    mask_secret,
    redact_secrets,
    resolve_credential,
    resolve_endpoints,
    start_connect,
    validate_origin,
)
from researchclaw.llm.orcarouter_pkce import (
    ExchangeResult,
    build_exchange_url,
    start_oob_login,
)

FAKE_KEY = "sk-orca-fake-0000000000000000000001"
FAKE_KEY_2 = "sk-orca-fake-0000000000000000000002"


# --------------------------------------------------------------------------
# Origins
# --------------------------------------------------------------------------


def test_defaults_use_two_distinct_public_origins() -> None:
    endpoints = resolve_endpoints({})
    assert endpoints.auth_base == "https://www.orcarouter.ai"
    assert endpoints.api_base == "https://api.orcarouter.ai/v1"
    assert endpoints.auth_source == "default"
    assert endpoints.api_source == "default"


def test_the_two_origins_are_never_derived_from_each_other() -> None:
    endpoints = resolve_endpoints({})
    # The single most common integration mistake: swapping the hostname.
    assert endpoints.auth_base.replace("www.", "api.") != endpoints.api_base
    assert not endpoints.auth_base.endswith("/v1")
    assert endpoints.api_base.endswith("/v1")


def test_shared_self_hosted_base_is_a_fallback_for_both() -> None:
    endpoints = resolve_endpoints({"ORCA_BASE_URL": "https://orca.internal"})
    assert endpoints.auth_base == "https://orca.internal"
    assert endpoints.api_base == "https://orca.internal/v1"
    assert endpoints.auth_source == endpoints.api_source == "shared"


def test_explicit_overrides_win_over_the_shared_base() -> None:
    endpoints = resolve_endpoints(
        {
            "ORCA_BASE_URL": "https://orca.internal",
            "ORCA_AUTH_BASE_URL": "https://login.internal",
            "ORCA_API_BASE_URL": "https://relay.internal/v1",
        }
    )
    assert endpoints.auth_base == "https://login.internal"
    assert endpoints.api_base == "https://relay.internal/v1"
    assert endpoints.auth_source == endpoints.api_source == "explicit"

    # ...and the shared base survives for whichever one is not overridden.
    one_sided = resolve_endpoints(
        {"ORCA_BASE_URL": "https://orca.internal", "ORCA_AUTH_BASE_URL": "https://login.internal"}
    )
    assert one_sided.auth_base == "https://login.internal"
    assert one_sided.api_base == "https://orca.internal/v1"


@pytest.mark.parametrize(
    "url",
    ["http://orca.example.com", "ftp://orca.example.com", "https://u:p@orca.example.com", "not-a-url"],
)
def test_remote_origins_must_be_https_without_userinfo(url: str) -> None:
    with pytest.raises(OrcaConfigError):
        validate_origin(url, name="ORCA_AUTH_BASE_URL")


@pytest.mark.parametrize(
    "url", ["http://localhost:8080", "http://127.0.0.1:51733", "http://[::1]:9000"]
)
def test_http_is_allowed_only_for_loopback(url: str) -> None:
    assert validate_origin(url, name="x").startswith("http://")


# --------------------------------------------------------------------------
# The seam: two adapters, one credential
# --------------------------------------------------------------------------


def test_both_adapters_yield_the_same_credential_shape(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")

    api_adapter, pkce_adapter = build_credential_sources(
        store=store, environ={"ORCAROUTER_API_KEY": FAKE_KEY}
    )
    pkce_adapter.persist(
        ExchangeResult(api_key=FAKE_KEY_2, scope="api", user_id="12345")
    )

    from_api = api_adapter.acquire()
    from_pkce = pkce_adapter.acquire()

    assert isinstance(from_api, OrcaCredential)
    assert isinstance(from_pkce, OrcaCredential)
    assert from_api.api_key == FAKE_KEY
    assert from_pkce.api_key == FAKE_KEY_2
    # Same type, same fields, same downstream treatment.
    assert set(from_api.__dataclass_fields__) == set(from_pkce.__dataclass_fields__)
    assert from_api.source == "api_key"
    assert from_pkce.source == "pkce"


def test_downstream_client_and_catalogue_are_indifferent_to_credential_source(
    tmp_path: Path,
) -> None:
    """The provider client and model discovery never branch on the source."""
    store = CredentialStore(tmp_path / "creds.json")
    api_adapter, pkce_adapter = build_credential_sources(
        store=store, environ={"ORCAROUTER_API_KEY": FAKE_KEY}
    )
    pkce_adapter.persist(ExchangeResult(api_key=FAKE_KEY_2, scope="api", user_id="7"))

    def _client_for(credential: OrcaCredential) -> LLMClient:
        from researchclaw.llm.orcarouter import OrcaRouterProvider

        return OrcaRouterProvider(credential=credential).build_client()

    api_client = _client_for(api_adapter.acquire())
    pkce_client = _client_for(pkce_adapter.acquire())

    assert api_client.config.base_url == pkce_client.config.base_url
    assert api_client.config.base_url == "https://api.orcarouter.ai/v1"
    # Only the secret differs; the wire configuration is identical.
    assert api_client.config.api_key != pkce_client.config.api_key
    assert (
        api_client._endpoint_path() == pkce_client._endpoint_path() == "/chat/completions"
    )


def test_the_project_owns_no_second_key_store(tmp_path: Path) -> None:
    """The store lives in the project's existing ~/.researchclaw tree."""
    default = CredentialStore()
    assert default.path == Path.home() / ".researchclaw" / "orcarouter" / "credentials.json"


# --------------------------------------------------------------------------
# API-key adapter: save / read / clear / mask
# --------------------------------------------------------------------------


def test_api_key_adapter_save_read_clear(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    adapter = ApiKeySource(store, api_key_env="ORCAROUTER_API_KEY", environ={})

    assert adapter.status().configured is False
    with pytest.raises(OrcaAuthRequired):
        adapter.acquire()

    status = adapter.save(FAKE_KEY)
    assert status.configured is True
    assert status.masked == mask_secret(FAKE_KEY)
    assert FAKE_KEY not in json.dumps(status.as_dict())
    assert adapter.acquire().api_key == FAKE_KEY

    adapter.clear()
    assert adapter.status().configured is False
    assert store.get("orcarouter") == {}


def test_api_key_adapter_updates_in_place_and_bumps_generation(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    adapter = ApiKeySource(store, environ={})
    first = adapter.save(FAKE_KEY)
    second = adapter.save(FAKE_KEY_2)
    assert second.generation == first.generation + 1
    assert adapter.acquire().api_key == FAKE_KEY_2


@pytest.mark.parametrize("bad", ["", "   ", "sk-openai-not-orca", "orca-1234"])
def test_api_key_adapter_rejects_obviously_wrong_input(tmp_path: Path, bad: str) -> None:
    adapter = ApiKeySource(CredentialStore(tmp_path / "c.json"), environ={})
    with pytest.raises(OrcaConfigError):
        adapter.save(bad)


def test_stored_key_file_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "creds.json"
    ApiKeySource(CredentialStore(path), environ={}).save(FAKE_KEY)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"credential file must be 0600, got {oct(mode)}"


def test_config_and_env_take_precedence_over_the_store(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    ApiKeySource(store, environ={}).save(FAKE_KEY)

    from_env = ApiKeySource(store, environ={"ORCAROUTER_API_KEY": FAKE_KEY_2})
    assert from_env.acquire().api_key == FAKE_KEY_2

    from_config = ApiKeySource(store, config_value="sk-orca-from-config-0001", environ={})
    assert from_config.acquire().api_key == "sk-orca-from-config-0001"


def test_pkce_adapter_does_not_use_the_api_key_env(tmp_path: Path) -> None:
    """Choosing OrcaRouter — Auth must not silently fall back to an env key."""
    store = CredentialStore(tmp_path / "creds.json")
    with pytest.raises(OrcaAuthRequired) as excinfo:
        PkceSource(store).acquire()
    assert excinfo.value.reason == "not_connected"


# --------------------------------------------------------------------------
# PKCE persistence through the project's own connect adapter
# --------------------------------------------------------------------------


def test_complete_connect_persists_a_durable_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    pending = start_oob_login("https://auth.example.test")
    credential = complete_connect(
        pending,
        "auth-code",
        store=store,
        endpoints=resolve_endpoints({}),
        post_json=lambda url, payload, timeout: (
            200,
            json.dumps({"key": FAKE_KEY, "user_id": "12345", "scope": "api"}).encode(),
        ),
    )

    assert credential.api_key == FAKE_KEY
    assert credential.source == "pkce"
    assert credential.grant_id == "12345"

    # Restarting reuses the stored key instead of minting a second one.
    reused = PkceSource(store).acquire()
    assert reused.api_key == FAKE_KEY
    assert reused.generation == credential.generation

    # The exchange went to the auth origin, never the inference origin.
    assert build_exchange_url("https://www.orcarouter.ai").endswith("/api/v1/auth/keys")


def test_exchange_targets_the_auth_origin_config_under_test(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    seen: dict[str, str] = {}
    pending = start_oob_login("https://auth.example.test")

    def _post(url: str, payload: Mapping[str, Any], *, timeout: float):
        seen["url"] = url
        return 200, json.dumps({"key": FAKE_KEY, "scope": "api"}).encode()

    complete_connect(
        pending,
        "auth-code",
        store=store,
        endpoints=resolve_endpoints({}),
        post_json=_post,
    )
    assert seen["url"] == "https://www.orcarouter.ai/api/v1/auth/keys"


# --------------------------------------------------------------------------
# Durable key lifecycle: no refresh, generation-safe 401
# --------------------------------------------------------------------------


def test_revoked_key_enters_needs_reauth_and_never_refreshes(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    adapter = PkceSource(store)
    credential = adapter.persist(ExchangeResult(api_key=FAKE_KEY, user_id="9"))

    assert handle_unauthorized(credential, store=store) is True
    assert store.status("orcarouter-oauth").needs_reauth is True

    # The secret is retained: a transient failure must not be irreversible.
    assert store.get("orcarouter-oauth")["api_key"] == FAKE_KEY

    with pytest.raises(OrcaAuthRequired) as excinfo:
        adapter.acquire()
    assert excinfo.value.reason == "needs_reauth"
    assert "revoke" in str(excinfo.value).lower() or "reconnect" in str(excinfo.value).lower()


def test_there_is_no_refresh_grant_to_call(tmp_path: Path) -> None:
    """A PKCE-issued key is durable, not a refreshable OAuth token."""
    source = Path("researchclaw/llm/orcarouter.py").read_text(encoding="utf-8")
    source += Path("researchclaw/llm/orcarouter_pkce.py").read_text(encoding="utf-8")
    for forbidden in ("grant_type=refresh_token", "refresh_token", "/oauth/token"):
        assert forbidden not in source, f"fake refresh machinery: {forbidden}"


def test_a_late_401_does_not_poison_a_newer_credential(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    adapter = PkceSource(store)
    stale = adapter.persist(ExchangeResult(api_key=FAKE_KEY, user_id="9"))

    # The user reauthorizes; the generation moves on.
    fresh = adapter.persist(ExchangeResult(api_key=FAKE_KEY_2, user_id="9"))
    assert fresh.generation == stale.generation + 1

    # A delayed 401 from the *old* request arrives now.
    assert handle_unauthorized(stale, store=store) is False
    status = store.status("orcarouter-oauth")
    assert status.needs_reauth is False, "the new credential must stay usable"
    assert ad_adapter_key(store) == FAKE_KEY_2


def ad_adapter_key(store: CredentialStore) -> str:
    return str(store.get("orcarouter-oauth")["api_key"])


def test_401_for_an_unmarked_entry_is_a_noop(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    stranger = OrcaCredential(
        api_key=FAKE_KEY, source="pkce", entry_id="orcarouter-oauth", generation=99
    )
    assert handle_unauthorized(stranger, store=store) is False


def test_a_repeat_401_is_not_rewritten(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    credential = PkceSource(store).persist(ExchangeResult(api_key=FAKE_KEY, user_id="1"))
    assert handle_unauthorized(credential, store=store) is True
    assert handle_unauthorized(credential, store=store) is False


def test_corrupt_credential_file_is_terminal_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "creds.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = CredentialStore(path)
    assert store.status("orcarouter-oauth").configured is False
    with pytest.raises(OrcaAuthRequired):
        PkceSource(store).acquire()


# --------------------------------------------------------------------------
# Provider registry integration
# --------------------------------------------------------------------------


def _rc_config(provider: str, **overrides: Any):
    from types import SimpleNamespace

    llm = dict(
        provider=provider,
        base_url="",
        api_key="",
        api_key_env="ORCAROUTER_API_KEY",
        wire_api="chat_completions",
        primary_model="orcarouter/auto",
        fallback_models=("deepseek/deepseek-v4-pro",),
        timeout_sec=60,
        reviewer_model="",
        reviewer_provider="",
        reviewer_base_url="",
        reviewer_api_key="",
        reviewer_api_key_env="",
    )
    llm.update(overrides)
    return SimpleNamespace(llm=SimpleNamespace(**llm))


def test_both_entries_are_first_class_registered_providers() -> None:
    assert PROVIDER_PRESETS["orcarouter"]["base_url"] == "https://api.orcarouter.ai/v1"
    assert PROVIDER_PRESETS["orcarouter-oauth"]["base_url"] == "https://api.orcarouter.ai/v1"
    # They are separate, explicitly labelled choices — not one ambiguous button.
    assert PROVIDER_PRESETS["orcarouter"]["label"] != PROVIDER_PRESETS["orcarouter-oauth"]["label"]
    assert PROVIDER_PRESETS["orcarouter"]["auth"] == "api_key"
    assert PROVIDER_PRESETS["orcarouter-oauth"]["auth"] == "pkce"

    from researchclaw.cli import _PROVIDER_CHOICES, _PROVIDER_MODELS, _PROVIDER_URLS

    chosen = {value[0] for value in _PROVIDER_CHOICES.values()}
    assert {"orcarouter", "orcarouter-oauth"} <= chosen
    assert _PROVIDER_URLS["orcarouter"] == "https://api.orcarouter.ai/v1"
    # Both entries seed the same chain: a model verified in the live
    # catalogue, with the routing alias kept as a fallback.
    assert _PROVIDER_MODELS["orcarouter"] == _PROVIDER_MODELS["orcarouter-oauth"]
    assert _PROVIDER_MODELS["orcarouter"][0] in (
        "deepseek/deepseek-v4-pro",
        "orcarouter/auto",
    )
    assert "orcarouter/auto" in _PROVIDER_MODELS["orcarouter"][1]


def test_provider_factory_routes_inference_to_the_orcarouter_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    for provider in ("orcarouter", "orcarouter-oauth"):
        client = create_llm_client(_rc_config(provider))
        assert isinstance(client, LLMClient)
        assert client.config.base_url == "https://api.orcarouter.ai/v1"
        assert client.config.api_key == FAKE_KEY
        assert client._model_chain == ["orcarouter/auto", "deepseek/deepseek-v4-pro"]


def test_provider_factory_uses_the_pkce_grant_for_the_oauth_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    CredentialStore(tmp_path / "creds.json")
    PkceSource(CredentialStore(tmp_path / "creds.json")).persist(
        ExchangeResult(api_key=FAKE_KEY_2, scope="api", user_id="5")
    )

    client = create_llm_client(_rc_config("orcarouter-oauth"))
    assert client.config.api_key == FAKE_KEY_2
    assert client.config.base_url == "https://api.orcarouter.ai/v1"


def test_missing_credential_fails_closed_with_an_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    with pytest.raises(OrcaAuthRequired) as excinfo:
        create_llm_client(_rc_config("orcarouter"))
    message = str(excinfo.value)
    assert "Connect with OrcaRouter" in message or "sk-orca" in message


def test_reviewer_and_panel_paths_reuse_the_same_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No AI entry point re-implements OrcaRouter authentication."""
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.setenv("ORCAROUTER_API_KEY", FAKE_KEY)
    config = _rc_config("orcarouter", reviewer_model="orcarouter/auto")
    reviewer = LLMClient.reviewer_from_rc_config(config)
    assert reviewer is not None
    assert reviewer.config.base_url == "https://api.orcarouter.ai/v1"
    assert reviewer.config.api_key == FAKE_KEY


def test_bearer_header_is_sent_to_the_orcarouter_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    captured: dict[str, Any] = {}

    class _Response:
        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}]}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(request: urllib.request.Request, timeout: int):
        captured["request"] = request
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    client = build_orcarouter_client(
        _rc_config("orcarouter", api_key=FAKE_KEY, api_key_env="")
    )
    client.chat([{"role": "user", "content": "ping"}])

    request = captured["request"]
    assert request.full_url == "https://api.orcarouter.ai/v1/chat/completions"
    assert request.get_header("Authorization") == f"Bearer {FAKE_KEY}"


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


def test_redaction_hides_key_shaped_text() -> None:
    text = f"failed with {FAKE_KEY} for user"
    assert FAKE_KEY not in redact_secrets(text)
    assert "sk-orca-***" in redact_secrets(text)
    assert FAKE_KEY not in redact_secrets(f"Bearer {FAKE_KEY}")
    assert FAKE_KEY not in redact_secrets(json.dumps({"key": FAKE_KEY}))


def test_mask_is_not_reversible_and_not_the_key() -> None:
    masked = mask_secret(FAKE_KEY)
    assert FAKE_KEY not in masked
    assert masked.endswith(FAKE_KEY[-4:])
    assert mask_secret("") == ""
    assert mask_secret("short") == "•" * 5


def test_status_payloads_never_carry_a_secret(tmp_path: Path) -> None:
    store = CredentialStore(tmp_path / "creds.json")
    adapter = PkceSource(store)
    credential = adapter.persist(ExchangeResult(api_key=FAKE_KEY, user_id="3"))
    payload = json.dumps(
        {
            "api": build_credential_sources(
                store=store, environ={"ORCAROUTER_API_KEY": FAKE_KEY}
            )[0].status().as_dict(),
            "pkce": adapter.status().as_dict(),
            "credential": credential.masked,
            "endpoints": resolve_endpoints({}).describe(),
        }
    )
    assert FAKE_KEY not in payload
    assert "code_verifier" not in payload


def test_no_hardcoded_real_key_anywhere_in_the_implementation() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-nE", r"sk-orca-[A-Za-z0-9]{16,}", "--", "researchclaw/"],
        capture_output=True,
        text=True,
    )
    hits = [line for line in result.stdout.splitlines() if line.strip()]
    # Only the documented placeholder prefix and test-shaped fixtures may match.
    assert not hits, f"hardcoded OrcaRouter key material: {hits}"
