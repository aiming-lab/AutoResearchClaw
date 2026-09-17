"""OrcaRouter provider API and the browser lifecycle of a PKCE login.

The server holds the single login lock; these tests drive every terminal
path through the real routes and assert the lock is released each time and
that a late response cannot repaint a newer attempt.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from researchclaw.llm import orcarouter as orca  # noqa: E402
from researchclaw.llm.orcarouter import CredentialStore  # noqa: E402
from researchclaw.llm.orcarouter_pkce import ExchangeResult  # noqa: E402
from researchclaw.server.routes import providers as providers_route  # noqa: E402

FAKE_KEY = "sk-orca-fake-0000000000000000000001"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(tmp_path / "creds.json"))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    for var in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    providers_route.reset_for_tests()

    from researchclaw.config import RCConfig
    from researchclaw.server.app import create_app

    config = RCConfig.load("config.researchclaw.example.yaml", check_paths=False)
    with TestClient(create_app(config)) as test_client:
        yield test_client
    providers_route.reset_for_tests()


# --------------------------------------------------------------------------
# Provider state
# --------------------------------------------------------------------------


def test_both_orcarouter_entries_are_listed_with_stable_ids(client) -> None:
    body = client.get("/api/providers").json()
    by_id = {p["id"]: p for p in body["providers"]}
    assert set(by_id) == {"orcarouter", "orcarouter-oauth"}
    assert by_id["orcarouter"]["kind"] == "api_key"
    assert by_id["orcarouter-oauth"]["kind"] == "pkce"
    assert by_id["orcarouter"]["label"] != by_id["orcarouter-oauth"]["label"]
    assert by_id["orcarouter"]["base_url"] == "https://api.orcarouter.ai/v1"
    assert body["endpoints"]["auth_base"] == "https://www.orcarouter.ai"


def test_api_key_round_trip_masks_the_secret(client) -> None:
    response = client.post("/api/providers/orcarouter/key", json={"api_key": FAKE_KEY})
    assert response.status_code == 200
    status = response.json()["status"]
    assert status["configured"] is True
    assert status["secret_masked"] != FAKE_KEY
    assert FAKE_KEY not in response.text

    listed = client.get("/api/providers").json()
    api_entry = next(p for p in listed["providers"] if p["id"] == "orcarouter")
    assert FAKE_KEY not in json.dumps(listed)
    assert api_entry["status"]["secret_masked"] == status["secret_masked"]

    cleared = client.request("DELETE", "/api/providers/orcarouter/key").json()
    assert cleared["status"]["configured"] is False


def test_invalid_key_is_rejected_without_echoing_it(client) -> None:
    response = client.post("/api/providers/orcarouter/key", json={"api_key": "not-a-key"})
    assert response.status_code == 400
    assert "sk-orca" in response.json()["detail"]


def test_models_requires_a_credential(client) -> None:
    response = client.get("/api/providers/orcarouter/models")
    assert response.status_code == 409
    assert "OrcaRouter" in response.json()["detail"]


# --------------------------------------------------------------------------
# Model catalogue endpoint
# --------------------------------------------------------------------------


def _seed_credential(tmp_path_unused: Any = None) -> None:
    orca.CredentialStore().save(orca.PROVIDER_ID, FAKE_KEY, source="api_key")


def test_models_endpoint_filters_by_capability(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_credential()
    live = {
        "data": [
            {
                "id": "deepseek/deepseek-v4-pro",
                "supported_endpoint_types": ["openai"],
                "architecture": {"input_modalities": ["text"]},
            },
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "supported_endpoint_types": ["openai"],
                "architecture": {"input_modalities": ["text", "image"]},
            },
            {
                "id": "vendor/nano-banana",
                "supported_endpoint_types": ["image-generation"],
            },
        ]
    }
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_catalog._default_fetcher",
        lambda url, key, timeout, max_bytes: live,
    )

    text = client.get("/api/providers/orcarouter/models?capability=chat").json()
    assert [m["id"] for m in text["models"]] == [
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4.1-flash",
    ]
    assert text["source"] == "live"
    assert text["catalog_source"].endswith("/v1/models?capability=chat")

    multimodal = client.get(
        "/api/providers/orcarouter/models?capability=chat&modality=image"
    ).json()
    assert [m["id"] for m in multimodal["models"]] == ["deepseek/deepseek-v4.1-flash"]

    images = client.get("/api/providers/orcarouter/models?capability=image").json()
    assert [m["id"] for m in images["models"]] == ["vendor/nano-banana"]


def test_models_endpoint_sends_the_key_server_side_only(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_credential()
    seen: dict[str, str] = {}

    def _fetch(url: str, api_key: str, timeout: float, max_bytes: int):
        seen["key"] = api_key
        seen["url"] = url
        return {"data": []}

    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_catalog._default_fetcher", _fetch
    )
    body = client.get("/api/providers/orcarouter/models").json()
    assert seen["key"] == FAKE_KEY
    assert seen["url"].startswith("https://api.orcarouter.ai/v1/models")
    assert FAKE_KEY not in json.dumps(body), "the key must never reach the browser"


def test_degraded_catalogue_is_labelled_with_its_source(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_credential()
    import urllib.error

    def _explode(url: str, api_key: str, timeout: float, max_bytes: int):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_catalog._default_fetcher", _explode
    )
    body = client.get("/api/providers/orcarouter/models").json()
    assert body["degraded"] is True
    assert body["source"] == "seed"
    assert body["count"] > 0
    assert "verified" in json.dumps(body).lower()


# --------------------------------------------------------------------------
# PKCE login lifecycle through the routes
# --------------------------------------------------------------------------


def test_login_start_returns_an_authorize_url_and_locks_the_attempt(client) -> None:
    response = client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "oob"}
    )
    assert response.status_code == 200
    attempt = response.json()["attempt"]
    assert attempt["status"] == "pending"
    assert attempt["busy"] is True
    assert attempt["flow"] == "oob"
    assert attempt["authorize_url"].startswith("https://www.orcarouter.ai/auth?")
    assert "code_challenge_method=S256" in attempt["authorize_url"]
    assert attempt["needs_code"] is True
    # A verifier is never handed to the browser, in any field.
    assert "code_verifier" not in json.dumps(attempt)
    assert "verifier" not in json.dumps(attempt).lower()


def test_a_second_login_is_refused_while_one_is_pending(client) -> None:
    assert client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).status_code == 200
    second = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"})
    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"]


def test_successful_exchange_persists_and_releases_the_lock(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).json()
    attempt_id = started["attempt"]["attempt_id"]

    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (
            200,
            json.dumps({"key": FAKE_KEY, "user_id": "12345", "scope": "api"}).encode(),
        ),
    )
    done = client.post(
        f"/api/providers/orcarouter/auth/{attempt_id}/code", json={"code": "the-code"}
    ).json()["attempt"]

    assert done["status"] == "connected"
    assert done["busy"] is False
    assert done["secret_masked"] != FAKE_KEY
    assert FAKE_KEY not in json.dumps(done)
    assert done["account"] == "12345"

    # The credential is persisted for the OrcaRouter — Auth entry.
    auth_state = client.get("/api/providers").json()
    pkce = next(p for p in auth_state["providers"] if p["id"] == "orcarouter-oauth")
    assert pkce["status"]["configured"] is True

    # The lock is free again, so a fresh login (e.g. after revocation) works.
    assert client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "oob"}
    ).status_code == 200


def test_denied_authorization_releases_the_lock(client, monkeypatch) -> None:
    started = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).json()
    attempt_id = started["attempt"]["attempt_id"]
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (403, b'{"error":"invalid_grant"}'),
    )
    done = client.post(
        f"/api/providers/orcarouter/auth/{attempt_id}/code", json={"code": "bad"}
    ).json()["attempt"]

    assert done["status"] == "error"
    assert done["busy"] is False
    assert "login" in done["error"].lower() or "code" in done["error"].lower()
    assert client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "oob"}
    ).status_code == 200


def test_explicit_cancel_releases_the_lock_and_is_idempotent(client) -> None:
    started = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).json()
    attempt_id = started["attempt"]["attempt_id"]

    first = client.post(f"/api/providers/orcarouter/auth/{attempt_id}/cancel").json()
    assert first["attempt"]["status"] == "cancelled"
    assert first["attempt"]["busy"] is False
    assert first["attempt"]["authorize_url"] == ""

    # The pagehide handler may fire after an explicit cancel: never an error.
    second = client.post(f"/api/providers/orcarouter/auth/{attempt_id}/cancel")
    assert second.status_code == 200

    assert client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "oob"}
    ).status_code == 200


def test_switching_flow_mid_login_requires_and_allows_a_cancel(client) -> None:
    loopback = client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "loopback"}
    ).json()["attempt"]
    assert loopback["flow"] == "loopback"
    client.post(f"/api/providers/orcarouter/auth/{loopback['attempt_id']}/cancel")
    assert client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "oob"}
    ).status_code == 200


def test_timeout_releases_the_lock(client, monkeypatch) -> None:
    started = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).json()
    attempt_id = started["attempt"]["attempt_id"]

    class _Boom(Exception):
        pass

    def _timeout(url: str, payload: Any, *, timeout: float):
        raise TimeoutError("no answer")

    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json", _timeout
    )
    done = client.post(
        f"/api/providers/orcarouter/auth/{attempt_id}/code", json={"code": "c"}
    ).json()["attempt"]
    assert done["status"] == "error"
    assert done["busy"] is False
    assert FAKE_KEY not in json.dumps(done)


def test_a_submitted_code_cannot_be_resubmitted(client, monkeypatch) -> None:
    started = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).json()
    attempt_id = started["attempt"]["attempt_id"]
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (
            200,
            json.dumps({"key": FAKE_KEY, "scope": "api"}).encode(),
        ),
    )
    assert client.post(
        f"/api/providers/orcarouter/auth/{attempt_id}/code", json={"code": "c"}
    ).status_code == 200
    again = client.post(
        f"/api/providers/orcarouter/auth/{attempt_id}/code", json={"code": "c"}
    )
    assert again.status_code == 409


def test_unknown_attempt_id_is_a_404(client) -> None:
    assert client.get("/api/providers/orcarouter/auth/nope").status_code == 404
    assert client.post("/api/providers/orcarouter/auth/nope/cancel").status_code == 404


def test_a_stale_response_cannot_overwrite_a_newer_attempt(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation guard: the server refuses to finish a superseded attempt."""
    stale = client.post(
        "/api/providers/orcarouter/auth/login", json={"flow": "oob"}
    ).json()["attempt"]

    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (
            200,
            json.dumps({"key": FAKE_KEY, "scope": "api"}).encode(),
        ),
    )

    # A newer attempt replaces the old one in the registry before the old
    # response lands.
    replacement = providers_route.LoginAttempt(
        attempt_id="newer",
        generation=stale["generation"] + 1,
        flow="oob",
        authorize_url="https://www.orcarouter.ai/auth?x=1",
        callback_url="oob",
        pending=None,
        created_at=0.0,
    )
    with providers_route._registry._lock:  # noqa: SLF001 - deliberate test hook
        providers_route._registry._active.close()
        providers_route._registry._active = replacement

    response = client.post(
        f"/api/providers/orcarouter/auth/{stale['attempt_id']}/code",
        json={"code": "late-code"},
    )
    assert response.status_code == 404, "a superseded attempt must not be completable"

    # ...and nothing was written for the newer attempt either.
    assert CredentialStore().status("orcarouter-oauth").configured is False


def test_pagehide_cancel_after_success_does_not_drop_the_credential(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = client.post("/api/providers/orcarouter/auth/login", json={"flow": "oob"}).json()
    attempt_id = started["attempt"]["attempt_id"]
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (
            200,
            json.dumps({"key": FAKE_KEY, "scope": "api", "user_id": "1"}).encode(),
        ),
    )
    client.post(f"/api/providers/orcarouter/auth/{attempt_id}/code", json={"code": "c"})
    client.post(f"/api/providers/orcarouter/auth/{attempt_id}/cancel")

    assert CredentialStore().status("orcarouter-oauth").configured is True


# --------------------------------------------------------------------------
# Reauthentication endpoint
# --------------------------------------------------------------------------


def test_reauth_marks_only_the_rejected_generation(client) -> None:
    store = CredentialStore()
    first = store.save("orcarouter-oauth", FAKE_KEY, source="pkce", grant_id="1")
    second = store.save("orcarouter-oauth", FAKE_KEY + "b", source="pkce", grant_id="1")

    stale = client.post(
        "/api/providers/orcarouter/reauth",
        json={"entry_id": "orcarouter-oauth", "generation": first.generation},
    ).json()
    assert stale["applied"] is False
    assert stale["status"]["needs_reauth"] is False

    fresh = client.post(
        "/api/providers/orcarouter/reauth",
        json={"entry_id": "orcarouter-oauth", "generation": second.generation},
    ).json()
    assert fresh["applied"] is True
    assert fresh["status"]["needs_reauth"] is True
    # The stored key is kept, so a misclassified failure is reversible.
    assert CredentialStore().get("orcarouter-oauth")["api_key"] == FAKE_KEY + "b"


# --------------------------------------------------------------------------
# The shipped settings page
# --------------------------------------------------------------------------


def test_settings_page_exposes_both_auth_entries(client) -> None:
    response = client.get("/providers")
    assert response.status_code == 200
    html = response.text
    assert 'data-testid="api-key-input"' in html
    assert 'data-testid="pkce-connect"' in html
    assert "OrcaRouter — API" in html
    assert "OrcaRouter — Auth" in html
    assert "orca-logo-classic.png" in html


def test_settings_assets_are_served(client) -> None:
    assert client.get("/providers/providers.js").status_code == 200
    assert client.get("/providers/providers.css").status_code == 200


def test_settings_page_never_renders_a_secret(client) -> None:
    client.post("/api/providers/orcarouter/key", json={"api_key": FAKE_KEY})
    page = client.get("/providers").text
    assert FAKE_KEY not in page
