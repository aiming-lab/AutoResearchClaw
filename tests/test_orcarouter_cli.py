"""The ``researchclaw orcarouter`` command group.

A CLI project must expose *both* credential choices as discoverable
commands, so these tests exercise both paths and the catalogue command that
feeds model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from researchclaw.cli import build_parser, cmd_orcarouter, cmd_init
from researchclaw.llm.orcarouter import CredentialStore

FAKE_KEY = "sk-orca-fake-0000000000000000000001"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CredentialStore:
    path = tmp_path / "creds.json"
    monkeypatch.setenv("ORCA_CREDENTIALS_PATH", str(path))
    monkeypatch.setenv("ORCA_CATALOG_CACHE_DIR", str(tmp_path / "catalog"))
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    for var in ("ORCA_BASE_URL", "ORCA_AUTH_BASE_URL", "ORCA_API_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return CredentialStore(path)


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# --------------------------------------------------------------------------
# Discoverability
# --------------------------------------------------------------------------


def test_the_command_group_parses_with_both_entry_points() -> None:
    parser = build_parser()
    key_cmd = parser.parse_args(["orcarouter", "key", "--set"])
    assert key_cmd.command == "orcarouter"
    assert key_cmd.orcarouter_command == "key"
    assert key_cmd.set is True

    login_cmd = parser.parse_args(["orcarouter", "login", "--flow", "oob"])
    assert login_cmd.orcarouter_command == "login"
    assert login_cmd.flow == "oob"

    models_cmd = parser.parse_args(
        ["orcarouter", "models", "--capability", "chat", "--modality", "image"]
    )
    assert models_cmd.modality == ["image"]

    for sub in ("status", "models", "logout", "login", "key"):
        assert parser.parse_args(["orcarouter", sub]).orcarouter_command == sub


def test_init_wizard_offers_both_orcarouter_choices() -> None:
    from researchclaw.cli import _PROVIDER_CHOICES

    kinds = {value[0] for value in _PROVIDER_CHOICES.values()}
    assert "orcarouter" in kinds, "the pasted-key entry must be selectable"
    assert "orcarouter-oauth" in kinds, "the account-login entry must be selectable"


def test_init_wizard_writes_orcarouter_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    class _TTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", _TTY())
    from researchclaw.cli import _PROVIDER_CHOICES

    choice = next(k for k, v in _PROVIDER_CHOICES.items() if v[0] == "orcarouter")
    monkeypatch.setattr("builtins.input", lambda _prompt: choice)
    monkeypatch.setattr("researchclaw.cli._prompt_open_install", lambda: False, raising=False)
    monkeypatch.setattr("researchclaw.cli._prompt_opencode_install", lambda: False)

    assert cmd_init(_args(force=False)) == 0
    content = (tmp_path / "config.arc.yaml").read_text(encoding="utf-8")
    assert 'provider: "orcarouter"' in content
    assert 'base_url: "https://api.orcarouter.ai/v1"' in content
    assert 'api_key_env: "ORCAROUTER_API_KEY"' in content


def test_init_wizard_oauth_entry_writes_no_api_key_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    class _TTY:
        def isatty(self) -> bool:
            return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", _TTY())
    from researchclaw.cli import _PROVIDER_CHOICES

    choice = next(k for k, v in _PROVIDER_CHOICES.items() if v[0] == "orcarouter-oauth")
    monkeypatch.setattr("builtins.input", lambda _prompt: choice)
    monkeypatch.setattr("researchclaw.cli._prompt_opencode_install", lambda: False)

    assert cmd_init(_args(force=False)) == 0
    content = (tmp_path / "config.arc.yaml").read_text(encoding="utf-8")
    assert 'provider: "orcarouter-oauth"' in content
    assert 'api_key_env: ""' in content


# --------------------------------------------------------------------------
# key subcommand
# --------------------------------------------------------------------------


def test_key_set_from_stdin_stores_and_masks(store, monkeypatch, capsys) -> None:
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{FAKE_KEY}\n"))
    assert cmd_orcarouter(_args(orcarouter_command="key", set=True, stdin=True, clear=False)) == 0
    out = capsys.readouterr().out
    assert FAKE_KEY not in out
    assert "Stored OrcaRouter API key" in out
    assert store.status("orcarouter").configured is True


def test_key_rejects_a_malformed_value(store, monkeypatch, capsys) -> None:
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("sk-openai-nope\n"))
    assert cmd_orcarouter(_args(orcarouter_command="key", set=True, stdin=True, clear=False)) == 1
    assert "sk-orca" in capsys.readouterr().err


def test_key_status_and_clear(store, capsys) -> None:
    assert cmd_orcarouter(_args(orcarouter_command="key", set=False, stdin=False, clear=False)) == 0
    assert "not configured" in capsys.readouterr().out

    store.save("orcarouter", FAKE_KEY, source="api_key")
    assert cmd_orcarouter(_args(orcarouter_command="key", set=False, stdin=False, clear=False)) == 0
    out = capsys.readouterr().out
    assert FAKE_KEY not in out
    assert "sk-orca" in out

    assert cmd_orcarouter(_args(orcarouter_command="key", set=False, stdin=False, clear=True)) == 0
    assert store.status("orcarouter").configured is False


# --------------------------------------------------------------------------
# status subcommand
# --------------------------------------------------------------------------


def test_status_reports_both_entries_without_secrets(store, capsys) -> None:
    store.save("orcarouter", FAKE_KEY, source="api_key")
    assert cmd_orcarouter(_args(orcarouter_command="status", json=False)) == 0
    out = capsys.readouterr().out
    assert "OrcaRouter — API" in out
    assert "OrcaRouter — Auth" in out
    assert "www.orcarouter.ai" in out
    assert "api.orcarouter.ai/v1" in out
    assert FAKE_KEY not in out


def test_status_json_is_machine_readable_and_secret_free(store, capsys) -> None:
    store.save("orcarouter", FAKE_KEY, source="api_key")
    assert cmd_orcarouter(_args(orcarouter_command="status", json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["endpoints"]["auth_base"] == "https://www.orcarouter.ai"
    assert payload["providers"]["orcarouter"]["configured"] is True
    assert FAKE_KEY not in json.dumps(payload)
    assert "secret_masked" in payload["providers"]["orcarouter"]


def test_status_flags_a_revoked_credential(store, capsys) -> None:
    credential = store.save("orcarouter-oauth", FAKE_KEY, source="pkce", grant_id="1")
    store.mark_needs_reauth("orcarouter-oauth", credential.generation)
    assert cmd_orcarouter(_args(orcarouter_command="status", json=False)) == 0
    out = capsys.readouterr().out
    assert "reauthorization" in out
    assert "orcarouter login" in out


# --------------------------------------------------------------------------
# models subcommand
# --------------------------------------------------------------------------


def test_models_requires_a_credential(store, capsys) -> None:
    assert cmd_orcarouter(_args(orcarouter_command="models", capability="chat", modality=[], json=False, refresh=False)) == 1
    err = capsys.readouterr().err
    assert "orcarouter key --set" in err
    assert "orcarouter login" in err


def test_models_lists_the_live_catalogue(store, capsys, monkeypatch) -> None:
    store.save("orcarouter", FAKE_KEY, source="api_key")
    live = {
        "data": [
            {
                "id": "deepseek/deepseek-v4-pro",
                "supported_endpoint_types": ["openai"],
                "architecture": {"input_modalities": ["text"]},
                "context_length": 1048576,
            },
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "supported_endpoint_types": ["openai"],
                "architecture": {"input_modalities": ["text", "image"]},
            },
        ]
    }
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_catalog._default_fetcher",
        lambda url, key, timeout, max_bytes: live,
    )
    assert cmd_orcarouter(_args(orcarouter_command="models", capability="chat", modality=[], json=False, refresh=True)) == 0
    out = capsys.readouterr().out
    assert "deepseek/deepseek-v4-pro" in out
    assert "live catalogue" in out
    assert FAKE_KEY not in out

    # With an image attached, only the model that declares image input remains.
    assert cmd_orcarouter(_args(orcarouter_command="models", capability="chat", modality=["image"], json=False, refresh=True)) == 0
    out = capsys.readouterr().out
    assert "deepseek/deepseek-v4.1-flash" in out
    assert "deepseek-v4-pro" not in out


def test_models_json_reports_source_and_degradation(store, capsys, monkeypatch) -> None:
    store.save("orcarouter", FAKE_KEY, source="api_key")
    import urllib.error

    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_catalog._default_fetcher",
        lambda url, key, timeout, max_bytes: (_ for _ in ()).throw(
            urllib.error.URLError("down")
        ),
    )
    assert cmd_orcarouter(_args(orcarouter_command="models", capability="chat", modality=[], json=True, refresh=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["degraded"] is True
    assert payload["source"] == "seed"
    assert FAKE_KEY not in json.dumps(payload)


# --------------------------------------------------------------------------
# login subcommand
# --------------------------------------------------------------------------


def test_login_reuses_an_existing_durable_credential(store, capsys) -> None:
    """Re-authorizing on every launch would burn the 10-keys-per-day cap."""
    store.save("orcarouter-oauth", FAKE_KEY, source="pkce", grant_id="42")
    assert cmd_orcarouter(
        _args(orcarouter_command="login", flow="auto", app_name="", scope="api",
              login_hint="", no_browser=True)
    ) == 0
    out = capsys.readouterr().out
    assert "Already connected" in out
    assert "reused until revoked" in out
    assert FAKE_KEY not in out


def test_login_oob_prints_the_authorize_url_and_persists_the_key(
    store, capsys, monkeypatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _p: "the-code")
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (
            200,
            json.dumps({"key": FAKE_KEY, "user_id": "5", "scope": "api"}).encode(),
        ),
    )
    assert cmd_orcarouter(
        _args(orcarouter_command="login", flow="oob", app_name="", scope="api",
              login_hint="", no_browser=True)
    ) == 0
    out = capsys.readouterr().out
    assert "https://www.orcarouter.ai/auth?" in out
    assert "code_challenge_method=S256" in out
    assert "api.orcarouter.ai/v1/auth/keys" not in out
    assert FAKE_KEY not in out
    assert store.status("orcarouter-oauth").configured is True


def test_login_denied_exits_cleanly(store, capsys, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _p: "the-code")
    monkeypatch.setattr(
        "researchclaw.llm.orcarouter_pkce._default_post_json",
        lambda url, payload, timeout: (403, b'{"error":"invalid_grant"}'),
    )
    assert cmd_orcarouter(
        _args(orcarouter_command="login", flow="oob", app_name="", scope="api",
              login_hint="", no_browser=True)
    ) == 1
    err = capsys.readouterr().err
    assert "expired" in err or "already used" in err
    assert store.status("orcarouter-oauth").configured is False


def test_login_rejects_an_api_shaped_auth_origin(store, capsys, monkeypatch) -> None:
    """The classic bug: pointing auth at the inference origin."""
    monkeypatch.setenv("ORCA_AUTH_BASE_URL", "https://api.orcarouter.ai/v1")
    monkeypatch.setattr("builtins.input", lambda _p: "the-code")
    assert cmd_orcarouter(
        _args(orcarouter_command="login", flow="oob", app_name="", scope="api",
              login_hint="", no_browser=True)
    ) == 1
    assert "404" in capsys.readouterr().err


def test_login_cancel_does_not_hang(store, capsys, monkeypatch) -> None:
    def _interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)
    assert cmd_orcarouter(
        _args(orcarouter_command="login", flow="oob", app_name="", scope="api",
              login_hint="", no_browser=True)
    ) == 1
    assert "Cancelled" in capsys.readouterr().err


# --------------------------------------------------------------------------
# logout subcommand
# --------------------------------------------------------------------------


def test_logout_forgets_the_pkce_credential_only(store, capsys) -> None:
    store.save("orcarouter", FAKE_KEY, source="api_key")
    store.save("orcarouter-oauth", FAKE_KEY, source="pkce", grant_id="1")
    assert cmd_orcarouter(_args(orcarouter_command="logout", yes=True)) == 0
    assert store.status("orcarouter-oauth").configured is False
    assert store.status("orcarouter").configured is True
    out = capsys.readouterr().out
    assert "console/authorized-apps" in out


def test_logout_with_nothing_stored_is_a_noop(store, capsys) -> None:
    assert cmd_orcarouter(_args(orcarouter_command="logout", yes=True)) == 0
    assert "No PKCE-issued" in capsys.readouterr().out
