"""PKCE protocol tests for the OrcaRouter connect flow.

These drive the real adapter code path — authorize URL construction, the
loopback listener, and the exchange — against a local fake auth server. They
never touch the network and never use a real credential.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import threading
import urllib.parse
import urllib.request

import pytest

from researchclaw.llm.orcarouter_pkce import (
    AUTHORIZE_PATH,
    EXCHANGE_PATH,
    PkceDenied,
    PkceExchangeRejected,
    PkceNetworkError,
    PkceStateMismatch,
    PkceTimeout,
    build_authorize_url,
    build_exchange_url,
    challenge_for,
    exchange_code,
    generate_state,
    generate_verifier,
    start_login,
    start_loopback_login,
    start_oob_login,
)

FAKE_KEY = "sk-orca-fake-key-for-tests-0001"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


# --------------------------------------------------------------------------
# Fake auth server: a real HTTP server, so the adapter's own HTTP code runs.
# --------------------------------------------------------------------------


class FakeAuthServer:
    """A minimal OrcaRouter auth origin. Records what it was asked."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responder = None  # callable(payload) -> (status, dict)
        self._server = http.server.HTTPServer(("127.0.0.1", 0), self._handler())
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _handler(self):
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # noqa: D102
                return

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {"_raw": raw.decode("utf-8", "replace")}
                outer.requests.append({"path": self.path, "payload": payload,
                                       "content_type": self.headers.get("Content-Type")})
                status, body = outer.responder(payload) if outer.responder else (200, {})
                encoded = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return _Handler

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def auth_server():
    server = FakeAuthServer()
    try:
        yield server
    finally:
        server.close()


# --------------------------------------------------------------------------
# verifier / challenge / state
# --------------------------------------------------------------------------


def test_verifier_is_fresh_crypto_randomness() -> None:
    verifiers = {generate_verifier() for _ in range(200)}
    assert len(verifiers) == 200, "verifier must be fresh per attempt"
    for verifier in verifiers:
        # RFC 7636 §4.1: 43-128 chars from the unreserved set.
        assert 43 <= len(verifier) <= 128
        assert set(verifier) <= set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
        )
        assert "=" not in verifier, "base64url must be unpadded"

    assert len({generate_state() for _ in range(200)}) == 200


def test_challenge_is_unpadded_base64url_sha256() -> None:
    verifier = "A" * 43
    expected = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    assert challenge_for(verifier) == expected
    assert "=" not in challenge_for(verifier)


def test_authorize_url_sends_only_the_challenge_never_the_verifier() -> None:
    pending = start_oob_login("https://auth.example.test", app_name="Tool X")
    parsed = urllib.parse.urlparse(pending.authorize_url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.path == AUTHORIZE_PATH
    assert parsed.netloc == "auth.example.test"
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"] == [challenge_for(pending.verifier)]
    assert params["callback_url"] == ["oob"]
    assert params["app_name"] == ["Tool X"]
    assert params["scope"] == ["api"]
    assert pending.state in params["state"]

    # The verifier must not appear anywhere in the URL, in any encoding.
    assert pending.verifier not in pending.authorize_url
    assert urllib.parse.quote(pending.verifier, safe="") not in pending.authorize_url


def test_loopback_authorize_url_points_at_the_bound_port() -> None:
    pending = start_loopback_login("https://auth.example.test")
    try:
        assert pending.flow == "loopback"
        assert pending.receiver is not None
        params = urllib.parse.parse_qs(
            urllib.parse.urlparse(pending.authorize_url).query
        )
        assert params["callback_url"] == [f"http://127.0.0.1:{pending.receiver.port}/cb"]
        assert params["code_challenge_method"] == ["S256"]
    finally:
        pending.close()


def test_auto_flow_falls_back_to_oob_without_loopback(monkeypatch) -> None:
    import researchclaw.llm.orcarouter_pkce as mod

    monkeypatch.setattr(mod, "loopback_available", lambda: False)
    pending = mod.start_login("auto", auth_base="https://auth.example.test")
    assert pending.flow == "oob"


def test_unknown_flow_is_rejected() -> None:
    with pytest.raises(ValueError):
        start_login("device", auth_base="https://auth.example.test")


# --------------------------------------------------------------------------
# Exchange
# --------------------------------------------------------------------------


def test_exchange_uses_the_auth_origin_and_the_documented_body(auth_server) -> None:
    auth_server.responder = lambda payload: (
        200,
        {"key": FAKE_KEY, "user_id": "42", "scope": "api"},
    )
    result = exchange_code(
        auth_server.base_url, code="the-code", verifier="the-verifier"
    )

    assert len(auth_server.requests) == 1
    sent = auth_server.requests[0]
    assert sent["path"] == EXCHANGE_PATH, "exchange must POST to /api/v1/auth/keys"
    assert sent["path"] != "/v1/auth/keys"
    assert sent["payload"] == {
        "code": "the-code",
        "code_verifier": "the-verifier",
        "code_challenge_method": "S256",
    }
    assert result.api_key == FAKE_KEY
    assert result.scope == "api"
    assert result.user_id == "42"


def test_exchange_never_returns_the_wrong_origin_path() -> None:
    url = build_exchange_url("https://www.orcarouter.ai")
    assert url == "https://www.orcarouter.ai/api/v1/auth/keys"
    assert "api.orcarouter.ai/v1/auth/keys" not in url

    with pytest.raises(ValueError) as excinfo:
        build_exchange_url("https://api.orcarouter.ai/v1")
    assert "404" in str(excinfo.value)


@pytest.mark.parametrize(
    ("status", "reason_fragment"),
    [
        (400, "PKCE method"),
        (403, "expired"),
        (429, "too many"),
    ],
)
def test_exchange_terminal_errors_are_actionable(
    auth_server, status, reason_fragment
) -> None:
    auth_server.responder = lambda payload: (status, {"error": "x"})
    with pytest.raises(PkceExchangeRejected) as excinfo:
        exchange_code(auth_server.base_url, code="c", verifier="v")
    assert excinfo.value.status == status
    message = str(excinfo.value)
    assert any(word in message.lower() for word in ("login", "authoriz", "wait"))
    # The verifier never leaks into an error message.
    assert "v" != message
    assert "code_verifier" not in message


def test_exchange_403_is_classified_as_invalid_grant(auth_server) -> None:
    auth_server.responder = lambda payload: (403, {"error": "invalid_grant"})
    with pytest.raises(PkceExchangeRejected) as excinfo:
        exchange_code(auth_server.base_url, code="used-code", verifier="v")
    assert excinfo.value.reason == "invalid_grant"


def test_exchange_malformed_200_is_rejected(auth_server) -> None:
    auth_server.responder = lambda payload: (200, {"scope": "api"})
    with pytest.raises(PkceExchangeRejected) as excinfo:
        exchange_code(auth_server.base_url, code="c", verifier="v")
    assert excinfo.value.reason == "malformed_response"


def test_exchange_transport_failure_is_terminal_not_a_hot_loop() -> None:
    # Nothing is listening on this port.
    with pytest.raises(PkceNetworkError):
        exchange_code("http://127.0.0.1:9", code="c", verifier="v", timeout=2)


def test_scope_downgrade_is_surfaced(auth_server, caplog) -> None:
    auth_server.responder = lambda payload: (
        200,
        {"key": FAKE_KEY, "user_id": "7", "scope": "read"},
    )
    with caplog.at_level("WARNING"):
        result = exchange_code(auth_server.base_url, code="c", verifier="v")
    assert result.scope == "read"
    assert any("scope" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------
# Flow A end-to-end through the adapter (fake auth origin, real listener)
# --------------------------------------------------------------------------


def _drive_callback(callback_url: str, query: str) -> int:
    with urllib.request.urlopen(f"{callback_url}?{query}", timeout=5) as response:
        return response.status


def test_flow_a_happy_path_through_the_adapter(auth_server) -> None:
    auth_server.responder = lambda payload: (
        200,
        {"key": FAKE_KEY, "user_id": "9", "scope": "api"},
    )
    pending = start_loopback_login(auth_server.base_url)
    try:
        assert pending.receiver is not None
        holder: dict = {}

        def deliver() -> None:
            holder["status"] = _drive_callback(
                pending.callback_url, f"code=one-time-code&state={pending.state}"
            )

        thread = threading.Thread(target=deliver, daemon=True)
        thread.start()

        code = pending.receiver.wait(timeout=10)
        thread.join(timeout=5)
        assert holder["status"] == 200, "the browser must get a closeable page"

        credential = pending.exchange(auth_server.base_url, code)
        assert credential.api_key == FAKE_KEY
        assert auth_server.requests[-1]["payload"]["code_verifier"] == pending.verifier
    finally:
        pending.close()


def test_flow_a_state_mismatch_discards_the_code() -> None:
    pending = start_loopback_login("https://auth.example.test")
    try:
        assert pending.receiver is not None

        def deliver() -> None:
            try:
                _drive_callback(pending.callback_url, "code=stolen&state=not-our-state")
            except Exception:  # noqa: BLE001 - the response may be cut short
                pass

        threading.Thread(target=deliver, daemon=True).start()
        with pytest.raises(PkceStateMismatch):
            pending.receiver.wait(timeout=10)
    finally:
        pending.close()


def test_flow_a_denial_is_reported_as_denial() -> None:
    pending = start_loopback_login("https://auth.example.test")
    try:
        assert pending.receiver is not None
        threading.Thread(
            target=lambda: _drive_callback(
                pending.callback_url, f"error=access_denied&state={pending.state}"
            ),
            daemon=True,
        ).start()
        with pytest.raises(PkceDenied):
            pending.receiver.wait(timeout=10)
    finally:
        pending.close()


def test_flow_a_cancel_and_timeout_release_the_listener() -> None:
    pending = start_loopback_login("https://auth.example.test")
    receiver = pending.receiver
    assert receiver is not None
    pending.cancel()
    with pytest.raises(Exception):
        receiver.wait(timeout=5)
    pending.close()
    assert pending.receiver is None

    timed_out = start_loopback_login("https://auth.example.test")
    try:
        with pytest.raises(PkceTimeout):
            timed_out.receiver.wait(timeout=0.3)
    finally:
        timed_out.close()


def test_code_cannot_be_redeemed_twice_from_one_attempt(auth_server) -> None:
    auth_server.responder = lambda payload: (200, {"key": FAKE_KEY, "scope": "api"})
    pending = start_oob_login(auth_server.base_url)
    pending.exchange(auth_server.base_url, "one-time-code")
    with pytest.raises(PkceExchangeRejected) as excinfo:
        pending.exchange(auth_server.base_url, "one-time-code")
    assert excinfo.value.status == 403
    assert len(auth_server.requests) == 1, "a reused code must not be re-sent"


# --------------------------------------------------------------------------
# Secrets stay out of logs and errors
# --------------------------------------------------------------------------


def test_verifier_and_key_never_appear_in_logs_or_errors(auth_server, caplog, capsys) -> None:
    auth_server.responder = lambda payload: (403, {"error": "invalid_grant"})
    pending = start_oob_login(auth_server.base_url)
    with caplog.at_level("DEBUG"):
        with pytest.raises(PkceExchangeRejected) as excinfo:
            pending.exchange(auth_server.base_url, "code-value")
        message = str(excinfo.value)

    captured = capsys.readouterr()
    for haystack in (
        caplog.text,
        captured.out,
        captured.err,
        message,
    ):
        assert pending.verifier not in haystack
        assert "code-value" not in haystack
        assert FAKE_KEY not in haystack
    # The authorize URL deliberately carries the challenge and the state (the
    # server echoes the state back so we can compare it) — but never the
    # verifier, which is what makes an intercepted code unredeemable.
    assert pending.verifier not in pending.authorize_url
    assert challenge_for(pending.verifier) in pending.authorize_url
    assert pending.state in pending.authorize_url


def test_build_authorize_url_requires_a_challenge() -> None:
    with pytest.raises(ValueError):
        build_authorize_url("https://a.test", callback_url="oob", challenge="", state="s")
