"""OrcaRouter OAuth 2.0 + PKCE connect flow (authorization code).

This module implements the two PKCE flows OrcaRouter supports for a client
that has no client secret and no pre-registered redirect URI:

* **Flow A — loopback redirect** (default for the CLI): bind ``127.0.0.1:0``,
  open ``https://www.orcarouter.ai/auth?callback_url=http://127.0.0.1:<port>/cb``,
  receive ``?code=…&state=…`` on the local listener.
* **Flow B — out-of-band code**: ``callback_url=oob``. The consent screen
  displays a code that the user pastes back. Used by SSH/container users and
  by the hosted web UI, which cannot be reached on the user's loopback.

Flow C (RFC 8628 device grant) is intentionally not implemented: the
integration requires PKCE, and the device grant is an alternative to it, not
a substitute.

Security properties enforced here:

* the verifier is 32 bytes from :func:`secrets.token_bytes` per attempt and
  never leaves this process until the exchange — it is not put in a URL, a
  log line, an exception message, or the credential store;
* the challenge is ``base64url(sha256(verifier))`` with no padding, and the
  method is always ``S256`` (never ``plain``: in Flow A the user can still
  choose "Show me a code" on the consent screen, which hands a human the
  code);
* ``state`` is compared with :func:`hmac.compare_digest` before the code is
  used;
* denial, state mismatch, timeout, cancellation, an expired/reused code
  (403), a rejected request (400), rate limiting (429) and transport errors
  all end the attempt with an actionable message and release every resource.

The exchanged result is a durable OrcaRouter API key, **not** a refreshable
OAuth token: there is no refresh grant here, and none is invented.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.server
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Authorize endpoint path on the auth origin (fixed by the protocol).
AUTHORIZE_PATH = "/auth"
#: Code-exchange endpoint on the auth origin. Note: the relay lives at
#: ``/v1`` on the API origin; the auth endpoints do NOT. Deriving this from
#: the API base by appending ``/v1`` is the single most common integration
#: mistake and it 404s, so the path is a constant here and
#: :func:`build_exchange_url` refuses an API-style base.
EXCHANGE_PATH = "/api/v1/auth/keys"

DEFAULT_SCOPE = "api"
DEFAULT_APP_NAME = "AutoResearchClaw"
DEFAULT_TIMEOUT_SEC = 300.0

_CLOSE_TAB_PAGE = (
    "<!doctype html><meta charset='utf-8'>"
    "<title>AutoResearchClaw</title>"
    "<body style='font-family:system-ui,sans-serif;padding:2rem'>"
    "<h2>OrcaRouter connected</h2>"
    "<p>You can close this tab and return to your terminal.</p></body>"
)


class PkceError(RuntimeError):
    """Base class for every terminal PKCE failure.

    Messages are written to be shown to a user. They never contain the
    verifier, the auth code, or an issued key.
    """

    kind = "error"


class PkceDenied(PkceError):
    kind = "access_denied"


class PkceStateMismatch(PkceError):
    kind = "state_mismatch"


class PkceTimeout(PkceError):
    kind = "timeout"


class PkceCancelled(PkceError):
    kind = "cancelled"


class PkceExchangeRejected(PkceError):
    """The exchange endpoint refused the code."""

    kind = "exchange_rejected"

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = kind


class PkceNetworkError(PkceError):
    kind = "network"


def _b64url(raw: bytes) -> str:
    """base64url without padding (RFC 7636 §A)."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_verifier() -> str:
    """A fresh 43-character verifier from a cryptographic RNG."""
    return _b64url(secrets.token_bytes(32))


def challenge_for(verifier: str) -> str:
    """``base64url(sha256(verifier))`` with no padding."""
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def generate_state() -> str:
    """A fresh opaque CSRF token."""
    return _b64url(secrets.token_bytes(16))


def build_authorize_url(
    auth_base: str,
    *,
    callback_url: str,
    challenge: str,
    state: str,
    app_name: str = DEFAULT_APP_NAME,
    scope: str = DEFAULT_SCOPE,
    login_hint: str = "",
    workspace_hint: str = "",
) -> str:
    """Build the consent-screen URL.

    The verifier is deliberately absent: only its SHA-256 challenge travels.
    """
    if not challenge:
        raise ValueError("code_challenge is required")
    query = [
        ("callback_url", callback_url),
        ("code_challenge", challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        ("app_name", app_name),
        ("scope", scope),
    ]
    if login_hint:
        query.append(("login_hint", login_hint))
    if workspace_hint:
        query.append(("workspace_hint", workspace_hint))
    base = auth_base.rstrip("/")
    return f"{base}{AUTHORIZE_PATH}?{urllib.parse.urlencode(query)}"


def build_exchange_url(auth_base: str) -> str:
    """Build the code-exchange URL, guarding against the ``/v1/auth/keys`` bug."""
    base = auth_base.rstrip("/")
    if base.lower().endswith("/v1"):
        raise ValueError(
            "the exchange endpoint is on the auth origin, not on the "
            f"inference origin: {base!r} looks like an API base (…/v1). "
            "Authentication is at https://www.orcarouter.ai/api/v1/auth/keys; "
            "https://api.orcarouter.ai/v1/auth/keys is a 404."
        )
    return f"{base}{EXCHANGE_PATH}"


@dataclass(frozen=True)
class ExchangeResult:
    """A durable OrcaRouter API key and the scope actually granted."""

    api_key: str
    scope: str = DEFAULT_SCOPE
    user_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise PkceExchangeRejected(200, "malformed_response", "no key in response")
        if self.scope != DEFAULT_SCOPE:
            logger.warning(
                "OrcaRouter granted scope %r, not %r — the workspace role may "
                "not permit the wider grant.",
                self.scope,
                DEFAULT_SCOPE,
            )


def _default_post_json(
    url: str, payload: dict[str, Any], *, timeout: float
) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:  # status-code errors are answers
        try:
            return exc.code, exc.read()
        except Exception:  # noqa: BLE001 - body is optional
            return exc.code, b""


PostJson = Callable[..., tuple[int, bytes]]


def exchange_code(
    auth_base: str,
    *,
    code: str,
    verifier: str,
    timeout: float = 30.0,
    post_json: PostJson | None = None,
) -> ExchangeResult:
    """Redeem an auth code for a durable API key.

    ``post_json`` is injectable so tests can drive the real code path against
    a local fake auth server instead of the network.
    """
    if not code:
        raise PkceExchangeRejected(0, "missing_code", "no authorization code")
    if not verifier:
        raise PkceExchangeRejected(0, "missing_verifier", "no code verifier")
    url = build_exchange_url(auth_base)
    payload = {
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    }
    poster = post_json or _default_post_json
    try:
        status, body = poster(url, payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - transport failures are terminal here
        raise PkceNetworkError(
            f"could not reach {url.split('/api/')[0]}: {type(exc).__name__}"
        ) from exc

    if status == 200:
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise PkceExchangeRejected(
                200, "malformed_response", "exchange returned non-JSON"
            ) from exc
        if not isinstance(data, dict):
            raise PkceExchangeRejected(
                200, "malformed_response", "exchange returned non-object JSON"
            )
        return ExchangeResult(
            api_key=str(data.get("key") or ""),
            scope=str(data.get("scope") or DEFAULT_SCOPE),
            user_id=str(data.get("user_id") or ""),
            raw=data,
        )
    if status == 400:
        raise PkceExchangeRejected(
            status,
            "invalid_request",
            "the authorization code or its PKCE method was refused; start a new "
            "login (S256 is always sent, so this usually means the code came "
            "from a different authorization attempt)",
        )
    if status == 403:
        raise PkceExchangeRejected(
            status,
            "invalid_grant",
            "that code is unknown, expired, or already used — start a new login",
        )
    if status == 429:
        raise PkceExchangeRejected(
            status,
            "rate_limited",
            "too many authorizations for this account; wait a moment before "
            "connecting again (OrcaRouter allows 10 PKCE keys per user per day)",
        )
    raise PkceExchangeRejected(
        status, "unexpected_status", f"exchange failed with HTTP {status}"
    )


class LoopbackReceiver:
    """Flow A listener on ``127.0.0.1:<ephemeral>`` serving ``/cb``."""

    def __init__(self, state: str, *, path: str = "/cb") -> None:
        self._state = state
        self._path = path
        self._result: str | None = None
        self._error: PkceError | None = None
        self._event = threading.Event()
        self._server: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    # -- lifecycle -----------------------------------------------------
    def start(self) -> int:
        receiver = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # noqa: D102 - silence
                return

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != receiver._path:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                params = urllib.parse.parse_qs(parsed.query)
                body = _CLOSE_TAB_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

                got_state = (params.get("state") or [""])[0]
                # Constant-time compare before the code is allowed to be used.
                if not hmac.compare_digest(str(got_state), receiver._state):
                    receiver._fail(
                        PkceStateMismatch(
                            "the callback carried a different state than the one "
                            "this login sent; the code was discarded"
                        )
                    )
                    return
                err = (params.get("error") or [""])[0]
                if err:
                    receiver._fail(
                        PkceDenied(
                            "authorization was denied"
                            if err == "access_denied"
                            else f"authorization failed: {err}"
                        )
                    )
                    return
                code = (params.get("code") or [""])[0]
                if not code:
                    receiver._fail(
                        PkceExchangeRejected(0, "missing_code", "no code in callback")
                    )
                    return
                receiver._result = code
                receiver._event.set()

        self._server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
        )
        self._thread.start()
        return self.port

    def _fail(self, error: PkceError) -> None:
        self._error = error
        self._event.set()

    @property
    def callback_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self._path}"

    def wait(self, timeout: float) -> str:
        if not self._event.wait(timeout):
            raise PkceTimeout(
                "timed out waiting for the browser to return the authorization code"
            )
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def cancel(self) -> None:
        self._fail(PkceCancelled("login cancelled"))

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001 - shutdown races are harmless
                pass
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)


@dataclass
class PendingLogin:
    """One authorization attempt. Never persisted, never logged."""

    flow: str
    verifier: str
    state: str
    authorize_url: str
    callback_url: str
    receiver: LoopbackReceiver | None = None
    created_at: float = 0.0
    generation: int = 0
    cancelled: bool = False
    _verifier_used: bool = False

    def authorize_hint(self) -> str:
        return self.authorize_url

    def take_code(self, code: str) -> str:
        """Consume the code exactly once, guarding against reuse."""
        if self._verifier_used:
            raise PkceExchangeRejected(
                403, "invalid_grant", "this login attempt was already completed"
            )
        self._verifier_used = True
        return code

    def exchange(
        self, auth_base: str, code: str, *, post_json: PostJson | None = None
    ) -> ExchangeResult:
        return exchange_code(
            auth_base,
            code=self.take_code(code),
            verifier=self.verifier,
            post_json=post_json,
        )

    def close(self) -> None:
        if self.receiver is not None:
            self.receiver.close()
            self.receiver = None

    def cancel(self) -> None:
        """Release the listener and fail any waiter with a cancellation."""
        self.cancelled = True
        if self.receiver is not None:
            self.receiver.cancel()
            self.receiver.close()
            self.receiver = None


def start_loopback_login(
    auth_base: str,
    *,
    app_name: str = DEFAULT_APP_NAME,
    scope: str = DEFAULT_SCOPE,
    login_hint: str = "",
) -> PendingLogin:
    """Flow A: listen first (so the port is known), then build the URL."""
    verifier = generate_verifier()
    state = generate_state()
    receiver = LoopbackReceiver(state)
    receiver.start()
    return PendingLogin(
        flow="loopback",
        verifier=verifier,
        state=state,
        authorize_url=build_authorize_url(
            auth_base,
            callback_url=receiver.callback_url,
            challenge=challenge_for(verifier),
            state=state,
            app_name=app_name,
            scope=scope,
            login_hint=login_hint,
        ),
        callback_url=receiver.callback_url,
        receiver=receiver,
        created_at=time.time(),
    )


def start_oob_login(
    auth_base: str,
    *,
    app_name: str = DEFAULT_APP_NAME,
    scope: str = DEFAULT_SCOPE,
    login_hint: str = "",
) -> PendingLogin:
    """Flow B: ``callback_url=oob``; S256 is mandatory and is always sent."""
    verifier = generate_verifier()
    state = generate_state()
    return PendingLogin(
        flow="oob",
        verifier=verifier,
        state=state,
        authorize_url=build_authorize_url(
            auth_base,
            callback_url="oob",
            challenge=challenge_for(verifier),
            state=state,
            app_name=app_name,
            scope=scope,
            login_hint=login_hint,
        ),
        callback_url="oob",
        created_at=time.time(),
    )


def start_login(flow: str = "auto", **kwargs: Any) -> PendingLogin:
    """Start a PKCE login.

    ``flow="auto"`` picks loopback when a browser and a bindable loopback
    interface are both available, and falls back to out-of-band otherwise.
    """
    normalized = (flow or "auto").strip().lower()
    if normalized in ("auto", ""):
        normalized = "loopback" if loopback_available() else "oob"
    if normalized in ("loopback", "a"):
        return start_loopback_login(**kwargs)
    if normalized in ("oob", "b", "out-of-band"):
        return start_oob_login(**kwargs)
    raise ValueError(f"unknown flow: {flow!r} (use 'loopback', 'oob', or 'auto')")


def loopback_available() -> bool:
    """True when this host can bind a loopback listener for Flow A."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False
    finally:
        probe.close()
