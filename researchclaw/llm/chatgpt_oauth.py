"""ChatGPT Plus/Pro subscription OAuth 2.0 PKCE authentication.

Uses the same public OAuth endpoints as the official Codex CLI.
Tokens are stored securely via the OS keychain (keyring library).
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_KEYRING_SERVICE = "researchclaw-chatgpt"
_KEYRING_USERNAME = "oauth-tokens"

CLIENT_ID = "DRivsnm2Mu42T3KOpqdtwB3NYviHYzwD"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
AUDIENCE = "https://api.openai.com/v1"

try:
    import keyring

    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False


@dataclass
class AuthTokens:
    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str = ""

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "account_id": self.account_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthTokens:
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data.get("expires_at", 0),
            account_id=data.get("account_id", ""),
        )


def _require_keyring() -> None:
    if not HAS_KEYRING:
        raise ImportError(
            "keyring is required for ChatGPT authentication. "
            "Install: pip install 'researchclaw[chatgpt]'"
        )


def extract_account_id(access_token: str) -> str:
    """Decode the JWT payload to extract the account/subject ID."""
    try:
        payload_b64 = access_token.split(".")[1]
        padding = (-len(payload_b64)) % 4
        if padding:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("sub", payload.get("account_id", ""))
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Secure token storage via OS keychain
# ---------------------------------------------------------------------------

_NO_KEYRING_MSG = (
    "No OS keychain backend available. On Linux, install one of: "
    "gnome-keyring, kwallet, or secretstorage. On headless servers, "
    "use an API key provider instead of chatgpt."
)


def _safe_keyring_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Wrap a keyring operation with user-friendly error handling."""
    _require_keyring()
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        exc_name = type(exc).__name__
        if "NoKeyringError" in exc_name or "InitError" in exc_name:
            raise RuntimeError(_NO_KEYRING_MSG) from exc
        raise


def save_auth(tokens: AuthTokens) -> None:
    _safe_keyring_call(
        keyring.set_password,
        _KEYRING_SERVICE, _KEYRING_USERNAME, json.dumps(tokens.to_dict()),
    )
    logger.debug("Saved auth tokens to OS keychain")


def load_auth() -> AuthTokens | None:
    raw = _safe_keyring_call(keyring.get_password, _KEYRING_SERVICE, _KEYRING_USERNAME)
    if not raw:
        return None
    try:
        return AuthTokens.from_dict(json.loads(raw))
    except (json.JSONDecodeError, KeyError):
        return None


def clear_auth() -> None:
    try:
        _safe_keyring_call(keyring.delete_password, _KEYRING_SERVICE, _KEYRING_USERNAME)
    except (RuntimeError, keyring.errors.PasswordDeleteError):
        pass


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url() -> tuple[str, str, str]:
    """Build the OAuth authorization URL with PKCE.

    Returns (url, state, code_verifier).
    """
    code_verifier, code_challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "audience": AUDIENCE,
    }
    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    return url, state, code_verifier


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------

def _post_token_request(data: dict[str, str]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode("ascii")
    req = urllib.request.Request(
        TOKEN_URL,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def exchange_code(code: str, code_verifier: str) -> AuthTokens:
    """Exchange an authorization code for access/refresh tokens."""
    result = _post_token_request({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": REDIRECT_URI,
    })
    access = result["access_token"]
    return AuthTokens(
        access_token=access,
        refresh_token=result["refresh_token"],
        expires_at=time.time() + result.get("expires_in", 3600),
        account_id=extract_account_id(access),
    )


def refresh_tokens(refresh_token: str) -> AuthTokens:
    """Refresh an expired access token."""
    result = _post_token_request({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    })
    access = result["access_token"]
    return AuthTokens(
        access_token=access,
        refresh_token=result.get("refresh_token", refresh_token),
        expires_at=time.time() + result.get("expires_in", 3600),
        account_id=extract_account_id(access),
    )


def get_valid_tokens() -> AuthTokens:
    """Load tokens from keychain, refreshing if expired."""
    tokens = load_auth()
    if tokens is None:
        raise RuntimeError(
            "Not logged in. Run 'researchclaw login' first."
        )
    if tokens.expired:
        logger.info("Access token expired, refreshing...")
        tokens = refresh_tokens(tokens.refresh_token)
        save_auth(tokens)
    return tokens


# ---------------------------------------------------------------------------
# Local OAuth callback server
# ---------------------------------------------------------------------------

class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handles the OAuth redirect callback on localhost."""

    auth_code: str | None = None
    auth_state: str | None = None
    error: str | None = None
    _event: threading.Event

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_error(404, "Not Found")
            return

        params = urllib.parse.parse_qs(parsed.query)

        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]

        if error:
            _OAuthCallbackHandler.error = error
            self._send_html(
                "<h1>Authentication Failed</h1>"
                f"<p>Error: {error}</p>"
                "<p>You can close this window.</p>"
            )
            _OAuthCallbackHandler._event.set()
        elif code:
            _OAuthCallbackHandler.auth_code = code
            _OAuthCallbackHandler.auth_state = state
            self._send_html(
                "<h1>Authentication Successful!</h1>"
                "<p>You can close this window and return to the terminal.</p>"
                "<script>window.close()</script>"
            )
            _OAuthCallbackHandler._event.set()
        else:
            self._send_html(
                "<h1>Unknown Response</h1>"
                "<p>No authorization code received. Waiting for valid callback...</p>"
            )

    def _send_html(self, body: str) -> None:
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>ResearchClaw Auth</title>"
            "<style>body{font-family:system-ui;text-align:center;padding:60px 20px;}</style>"
            f"</head><body>{body}</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def run_oauth_flow(timeout: int = 120) -> AuthTokens:
    """Run the complete browser-based OAuth flow.

    1. Start a local HTTP server on port 1455
    2. Open the browser to the authorization URL
    3. Wait for the callback with the authorization code
    4. Exchange the code for tokens
    5. Store tokens in OS keychain
    """
    _require_keyring()

    auth_url, state, code_verifier = build_authorize_url()

    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.auth_state = None
    _OAuthCallbackHandler.error = None
    _OAuthCallbackHandler._event = threading.Event()

    port = int(urllib.parse.urlparse(REDIRECT_URI).port or 1455)
    try:
        server = http.server.HTTPServer(("127.0.0.1", port), _OAuthCallbackHandler)
    except OSError as exc:
        raise RuntimeError(
            f"Port {port} is already in use. Close the other process "
            f"occupying it and try again: {exc}"
        ) from exc
    server.timeout = timeout

    print(f"\nOpening browser for authentication...\n  {auth_url}\n")
    print("If the browser doesn't open, copy the URL above and paste it manually.\n")

    import webbrowser
    webbrowser.open(auth_url)

    def _serve_until_done() -> None:
        while not _OAuthCallbackHandler._event.is_set():
            server.handle_request()

    thread = threading.Thread(target=_serve_until_done, daemon=True)
    thread.start()

    if not _OAuthCallbackHandler._event.wait(timeout=timeout):
        server.server_close()
        raise RuntimeError("Authentication timed out. Please try again.")

    server.server_close()

    if _OAuthCallbackHandler.error:
        raise RuntimeError(
            f"Authentication failed: {_OAuthCallbackHandler.error}"
        )

    code = _OAuthCallbackHandler.auth_code
    callback_state = _OAuthCallbackHandler.auth_state

    if not code:
        raise RuntimeError("No authorization code received.")

    if callback_state != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF attack.")

    print("Exchanging authorization code for tokens...")
    tokens = exchange_code(code, code_verifier)
    save_auth(tokens)
    return tokens
