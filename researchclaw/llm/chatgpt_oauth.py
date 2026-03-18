"""ChatGPT OAuth PKCE authentication for Plus/Pro subscribers.

Implements the same OAuth flow used by the official OpenAI Codex CLI,
allowing ChatGPT Plus/Pro subscribers to authenticate with their
subscription instead of a separate API key.

Tokens are cached at ``~/.researchclaw/auth.json`` and refreshed
automatically before they expire.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OAuth constants — same as official OpenAI Codex CLI
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"

AUTH_DIR = Path.home() / ".researchclaw"
AUTH_FILE = AUTH_DIR / "auth.json"

_TOKEN_REFRESH_MARGIN_SEC = 300  # refresh 5 minutes before expiry


@dataclass
class AuthTokens:
    """Stored OAuth tokens."""

    access_token: str
    refresh_token: str
    expires_at: float  # epoch seconds
    account_id: str = ""

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - _TOKEN_REFRESH_MARGIN_SEC

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
            expires_at=float(data["expires_at"]),
            account_id=data.get("account_id", ""),
        )


# ---------------------------------------------------------------------------
# PKCE helpers (stdlib only — no external dependencies)
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier_bytes = secrets.token_bytes(32)
    code_verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def _generate_state() -> str:
    return secrets.token_hex(16)


def _build_authorize_url(state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# JWT decoding (extract chatgpt_account_id)
# ---------------------------------------------------------------------------

def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verification (we only need claims)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        payload = parts[1]
        # Add padding
        padded = payload + "=" * (4 - len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        return json.loads(decoded)
    except Exception:  # noqa: BLE001
        return {}


def extract_account_id(access_token: str) -> str:
    """Extract chatgpt_account_id from an OpenAI JWT access token."""
    payload = _decode_jwt_payload(access_token)
    auth_claim = payload.get("https://api.openai.com/auth", {})
    return auth_claim.get("chatgpt_account_id", "")


# ---------------------------------------------------------------------------
# Token exchange and refresh (stdlib urllib)
# ---------------------------------------------------------------------------

def _post_token_request(params: dict[str, str]) -> dict[str, Any]:
    """POST to the OpenAI token endpoint."""
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
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


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------

def save_auth(tokens: AuthTokens) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(tokens.to_dict(), indent=2), encoding="utf-8")
    # Restrict permissions (owner-only read/write)
    try:
        AUTH_FILE.chmod(0o600)
    except OSError:
        pass


def load_auth() -> AuthTokens | None:
    if not AUTH_FILE.exists():
        return None
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        return AuthTokens.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Corrupted auth.json — removing")
        AUTH_FILE.unlink(missing_ok=True)
        return None


def delete_auth() -> bool:
    if AUTH_FILE.exists():
        AUTH_FILE.unlink()
        return True
    return False


def ensure_valid_token() -> AuthTokens:
    """Load stored tokens, refresh if expired, and return valid tokens.

    Raises RuntimeError if no tokens are stored or refresh fails.
    """
    tokens = load_auth()
    if tokens is None:
        raise RuntimeError(
            "Not logged in. Run 'researchclaw login' to authenticate "
            "with your ChatGPT Plus/Pro subscription."
        )
    if tokens.expired:
        logger.info("Access token expired, refreshing...")
        try:
            tokens = refresh_tokens(tokens.refresh_token)
            save_auth(tokens)
            logger.info("Token refreshed successfully")
        except Exception as exc:
            raise RuntimeError(
                f"Token refresh failed: {exc}\n"
                "Run 'researchclaw login' to re-authenticate."
            ) from exc
    return tokens


# ---------------------------------------------------------------------------
# Local OAuth callback server
# ---------------------------------------------------------------------------

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth redirect callback on localhost."""

    auth_code: str | None = None
    auth_state: str | None = None
    error: str | None = None
    _event: threading.Event

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
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
        elif code:
            _OAuthCallbackHandler.auth_code = code
            _OAuthCallbackHandler.auth_state = state
            self._send_html(
                "<h1>Authentication Successful!</h1>"
                "<p>You can close this window and return to the terminal.</p>"
                "<script>window.close()</script>"
            )
        else:
            self._send_html(
                "<h1>Unknown Response</h1>"
                "<p>No authorization code received.</p>"
            )

        _OAuthCallbackHandler._event.set()

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

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress default access log


def run_oauth_flow() -> AuthTokens:
    """Run the full browser-based OAuth PKCE flow.

    1. Start local callback server on port 1455
    2. Open browser to OpenAI authorization page
    3. Wait for the user to complete login
    4. Exchange authorization code for tokens
    5. Save tokens to auth.json

    Returns AuthTokens on success. Raises RuntimeError on failure.
    """
    code_verifier, code_challenge = _generate_pkce()
    state = _generate_state()
    authorize_url = _build_authorize_url(state, code_challenge)

    # Reset handler state
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.auth_state = None
    _OAuthCallbackHandler.error = None
    event = threading.Event()
    _OAuthCallbackHandler._event = event

    # Start local server
    try:
        server = HTTPServer(("127.0.0.1", 1455), _OAuthCallbackHandler)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot start OAuth callback server on port 1455: {exc}\n"
            "Make sure the port is not in use."
        ) from exc

    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    # Open browser
    print(f"\nOpening browser for ChatGPT login...")
    print(f"If the browser doesn't open, visit this URL manually:\n")
    print(f"  {authorize_url}\n")
    webbrowser.open(authorize_url)

    # Wait for callback (timeout after 5 minutes)
    print("Waiting for authentication... (timeout: 5 minutes)")
    if not event.wait(timeout=300):
        server.server_close()
        raise RuntimeError("Authentication timed out. Please try again.")

    server.server_close()

    # Check result
    if _OAuthCallbackHandler.error:
        raise RuntimeError(
            f"Authentication failed: {_OAuthCallbackHandler.error}"
        )

    code = _OAuthCallbackHandler.auth_code
    callback_state = _OAuthCallbackHandler.auth_state

    if not code:
        raise RuntimeError("No authorization code received.")

    if callback_state and callback_state != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF attack.")

    # Exchange code for tokens
    print("Exchanging authorization code for tokens...")
    tokens = exchange_code(code, code_verifier)

    if not tokens.account_id:
        logger.warning("Could not extract chatgpt_account_id from token")

    # Save tokens
    save_auth(tokens)
    return tokens
