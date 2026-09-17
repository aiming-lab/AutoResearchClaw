"""OrcaRouter as a first-class provider: origins, credentials, and the seam.

OrcaRouter is an OpenAI-compatible AI gateway. This module holds the pieces
that are independent of the wire protocol:

* **Origins.** Authentication and inference live on *different* public
  origins — ``https://www.orcarouter.ai`` (consent screen at ``/auth``,
  exchange at ``/api/v1/auth/keys``) and ``https://api.orcarouter.ai/v1``
  (inference and model discovery). Neither is derived from the other by
  swapping a hostname or appending ``/v1``.
* **The credential seam.** :class:`CredentialSource` is a two-method
  interface for *obtaining a credential*. :class:`ApiKeySource` (the user
  pastes an ``sk-orca-…`` key) and :class:`PkceSource` (OAuth 2.0 + PKCE
  mints one) are its two adapters. Both hand downstream code the same
  :class:`OrcaCredential`, so the provider client, model discovery, and every
  AI entry point stay ignorant of how the key was obtained and never
  duplicate authentication logic.
* **Credential lifecycle.** A PKCE-issued key is a *durable API key*, not a
  refreshable OAuth token: it is reused until OrcaRouter revokes it, and a
  ``401`` from the relay is a terminal reauthentication for the exact
  account *generation* that made the rejected request — never a refresh, and
  never a state change for a newer credential.

Keys are stored where the project already keeps user-level state,
``~/.researchclaw/``, in a ``0600`` file. No new secret store technology is
introduced, and no key is ever logged, printed, or returned to a browser.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from urllib.parse import urlparse

from researchclaw.llm.orcarouter_pkce import (
    DEFAULT_APP_NAME,
    DEFAULT_SCOPE,
    PkceError,
    PkceExchangeRejected,
    PendingLogin,
    ExchangeResult,
    start_login,
)

logger = logging.getLogger(__name__)

PROVIDER_ID = "orcarouter"
PROVIDER_ID_PKCE = "orcarouter-oauth"
PROVIDER_LABEL = "OrcaRouter — API"
PROVIDER_LABEL_PKCE = "OrcaRouter — Auth"

DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
DEFAULT_API_KEY_ENV = "ORCAROUTER_API_KEY"
KEY_PREFIX = "sk-orca-"
KEY_DASHBOARD_URL = "https://www.orcarouter.ai/console/token"
REVOCATION_URL = "https://www.orcarouter.ai/console/authorized-apps"

ENV_SHARED_BASE = "ORCA_BASE_URL"
ENV_AUTH_BASE = "ORCA_AUTH_BASE_URL"
ENV_API_BASE = "ORCA_API_BASE_URL"
ENV_CREDENTIALS_PATH = "ORCA_CREDENTIALS_PATH"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

_KEY_PATTERN = re.compile(r"sk-orca-[A-Za-z0-9._\-]{4,}")
_SK_PATTERN = re.compile(r"\bsk-[A-Za-z0-9._\-]{8,}")


class OrcaConfigError(ValueError):
    """An origin or credential setting is unusable."""


class OrcaAuthRequired(RuntimeError):
    """No usable credential; the caller must run a login or paste a key."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


def redact_secrets(text: str) -> str:
    """Strip API-key-shaped substrings from text before it is logged/shown."""
    return _SK_PATTERN.sub("sk-***", _KEY_PATTERN.sub("sk-orca-***", text or ""))


def mask_secret(secret: str) -> str:
    """A display-only mask. Never reversible, never the key itself."""
    if not secret:
        return ""
    if len(secret) <= 12:
        return "•" * len(secret)
    return f"{secret[:7]}…{secret[-4:]}"


def validate_origin(url: str, *, name: str) -> str:
    """Require HTTPS for remote origins; HTTP only for loopback."""
    candidate = (url or "").strip().rstrip("/")
    if not candidate:
        raise OrcaConfigError(f"{name} must not be empty")
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        raise OrcaConfigError(f"{name} must be an http(s) URL, got {candidate!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise OrcaConfigError(f"{name} has no host: {candidate!r}")
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        raise OrcaConfigError(
            f"{name} must use HTTPS unless it is loopback, got {candidate!r}"
        )
    if parsed.username or parsed.password:
        raise OrcaConfigError(f"{name} must not contain userinfo")
    return candidate


@dataclass(frozen=True)
class OrcaEndpoints:
    """Resolved, validated origins for one OrcaRouter installation."""

    auth_base: str = DEFAULT_AUTH_BASE
    api_base: str = DEFAULT_API_BASE
    auth_source: str = "default"
    api_source: str = "default"

    @property
    def authorize_url_base(self) -> str:
        return self.auth_base

    def describe(self) -> dict[str, str]:
        return {
            "auth_base": self.auth_base,
            "api_base": self.api_base,
            "auth_source": self.auth_source,
            "api_source": self.api_source,
        }


def resolve_endpoints(env: Mapping[str, str] | None = None) -> OrcaEndpoints:
    """Resolve origins from the environment.

    Precedence, explicit first: ``ORCA_AUTH_BASE_URL`` / ``ORCA_API_BASE_URL``,
    then the shared self-hosted fallback ``ORCA_BASE_URL``, then the public
    defaults. A single-origin self-hosted deployment sets only
    ``ORCA_BASE_URL`` and both flows follow it.
    """
    environ = os.environ if env is None else env
    shared = (environ.get(ENV_SHARED_BASE) or "").strip()
    auth_raw = (environ.get(ENV_AUTH_BASE) or "").strip()
    api_raw = (environ.get(ENV_API_BASE) or "").strip()

    auth_source = "explicit" if auth_raw else ("shared" if shared else "default")
    api_source = "explicit" if api_raw else ("shared" if shared else "default")

    auth_base = validate_origin(
        auth_raw or shared or DEFAULT_AUTH_BASE, name=ENV_AUTH_BASE
    )
    api_base = validate_origin(api_raw or shared or DEFAULT_API_BASE, name=ENV_API_BASE)

    # The relay lives at /v1 on the API origin; a shared self-hosted base is
    # an origin, not a relay path, so the /v1 segment is added once here.
    if not api_base.rstrip("/").endswith("/v1"):
        api_base = f"{api_base.rstrip('/')}/v1"
    return OrcaEndpoints(
        auth_base=auth_base,
        api_base=api_base,
        auth_source=auth_source,
        api_source=api_source,
    )


# ---------------------------------------------------------------------------
# Credential
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrcaCredential:
    """The single downstream currency: a normal OrcaRouter API key.

    ``entry_id``/``generation`` identify the exact account credential a
    request was made with, so a late ``401`` can be attributed to the
    generation that earned it.
    """

    api_key: str
    source: str  # "api_key" | "pkce"
    entry_id: str
    generation: int = 0
    grant_id: str = ""
    scope: str = DEFAULT_SCOPE
    created_at: float = 0.0

    @property
    def masked(self) -> str:
        return mask_secret(self.api_key)

    @property
    def account_key(self) -> tuple[str, str, int]:
        return (self.entry_id, self.grant_id, self.generation)


@dataclass(frozen=True)
class CredentialStatus:
    """Everything a UI may show about one credential entry — no secrets."""

    entry_id: str
    source: str
    configured: bool = False
    masked: str = ""
    scope: str = ""
    grant_id: str = ""
    generation: int = 0
    needs_reauth: bool = False
    created_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source": self.source,
            "configured": self.configured,
            "secret_masked": self.masked,
            "scope": self.scope,
            "grant_id": self.grant_id,
            "generation": self.generation,
            "needs_reauth": self.needs_reauth,
            "created_at": self.created_at,
        }


class CredentialStore:
    """The project's existing user-level state dir, with 0600 file mode.

    The file is *not* a new secret store: it is the same
    ``~/.researchclaw/`` tree the CLI already uses for skills, profiles and
    hooks.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            override = (os.environ.get(ENV_CREDENTIALS_PATH) or "").strip()
            path = (
                Path(override).expanduser()
                if override
                else Path.home() / ".researchclaw" / "orcarouter" / "credentials.json"
            )
        self.path = Path(path).expanduser()

    # -- io ------------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": 1, "accounts": {}}
        except OSError as exc:
            logger.warning("Could not read OrcaRouter credentials: %s", exc)
            return {"version": 1, "accounts": {}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OrcaRouter credential file is corrupt; ignoring it")
            return {"version": 1, "accounts": {}}
        if not isinstance(data, dict):
            return {"version": 1, "accounts": {}}
        accounts = data.get("accounts")
        if not isinstance(accounts, dict):
            data["accounts"] = {}
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    # -- api -----------------------------------------------------------
    def get(self, entry_id: str) -> dict[str, Any]:
        entry = self._read()["accounts"].get(entry_id)
        return dict(entry) if isinstance(entry, dict) else {}

    def entries(self) -> dict[str, dict[str, Any]]:
        accounts = self._read()["accounts"]
        return {k: dict(v) for k, v in accounts.items() if isinstance(v, dict)}

    def credential(self, entry_id: str) -> OrcaCredential | None:
        entry = self.get(entry_id)
        key = str(entry.get("api_key") or "")
        if not key:
            return None
        return OrcaCredential(
            api_key=key,
            source=str(entry.get("source") or entry_id),
            entry_id=entry_id,
            generation=int(entry.get("generation") or 0),
            grant_id=str(entry.get("grant_id") or ""),
            scope=str(entry.get("scope") or DEFAULT_SCOPE),
            created_at=float(entry.get("created_at") or 0.0),
        )

    def status(self, entry_id: str) -> CredentialStatus:
        entry = self.get(entry_id)
        key = str(entry.get("api_key") or "")
        return CredentialStatus(
            entry_id=entry_id,
            source=str(entry.get("source") or entry_id),
            configured=bool(key),
            masked=mask_secret(key),
            scope=str(entry.get("scope") or ""),
            grant_id=str(entry.get("grant_id") or ""),
            generation=int(entry.get("generation") or 0),
            needs_reauth=bool(entry.get("needs_reauth")),
            created_at=float(entry.get("created_at") or 0.0),
        )

    def save(
        self,
        entry_id: str,
        api_key: str,
        *,
        source: str,
        grant_id: str = "",
        scope: str = DEFAULT_SCOPE,
    ) -> OrcaCredential:
        """Persist a key and bump the generation.

        Bumping the generation is what makes the later ``401`` transition
        generation-safe: a failure reported by a request made with an older
        generation cannot mark the new credential broken.
        """
        data = self._read()
        previous = data["accounts"].get(entry_id) or {}
        generation = int(previous.get("generation") or 0) + 1
        data["accounts"][entry_id] = {
            "api_key": api_key,
            "source": source,
            "grant_id": grant_id,
            "scope": scope,
            "generation": generation,
            "created_at": time.time(),
            "needs_reauth": False,
        }
        self._write(data)
        return OrcaCredential(
            api_key=api_key,
            source=source,
            entry_id=entry_id,
            generation=generation,
            grant_id=grant_id,
            scope=scope,
            created_at=float(data["accounts"][entry_id]["created_at"]),
        )

    def clear(self, entry_id: str) -> bool:
        data = self._read()
        existed = entry_id in data["accounts"]
        data["accounts"].pop(entry_id, None)
        if existed:
            self._write(data)
        return existed

    def mark_needs_reauth(self, entry_id: str, generation: int) -> bool:
        """Mark *exactly* this credential generation as unusable.

        Returns True when the mark was applied. A stale generation is a
        no-op: a late failure from an old request must never invalidate a
        credential that has since been reauthorized. The stored secret is
        deliberately kept so the user can see what was rejected and a
        transient misclassification is not irreversible.
        """
        data = self._read()
        entry = data["accounts"].get(entry_id)
        if not isinstance(entry, dict):
            return False
        if int(entry.get("generation") or 0) != int(generation):
            logger.info(
                "Ignoring stale OrcaRouter 401 for %s generation %s (current %s)",
                entry_id,
                generation,
                entry.get("generation"),
            )
            return False
        if entry.get("needs_reauth"):
            return False
        entry["needs_reauth"] = True
        self._write(data)
        return True


# ---------------------------------------------------------------------------
# The credential seam: two adapters, one result type
# ---------------------------------------------------------------------------


@runtime_checkable
class CredentialSource(Protocol):
    """Where a credential comes from. Nothing downstream needs more."""

    id: str

    def acquire(self) -> OrcaCredential:  # pragma: no cover - protocol
        ...

    def status(self) -> CredentialStatus:  # pragma: no cover - protocol
        ...

    def clear(self) -> None:  # pragma: no cover - protocol
        ...


class ApiKeySource:
    """Adapter 1 — a user-supplied ``sk-orca-…`` key.

    Resolution order matches the rest of the project: explicit config value,
    then the env var named for the provider, then whatever the user stored
    through the UI/CLI. No network call is made to "validate" the key: an
    ``sk-orca-`` prefix is a format check, not proof, and OrcaRouter exposes
    no non-billing validation request.
    """

    id = PROVIDER_ID
    label = PROVIDER_LABEL
    kind = "api_key"

    def __init__(
        self,
        store: CredentialStore,
        *,
        config_value: str = "",
        api_key_env: str = DEFAULT_API_KEY_ENV,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.store = store
        self.config_value = config_value or ""
        self.api_key_env = api_key_env or DEFAULT_API_KEY_ENV
        self._environ = environ

    def _env(self) -> Mapping[str, str]:
        return os.environ if self._environ is None else self._environ

    @staticmethod
    def looks_like_key(value: str) -> bool:
        return bool(_KEY_PATTERN.fullmatch((value or "").strip()))

    def stored_key(self) -> str:
        return str(self.store.get(self.id).get("api_key") or "")

    def raw_key(self) -> str:
        return (
            self.config_value.strip()
            or (self._env().get(self.api_key_env) or "").strip()
            or self.stored_key()
        )

    def acquire(self) -> OrcaCredential:
        key = self.raw_key()
        if not key:
            raise OrcaAuthRequired(
                "no_api_key",
                "No OrcaRouter API key configured. Paste an sk-orca-… key, or "
                "use 'Connect with OrcaRouter' to authorize with your account.",
            )
        entry = self.store.get(self.id)
        if key == str(entry.get("api_key") or ""):
            return self.store.credential(self.id)  # type: ignore[return-value]
        # A key from config/env is not persisted by us; it still gets a
        # stable identity so 401 attribution stays generation-safe.
        return OrcaCredential(
            api_key=key,
            source="api_key",
            entry_id=f"{self.id}:{self.api_key_env}" if not self.config_value else self.id,
            generation=0,
            scope=DEFAULT_SCOPE,
        )

    def save(self, api_key: str) -> CredentialStatus:
        value = (api_key or "").strip()
        if not value:
            raise OrcaConfigError("API key must not be empty")
        if not self.looks_like_key(value):
            raise OrcaConfigError(
                "That does not look like an OrcaRouter key — they start with "
                f"'{KEY_PREFIX}'. Copy one from {KEY_DASHBOARD_URL}."
            )
        self.store.save(self.id, value, source="api_key")
        return self.status()

    def status(self) -> CredentialStatus:
        stored = self.store.status(self.id)
        if stored.configured:
            return stored
        env_key = (self._env().get(self.api_key_env) or "").strip()
        key = self.config_value.strip() or env_key
        if not key:
            return CredentialStatus(entry_id=self.id, source="api_key")
        origin = "config" if self.config_value.strip() else self.api_key_env
        return CredentialStatus(
            entry_id=self.id,
            source="api_key",
            configured=True,
            masked=mask_secret(key),
            scope=DEFAULT_SCOPE,
            grant_id=origin,
        )

    def clear(self) -> None:
        self.store.clear(self.id)


class PkceSource:
    """Adapter 2 — OAuth 2.0 + PKCE mints the same kind of key.

    The grant is durable: it is reused on every start until OrcaRouter
    revokes it. There is no refresh grant to call, and this adapter never
    invents one.
    """

    id = PROVIDER_ID_PKCE
    label = PROVIDER_LABEL_PKCE
    kind = "pkce"

    def __init__(self, store: CredentialStore) -> None:
        self.store = store

    def acquire(self) -> OrcaCredential:
        credential = self.store.credential(self.id)
        if credential is None:
            raise OrcaAuthRequired(
                "not_connected",
                "Not connected to OrcaRouter. Use 'Connect with OrcaRouter' to "
                "authorize this app with your account.",
            )
        if self.store.status(self.id).needs_reauth:
            raise OrcaAuthRequired(
                "needs_reauth",
                "Your OrcaRouter authorization was revoked or rejected. "
                f"Reconnect, or check {REVOCATION_URL}.",
            )
        return credential

    def status(self) -> CredentialStatus:
        return self.store.status(self.id)

    def persist(self, result: ExchangeResult) -> OrcaCredential:
        """Store an exchange result as a durable credential."""
        return self.store.save(
            self.id,
            result.api_key,
            source="pkce",
            grant_id=result.user_id,
            scope=result.scope or DEFAULT_SCOPE,
        )

    def clear(self) -> None:
        self.store.clear(self.id)


def build_credential_sources(
    *,
    store: CredentialStore | None = None,
    config_value: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    environ: Mapping[str, str] | None = None,
) -> tuple[ApiKeySource, PkceSource]:
    """The two adapters over one credential seam, in a stable order."""
    shared_store = store or CredentialStore()
    return (
        ApiKeySource(
            shared_store,
            config_value=config_value,
            api_key_env=api_key_env,
            environ=environ,
        ),
        PkceSource(shared_store),
    )


def resolve_credential(
    *,
    store: CredentialStore | None = None,
    config_value: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    prefer: str = "",
    environ: Mapping[str, str] | None = None,
) -> OrcaCredential:
    """Resolve *one* credential through the seam, for any consumer.

    ``prefer`` pins a specific adapter id (the provider the user selected);
    otherwise an existing API key wins over a stored PKCE grant, because the
    API key is the more explicit choice.
    """
    api_source, pkce_source = build_credential_sources(
        store=store,
        config_value=config_value,
        api_key_env=api_key_env,
        environ=environ,
    )
    ordered: list[CredentialSource] = [api_source, pkce_source]
    if prefer:
        ordered.sort(key=lambda s: 0 if s.id == prefer else 1)
    errors: list[OrcaAuthRequired] = []
    for source in ordered:
        try:
            return source.acquire()
        except OrcaAuthRequired as exc:
            errors.append(exc)
    raise errors[-1] if errors else OrcaAuthRequired("no_credential")


def handle_unauthorized(
    credential: OrcaCredential, *, store: CredentialStore | None = None
) -> bool:
    """Terminal handling for a relay ``401``.

    Marks exactly the rejected credential generation as ``needs_reauth``.
    Deliberately does NOT delete the stored secret (a transient or
    misclassified failure must not become irreversible account loss) and
    does NOT attempt a refresh — there is no refresh grant.
    """
    target = store or CredentialStore()
    if not credential.entry_id or credential.entry_id.startswith(
        f"{PROVIDER_ID}:"
    ):
        # Key came from config/env; there is nothing of ours to mark.
        return False
    return target.mark_needs_reauth(credential.entry_id, credential.generation)


# ---------------------------------------------------------------------------
# Provider glue
# ---------------------------------------------------------------------------


@dataclass
class OrcaRouterProvider:
    """OrcaRouter bound to one resolved credential and one endpoint pair.

    Every AI entry point in the project reaches OrcaRouter through here (via
    :func:`build_orcarouter_client`), so none of them re-implement
    authentication or catalogue logic.
    """

    credential: OrcaCredential
    endpoints: OrcaEndpoints = field(default_factory=OrcaEndpoints)
    wire_api: str = "chat_completions"
    timeout_sec: int = 600

    def llm_config(
        self,
        *,
        primary_model: str = "",
        fallback_models: Sequence[str] = (),
    ):
        """An :class:`LLMConfig` for the project's existing OpenAI client."""
        from researchclaw.llm.client import LLMConfig

        return LLMConfig(
            base_url=self.endpoints.api_base,
            api_key=self.credential.api_key,
            wire_api=self.wire_api,
            primary_model=primary_model or "orcarouter/auto",
            fallback_models=list(fallback_models),
            timeout_sec=self.timeout_sec,
        )

    def build_client(
        self,
        *,
        primary_model: str = "",
        fallback_models: Sequence[str] = (),
    ):
        """The project's standard :class:`LLMClient`, pointed at OrcaRouter."""
        from researchclaw.llm.client import LLMClient

        return LLMClient(
            self.llm_config(
                primary_model=primary_model, fallback_models=fallback_models
            )
        )

    @property
    def masked(self) -> str:
        return self.credential.masked


def build_orcarouter_client(
    config: Any = None,
    *,
    store: CredentialStore | None = None,
    prefer: str = "",
    endpoints: OrcaEndpoints | None = None,
) -> Any:
    """Build the project's OpenAI-compatible client for OrcaRouter.

    Used by :func:`researchclaw.llm.create_llm_client` for both the
    ``orcarouter`` and ``orcarouter-oauth`` provider ids — the two entries
    differ only in which credential adapter is preferred.
    """
    llm = getattr(config, "llm", None)
    credential = resolve_credential(
        store=store,
        config_value=str(getattr(llm, "api_key", "") or ""),
        api_key_env=str(
            getattr(llm, "api_key_env", "") or DEFAULT_API_KEY_ENV
        ),
        prefer=prefer,
    )
    provider = OrcaRouterProvider(
        credential=credential,
        endpoints=endpoints or resolve_endpoints(),
        wire_api=str(getattr(llm, "wire_api", "") or "chat_completions"),
        timeout_sec=int(getattr(llm, "timeout_sec", 600) or 600),
    )
    primary = str(getattr(llm, "primary_model", "") or "").strip()
    fallbacks = tuple(getattr(llm, "fallback_models", ()) or ())
    if not primary:
        # No model configured: resolve one from the account's own catalogue
        # rather than inventing an id the workspace may not be able to call.
        primary, fallbacks = _discover_default_models(provider)
    return provider.build_client(primary_model=primary, fallback_models=fallbacks)


def _discover_default_models(
    provider: OrcaRouterProvider,
) -> tuple[str, tuple[str, ...]]:
    """A usable model chain from the live catalogue (or the verified seed).

    Only ids the catalogue actually offered are returned.
    """
    from researchclaw.llm import orcarouter_catalog as _catalog

    try:
        result = _catalog.discover_models(
            provider.endpoints.api_base,
            provider.credential.api_key,
            capability=_catalog.CAPABILITY_CHAT,
        )
    except Exception:  # noqa: BLE001 - never block client construction
        logger.debug("OrcaRouter catalogue unavailable while picking a model")
        return "", ()
    ids = [model.id for model in result.models]
    if not ids:
        return "", ()
    # Prefer models that *declare* a text modality. The catalogue lists plain
    # routing aliases too, and a given workspace key may not be able to call
    # one (the relay answers 403 model_access_denied) — so an alias is a
    # fallback of last resort, not the default.
    declared = [
        model.id for model in result.models if "text" in model.input_modalities
    ] or ids
    primary = declared[0]
    fallbacks = tuple(model_id for model_id in declared if model_id != primary)[:3]
    return primary, fallbacks


def start_connect(
    *,
    flow: str = "auto",
    app_name: str = DEFAULT_APP_NAME,
    scope: str = DEFAULT_SCOPE,
    endpoints: OrcaEndpoints | None = None,
    login_hint: str = "",
) -> PendingLogin:
    """Begin a PKCE login against the configured auth origin."""
    resolved = endpoints or resolve_endpoints()
    return start_login(
        flow,
        auth_base=resolved.auth_base,
        app_name=app_name,
        scope=scope,
        login_hint=login_hint,
    )


def complete_connect(
    pending: PendingLogin,
    code: str,
    *,
    store: CredentialStore | None = None,
    endpoints: OrcaEndpoints | None = None,
    post_json: Any = None,
) -> OrcaCredential:
    """Exchange the code and persist the resulting durable key."""
    resolved = endpoints or resolve_endpoints()
    result = pending.exchange(resolved.auth_base, code, post_json=post_json)
    return PkceSource(store or CredentialStore()).persist(result)


def model_options_for_entry_point(
    config: Any,
    *,
    entry_point: str = "chat",
    attachments: Sequence[str] = (),
    current_model: str = "",
    **kwargs: Any,
):
    """The capability-filtered model options for one AI entry point.

    Returns ``None`` for any provider that is not OrcaRouter, so callers keep
    their existing model control untouched.
    """
    from researchclaw.llm.model_select import build_model_selector

    return build_model_selector(
        config,
        entry_point=entry_point,
        attachments=attachments,
        current_model=current_model,
        **kwargs,
    )


__all__ = [
    "ApiKeySource",
    "CredentialSource",
    "CredentialStatus",
    "CredentialStore",
    "DEFAULT_API_BASE",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_AUTH_BASE",
    "KEY_DASHBOARD_URL",
    "OrcaAuthRequired",
    "OrcaConfigError",
    "OrcaCredential",
    "OrcaEndpoints",
    "OrcaRouterProvider",
    "PkceError",
    "PkceExchangeRejected",
    "PkceSource",
    "PROVIDER_ID",
    "PROVIDER_ID_PKCE",
    "PROVIDER_LABEL",
    "PROVIDER_LABEL_PKCE",
    "REVOCATION_URL",
    "build_credential_sources",
    "build_orcarouter_client",
    "model_options_for_entry_point",
    "complete_connect",
    "handle_unauthorized",
    "mask_secret",
    "model_options_for_entry_point",
    "redact_secrets",
    "resolve_credential",
    "resolve_endpoints",
    "start_connect",
    "start_login",
    "validate_origin",
]
