"""OrcaRouter provider routes: status, model discovery, and PKCE connect.

The browser never holds an OrcaRouter credential. The key lives in the
server's credential store (``~/.researchclaw/``, mode 0600) and the
catalogue endpoint returns minimal model metadata only, so a page cannot
leak a key it was never given.

The connect flow keeps a single server-side login lock. Every terminal path
— success, denial, exchange error, timeout, explicit cancel, and a browser
``pagehide`` cancel — releases it, and attempts carry a monotonically
increasing generation so a late response can never overwrite a newer login.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from researchclaw.llm import orcarouter as orca
from researchclaw.llm import orcarouter_catalog as catalog
from researchclaw.llm.orcarouter_pkce import (
    PkceCancelled,
    PkceDenied,
    PkceError,
    PkceStateMismatch,
    PkceTimeout,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])

_LOGIN_TTL_SEC = 600.0
_LOGIN_WAIT_SEC = 300.0


@dataclass
class LoginAttempt:
    """One in-flight PKCE login. Holds a verifier — never serialized."""

    attempt_id: str
    generation: int
    flow: str
    authorize_url: str
    callback_url: str
    pending: Any
    busy: bool = True
    status: str = "pending"  # pending | connected | denied | error | cancelled
    hint: str = ""
    error: str = ""
    secret_masked: str = ""
    account: str = ""
    scope: str = ""
    created_at: float = 0.0
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict[str, Any]:
        """State safe to hand a browser: no verifier, no code, no key."""
        with self.lock:
            return {
                "attempt_id": self.attempt_id,
                "generation": self.generation,
                "flow": self.flow,
                "status": self.status,
                "busy": self.busy,
                "hint": self.hint,
                "error": self.error,
                "secret_masked": self.secret_masked,
                "account": self.account,
                "scope": self.scope,
                "needs_code": self.flow == "oob" and self.status == "pending",
                "authorize_url": self.authorize_url if self.status == "pending" else "",
            }

    def finish(
        self,
        status: str,
        *,
        error: str = "",
        credential: Any = None,
    ) -> None:
        with self.lock:
            if self.status != "pending":
                return  # a terminal state was already recorded
            self.status = status
            self.busy = False
            self.error = error
            self.hint = ""
            if credential is not None:
                self.secret_masked = credential.masked
                self.account = credential.grant_id
                self.scope = credential.scope

    def cancel(self) -> None:
        pending = self.pending
        if pending is not None and hasattr(pending, "cancel"):
            try:
                pending.cancel()
            except Exception:  # noqa: BLE001 - cancelling must never raise
                pass
        self.close()
        self.finish("cancelled", error="cancelled")

    def close(self) -> None:
        pending = self.pending
        if pending is not None and hasattr(pending, "close"):
            try:
                pending.close()
            except Exception:  # noqa: BLE001
                pass
        self.pending = None


class _LoginRegistry:
    """The single server-side login lock, plus attempt history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: LoginAttempt | None = None
        self._generation = 0

    def _gc(self) -> None:
        active = self._active
        if active is None:
            return
        age = time.time() - active.created_at
        if active.status == "pending" and age > _LOGIN_TTL_SEC:
            active.cancel()
        elif active.status != "pending" and age > _LOGIN_TTL_SEC:
            active.close()
            self._active = None

    def start(self, *, flow: str, app_name: str, scope: str) -> LoginAttempt:
        with self._lock:
            self._gc()
            if self._active is not None and self._active.status == "pending":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A login is already in progress. Cancel it before "
                        "starting another."
                    ),
                )
            self._generation += 1
            endpoints = orca.resolve_endpoints()
            pending = orca.start_connect(
                flow=flow, app_name=app_name, scope=scope, endpoints=endpoints
            )
            attempt = LoginAttempt(
                attempt_id=uuid.uuid4().hex,
                generation=self._generation,
                flow=pending.flow,
                authorize_url=pending.authorize_url,
                callback_url=pending.callback_url,
                pending=pending,
                created_at=time.time(),
                hint="Waiting for you to approve access in the browser.",
            )
            self._active = attempt
            return attempt

    def get(self, attempt_id: str) -> LoginAttempt:
        with self._lock:
            active = self._active
        if active is None or active.attempt_id != attempt_id:
            raise HTTPException(status_code=404, detail="Unknown login attempt")
        return active

    def active(self) -> LoginAttempt | None:
        with self._lock:
            self._gc()
            return self._active

    def release(self, attempt: LoginAttempt) -> None:
        with self._lock:
            if self._active is attempt and attempt.status != "pending":
                attempt.close()


_registry = _LoginRegistry()


def reset_for_tests() -> None:
    """Drop all login state. Used by tests; not part of the public API."""
    with _registry._lock:  # noqa: SLF001 - deliberate test hook
        if _registry._active is not None:
            _registry._active.close()
        _registry._active = None
        _registry._generation = 0


class LoginRequest(BaseModel):
    flow: str = "auto"
    app_name: str = "AutoResearchClaw"
    scope: str = "api"


class CodeRequest(BaseModel):
    code: str


def _store() -> orca.CredentialStore:
    return orca.CredentialStore()


@router.get("")
def list_providers() -> dict[str, Any]:
    """Both OrcaRouter entries, with their non-secret credential state."""
    store = _store()
    api_source, pkce_source = orca.build_credential_sources(store=store)
    endpoints = orca.resolve_endpoints()
    return {
        "providers": [
            {
                "id": orca.PROVIDER_ID,
                "label": orca.PROVIDER_LABEL,
                "kind": "api_key",
                "base_url": endpoints.api_base,
                "status": api_source.status().as_dict(),
            },
            {
                "id": orca.PROVIDER_ID_PKCE,
                "label": orca.PROVIDER_LABEL_PKCE,
                "kind": "pkce",
                "base_url": endpoints.api_base,
                "status": pkce_source.status().as_dict(),
            },
        ],
        "endpoints": endpoints.describe(),
        "key_dashboard_url": orca.KEY_DASHBOARD_URL,
        "revocation_url": orca.REVOCATION_URL,
    }


@router.post("/orcarouter/key")
def save_api_key(payload: dict[str, str]) -> dict[str, Any]:
    """Store a pasted ``sk-orca-…`` key. The key is never echoed back."""
    api_key = str(payload.get("api_key") or "")
    source = orca.ApiKeySource(_store())
    try:
        status = source.save(api_key)
    except orca.OrcaConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": status.as_dict()}


@router.delete("/orcarouter/key")
def clear_api_key() -> dict[str, Any]:
    source = orca.ApiKeySource(_store())
    source.clear()
    return {"status": source.status().as_dict()}


@router.get("/orcarouter/models")
def list_models(
    capability: str = catalog.CAPABILITY_CHAT,
    modality: str = "",
    refresh: bool = False,
) -> dict[str, Any]:
    """Capability-filtered model list for the selectors.

    Returns minimal metadata from the configured origin's ``/v1/models``;
    the credential stays on the server.
    """
    required = tuple(m for m in (modality or "").split(",") if m)
    try:
        credential = orca.resolve_credential(store=_store())
    except orca.OrcaAuthRequired as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    endpoints = orca.resolve_endpoints()
    result = catalog.discover_models(
        endpoints.api_base,
        credential.api_key,
        capability=capability,
        required_input_modalities=required,
        use_cache=not refresh,
    )
    payload = result.as_dict()
    payload["catalog_source"] = catalog.catalog_url(endpoints.api_base, capability)
    payload["credential_source"] = credential.source
    return payload


@router.get("/orcarouter/auth")
def auth_state() -> dict[str, Any]:
    attempt = _registry.active()
    return {"attempt": attempt.public() if attempt else None}


@router.post("/orcarouter/auth/login")
def auth_login(request: LoginRequest) -> dict[str, Any]:
    attempt = _registry.start(
        flow=request.flow, app_name=request.app_name, scope=request.scope
    )
    if attempt.flow == "loopback":
        attempt.thread = threading.Thread(
            target=_await_loopback_callback,
            args=(attempt,),
            daemon=True,
        )
        attempt.thread.start()
    return {"attempt": attempt.public()}


def _await_loopback_callback(attempt: LoginAttempt) -> None:
    """Background waiter: the HTTP request must not block on the browser."""
    pending = attempt.pending
    try:
        code = pending.receiver.wait(timeout=_LOGIN_WAIT_SEC)
        _complete(attempt, code)
    except PkceTimeout:
        attempt.finish("error", error="Timed out waiting for the browser callback.")
    except PkceStateMismatch as exc:
        attempt.finish("error", error=str(exc))
    except PkceDenied as exc:
        attempt.finish("denied", error=str(exc))
    except PkceCancelled:
        attempt.finish("cancelled", error="cancelled")
    except PkceError as exc:
        attempt.finish("error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - never leave the lock held
        logger.exception("OrcaRouter loopback login failed")
        attempt.finish("error", error=orca.redact_secrets(str(exc)))
    finally:
        attempt.close()


def _complete(attempt: LoginAttempt, code: str) -> None:
    """Exchange a code and persist it, generation-guarded."""
    current = _registry.active()
    if current is not attempt or attempt.generation != current.generation:
        # A newer login superseded this one; drop the result on the floor.
        return
    try:
        credential = orca.complete_connect(attempt.pending, code, store=_store())
    except PkceError as exc:
        attempt.finish("error", error=str(exc))
        return
    except orca.OrcaConfigError as exc:
        attempt.finish("error", error=str(exc))
        return
    current2 = _registry.active()
    if current2 is not attempt or attempt.generation != current2.generation:
        return
    attempt.finish("connected", credential=credential)


@router.post("/orcarouter/auth/{attempt_id}/code")
def auth_submit_code(attempt_id: str, request: CodeRequest) -> dict[str, Any]:
    attempt = _registry.get(attempt_id)
    if attempt.status != "pending":
        raise HTTPException(status_code=409, detail="This login attempt already ended")
    _complete(attempt, request.code.strip())
    return {"attempt": attempt.public()}


@router.post("/orcarouter/auth/{attempt_id}/cancel")
def auth_cancel(attempt_id: str) -> dict[str, Any]:
    """Release the login lock. Safe to call twice (e.g. unload + pagehide)."""
    attempt = _registry.get(attempt_id)
    attempt.cancel()
    _registry.release(attempt)
    return {"attempt": attempt.public()}


@router.post("/orcarouter/reauth")
def mark_reauth(payload: dict[str, Any]) -> dict[str, Any]:
    """Terminal 401 handling for one exact credential generation."""
    entry_id = str(payload.get("entry_id") or "")
    generation = int(payload.get("generation") or 0)
    store = _store()
    applied = store.mark_needs_reauth(entry_id, generation)
    return {"applied": applied, "status": store.status(entry_id).as_dict()}
