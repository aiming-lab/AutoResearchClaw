"""The model dropdown: one capability-filtered option list per entry point.

This is the single place the rest of the project asks "which models may this
entry point offer right now?", so the filtering rules live here rather than
being re-implemented per surface. It is deliberately provider-aware: for any
provider other than OrcaRouter it returns ``None`` and callers keep their
existing behaviour untouched.

Two OrcaRouter-specific guarantees:

* When the user's attachment or task type changes, the *options handed to the
  selector* change — filtering the dropdown is the control, not a pre-send
  guard.
* A previously selected model that is no longer compatible is cleared, never
  silently kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from researchclaw.llm import orcarouter as orca
from researchclaw.llm import orcarouter_catalog as catalog
from researchclaw.llm.orcarouter_catalog import (
    CAPABILITY_CHAT,
    CatalogModel,
    CatalogResult,
)

logger = logging.getLogger(__name__)

ORCAROUTER_PROVIDERS = frozenset({orca.PROVIDER_ID, orca.PROVIDER_ID_PKCE})

#: Which declared input modality an AI entry point actually uploads.
MODALITY_IMAGE = "image"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"


@dataclass(frozen=True)
class ModelOption:
    """One selectable model, minimal metadata only."""

    id: str
    label: str
    context_length: int = 0
    input_modalities: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "context_length": self.context_length,
            "input_modalities": list(self.input_modalities),
            "reasoning_efforts": list(self.reasoning_efforts),
            "verified": self.verified,
        }


@dataclass
class ModelSelectorState:
    """The options for one model selector, plus what happened to the old value."""

    provider: str
    capability: str
    required_input_modalities: tuple[str, ...] = ()
    options: tuple[ModelOption, ...] = ()
    selected: str = ""
    selection_cleared: bool = False
    clear_reason: str = ""
    source: str = "live"
    degraded: bool = False
    error: str = ""
    catalog_source_url: str = ""

    @property
    def ids(self) -> list[str]:
        return [option.id for option in self.options]

    @property
    def is_free_text(self) -> bool:
        """A selector over a real catalogue is never free text."""
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "capability": self.capability,
            "required_input_modalities": list(self.required_input_modalities),
            "options": [option.as_dict() for option in self.options],
            "selected": self.selected,
            "selection_cleared": self.selection_cleared,
            "clear_reason": self.clear_reason,
            "source": self.source,
            "degraded": self.degraded,
            "error": self.error,
            "catalog_source_url": self.catalog_source_url,
        }


def is_orcarouter(config: Any) -> bool:
    return str(getattr(getattr(config, "llm", None), "provider", "") or "") in (
        ORCAROUTER_PROVIDERS
    )


def capability_for_entry_point(entry_point: str) -> str:
    """Map a project AI entry point onto an OrcaRouter catalogue capability."""
    normalized = (entry_point or "").strip().lower()
    if normalized in ("chat", "agent", "completion", "code", "review", "debate"):
        return CAPABILITY_CHAT
    if normalized in ("embedding", "embeddings", "rag"):
        return catalog.CAPABILITY_EMBEDDING
    if normalized in ("image", "image-generation", "figure"):
        return catalog.CAPABILITY_IMAGE
    if normalized in ("video", "video-generation"):
        return catalog.CAPABILITY_VIDEO
    if normalized in ("rerank", "reranking"):
        return catalog.CAPABILITY_RERANK
    raise ValueError(f"unknown AI entry point: {entry_point!r}")


def required_modalities_for_attachments(attachments: Sequence[str]) -> tuple[str, ...]:
    """Declared modalities a request actually uploads.

    Anything that is not ``text`` is a hard requirement: a model that does
    not declare it is excluded (fail closed).
    """
    required: list[str] = []
    for attachment in attachments or ():
        kind = (attachment or "").strip().lower()
        if kind in ("", "text"):
            continue
        if kind not in (MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO):
            # Unknown attachment kinds are their own modality, so no model
            # can satisfy them and the list correctly comes back empty
            # rather than quietly offering an incompatible model.
            required.append(kind)
            continue
        if kind not in required:
            required.append(kind)
    return tuple(required)


def build_model_selector(
    config: Any,
    *,
    entry_point: str = "chat",
    attachments: Sequence[str] = (),
    current_model: str = "",
    store: orca.CredentialStore | None = None,
    fetcher: catalog.Fetcher | None = None,
    use_cache: bool = True,
) -> ModelSelectorState | None:
    """Build the option list for one selector.

    Returns ``None`` for non-OrcaRouter providers so callers leave their
    existing model controls alone.
    """
    if not is_orcarouter(config):
        return None

    provider = str(config.llm.provider)
    capability = capability_for_entry_point(entry_point)
    required = required_modalities_for_attachments(attachments)
    endpoints = orca.resolve_endpoints()

    try:
        credential = orca.resolve_credential(
            store=store,
            config_value=str(getattr(config.llm, "api_key", "") or ""),
            api_key_env=str(
                getattr(config.llm, "api_key_env", "") or orca.DEFAULT_API_KEY_ENV
            ),
        )
        api_key = credential.api_key
        error = ""
    except orca.OrcaAuthRequired as exc:
        api_key = ""
        error = str(exc)

    result: CatalogResult = catalog.discover_models(
        endpoints.api_base,
        api_key,
        capability=capability,
        required_input_modalities=required,
        fetcher=fetcher,
        use_cache=use_cache,
    )

    options = tuple(
        ModelOption(
            id=model.id,
            label=model.label,
            context_length=model.context_length,
            input_modalities=model.input_modalities,
            reasoning_efforts=model.reasoning_efforts,
            verified=model.verified,
        )
        for model in result.models
    )

    state = ModelSelectorState(
        provider=provider,
        capability=capability,
        required_input_modalities=required,
        options=options,
        source=result.source,
        degraded=result.degraded,
        error=result.error or error,
        catalog_source_url=catalog.catalog_url(endpoints.api_base, capability),
    )
    apply_selection(state, current_model)
    return state


def apply_selection(state: ModelSelectorState, current_model: str) -> ModelSelectorState:
    """Keep a remembered model only if it is still in the filtered options."""
    candidate = (current_model or "").strip()
    if not candidate:
        return state
    if candidate in state.ids:
        state.selected = candidate
        return state
    state.selection_cleared = True
    state.clear_reason = (
        f"{candidate!r} is not available for {state.capability}"
        + (
            f" with input modalities {list(state.required_input_modalities)}"
            if state.required_input_modalities
            else ""
        )
        + " on OrcaRouter; choose another model."
    )
    return state


def model_options_for(
    config: Any,
    *,
    entry_point: str = "chat",
    attachments: Sequence[str] = (),
    current_model: str = "",
    **kwargs: Any,
) -> list[ModelOption] | None:
    """Convenience wrapper for callers that only need the option list."""
    state = build_model_selector(
        config,
        entry_point=entry_point,
        attachments=attachments,
        current_model=current_model,
        **kwargs,
    )
    return None if state is None else list(state.options)


def resolve_primary_model(config: Any, models: Sequence[CatalogModel]) -> str:
    """Pick the model the pipeline should use when none is configured.

    Prefers the configured model when it is still offered, then the routing
    alias, then the first available model — never an invented id.
    """
    configured = str(getattr(config.llm, "primary_model", "") or "").strip()
    ids = [model.id for model in models]
    if configured and configured in ids:
        return configured
    if "orcarouter/auto" in ids:
        return "orcarouter/auto"
    return ids[0] if ids else ""


__all__ = [
    "MODALITY_AUDIO",
    "MODALITY_IMAGE",
    "MODALITY_VIDEO",
    "ModelOption",
    "ModelSelectorState",
    "ORCAROUTER_PROVIDERS",
    "apply_selection",
    "build_model_selector",
    "capability_for_entry_point",
    "is_orcarouter",
    "model_options_for",
    "required_modalities_for_attachments",
    "resolve_primary_model",
]
