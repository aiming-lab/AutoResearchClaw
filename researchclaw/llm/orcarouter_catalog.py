"""OrcaRouter model discovery and capability filtering.

The single source of truth for the model list is ``GET {api_base}/models`` on
the *inference* origin (``https://api.orcarouter.ai/v1/models``). The request
carries the user's own OrcaRouter bearer key, so the answer is the catalogue
that workspace can actually call.

Rules implemented here, and the reason each exists:

* **Bounded.** A catalogue response cannot consume unbounded time, bytes, or
  memory: 10 s timeout, 512 KiB cap, 500 items, and only records whose shape
  is understood.
* **Live is authoritative.** When discovery succeeds, only discovered models
  are offered — a fallback seed is never mixed into a successful result. A
  verified seed exists only for a cold start or an outage, is labelled as
  degraded, and keeps its verified metadata.
* **Capability-filtered per entry point.** Text chat requires a text wire
  endpoint and excludes image/video/rerank-only models; multimodal requires
  the *declared* input modality, and fails closed when nothing is declared;
  embedding/image/video/rerank require their exact endpoint type. Capability
  is never inferred from a model's name.
* **Other providers are untouched.** Nothing here runs unless the selected
  provider is OrcaRouter.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: Wire endpoint types OrcaRouter advertises per model.
TEXT_ENDPOINT_TYPES = frozenset(
    {"openai", "anthropic", "gemini", "openai-response"}
)
NON_TEXT_ENDPOINT_TYPES = frozenset(
    {"image-generation", "openai-video", "jina-rerank"}
)

CAPABILITY_CHAT = "chat"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_IMAGE = "image"
CAPABILITY_VIDEO = "video"
CAPABILITY_RERANK = "rerank"

#: capability -> (server-side ?capability= value, required endpoint types)
_CAPABILITY_RULES: dict[str, tuple[str | None, frozenset[str]]] = {
    CAPABILITY_CHAT: (CAPABILITY_CHAT, TEXT_ENDPOINT_TYPES),
    CAPABILITY_EMBEDDING: (CAPABILITY_EMBEDDING, frozenset({"embeddings"})),
    CAPABILITY_IMAGE: (CAPABILITY_IMAGE, frozenset({"image-generation"})),
    CAPABILITY_VIDEO: (None, frozenset({"openai-video"})),
    CAPABILITY_RERANK: (None, frozenset({"jina-rerank"})),
}

DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_MAX_BYTES = 512 * 1024
DEFAULT_MAX_ITEMS = 500
DEFAULT_CACHE_TTL_SEC = 6 * 3600.0

ENV_CATALOG_TTL = "ORCA_CATALOG_TTL_SEC"
ENV_CACHE_DIR = "ORCA_CATALOG_CACHE_DIR"


@dataclass(frozen=True)
class CatalogModel:
    """One catalogue row, reduced to what the client can act on."""

    id: str
    name: str = ""
    context_length: int = 0
    max_completion_tokens: int = 0
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    supported_endpoint_types: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    verified: bool = False
    provenance: str = ""

    @property
    def label(self) -> str:
        return self.name or self.id

    @property
    def supports_chat(self) -> bool:
        return bool(set(self.supported_endpoint_types) & TEXT_ENDPOINT_TYPES)

    def supports_input(self, modality: str) -> bool:
        return modality in self.input_modalities

    def as_dict(self) -> dict[str, Any]:
        """Minimal metadata for a browser — never a credential."""
        return {
            "id": self.id,
            "name": self.label,
            "context_length": self.context_length,
            "max_completion_tokens": self.max_completion_tokens,
            "input_modalities": list(self.input_modalities),
            "output_modalities": list(self.output_modalities),
            "supported_endpoint_types": list(self.supported_endpoint_types),
            "reasoning_efforts": list(self.reasoning_efforts),
            "verified": self.verified,
        }


def _seed() -> tuple[CatalogModel, ...]:
    """A small, verified cold-start catalogue.

    Provenance, per entry:

    * ``orcarouter/auto`` — the routing alias documented by OrcaRouter; also
      present in the live catalogue.
    * ``openai/gpt-5.5`` — canonical OrcaRouter seed entry, retaining its
      verified reasoning-effort ladder (low/medium/high/xhigh). Effort
      metadata is not exposed by ``/v1/models``, so it is preserved here
      rather than dropped.
    * ``anthropic/claude-opus-4.8``, ``google/gemini-3.5-flash`` — canonical
      OrcaRouter seed entries (no effort ladder claimed for them).
    * ``deepseek/deepseek-v4-pro``, ``deepseek/deepseek-v4-flash`` — present
      in the live catalogue on 2026-09-16 with a 1 048 576-token context and
      declared ``text`` input.
    """
    return (
        CatalogModel(
            id="orcarouter/auto",
            name="OrcaRouter: Auto",
            supported_endpoint_types=("openai", "openai-response", "anthropic", "gemini"),
            verified=True,
            provenance="orcarouter routing alias",
        ),
        CatalogModel(
            id="openai/gpt-5.5",
            name="OpenAI: GPT-5.5",
            context_length=400000,
            input_modalities=("text", "image"),
            output_modalities=("text",),
            supported_endpoint_types=("openai", "openai-response"),
            reasoning_efforts=("low", "medium", "high", "xhigh"),
            verified=True,
            provenance="OrcaRouter canonical seed (reasoning ladder verified)",
        ),
        CatalogModel(
            id="anthropic/claude-opus-4.8",
            name="Anthropic: Claude Opus 4.8",
            context_length=200000,
            input_modalities=("text", "image"),
            output_modalities=("text",),
            supported_endpoint_types=("anthropic", "openai"),
            verified=True,
            provenance="OrcaRouter canonical seed",
        ),
        CatalogModel(
            id="google/gemini-3.5-flash",
            name="Google: Gemini 3.5 Flash",
            context_length=1000000,
            input_modalities=("text", "image", "audio", "video"),
            output_modalities=("text",),
            supported_endpoint_types=("gemini", "openai"),
            verified=True,
            provenance="OrcaRouter canonical seed",
        ),
        CatalogModel(
            id="deepseek/deepseek-v4-pro",
            name="DeepSeek: DeepSeek V4 Pro",
            context_length=1048576,
            max_completion_tokens=384000,
            input_modalities=("text",),
            output_modalities=("text",),
            supported_endpoint_types=("openai", "openai-response"),
            verified=True,
            provenance="live catalogue 2026-09-16",
        ),
        CatalogModel(
            id="deepseek/deepseek-v4-flash",
            name="DeepSeek: DeepSeek V4 Flash",
            context_length=1048576,
            max_completion_tokens=384000,
            input_modalities=("text",),
            output_modalities=("text",),
            supported_endpoint_types=("openai", "openai-response"),
            verified=True,
            provenance="live catalogue 2026-09-16",
        ),
    )


VERIFIED_SEED: tuple[CatalogModel, ...] = _seed()


def seed_for_capability(capability: str, required_input_modalities: Sequence[str] = ()) -> tuple[CatalogModel, ...]:
    """The verified seed, filtered by the same capability rules as live."""
    return tuple(
        filter_models(VERIFIED_SEED, capability, required_input_modalities=required_input_modalities)
    )


def _declares_only_non_text(model: CatalogModel) -> bool:
    declared_in = set(model.input_modalities)
    declared_out = set(model.output_modalities)
    if declared_in and "text" not in declared_in:
        return True
    if declared_out and "text" not in declared_out:
        return True
    return False


def matches_capability(
    model: CatalogModel,
    capability: str,
    *,
    required_input_modalities: Sequence[str] = (),
) -> bool:
    """Does *model* belong in a selector for *capability*?

    Fail-closed: a model that does not *declare* a required non-text input
    modality is excluded, never assumed compatible.
    """
    rule = _CAPABILITY_RULES.get(capability)
    if rule is None:
        raise ValueError(f"unknown capability: {capability!r}")
    _, required_endpoints = rule
    endpoints = set(model.supported_endpoint_types)

    if capability == CAPABILITY_CHAT:
        if not (endpoints & TEXT_ENDPOINT_TYPES):
            return False
        if endpoints & NON_TEXT_ENDPOINT_TYPES:
            return False
        if _declares_only_non_text(model):
            return False
        for modality in required_input_modalities:
            if modality == "text":
                continue
            if modality not in set(model.input_modalities):
                return False
        return True

    # Non-text capabilities require their exact wire endpoint.
    if not (endpoints & required_endpoints):
        return False
    for modality in required_input_modalities:
        if modality not in set(model.input_modalities):
            return False
    return True


def filter_models(
    models: Iterable[CatalogModel],
    capability: str,
    *,
    required_input_modalities: Sequence[str] = (),
) -> list[CatalogModel]:
    return [
        model
        for model in models
        if matches_capability(
            model,
            capability,
            required_input_modalities=required_input_modalities,
        )
    ]


def parse_models(payload: Any, *, max_items: int = DEFAULT_MAX_ITEMS) -> list[CatalogModel]:
    """Translate a ``/v1/models`` payload into :class:`CatalogModel` rows.

    Unknown records are skipped instead of trusted; the vendor/model
    namespace in ``id`` is preserved verbatim.
    """
    if isinstance(payload, dict):
        rows = payload.get("data")
    else:
        rows = payload
    if not isinstance(rows, list):
        return []
    parsed: list[CatalogModel] = []
    for row in rows[:max_items]:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        architecture = row.get("architecture")
        architecture = architecture if isinstance(architecture, dict) else {}

        def _modalities(key: str) -> tuple[str, ...]:
            value = architecture.get(key)
            if not isinstance(value, list):
                return ()
            return tuple(str(v) for v in value if isinstance(v, str) and v)

        endpoints = row.get("supported_endpoint_types")
        endpoints = (
            tuple(str(v) for v in endpoints if isinstance(v, str) and v)
            if isinstance(endpoints, list)
            else ()
        )
        efforts = row.get("reasoning_efforts") or architecture.get("reasoning_efforts")
        efforts = (
            tuple(str(v) for v in efforts if isinstance(v, str) and v)
            if isinstance(efforts, list)
            else ()
        )

        def _int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        parsed.append(
            CatalogModel(
                id=model_id.strip(),
                name=str(row.get("name") or ""),
                context_length=_int(row.get("context_length")),
                max_completion_tokens=_int(row.get("max_completion_tokens")),
                input_modalities=_modalities("input_modalities"),
                output_modalities=_modalities("output_modalities"),
                supported_endpoint_types=endpoints,
                reasoning_efforts=efforts,
                verified=False,
                provenance="live",
            )
        )
    return parsed


def merge_verified_metadata(
    models: Sequence[CatalogModel], seed: Sequence[CatalogModel] = VERIFIED_SEED
) -> list[CatalogModel]:
    """Keep verified metadata for models the live catalogue also lists.

    Live discovery must not *reduce* a known model's capabilities (reasoning
    ladder, declared modalities, context window), and must not *add* a model
    the live catalogue did not return.
    """
    by_id = {model.id: model for model in seed}
    merged: list[CatalogModel] = []
    for model in models:
        known = by_id.get(model.id)
        if known is None:
            merged.append(model)
            continue
        merged.append(
            replace(
                model,
                name=model.name or known.name,
                context_length=model.context_length or known.context_length,
                max_completion_tokens=(
                    model.max_completion_tokens or known.max_completion_tokens
                ),
                input_modalities=model.input_modalities or known.input_modalities,
                output_modalities=model.output_modalities or known.output_modalities,
                supported_endpoint_types=(
                    model.supported_endpoint_types or known.supported_endpoint_types
                ),
                reasoning_efforts=model.reasoning_efforts or known.reasoning_efforts,
                verified=True,
                provenance="live+verified",
            )
        )
    return merged


@dataclass(frozen=True)
class CatalogResult:
    """Outcome of one catalogue lookup — always usable, never a bare error."""

    models: tuple[CatalogModel, ...]
    source: str  # "live" | "cache" | "seed"
    capability: str
    required_input_modalities: tuple[str, ...] = ()
    degraded: bool = False
    error: str = ""
    fetched_at: float = 0.0
    live_model_count: int = 0
    considered: int = 0

    @property
    def count(self) -> int:
        return len(self.models)

    @property
    def ids(self) -> list[str]:
        return [model.id for model in self.models]

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [model.as_dict() for model in self.models],
            "source": self.source,
            "capability": self.capability,
            "required_input_modalities": list(self.required_input_modalities),
            "degraded": self.degraded,
            "error": self.error,
            "fetched_at": self.fetched_at,
            "live_model_count": self.live_model_count,
            "count": self.count,
        }


Fetcher = Callable[[str, str, float, int], Any]


def _default_fetcher(url: str, api_key: str, timeout: float, max_bytes: int) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "AutoResearchClaw-orcarouter/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("model catalogue exceeded the size cap")
    return json.loads(raw.decode("utf-8"))


def catalog_url(api_base: str, capability: str) -> str:
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    rule = _CAPABILITY_RULES.get(capability)
    server_capability = rule[0] if rule else None
    if server_capability:
        return f"{base}/models?capability={server_capability}"
    return f"{base}/models"


def _cache_path(capability: str, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        override = (os.environ.get(ENV_CACHE_DIR) or "").strip()
        cache_dir = (
            Path(override).expanduser()
            if override
            else Path.home() / ".researchclaw" / "orcarouter"
        )
    return Path(cache_dir) / f"catalog_{capability}.json"


def read_cache(
    capability: str, *, cache_dir: Path | None = None, ttl: float | None = None
) -> list[CatalogModel] | None:
    """Last-known-good catalogue, if it is still fresh. Not a secret."""
    path = _cache_path(capability, cache_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    age = time.time() - float(data.get("fetched_at") or 0.0)
    effective_ttl = (
        ttl
        if ttl is not None
        else float(os.environ.get(ENV_CATALOG_TTL, DEFAULT_CACHE_TTL_SEC) or 0.0)
    )
    if effective_ttl <= 0 or age > effective_ttl:
        return None
    return parse_models(data.get("payload"))


def write_cache(
    capability: str, payload: Any, *, cache_dir: Path | None = None
) -> None:
    path = _cache_path(capability, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload}),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("Could not cache OrcaRouter catalogue: %s", exc)


def discover_models(
    api_base: str,
    api_key: str,
    *,
    capability: str = CAPABILITY_CHAT,
    required_input_modalities: Sequence[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SEC,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_items: int = DEFAULT_MAX_ITEMS,
    fetcher: Fetcher | None = None,
    cache_dir: Path | None = None,
    cache_ttl: float | None = None,
    use_cache: bool = True,
) -> CatalogResult:
    """Resolve one capability's model list, degrading instead of failing."""
    url = catalog_url(api_base, capability)
    fetch = fetcher or _default_fetcher
    error = ""

    if api_key:
        try:
            payload = fetch(url, api_key, timeout, max_bytes)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            error = _describe_fetch_error(exc)
        else:
            live = merge_verified_metadata(parse_models(payload, max_items=max_items))
            if use_cache:
                write_cache(capability, payload, cache_dir=cache_dir)
            selected = filter_models(
                live,
                capability,
                required_input_modalities=required_input_modalities,
            )
            return CatalogResult(
                models=tuple(selected),
                source="live",
                capability=capability,
                required_input_modalities=tuple(required_input_modalities),
                degraded=False,
                fetched_at=time.time(),
                live_model_count=len(live),
                considered=len(live),
            )
    else:
        error = "no OrcaRouter credential configured"

    if use_cache:
        cached = read_cache(capability, cache_dir=cache_dir, ttl=cache_ttl)
        if cached:
            selected = filter_models(
                merge_verified_metadata(cached),
                capability,
                required_input_modalities=required_input_modalities,
            )
            if selected:
                return CatalogResult(
                    models=tuple(selected),
                    source="cache",
                    capability=capability,
                    required_input_modalities=tuple(required_input_modalities),
                    degraded=True,
                    error=error,
                    fetched_at=time.time(),
                    live_model_count=len(cached),
                    considered=len(cached),
                )

    seed_models = seed_for_capability(
        capability, required_input_modalities=required_input_modalities
    )
    return CatalogResult(
        models=seed_models,
        source="seed",
        capability=capability,
        required_input_modalities=tuple(required_input_modalities),
        degraded=True,
        error=error,
        fetched_at=time.time(),
        live_model_count=0,
        considered=len(VERIFIED_SEED),
    )


def _describe_fetch_error(exc: BaseException) -> str:
    """A user-facing reason that never contains a credential."""
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return "OrcaRouter rejected the credential (HTTP 401)"
        if exc.code == 403:
            return "this key may not list models (HTTP 403)"
        if exc.code == 429:
            return "rate limited while listing models (HTTP 429)"
        return f"model catalogue request failed (HTTP {exc.code})"
    if isinstance(exc, (urllib.error.URLError, OSError)):
        return f"could not reach the model catalogue ({type(exc).__name__})"
    return "model catalogue response was not usable"


def is_model_available(
    model_id: str,
    models: Sequence[CatalogModel],
) -> bool:
    """Re-validate a remembered selection before restoring it."""
    return any(model.id == model_id for model in models)


__all__ = [
    "CAPABILITY_CHAT",
    "CAPABILITY_EMBEDDING",
    "CAPABILITY_IMAGE",
    "CAPABILITY_RERANK",
    "CAPABILITY_VIDEO",
    "CatalogModel",
    "CatalogResult",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_TIMEOUT_SEC",
    "ENV_CACHE_DIR",
    "ENV_CATALOG_TTL",
    "NON_TEXT_ENDPOINT_TYPES",
    "TEXT_ENDPOINT_TYPES",
    "VERIFIED_SEED",
    "catalog_url",
    "discover_models",
    "filter_models",
    "is_model_available",
    "matches_capability",
    "merge_verified_metadata",
    "parse_models",
    "read_cache",
    "seed_for_capability",
    "write_cache",
]
