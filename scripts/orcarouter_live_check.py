#!/usr/bin/env python3
"""Live end-to-end check through the implemented OrcaRouter provider path.

This is *not* a standalone curl: it builds the same client
``create_llm_client`` builds for ``provider: orcarouter`` and drives both the
inference call and catalogue discovery through the shipped code.

Requires ORCAROUTER_API_KEY in the environment. Prints no credential.

Usage:
    python scripts/orcarouter_live_check.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from researchclaw.llm import create_llm_client, orcarouter as orca  # noqa: E402
from researchclaw.llm import orcarouter_catalog as catalog  # noqa: E402
from researchclaw.llm.model_select import build_model_selector  # noqa: E402


def _config(**overrides):
    llm = dict(
        provider="orcarouter",
        base_url="",
        api_key="",
        api_key_env="ORCAROUTER_API_KEY",
        wire_api="chat_completions",
        primary_model="",
        fallback_models=(),
        timeout_sec=120,
    )
    llm.update(overrides)
    return SimpleNamespace(llm=SimpleNamespace(**llm))


def main() -> int:
    report: dict[str, object] = {}

    if not os.environ.get("ORCAROUTER_API_KEY"):
        print("ORCAROUTER_API_KEY is not set — cannot run the live check.", file=sys.stderr)
        return 2

    endpoints = orca.resolve_endpoints()
    report["auth_origin"] = endpoints.auth_base
    report["inference_origin"] = endpoints.api_base

    # 1. Catalogue through the shipped discovery path.
    credential = orca.resolve_credential()
    report["credential_source"] = credential.source
    report["credential_masked"] = credential.masked

    chat = catalog.discover_models(
        endpoints.api_base, credential.api_key, capability=catalog.CAPABILITY_CHAT
    )
    report["catalog_source"] = catalog.catalog_url(endpoints.api_base, "chat")
    report["chat_model_count"] = chat.count
    report["chat_models"] = chat.ids
    report["catalog_backend"] = chat.source

    # 2. The same options the UI/model selector receives.
    selector = build_model_selector(_config())
    assert selector is not None
    report["selector_option_count"] = len(selector.options)
    # The relay does not guarantee a stable tail order; membership and count
    # are what matter, and both come from the same live response.
    report["selector_matches_catalog"] = (
        selector.ids == chat.ids or set(selector.ids) == set(chat.ids)
    )
    report["selector_source"] = selector.source

    multimodal = build_model_selector(_config(), attachments=["image"])
    assert multimodal is not None
    report["image_capable_chat_models"] = multimodal.ids
    report["multimodal_is_subset"] = set(multimodal.ids) <= set(chat.ids)
    report["multimodal_declared_only"] = all(
        "image" in option.input_modalities for option in multimodal.options
    )

    # 3. A real inference call through the client the pipeline uses. The
    # catalogue lists what the gateway can route; the *key* may still be
    # scoped to a subset (the relay answers 403 model_access_denied), so the
    # client's own fallback chain is exercised here.
    # No model configured: the provider resolves one from the account's own
    # catalogue, which is exactly the path a fresh OrcaRouter config takes.
    client = create_llm_client(_config())
    report["inference_url"] = client._endpoint_url(client.config.base_url)
    report["inference_chain"] = client._model_chain
    report["inference_primary"] = client.config.primary_model
    report["primary_is_from_catalog"] = client.config.primary_model in chat.ids

    try:
        response = client.chat(
            [{"role": "user", "content": "Reply with the single word: pong"}],
            # Reasoning models spend budget on hidden reasoning first; a tiny
            # cap can leave the visible answer empty and look like a failure.
            max_tokens=256,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        report["inference_ok"] = False
        report["inference_error"] = orca.redact_secrets(str(exc))[:300]
    else:
        report["inference_ok"] = bool(response.content)
        report["inference_reply_model"] = response.model
        report["inference_reply"] = response.content[:60]
        report["inference_usage"] = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }

    # 4. Embedding/rerank entry points have no compatible model here.
    report["embedding_model_count"] = len(
        catalog.discover_models(
            endpoints.api_base,
            credential.api_key,
            capability=catalog.CAPABILITY_EMBEDDING,
        ).ids
    )
    report["image_generation_model_count"] = len(
        catalog.discover_models(
            endpoints.api_base,
            credential.api_key,
            capability=catalog.CAPABILITY_IMAGE,
        ).ids
    )

    ok = bool(report.get("inference_ok")) and report["selector_matches_catalog"]
    report["passed"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
