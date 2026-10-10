"""Explicit ACP stage routing without changing legacy provider behaviour."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig


def routing_enabled(config: RCConfig) -> bool:
    llm = getattr(config, "llm", None)
    return getattr(llm, "provider", None) == "acp" and bool(
        getattr(getattr(llm, "acp", None), "model_routing", False)
    )


def stage_model(config: RCConfig, stage: int) -> str:
    if not 1 <= int(stage) <= 23:
        raise ValueError(f"Invalid pipeline stage: {stage}")
    return dict(config.llm.acp.stage_models).get(int(stage), config.llm.primary_model)


def resolve_stage_config(
    config: RCConfig, stage: int, run_id: str, *, execution: bool = False,
    purpose: str = "stage", model: str | None = None,
) -> RCConfig:
    """Bind one model and isolate its session by run, stage and purpose.

    Stages exchange their saved artifacts, not a shared conversation whose
    model can be silently changed by another client.
    """
    if not routing_enabled(config):
        return config
    model = model or stage_model(config, stage)
    if execution:
        model = config.llm.acp.execution_model or model
    identity = json.dumps([run_id, int(stage), purpose, model, config.llm.acp.reasoning_effort])
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    acp = replace(config.llm.acp, session_name=f"{config.llm.acp.session_name}-{suffix}")
    return replace(config, llm=replace(config.llm, primary_model=model, fallback_models=(), acp=acp))


def prepare_routed_client(client: Any, config: RCConfig, evidence_path: Path) -> None:
    """Verify selection before a stage can catch errors and emit a template."""
    if not routing_enabled(config):
        return
    record = {
        "requested_model": config.llm.primary_model,
        "requested_reasoning_effort": config.llm.acp.reasoning_effort,
        "session_name": config.llm.acp.session_name,
        "verified": False,
    }
    try:
        client.prepare()
        record.update(client.session_selection)
        if not record.get("verified"):
            raise RuntimeError("ACP session model selection was not verified")
    except Exception as exc:
        record["verified"] = False
        record["error"] = str(exc)
        raise
    finally:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def check_selection_error(client: Any, evidence_path: Path) -> None:
    """Refresh evidence and reject failures swallowed by nested callers."""
    error = getattr(client, "selection_error", None)
    selection = getattr(client, "session_selection", None)
    if not error:
        if isinstance(selection, dict) and selection.get("verified"):
            evidence_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        return
    record = {**selection, "verified": False, "error": error}
    evidence_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError(f"Required ACP model selection failed: {error}")
