"""Pipeline control API routes."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

import re as _re
_RUN_ID_RE = _re.compile(r"^rc-\d{8}-\d{6}-[a-f0-9]+$")


def _validated_run_dir(run_id: str) -> Path:
    """Validate run_id format and return the run directory path."""
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")
    run_dir = Path("artifacts") / run_id
    # Ensure resolved path is under artifacts/
    if not run_dir.resolve().is_relative_to(Path("artifacts").resolve()):
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id}")
    return run_dir

router = APIRouter(prefix="/api", tags=["pipeline"])


class PipelineStartRequest(BaseModel):
    """Request body for starting a pipeline run."""

    topic: str | None = None
    config_overrides: dict[str, Any] | None = None
    auto_approve: bool = True


class PipelineStartResponse(BaseModel):
    """Response after starting a pipeline."""

    run_id: str
    status: str
    output_dir: str


# In-memory tracking of the active run (single-tenant MVP)
_active_run: dict[str, Any] | None = None
_run_task: asyncio.Task[Any] | None = None


def _get_app_state() -> dict[str, Any]:
    """Get shared application state (set by app.py)."""
    from researchclaw.server.app import _app_state
    return _app_state


@router.post("/pipeline/start", response_model=PipelineStartResponse)
async def start_pipeline(req: PipelineStartRequest) -> PipelineStartResponse:
    """Start a new pipeline run."""
    global _active_run, _run_task

    if _active_run and _active_run.get("status") == "running":
        raise HTTPException(status_code=409, detail="A pipeline is already running")

    state = _get_app_state()
    config = state["config"]

    if req.topic:
        import dataclasses
        new_research = dataclasses.replace(config.research, topic=req.topic)
        config = dataclasses.replace(config, research=new_research)

    import hashlib
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    topic_hash = hashlib.sha256(config.research.topic.encode()).hexdigest()[:6]
    run_id = f"rc-{ts}-{topic_hash}"
    run_dir = _validated_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    _active_run = {
        "run_id": run_id,
        "status": "running",
        "output_dir": str(run_dir),
        "topic": config.research.topic,
    }

    async def _run_in_background() -> None:
        global _active_run
        try:
            from researchclaw.adapters import AdapterBundle
            from researchclaw.pipeline.runner import execute_pipeline

            kb_root = Path(config.knowledge_base.root) if config.knowledge_base.root else None
            if kb_root:
                kb_root.mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: execute_pipeline(
                    run_dir=run_dir,
                    run_id=run_id,
                    config=config,
                    adapters=AdapterBundle(),
                    auto_approve_gates=req.auto_approve,
                    skip_noncritical=True,
                    kb_root=kb_root,
                ),
            )
            done = sum(1 for r in results if r.status.value == "done")
            failed = sum(1 for r in results if r.status.value == "failed")
            if _active_run:
                _active_run["status"] = "completed" if failed == 0 else "failed"
                _active_run["stages_done"] = done
                _active_run["stages_failed"] = failed
        except Exception as exc:
            logger.exception("Pipeline run failed")
            if _active_run:
                _active_run["status"] = "failed"
                _active_run["error"] = str(exc)

    _run_task = asyncio.create_task(_run_in_background())

    return PipelineStartResponse(
        run_id=run_id,
        status="running",
        output_dir=str(run_dir),
    )


@router.post("/pipeline/stop")
async def stop_pipeline() -> dict[str, str]:
    """Stop the currently running pipeline."""
    global _active_run, _run_task

    if not _run_task or not _active_run:
        raise HTTPException(status_code=404, detail="No pipeline is running")

    _run_task.cancel()
    _active_run["status"] = "stopped"
    return {"status": "stopped"}


@router.get("/pipeline/status")
async def pipeline_status() -> dict[str, Any]:
    """Get current pipeline run status."""
    if not _active_run:
        return {"status": "idle"}
    return _active_run


@router.get("/pipeline/stages")
async def pipeline_stages() -> dict[str, Any]:
    """Get the 23-stage pipeline definition."""
    from researchclaw.pipeline.stages import Stage

    stages = []
    for s in Stage:
        stages.append({
            "number": int(s),
            "name": s.name,
            "label": getattr(s, "label", s.name.replace("_", " ").title()),
            "phase": getattr(s, "phase", ""),
        })
    return {"stages": stages}


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
    """List historical pipeline runs from artifacts/ directory."""
    artifacts = Path("artifacts")
    runs: list[dict[str, Any]] = []
    if artifacts.exists():
        for d in sorted(artifacts.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("rc-"):
                info: dict[str, Any] = {"run_id": d.name, "path": str(d)}
                # Try reading checkpoint
                ckpt = d / "checkpoint.json"
                if ckpt.exists():
                    try:
                        with ckpt.open() as f:
                            info["checkpoint"] = json.load(f)
                    except Exception:
                        pass
                runs.append(info)
    return {"runs": runs[:50]}  # limit to 50 most recent


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Get details for a specific run."""
    run_dir = _validated_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    info: dict[str, Any] = {"run_id": run_id, "path": str(run_dir)}

    ckpt = run_dir / "checkpoint.json"
    if ckpt.exists():
        try:
            with ckpt.open() as f:
                info["checkpoint"] = json.load(f)
        except Exception:
            pass

    # List stage directories
    stage_dirs = sorted(
        [d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("stage-")]
    )
    info["stages_completed"] = stage_dirs

    # Check for paper
    for pattern in ["paper.md", "paper.tex", "paper.pdf"]:
        found = list(run_dir.rglob(pattern))
        if found:
            info[f"has_{pattern.split('.')[1]}"] = True

    return info


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str) -> dict[str, Any]:
    """Get experiment metrics for a run."""
    run_dir = _validated_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    metrics: dict[str, Any] = {}
    results_file = run_dir / "results.json"
    if results_file.exists():
        try:
            with results_file.open() as f:
                metrics = json.load(f)
        except Exception:
            pass

    return {"run_id": run_id, "metrics": metrics}
