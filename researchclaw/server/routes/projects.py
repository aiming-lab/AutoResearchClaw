"""Project listing / status API routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["projects"])


@router.get("/projects")
async def list_projects() -> dict[str, Any]:
    """List all project directories (artifacts/rc-*)."""
    artifacts = Path("artifacts")
    projects: list[dict[str, Any]] = []
    if artifacts.exists():
        for d in sorted(artifacts.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("rc-"):
                proj: dict[str, Any] = {
                    "id": d.name,
                    "path": str(d),
                }
                ckpt = d / "checkpoint.json"
                if ckpt.exists():
                    try:
                        with ckpt.open() as f:
                            ckpt_data = json.load(f)
                        proj["current_stage"] = ckpt_data.get("stage")
                        proj["status"] = ckpt_data.get("status", "unknown")
                    except Exception:
                        proj["status"] = "unknown"
                else:
                    proj["status"] = "no_checkpoint"
                projects.append(proj)
    return {"projects": projects}
