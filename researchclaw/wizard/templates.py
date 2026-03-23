"""Preset research configuration templates."""

from __future__ import annotations

from typing import Any

TEMPLATES: dict[str, dict[str, Any]] = {
    "quick-demo": {
        "description": "5-minute quick demo (simulated mode, no GPU needed)",
        "experiment.mode": "simulated",
        "experiment.time_budget_sec": 60,
        "experiment.max_iterations": 3,
    },
    "standard-cv": {
        "description": "Standard Computer Vision paper (Docker + CIFAR-10)",
        "research.domains": ["computer-vision"],
        "experiment.mode": "docker",
        "experiment.time_budget_sec": 600,
        "experiment.docker.gpu_enabled": True,
        "experiment.docker.network_policy": "setup_only",
    },
    "deep-nlp": {
        "description": "Deep NLP research (Docker + GPU + transformers)",
        "research.domains": ["nlp", "transformers"],
        "experiment.mode": "docker",
        "experiment.time_budget_sec": 1200,
        "experiment.docker.gpu_enabled": True,
        "experiment.docker.memory_limit_mb": 16384,
    },
    "rl-research": {
        "description": "Reinforcement Learning research (Docker + custom env)",
        "research.domains": ["reinforcement-learning"],
        "experiment.mode": "docker",
        "experiment.time_budget_sec": 900,
        "experiment.docker.gpu_enabled": True,
    },
    "ai4science": {
        "description": "AI for Science (large compute budget)",
        "research.domains": ["ai4science"],
        "experiment.mode": "docker",
        "experiment.time_budget_sec": 1800,
        "experiment.docker.gpu_enabled": True,
        "experiment.docker.memory_limit_mb": 32768,
    },
}


def get_template(name: str) -> dict[str, Any] | None:
    """Get a template by name."""
    return TEMPLATES.get(name)


def list_templates() -> list[dict[str, str]]:
    """List all available templates with descriptions."""
    return [
        {"name": name, "description": tpl.get("description", "")}
        for name, tpl in TEMPLATES.items()
    ]
