"""Skill data model definition (agentskills.io compatible)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Maps pipeline stage names to stage numbers.
STAGE_NAME_TO_NUMBER: dict[str, int] = {
    "topic_init": 1,
    "problem_decompose": 2,
    "search_strategy": 3,
    "literature_collect": 4,
    "literature_screen": 5,
    "knowledge_extract": 6,
    "synthesis": 7,
    "hypothesis_gen": 8,
    "experiment_design": 9,
    "code_generation": 10,
    "resource_planning": 11,
    "experiment_run": 12,
    "iterative_refine": 13,
    "result_analysis": 14,
    "research_decision": 15,
    "paper_outline": 16,
    "paper_draft": 17,
    "peer_review": 18,
    "paper_revision": 19,
    "quality_gate": 20,
    "knowledge_archive": 21,
    "export_publish": 22,
    "citation_verify": 23,
}

# Valid categories in the new taxonomy.
VALID_CATEGORIES = ("writing", "domain", "experiment", "tooling")


@dataclass
class Skill:
    """A single skill definition (agentskills.io compatible).

    Standard fields follow the agentskills.io specification.
    Legacy YAML fields are accessible via backward-compat properties
    that read from ``metadata``.
    """

    # agentskills.io standard fields
    name: str
    description: str
    body: str = ""
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    # filesystem context
    source_dir: Path | None = None
    source_format: str = "skillmd"  # "skillmd" | "yaml"

    # ── backward-compat property accessors ───────────────────────

    @property
    def id(self) -> str:  # noqa: A003
        """Alias for ``name`` (legacy)."""
        return self.name

    @property
    def category(self) -> str:
        return self.metadata.get("category", "domain")

    @property
    def trigger_keywords(self) -> list[str]:
        raw = self.metadata.get("trigger-keywords", "")
        return [k.strip() for k in raw.split(",") if k.strip()] if raw else []

    @property
    def applicable_stages(self) -> list[int]:
        raw = self.metadata.get("applicable-stages", "")
        if not raw:
            return []
        parts: list[int] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok.isdigit():
                parts.append(int(tok))
        return parts

    @property
    def priority(self) -> int:
        return int(self.metadata.get("priority", "5"))

    @property
    def prompt_template(self) -> str:
        """Alias for ``body`` (legacy)."""
        return self.body

    @property
    def code_template(self) -> str | None:
        return self.metadata.get("code-template") or None

    @property
    def references(self) -> list[str]:
        raw = self.metadata.get("references", "")
        return [r.strip() for r in raw.split(";") if r.strip()] if raw else []

    @property
    def version(self) -> str:
        return self.metadata.get("version", "1.0")

    # ── serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (legacy-compatible output)."""
        return {
            "id": self.name,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "trigger_keywords": self.trigger_keywords,
            "applicable_stages": self.applicable_stages,
            "prompt_template": self.body,
            "code_template": self.code_template,
            "references": self.references,
            "version": self.version,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        """Deserialize from a legacy YAML/JSON dictionary."""
        # Pack legacy top-level fields into metadata
        meta: dict[str, str] = {}
        if data.get("category"):
            meta["category"] = str(data["category"])
        kw = data.get("trigger_keywords") or []
        if kw:
            meta["trigger-keywords"] = ",".join(str(k) for k in kw)
        stages = data.get("applicable_stages") or []
        if stages:
            meta["applicable-stages"] = ",".join(str(s) for s in stages)
        if data.get("priority") is not None:
            meta["priority"] = str(data["priority"])
        if data.get("version"):
            meta["version"] = str(data["version"])
        if data.get("code_template"):
            meta["code-template"] = str(data["code_template"])
        refs = data.get("references") or []
        if refs:
            meta["references"] = "; ".join(str(r) for r in refs)
        # Merge any explicit metadata from the dict
        if isinstance(data.get("metadata"), dict):
            for k, v in data["metadata"].items():
                meta.setdefault(str(k), str(v))

        name = str(data.get("name") or data.get("id") or "")
        # For legacy YAML, use 'id' if 'name' looks like a display name
        # and 'id' looks like a slug
        raw_id = str(data.get("id", ""))
        if raw_id and "-" in raw_id:
            name = raw_id

        return cls(
            name=name,
            description=str(data.get("description", "")),
            body=str(data.get("prompt_template", "")),
            metadata=meta,
            source_format="yaml",
        )
