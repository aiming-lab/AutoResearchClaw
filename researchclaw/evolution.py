"""Self-evolution system for the ResearchClaw pipeline.

Records lessons from each pipeline run (failures, slow stages, quality issues)
and injects them into future runs as prompt overlays.  Inspired by Sibyl's
time-weighted evolution mechanism.

Architecture
------------
* ``LessonCategory`` — 6 issue categories for classification.
* ``LessonEntry`` — single lesson (stage, category, severity, description, ts).
* ``EvolutionStore`` — JSONL-backed persistent store with append + query.
* ``extract_lessons()`` — auto-extract lessons from ``StageResult`` lists.
* ``build_overlay()`` — generate per-stage prompt overlay text.

Usage
-----
::

    from researchclaw.evolution import EvolutionStore, extract_lessons

    store = EvolutionStore(Path("evolution"))
    lessons = extract_lessons(results)
    store.append_many(lessons)
    overlay = store.build_overlay("hypothesis_gen", max_lessons=5)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LessonCategory(str, Enum):
    """Issue classification for extracted lessons."""

    SYSTEM = "system"          # Environment / network / timeout
    EXPERIMENT = "experiment"  # Code validation, sandbox timeout
    WRITING = "writing"        # Paper quality issues
    ANALYSIS = "analysis"      # Weak analysis, missing comparison
    LITERATURE = "literature"  # Search / verification failures
    PIPELINE = "pipeline"      # Stage orchestration issues


@dataclass
class LessonEntry:
    """A single lesson extracted from a pipeline run."""

    stage_name: str
    stage_num: int
    category: str
    severity: str  # "info", "warning", "error"
    description: str
    timestamp: str  # ISO 8601
    run_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LessonEntry:
        return cls(
            stage_name=str(data.get("stage_name", "")),
            stage_num=int(data.get("stage_num", 0)),
            category=str(data.get("category", "pipeline")),
            severity=str(data.get("severity", "info")),
            description=str(data.get("description", "")),
            timestamp=str(data.get("timestamp", "")),
            run_id=str(data.get("run_id", "")),
        )


# ---------------------------------------------------------------------------
# Lesson classification keywords
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    LessonCategory.SYSTEM: [
        "timeout", "connection", "network", "oom", "memory",
        "permission", "ssh", "socket", "dns",
    ],
    LessonCategory.EXPERIMENT: [
        "sandbox", "validation", "import", "syntax", "subprocess",
        "experiment", "code", "execution",
    ],
    LessonCategory.WRITING: [
        "paper", "draft", "outline", "revision", "review",
        "template", "latex",
    ],
    LessonCategory.ANALYSIS: [
        "analysis", "metric", "statistic", "comparison", "baseline",
    ],
    LessonCategory.LITERATURE: [
        "search", "citation", "verify", "hallucin", "arxiv",
        "semantic_scholar", "literature", "collect",
    ],
}


def _classify_error(stage_name: str, error_text: str) -> str:
    """Classify an error into a LessonCategory based on keywords."""
    combined = f"{stage_name} {error_text}".lower()
    best_category = LessonCategory.PIPELINE
    best_score = 0
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


# ---------------------------------------------------------------------------
# Lesson extraction from pipeline results
# ---------------------------------------------------------------------------

# Stage name mapping (import-free to avoid circular deps)
_STAGE_NAMES: dict[int, str] = {
    1: "topic_init", 2: "problem_decompose", 3: "search_strategy",
    4: "literature_collect", 5: "literature_screen", 6: "knowledge_extract",
    7: "synthesis", 8: "hypothesis_gen", 9: "experiment_design",
    10: "code_generation", 11: "resource_planning", 12: "experiment_run",
    13: "iterative_refine", 14: "result_analysis", 15: "research_decision",
    16: "paper_outline", 17: "paper_draft", 18: "peer_review",
    19: "paper_revision", 20: "quality_gate", 21: "knowledge_archive",
    22: "export_publish", 23: "citation_verify",
}


def extract_lessons(
    results: list[object],
    run_id: str = "",
    run_dir: Path | None = None,
) -> list[LessonEntry]:
    """Extract lessons from a list of StageResult objects.

    Detects:
    - Failed stages → error lesson
    - Blocked stages → pipeline lesson
    - Decision pivots/refines → pipeline lesson (with rationale if available)
    - Runtime warnings from experiment stderr → code_bug lesson
    - Metric anomalies (NaN, identical convergence) → metric_anomaly lesson
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lessons: list[LessonEntry] = []

    for result in results:
        stage_num = int(getattr(result, "stage", 0))
        stage_name = _STAGE_NAMES.get(stage_num, f"stage_{stage_num}")
        status = str(getattr(result, "status", ""))
        error = getattr(result, "error", None)
        decision = str(getattr(result, "decision", "proceed"))

        # Failed stages
        if "failed" in status.lower() and error:
            category = _classify_error(stage_name, str(error))
            lessons.append(LessonEntry(
                stage_name=stage_name,
                stage_num=stage_num,
                category=category,
                severity="error",
                description=f"Stage {stage_name} failed: {str(error)[:300]}",
                timestamp=now,
                run_id=run_id,
            ))

        # Blocked stages
        if "blocked" in status.lower():
            lessons.append(LessonEntry(
                stage_name=stage_name,
                stage_num=stage_num,
                category=LessonCategory.PIPELINE,
                severity="warning",
                description=f"Stage {stage_name} blocked awaiting approval",
                timestamp=now,
                run_id=run_id,
            ))

        # PIVOT / REFINE decisions — extract rationale if available
        if decision in ("pivot", "refine"):
            rationale = _extract_decision_rationale(run_dir) if run_dir else ""
            desc = f"Research decision was {decision.upper()}"
            if rationale:
                desc += f": {rationale[:200]}"
            else:
                desc += " — prior hypotheses/experiments were insufficient"
            lessons.append(LessonEntry(
                stage_name=stage_name,
                stage_num=stage_num,
                category=LessonCategory.PIPELINE,
                severity="warning",
                description=desc,
                timestamp=now,
                run_id=run_id,
            ))

    # --- Extract lessons from experiment artifacts ---
    if run_dir is not None:
        lessons.extend(_extract_runtime_lessons(run_dir, now, run_id))

    return lessons


def _extract_decision_rationale(run_dir: Path) -> str:
    """Extract rationale from the most recent decision_structured.json.

    Supports multiple field formats:
    - ``rationale`` or ``reason`` key (direct)
    - ``raw_text_excerpt`` containing ``## Justification`` section (LLM output)
    """
    for stage_dir in sorted(run_dir.glob("stage-15*"), reverse=True):
        decision_file = stage_dir / "decision_structured.json"
        if decision_file.exists():
            try:
                data = json.loads(decision_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                # Try direct rationale/reason keys first
                direct = data.get("rationale", "") or data.get("reason", "")
                if direct:
                    return str(direct)
                # Parse raw_text_excerpt for Justification section
                raw = data.get("raw_text_excerpt", "")
                if raw:
                    return _parse_justification_from_excerpt(str(raw))
            except (json.JSONDecodeError, OSError):
                pass
    return ""


def _parse_justification_from_excerpt(text: str) -> str:
    """Extract the Justification/Rationale section from LLM decision text."""
    import re

    # Match ## Justification, ## Rationale, or similar headings
    pattern = re.compile(
        r"##\s*(?:Justification|Rationale|Reason)\s*\n(.*?)(?=\n##|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()[:300]
    # Fallback: skip the first line (## Decision / **REFINE**) and return the rest
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Skip heading lines starting with ## or **
    content_lines = [
        l for l in lines
        if not l.startswith("##") and not (l.startswith("**") and l.endswith("**"))
    ]
    if content_lines:
        return " ".join(content_lines)[:300]
    return ""


def _extract_runtime_lessons(
    run_dir: Path, timestamp: str, run_id: str
) -> list[LessonEntry]:
    """Extract fine-grained lessons from experiment run artifacts."""
    import math

    lessons: list[LessonEntry] = []

    # Check sandbox run results for stderr warnings and NaN
    for runs_dir in run_dir.glob("stage-*/runs"):
        for run_file in runs_dir.glob("*.json"):
            if run_file.name == "results.json":
                continue
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue

            # Check stderr for runtime warnings
            stderr = payload.get("stderr", "")
            if stderr and any(
                kw in stderr for kw in ("Warning", "Error", "divide", "overflow", "invalid value")
            ):
                lessons.append(LessonEntry(
                    stage_name="experiment_run",
                    stage_num=12,
                    category=LessonCategory.EXPERIMENT,
                    severity="warning",
                    description=f"Runtime warning in experiment: {stderr[:200]}",
                    timestamp=timestamp,
                    run_id=run_id,
                ))

            # Check metrics for NaN/Inf
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                for key, val in metrics.items():
                    try:
                        fval = float(val)
                        if math.isnan(fval) or math.isinf(fval):
                            lessons.append(LessonEntry(
                                stage_name="experiment_run",
                                stage_num=12,
                                category=LessonCategory.EXPERIMENT,
                                severity="error",
                                description=f"Metric '{key}' was {val} — code bug (division by zero or overflow)",
                                timestamp=timestamp,
                                run_id=run_id,
                            ))
                    except (TypeError, ValueError):
                        pass

    return lessons


# ---------------------------------------------------------------------------
# Time-decay weighting
# ---------------------------------------------------------------------------

HALF_LIFE_DAYS: float = 30.0
MAX_AGE_DAYS: float = 90.0


def _time_weight(timestamp_iso: str) -> float:
    """Compute exponential decay weight for a lesson based on age.

    Uses 30-day half-life: weight = exp(-age_days * ln(2) / 30).
    Returns 0.0 for lessons older than 90 days.
    """
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        age_days = age.total_seconds() / 86400.0
        if age_days > MAX_AGE_DAYS:
            return 0.0
        return math.exp(-age_days * math.log(2) / HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Evolution store
# ---------------------------------------------------------------------------


class EvolutionStore:
    """JSONL-backed store for pipeline lessons."""

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lessons_path = self._dir / "lessons.jsonl"

    @property
    def lessons_path(self) -> Path:
        return self._lessons_path

    def append(self, lesson: LessonEntry) -> None:
        """Append a single lesson to the store."""
        with self._lessons_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

    def append_many(self, lessons: list[LessonEntry]) -> None:
        """Append multiple lessons atomically."""
        if not lessons:
            return
        with self._lessons_path.open("a", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Appended %d lessons to evolution store", len(lessons))

    def load_all(self) -> list[LessonEntry]:
        """Load all lessons from disk."""
        if not self._lessons_path.exists():
            return []
        lessons: list[LessonEntry] = []
        for line in self._lessons_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                lessons.append(LessonEntry.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                continue
        return lessons

    def query_for_stage(
        self, stage_name: str, *, max_lessons: int = 5
    ) -> list[LessonEntry]:
        """Return the most relevant lessons for a stage, weighted by recency.

        Includes lessons that directly match the stage, plus high-severity
        lessons from related stages.
        """
        all_lessons = self.load_all()
        scored: list[tuple[float, LessonEntry]] = []
        for lesson in all_lessons:
            weight = _time_weight(lesson.timestamp)
            if weight <= 0.0:
                continue
            # Boost direct stage matches
            if lesson.stage_name == stage_name:
                weight *= 2.0
            # Boost errors over warnings/info
            if lesson.severity == "error":
                weight *= 1.5
            scored.append((weight, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_lessons]]

    def build_overlay(
        self,
        stage_name: str,
        *,
        max_lessons: int = 5,
        skills_dir: str = "",
    ) -> str:
        """Generate a prompt overlay string for a given stage.

        Combines two sources:
        1. Current-run lessons from ``lessons.jsonl`` (intra-run learning).
        2. Cross-run MetaClaw ``arc-*`` skills from *skills_dir* (inter-run
           learning via the MetaClaw skill-generation feedback loop).

        Returns empty string if no relevant lessons or skills exist.
        """
        parts: list[str] = []

        # --- Section 1: intra-run lessons ---
        lessons = self.query_for_stage(stage_name, max_lessons=max_lessons)
        if lessons:
            parts.append("## Lessons from Prior Runs")
            for i, lesson in enumerate(lessons, 1):
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                    lesson.severity, "•"
                )
                parts.append(
                    f"{i}. {severity_icon} [{lesson.category}] {lesson.description}"
                )
            parts.append(
                "\nUse these lessons to avoid repeating past mistakes."
            )

        # --- Section 2: cross-run MetaClaw arc-* skills ---
        if skills_dir:
            from pathlib import Path as _Path

            sd = _Path(skills_dir).expanduser()
            if sd.is_dir():
                arc_skills: list[str] = []
                for skill_dir in sorted(sd.iterdir()):
                    if skill_dir.is_dir() and skill_dir.name.startswith("arc-"):
                        skill_file = skill_dir / "SKILL.md"
                        if skill_file.is_file():
                            try:
                                text = skill_file.read_text(encoding="utf-8").strip()
                                if text:
                                    arc_skills.append(text)
                            except OSError:
                                continue
                if arc_skills:
                    parts.append("\n## Learned Skills from Prior Runs")
                    for skill_text in arc_skills[:5]:
                        parts.append(skill_text)
                    parts.append(
                        "\nApply these skills proactively to improve quality."
                    )

        return "\n".join(parts)

    def count(self) -> int:
        """Return total number of stored lessons."""
        return len(self.load_all())

    def export_to_memory(self, memory_store: object) -> int:
        """Export lessons to a memory store (duck-typed to avoid circular imports).

        The *memory_store* must expose an ``add(content, category, metadata)`` method
        (compatible with ``researchclaw.memory.store.MemoryStore``).

        Returns the number of lessons exported.
        """
        add_fn = getattr(memory_store, "add", None)
        if add_fn is None or not callable(add_fn):
            logger.warning("export_to_memory: memory_store has no add() method")
            return 0
        lessons = self.load_all()
        exported = 0
        for lesson in lessons:
            weight = _time_weight(lesson.timestamp)
            if weight <= 0.0:
                continue
            try:
                # Map lesson categories to valid MemoryStore categories
                _CAT_MAP = {
                    "system": "experiment", "analysis": "experiment",
                    "literature": "ideation", "pipeline": "experiment",
                    "experiment": "experiment", "writing": "writing",
                    "ideation": "ideation",
                }
                _mem_cat = _CAT_MAP.get(lesson.category, "experiment")
                add_fn(
                    content=lesson.description,
                    category=_mem_cat,
                    metadata={
                        "source": "evolution",
                        "stage": lesson.stage_name,
                        "severity": lesson.severity,
                        "run_id": lesson.run_id,
                        "timestamp": lesson.timestamp,
                    },
                )
                exported += 1
            except Exception:
                logger.debug("Failed to export lesson: %s", lesson.description[:80])
        return exported

    def get_lessons_for_stage_with_memory(
        self,
        stage_name: str,
        memory_store: object,
        *,
        max_lessons: int = 5,
    ) -> str:
        """Combine evolution overlay with memory context for a stage.

        *memory_store* must expose a ``recall(query, category, max_results)`` method
        returning objects with a ``.content`` attribute.
        """
        overlay = self.build_overlay(stage_name, max_lessons=max_lessons)
        recall_fn = getattr(memory_store, "recall", None)
        if recall_fn is None or not callable(recall_fn):
            return overlay
        try:
            memories = recall_fn(
                query=stage_name,
                category=None,
                max_results=max_lessons,
            )
            if memories:
                parts = ["\n## Recalled Memories"]
                for i, mem in enumerate(memories, 1):
                    content = getattr(mem, "content", str(mem))
                    parts.append(f"{i}. {content}")
                memory_text = "\n".join(parts)
                return f"{overlay}\n{memory_text}" if overlay else memory_text
        except Exception:
            logger.debug("Failed to recall memories for stage %s", stage_name)
        return overlay
