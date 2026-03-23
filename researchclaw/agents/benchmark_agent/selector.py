"""Selector Agent — filters and ranks benchmark candidates.

Applies hardware constraints, time budget, network policy, and tier
priorities to select the optimal combination of datasets and baselines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from researchclaw.agents.base import AgentStepResult, BaseAgent

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_knowledge.yaml"

# Maximum dataset size (MB) by tier and network policy
_SIZE_LIMITS: dict[str, int] = {
    "none": 0,          # No download allowed — tier 1 only
    "setup_only": 5000,  # Can download during setup phase
    "pip_only": 0,       # pip only, no data download
    "full": 50000,       # Generous limit
}


class SelectorAgent(BaseAgent):
    """Filters and ranks datasets/baselines based on constraints."""

    name = "selector"

    def __init__(
        self,
        llm: Any,
        *,
        gpu_memory_mb: int = 49000,
        time_budget_sec: int = 300,
        network_policy: str = "setup_only",
        tier_limit: int = 2,
        min_benchmarks: int = 1,
        min_baselines: int = 2,
        prefer_cached: bool = True,
    ) -> None:
        super().__init__(llm)
        self._gpu_mb = gpu_memory_mb
        self._time_sec = time_budget_sec
        self._network_policy = network_policy
        self._tier_limit = tier_limit
        self._min_bench = min_benchmarks
        self._min_base = min_baselines
        self._prefer_cached = prefer_cached

    # -- Filtering ---------------------------------------------------------

    def _filter_benchmarks(
        self, benchmarks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter benchmarks by tier, size, and network policy."""
        max_size = _SIZE_LIMITS.get(self._network_policy, 5000)
        filtered: list[dict[str, Any]] = []

        for b in benchmarks:
            tier = b.get("tier", 3)
            size = b.get("size_mb", 0)

            # Tier filter
            if tier > self._tier_limit:
                continue

            # Network policy filter
            if tier >= 2 and self._network_policy in ("none", "pip_only"):
                continue

            # Size filter (tier 2+ only — tier 1 is pre-cached)
            if tier >= 2 and size > max_size:
                continue

            filtered.append(b)

        return filtered

    def _filter_baselines(
        self, baselines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter baselines by pip availability."""
        filtered: list[dict[str, Any]] = []
        for bl in baselines:
            pip_deps = bl.get("pip", [])
            # If no network, only allow baselines with no extra pip deps
            if self._network_policy == "none" and pip_deps:
                continue
            filtered.append(bl)
        return filtered

    # -- Ranking -----------------------------------------------------------

    def _rank_benchmarks(
        self, benchmarks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort benchmarks by preference: tier 1 > tier 2, knowledge_base > hf, downloads."""
        def _score(b: dict[str, Any]) -> tuple[int, int, int]:
            tier = b.get("tier", 3)
            # Prefer lower tier (cached first)
            tier_score = -tier if self._prefer_cached else 0
            # Prefer knowledge_base over hf/llm
            origin_score = {
                "knowledge_base": 2,
                "huggingface_hub": 1,
                "llm_suggestion": 0,
            }.get(b.get("origin", ""), 0)
            # Downloads as tiebreaker
            downloads = b.get("downloads", 0)
            return (tier_score, origin_score, downloads)

        return sorted(benchmarks, key=_score, reverse=True)

    def _rank_baselines(
        self, baselines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort baselines: knowledge_base first, fewer deps preferred."""
        def _score(bl: dict[str, Any]) -> tuple[int, int]:
            origin_score = 1 if bl.get("origin") == "knowledge_base" else 0
            dep_score = -len(bl.get("pip", []))
            return (origin_score, dep_score)

        return sorted(baselines, key=_score, reverse=True)

    # -- Selection ---------------------------------------------------------

    def _select_with_llm(
        self,
        topic: str,
        benchmarks: list[dict[str, Any]],
        baselines: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask LLM to make final selection from filtered candidates."""
        bench_summary = "\n".join(
            f"- {b.get('name', 'Unknown')} (tier {b.get('tier', '?')}, "
            f"origin: {b.get('origin', '?')}, "
            f"metrics: {b.get('metrics', [])})"
            for b in benchmarks[:15]
        )
        base_summary = "\n".join(
            f"- {bl.get('name', 'Unknown')}: {bl.get('paper', 'N/A')}"
            for bl in baselines[:10]
        )

        system = (
            "You are an ML experiment design expert. Select the BEST combination "
            "of benchmarks and baselines for a research paper.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "primary_benchmark": "name",\n'
            '  "secondary_benchmarks": ["name1", "name2"],\n'
            '  "selected_baselines": ["name1", "name2", "name3"],\n'
            '  "rationale": "why these choices are optimal",\n'
            '  "experiment_notes": "specific setup guidance"\n'
            "}\n\n"
            "RULES:\n"
            "- Select 1 primary benchmark (the main evaluation dataset)\n"
            "- Select 0-2 secondary benchmarks (additional validation)\n"
            "- Select 2-4 baselines (must include at least 1 classic + 1 recent)\n"
            "- Primary benchmark MUST be the domain standard\n"
            "- Prefer benchmarks that top-venue papers commonly use\n"
            "- Consider dataset size vs time budget\n"
            "- CRITICAL: Only select benchmarks that are RELEVANT to the research "
            "topic's domain. Do NOT select image classification datasets (CIFAR, "
            "MNIST) for non-image tasks like PDE solvers, RL, or optimization.\n"
            "- CRITICAL: Baselines must be COMPETING METHODS, not optimizers. "
            "SGD/Adam/AdamW/Cosine LR are NOT baselines — they are training "
            "tools. Baselines must be alternative approaches to the same problem."
        )
        user = (
            f"Research Topic: {topic}\n\n"
            f"Available Benchmarks:\n{bench_summary}\n\n"
            f"Available Baselines:\n{base_summary}\n\n"
            f"Constraints: GPU={self._gpu_mb}MB, "
            f"time_budget={self._time_sec}s, "
            f"network_policy={self._network_policy}\n\n"
            "Make your selection."
        )
        return self._chat_json(system, user, max_tokens=2048)

    def _resolve_selection(
        self,
        selection: dict[str, Any],
        benchmarks: list[dict[str, Any]],
        baselines: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve LLM-selected names back to full benchmark/baseline dicts."""
        # Build name lookup
        bench_map = {b.get("name", f"bench_{i}"): b for i, b in enumerate(benchmarks)}
        base_map = {bl.get("name", f"base_{i}"): bl for i, bl in enumerate(baselines)}

        selected_bench: list[dict[str, Any]] = []
        primary = selection.get("primary_benchmark", "")
        if primary and primary in bench_map:
            entry = {**bench_map[primary], "role": "primary"}
            selected_bench.append(entry)

        for name in selection.get("secondary_benchmarks", []):
            if name in bench_map and name != primary:
                entry = {**bench_map[name], "role": "secondary"}
                selected_bench.append(entry)

        selected_base: list[dict[str, Any]] = []
        for name in selection.get("selected_baselines", []):
            if name in base_map:
                selected_base.append(base_map[name])

        return selected_bench, selected_base

    # -- Required baselines injection --------------------------------------

    def _inject_required_baselines(
        self,
        topic: str,
        selected: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Load required_baselines from knowledge base and inject missing ones.

        Returns the list of newly injected baseline dicts.
        """
        try:
            kb = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
            domains = kb.get("domains", {}) if isinstance(kb, dict) else {}
        except Exception:  # noqa: BLE001
            return []

        topic_lower = topic.lower()
        injected: list[dict[str, Any]] = []
        selected_names = {b.get("name", "").lower() for b in selected}

        for _domain_id, domain_data in domains.items():
            if not isinstance(domain_data, dict):
                continue
            keywords = domain_data.get("keywords", [])
            if not any(kw.lower() in topic_lower for kw in keywords):
                continue
            required = domain_data.get("required_baselines", [])
            if not required:
                continue
            # Find each required baseline in ranked list or create stub
            all_baselines = domain_data.get("common_baselines", [])
            bl_by_name = {b.get("name", ""): b for b in all_baselines}
            for req_name in required:
                if req_name.lower() in selected_names:
                    continue
                # Try to find full entry from knowledge base
                if req_name in bl_by_name:
                    entry = {**bl_by_name[req_name], "origin": "required_baseline"}
                else:
                    entry = {"name": req_name, "origin": "required_baseline", "pip": []}
                selected.append(entry)
                selected_names.add(req_name.lower())
                injected.append(entry)

        return injected

    # -- Main entry point --------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Select optimal benchmarks and baselines from survey results.

        Context keys:
            topic (str): Research topic
            survey (dict): Output from SurveyorAgent
        """
        topic = context.get("topic", "")
        survey = context.get("survey", {})

        benchmarks = survey.get("benchmarks", [])
        baselines = survey.get("baselines", [])

        if not benchmarks and not baselines:
            return self._make_result(False, error="No candidates to select from")

        # 1. Filter by constraints
        filtered_bench = self._filter_benchmarks(benchmarks)
        filtered_base = self._filter_baselines(baselines)

        self.logger.info(
            "Filtered: %d/%d benchmarks, %d/%d baselines",
            len(filtered_bench), len(benchmarks),
            len(filtered_base), len(baselines),
        )

        # 2. Rank
        ranked_bench = self._rank_benchmarks(filtered_bench)
        ranked_base = self._rank_baselines(filtered_base)

        # 3. LLM-assisted final selection (if enough candidates)
        if len(ranked_bench) >= 2 or len(ranked_base) >= 2:
            selection = self._select_with_llm(topic, ranked_bench, ranked_base)
            selected_bench, selected_base = self._resolve_selection(
                selection, ranked_bench, ranked_base,
            )
        else:
            # Not enough to warrant LLM call — use top ranked
            # BUG-DA6-06: Create copies to avoid mutating input dicts
            selected_bench = [{**b, "role": "primary"} if i == 0 else {**b, "role": "secondary"}
                              for i, b in enumerate(ranked_bench[:3])]
            selected_base = ranked_base[:self._min_base]
            selection = {}

        # 4. Fallback: ensure minimums
        if len(selected_bench) < self._min_bench and ranked_bench:
            for b in ranked_bench:
                if b not in selected_bench:
                    selected_bench.append({**b, "role": "secondary"})
                if len(selected_bench) >= self._min_bench:
                    break

        if len(selected_base) < self._min_base and ranked_base:
            for bl in ranked_base:
                if bl not in selected_base:
                    selected_base.append(bl)
                if len(selected_base) >= self._min_base:
                    break

        # 4b. Improvement E: Inject required baselines from knowledge base
        _injected_required = self._inject_required_baselines(
            topic, selected_base, ranked_base,
        )
        if _injected_required:
            self.logger.info(
                "Injected %d required baselines: %s",
                len(_injected_required),
                [b.get("name") for b in _injected_required],
            )

        # 5. Collect required pip packages
        required_pip: list[str] = []
        seen_pip: set[str] = set()
        for item in selected_bench + selected_base:
            for pkg in item.get("pip", []):
                if pkg not in seen_pip:
                    seen_pip.add(pkg)
                    required_pip.append(pkg)

        result = {
            "selected_benchmarks": selected_bench,
            "selected_baselines": selected_base,
            "required_pip": required_pip,
            "rationale": selection.get("rationale", ""),
            "experiment_notes": selection.get("experiment_notes", ""),
            "total_filtered": len(filtered_bench),
        }

        self.logger.info(
            "Selected: %d benchmarks, %d baselines, %d pip packages",
            len(selected_bench), len(selected_base), len(required_pip),
        )

        return self._make_result(True, data=result)
