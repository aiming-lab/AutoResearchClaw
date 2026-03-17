"""Surveyor Agent — searches for domain-relevant benchmarks and baselines.

Data sources (in priority order):
1. Local ``benchmark_knowledge.yaml`` — always available, no network.
2. HuggingFace Hub API (``huggingface_hub``) — dataset discovery by task/keyword.
3. LLM fallback — asks the LLM to suggest benchmarks when APIs unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from researchclaw.agents.base import AgentStepResult, BaseAgent

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_knowledge.yaml"

# ---------------------------------------------------------------------------
# HuggingFace Hub helpers (optional dependency)
# ---------------------------------------------------------------------------

_HF_AVAILABLE = False
try:
    from huggingface_hub import HfApi  # type: ignore[import-untyped]
    _HF_AVAILABLE = True
except ImportError:
    pass

# Mapping from our domain keywords to HuggingFace task_categories filters
_DOMAIN_TO_HF_TASK: dict[str, list[str]] = {
    "image_classification": ["image-classification"],
    "text_classification": ["text-classification", "sentiment-analysis"],
    "language_modeling": ["text-generation"],
    "question_answering": ["question-answering"],
    "generative_models": ["unconditional-image-generation"],
    "graph_neural_networks": ["graph-ml"],
    "reinforcement_learning": ["reinforcement-learning"],
    "tabular_learning": ["tabular-classification", "tabular-regression"],
    "llm_finetuning": ["text-generation"],
}


class SurveyorAgent(BaseAgent):
    """Searches local knowledge base and HuggingFace Hub for benchmarks."""

    name = "surveyor"

    def __init__(
        self,
        llm: Any,
        *,
        enable_hf_search: bool = True,
        max_hf_results: int = 10,
    ) -> None:
        super().__init__(llm)
        self._enable_hf = enable_hf_search and _HF_AVAILABLE
        self._max_hf = max_hf_results
        self._knowledge = self._load_knowledge()

    # -- Knowledge base ----------------------------------------------------

    @staticmethod
    def _load_knowledge() -> dict[str, Any]:
        """Load the local benchmark knowledge base."""
        try:
            data = yaml.safe_load(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
            return data.get("domains", {}) if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load benchmark_knowledge.yaml", exc_info=True)
            return {}

    def _match_domains(self, topic: str) -> list[str]:
        """Return domain IDs whose keywords appear in the topic."""
        topic_lower = topic.lower()
        matched: list[str] = []
        for domain_id, info in self._knowledge.items():
            keywords = info.get("keywords", [])
            for kw in keywords:
                if kw in topic_lower:
                    matched.append(domain_id)
                    break
        return matched

    def _get_local_candidates(self, domain_ids: list[str]) -> dict[str, Any]:
        """Retrieve benchmarks and baselines from local knowledge base."""
        benchmarks: list[dict[str, Any]] = []
        baselines: list[dict[str, Any]] = []
        seen_bench: set[str] = set()
        seen_base: set[str] = set()

        for did in domain_ids:
            info = self._knowledge.get(did, {})
            for b in info.get("standard_benchmarks", []):
                name = b.get("name", "")
                if name not in seen_bench:
                    seen_bench.add(name)
                    benchmarks.append({**b, "source_domain": did, "origin": "knowledge_base"})
            for bl in info.get("common_baselines", []):
                name = bl.get("name", "")
                if name not in seen_base:
                    seen_base.add(name)
                    baselines.append({**bl, "source_domain": did, "origin": "knowledge_base"})

        return {"benchmarks": benchmarks, "baselines": baselines}

    # -- HuggingFace Hub ---------------------------------------------------

    def _search_hf_datasets(self, topic: str, domain_ids: list[str]) -> list[dict[str, Any]]:
        """Search HuggingFace Hub for relevant datasets."""
        if not self._enable_hf:
            return []

        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        try:
            api = HfApi()

            # Strategy 1: Search by task category
            for did in domain_ids:
                for task_cat in _DOMAIN_TO_HF_TASK.get(did, []):
                    try:
                        datasets = api.list_datasets(
                            filter=[f"task_categories:{task_cat}"],
                            sort="downloads",
                            direction=-1,
                            limit=self._max_hf,
                        )
                        for ds in datasets:
                            if ds.id not in seen:
                                seen.add(ds.id)
                                results.append({
                                    "name": ds.id,
                                    "downloads": getattr(ds, "downloads", 0),
                                    "origin": "huggingface_hub",
                                    "api": f"datasets.load_dataset('{ds.id}', cache_dir='/workspace/data/hf')",
                                    "tier": 2,
                                })
                    except Exception:  # noqa: BLE001
                        logger.debug("HF task search failed for %s", task_cat)

            # Strategy 2: Keyword search on topic
            keywords = self._extract_search_keywords(topic)
            for kw in keywords[:3]:
                try:
                    datasets = api.list_datasets(
                        search=kw,
                        sort="downloads",
                        direction=-1,
                        limit=self._max_hf,
                    )
                    for ds in datasets:
                        if ds.id not in seen:
                            seen.add(ds.id)
                            results.append({
                                "name": ds.id,
                                "downloads": getattr(ds, "downloads", 0),
                                "origin": "huggingface_hub",
                                "api": f"datasets.load_dataset('{ds.id}', cache_dir='/workspace/data/hf')",
                                "tier": 2,
                            })
                except Exception:  # noqa: BLE001
                    logger.debug("HF keyword search failed for %s", kw)

        except Exception as exc:  # noqa: BLE001
            logger.warning("HuggingFace Hub search failed: %s", exc)

        return results

    @staticmethod
    def _extract_search_keywords(topic: str) -> list[str]:
        """Extract 1-3 word search keywords from a topic string."""
        # Remove common filler words to get meaningful search terms
        stop = {
            "a", "an", "the", "for", "in", "on", "of", "to", "with", "and",
            "or", "is", "are", "using", "via", "based", "towards", "novel",
            "new", "improved", "approach", "method", "methods", "study",
        }
        words = [w.lower().strip(".,;:!?()[]") for w in topic.split()]
        filtered = [w for w in words if w and w not in stop and len(w) > 2]
        # Return 2-3 keyword phrases
        keywords: list[str] = []
        if len(filtered) >= 2:
            keywords.append(" ".join(filtered[:2]))
        if len(filtered) >= 3:
            keywords.append(" ".join(filtered[:3]))
        if filtered:
            keywords.append(filtered[0])
        return keywords

    # -- LLM fallback ------------------------------------------------------

    def _llm_suggest_benchmarks(self, topic: str, hypothesis: str) -> dict[str, Any]:
        """Ask LLM to suggest benchmarks and baselines when APIs unavailable."""
        system = (
            "You are an expert ML researcher. Given a research topic and hypothesis, "
            "suggest appropriate benchmarks, datasets, and baseline methods.\n\n"
            "Return a JSON object with:\n"
            "- benchmarks: array of {name, domain, metrics: [], api (Python one-liner), "
            "  tier (1=pre-cached, 2=downloadable), size_mb}\n"
            "- baselines: array of {name, source (Python code), paper (citation), pip: []}\n"
            "- rationale: string explaining why these are the right choices\n\n"
            "CRITICAL RULES:\n"
            "- Benchmarks and baselines MUST be DOMAIN-APPROPRIATE for the topic.\n"
            "- Do NOT suggest image classification datasets (CIFAR, ImageNet, MNIST) "
            "for non-image topics like PDE solvers, RL, combinatorial optimization, etc.\n"
            "- Do NOT suggest optimizers (SGD, Adam, AdamW) as METHOD baselines — "
            "optimizers are training tools, NOT research methods to compare against.\n"
            "- Baselines must be COMPETING METHODS from the same research domain.\n\n"
            "DOMAIN-SPECIFIC GUIDANCE:\n"
            "- Physics/PDE/Scientific computing: Use SYNTHETIC data (Burgers eq, "
            "Darcy flow, Navier-Stokes, heat equation). Baselines: FNO, DeepONet, "
            "PINN, spectral methods.\n"
            "- Combinatorial optimization (TSP, graph coloring, scheduling): Use "
            "SYNTHETIC instances (random TSP, Erdos-Renyi graphs). Baselines: "
            "classical MCTS, LKH, OR-Tools, Concorde, RL-based methods.\n"
            "- Reinforcement learning: Use Gymnasium environments (CartPole, "
            "LunarLander, HalfCheetah). Baselines: PPO, SAC, DQN, TD3.\n"
            "- Graph learning: Use standard graph benchmarks (Cora, CiteSeer, "
            "ogbn-arxiv). Baselines: GCN, GAT, GraphSAGE.\n"
            "- If the domain naturally requires SYNTHETIC data (PDE, optimization, "
            "theoretical analysis), explicitly set tier=1 and api='synthetic' and "
            "describe the data generation procedure in the 'source' field.\n\n"
            "- Prefer well-known, widely-used benchmarks from top venues\n"
            "- Prefer baselines with open-source PyTorch implementations\n"
            "- Include at least 2 datasets and 2 baselines"
        )
        user = (
            f"Research Topic: {topic}\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Suggest appropriate benchmarks, datasets, and baseline methods. "
            "Make sure they are relevant to the specific domain of this research."
        )
        result = self._chat_json(system, user, max_tokens=4096)
        return result

    # -- Main entry point --------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Survey available benchmarks and baselines for the given topic.

        Context keys:
            topic (str): Research topic/title
            hypothesis (str): Research hypothesis
            experiment_plan (str): Experiment plan from previous stages
        """
        topic = context.get("topic", "")
        hypothesis = context.get("hypothesis", "")

        if not topic:
            return self._make_result(False, error="No topic provided")

        self.logger.info("Surveying benchmarks for topic: %s", topic[:80])

        # 1. Match domains from knowledge base
        domain_ids = self._match_domains(topic)
        if hypothesis:
            domain_ids = list(dict.fromkeys(
                domain_ids + self._match_domains(hypothesis)
            ))
        self.logger.info("Matched domains: %s", domain_ids)

        # 2. Get local candidates
        local = self._get_local_candidates(domain_ids)

        # 3. Search HuggingFace Hub (if available)
        hf_datasets = self._search_hf_datasets(topic, domain_ids)

        # 4. LLM fallback if no local matches
        llm_suggestions: dict[str, Any] = {}
        if not local["benchmarks"] and not hf_datasets:
            self.logger.info("No local/HF matches — falling back to LLM")
            llm_suggestions = self._llm_suggest_benchmarks(topic, hypothesis)

        # 5. Combine results
        all_benchmarks = local["benchmarks"] + hf_datasets
        if llm_suggestions.get("benchmarks"):
            for b in llm_suggestions["benchmarks"]:
                b["origin"] = "llm_suggestion"
                all_benchmarks.append(b)

        all_baselines = local["baselines"]
        if llm_suggestions.get("baselines"):
            for bl in llm_suggestions["baselines"]:
                bl["origin"] = "llm_suggestion"
                all_baselines.append(bl)

        survey_result = {
            "matched_domains": domain_ids,
            "benchmarks": all_benchmarks,
            "baselines": all_baselines,
            "hf_datasets_found": len(hf_datasets),
            "llm_fallback_used": bool(llm_suggestions),
            "rationale": llm_suggestions.get("rationale", ""),
        }

        self.logger.info(
            "Survey complete: %d benchmarks, %d baselines, %d HF datasets",
            len(all_benchmarks), len(all_baselines), len(hf_datasets),
        )

        return self._make_result(True, data=survey_result)
