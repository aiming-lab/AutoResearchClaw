"""Research trend analysis engine."""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# Common stopwords to exclude from keyword analysis
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "that", "this", "these", "those", "it", "its", "we", "our", "their",
    "which", "what", "how", "when", "where", "who", "whom", "why",
    "not", "no", "nor", "as", "if", "then", "than", "both", "each",
    "all", "any", "few", "more", "most", "some", "such", "only", "very",
    "also", "about", "up", "out", "so", "into", "over", "after", "before",
    "between", "under", "through", "during", "using", "based", "via",
    "paper", "propose", "proposed", "method", "approach", "results", "show",
    "new", "novel", "model", "models", "data", "dataset", "task", "tasks",
    "performance", "learning", "training",
})


class TrendAnalyzer:
    """Analyze research trends from paper collections."""

    def __init__(self, min_keyword_length: int = 3):
        self.min_keyword_length = min_keyword_length

    def analyze(
        self,
        papers: list[dict[str, Any]],
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Analyze trends in a collection of papers."""
        if not papers:
            return {
                "rising_keywords": [],
                "hot_authors": [],
                "popular_datasets": [],
                "method_trends": [],
                "paper_count": 0,
            }

        keywords = self._extract_keywords(papers)
        authors = self._extract_authors(papers)
        datasets = self._extract_datasets(papers)
        methods = self._extract_methods(papers)

        return {
            "rising_keywords": keywords[:20],
            "hot_authors": authors[:10],
            "popular_datasets": datasets[:10],
            "method_trends": methods[:10],
            "paper_count": len(papers),
            "source_distribution": self._source_distribution(papers),
        }

    def _extract_keywords(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract and rank keywords from paper titles and abstracts."""
        word_counts: Counter[str] = Counter()
        bigram_counts: Counter[str] = Counter()

        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            words = self._tokenize(text)

            for w in words:
                if w not in _STOPWORDS and len(w) >= self.min_keyword_length:
                    word_counts[w] += 1

            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if (
                    w1 not in _STOPWORDS
                    and w2 not in _STOPWORDS
                    and len(w1) >= self.min_keyword_length
                ):
                    bigram_counts[f"{w1} {w2}"] += 1

        results = []
        for keyword, count in bigram_counts.most_common(30):
            if count >= 2:
                results.append({"keyword": keyword, "count": count, "type": "bigram"})
        for keyword, count in word_counts.most_common(30):
            if count >= 2:
                results.append({"keyword": keyword, "count": count, "type": "unigram"})

        results.sort(key=lambda x: -x["count"])
        return results[:20]

    def _extract_authors(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract most prolific authors."""
        author_counts: Counter[str] = Counter()
        for paper in papers:
            authors = paper.get("authors", [])
            if isinstance(authors, list):
                for author in authors:
                    name = author if isinstance(author, str) else author.get("name", "")
                    if name:
                        author_counts[name] += 1

        return [
            {"author": name, "paper_count": count}
            for name, count in author_counts.most_common(10)
            if count >= 2
        ]

    def _extract_datasets(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract commonly mentioned datasets."""
        dataset_patterns = [
            "ImageNet", "CIFAR", "MNIST", "COCO", "SQuAD", "GLUE",
            "SuperGLUE", "WikiText", "Penn Treebank", "WMT",
            "OpenWebText", "Common Crawl", "BookCorpus",
            "MMLU", "HumanEval", "GSM8K", "ARC", "HellaSwag",
        ]
        dataset_counts: Counter[str] = Counter()
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            for ds in dataset_patterns:
                if ds.lower() in text.lower():
                    dataset_counts[ds] += 1

        return [
            {"dataset": ds, "mention_count": count}
            for ds, count in dataset_counts.most_common(10)
            if count >= 1
        ]

    def _extract_methods(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract commonly mentioned methods/architectures."""
        method_patterns = [
            "transformer", "attention", "diffusion", "GAN", "VAE",
            "reinforcement learning", "contrastive learning",
            "self-supervised", "few-shot", "zero-shot", "in-context",
            "fine-tuning", "pre-training", "RLHF", "DPO",
            "chain-of-thought", "retrieval-augmented", "RAG",
            "mixture of experts", "MoE", "LoRA", "quantization",
            "knowledge distillation", "pruning", "graph neural",
        ]
        method_counts: Counter[str] = Counter()
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            for method in method_patterns:
                if method.lower() in text.lower():
                    method_counts[method] += 1

        return [
            {"method": method, "mention_count": count}
            for method, count in method_counts.most_common(10)
            if count >= 1
        ]

    @staticmethod
    def _source_distribution(
        papers: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Count papers by source."""
        dist: Counter[str] = Counter()
        for paper in papers:
            dist[paper.get("source", "unknown")] += 1
        return dict(dist)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return [w.lower() for w in re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)]

    def generate_trend_report(
        self,
        analysis: dict[str, Any],
    ) -> str:
        """Format trend analysis as a readable report."""
        lines = [
            f"Research Trend Analysis ({analysis.get('paper_count', 0)} papers)",
            "=" * 50,
            "",
        ]

        keywords = analysis.get("rising_keywords", [])
        if keywords:
            lines.append("Top Keywords:")
            for kw in keywords[:10]:
                lines.append(f"  - {kw['keyword']} ({kw['count']} mentions)")
            lines.append("")

        authors = analysis.get("hot_authors", [])
        if authors:
            lines.append("Most Active Authors:")
            for a in authors[:5]:
                lines.append(f"  - {a['author']} ({a['paper_count']} papers)")
            lines.append("")

        methods = analysis.get("method_trends", [])
        if methods:
            lines.append("Method Trends:")
            for m in methods[:5]:
                lines.append(f"  - {m['method']} ({m['mention_count']} mentions)")
            lines.append("")

        return "\n".join(lines)
