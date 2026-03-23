"""Venue recommendation based on paper quality scores."""

from __future__ import annotations

from typing import Any


class VenueRecommender:
    """Recommend submission venues based on quality scores."""

    VENUE_TIERS: dict[str, dict[str, Any]] = {
        "tier_1": {
            "venues": ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL"],
            "min_score": 8.0,
            "domains": {
                "NeurIPS": ["ml", "ai", "deep-learning"],
                "ICML": ["ml"],
                "ICLR": ["ml", "deep-learning", "representation-learning"],
                "CVPR": ["cv", "deep-learning"],
                "ACL": ["nlp", "ai"],
            },
        },
        "tier_2": {
            "venues": ["AAAI", "IJCAI", "ECCV", "EMNLP", "KDD"],
            "min_score": 6.0,
            "domains": {
                "AAAI": ["ai", "ml"],
                "IJCAI": ["ai", "ml"],
                "ECCV": ["cv", "deep-learning"],
                "EMNLP": ["nlp"],
                "KDD": ["data-mining", "ml"],
            },
        },
        "tier_3": {
            "venues": ["ACML", "AISTATS", "WACV", "COLING"],
            "min_score": 4.0,
            "domains": {
                "ACML": ["ml"],
                "AISTATS": ["ml", "statistics"],
                "WACV": ["cv"],
                "COLING": ["nlp"],
            },
        },
        "workshop": {
            "venues": ["NeurIPS Workshop", "ICML Workshop", "ICLR Workshop"],
            "min_score": 3.0,
            "domains": {
                "NeurIPS Workshop": ["ml", "ai"],
                "ICML Workshop": ["ml"],
                "ICLR Workshop": ["ml", "deep-learning"],
            },
        },
    }

    def recommend(
        self,
        scores: dict[str, Any],
        domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend venues based on paper scores."""
        overall = scores.get("overall", 0.0)
        if not isinstance(overall, (int, float)):
            overall = 0.0

        recommendations: list[dict[str, Any]] = []

        for tier_name, tier_data in self.VENUE_TIERS.items():
            min_score = tier_data["min_score"]
            if overall < min_score:
                continue

            for venue in tier_data["venues"]:
                venue_domains = tier_data["domains"].get(venue, [])
                if domains and not any(d in venue_domains for d in domains):
                    continue

                recommendations.append({
                    "venue": venue,
                    "tier": tier_name,
                    "match_score": overall,
                    "suggestion": self._get_suggestion(venue, scores),
                    "venue_domains": venue_domains,
                })

        recommendations.sort(
            key=lambda r: (
                -r["match_score"],
                {"tier_1": 0, "tier_2": 1, "tier_3": 2, "workshop": 3}.get(
                    r["tier"], 4
                ),
            )
        )
        return recommendations

    @staticmethod
    def _get_suggestion(venue: str, scores: dict[str, Any]) -> str:
        """Generate a brief suggestion for improving chances at this venue."""
        dim_scores = scores.get("scores", {})
        if not dim_scores:
            return "Evaluate paper quality to get specific suggestions."

        weakest_dim = min(dim_scores, key=dim_scores.get)
        weakest_score = dim_scores[weakest_dim]

        if weakest_score < 5:
            return f"Strengthen '{weakest_dim}' (currently {weakest_score}/10) before submitting."
        if weakest_score < 7:
            return f"Consider improving '{weakest_dim}' ({weakest_score}/10) for better chances."
        return "Paper quality looks strong for this venue."

    def format_recommendations(
        self,
        recommendations: list[dict[str, Any]],
    ) -> str:
        """Format recommendations as a readable string."""
        if not recommendations:
            return "No suitable venues found for current paper quality."

        lines = ["Venue Recommendations:", ""]
        for rec in recommendations:
            lines.append(
                f"  {rec['venue']} ({rec['tier']}) — "
                f"score {rec['match_score']}/10"
            )
            lines.append(f"    {rec['suggestion']}")
            lines.append("")
        return "\n".join(lines)
