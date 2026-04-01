"""Paper quality assessment and venue recommendation."""

from researchclaw.assessor.rubrics import RUBRICS, Rubric
from researchclaw.assessor.scorer import PaperScorer
from researchclaw.assessor.venue_recommender import VenueRecommender
from researchclaw.assessor.comparator import HistoryComparator

__all__ = [
    "RUBRICS",
    "HistoryComparator",
    "PaperScorer",
    "Rubric",
    "VenueRecommender",
]
