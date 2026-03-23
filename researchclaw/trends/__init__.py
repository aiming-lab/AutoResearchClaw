"""Research trend tracking and automatic topic generation."""

from researchclaw.trends.daily_digest import DailyDigest
from researchclaw.trends.trend_analyzer import TrendAnalyzer
from researchclaw.trends.opportunity_finder import OpportunityFinder
from researchclaw.trends.auto_topic import AutoTopicGenerator
from researchclaw.trends.feeds import FeedManager

__all__ = [
    "AutoTopicGenerator",
    "DailyDigest",
    "FeedManager",
    "OpportunityFinder",
    "TrendAnalyzer",
]
