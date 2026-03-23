"""Submission timeline planner."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from researchclaw.calendar.deadlines import ConferenceCalendar


class SubmissionPlanner:
    """Generate submission timelines for target conferences."""

    # Stage proportions of total available time
    STAGE_PROPORTIONS = [
        ("Topic Selection", 0.0),
        ("Literature Review", 0.10),
        ("Experiment Design", 0.20),
        ("Experiments", 0.40),
        ("Paper Writing", 0.60),
        ("Revision", 0.80),
        ("Final Check", 0.95),
        ("Submission", 1.0),
    ]

    def __init__(self, calendar: ConferenceCalendar):
        self.calendar = calendar

    def plan(
        self,
        target_venue: str,
        start_date: date | None = None,
    ) -> dict[str, Any]:
        """Generate a submission timeline for a target venue."""
        conf = self.calendar.get_by_name(target_venue)
        if conf is None:
            return {"error": f"Conference '{target_venue}' not found"}

        deadline = conf.paper_deadline or conf.abstract_deadline
        if deadline is None:
            return {"error": f"No deadline found for '{conf.name}'"}

        start = start_date or date.today()
        total_days = (deadline - start).days

        if total_days <= 0:
            return {
                "error": f"Deadline {deadline} has passed",
                "venue": conf.name,
                "deadline": deadline.isoformat(),
            }

        milestones = []
        for stage_name, proportion in self.STAGE_PROPORTIONS:
            offset = int(total_days * proportion)
            milestone_date = start + timedelta(days=offset)
            days_left = (deadline - milestone_date).days
            milestones.append({
                "stage": stage_name,
                "date": milestone_date.isoformat(),
                "days_left": days_left,
            })

        return {
            "venue": conf.name,
            "deadline": deadline.isoformat(),
            "total_days": total_days,
            "start_date": start.isoformat(),
            "milestones": milestones,
            "conference_url": conf.url,
            "tier": conf.tier,
        }

    def format_plan(
        self,
        target_venue: str,
        start_date: date | None = None,
    ) -> str:
        """Format a submission plan as a readable string."""
        plan = self.plan(target_venue, start_date)
        if "error" in plan:
            return f"Error: {plan['error']}"

        lines = [
            f"Submission Plan for {plan['venue']}",
            f"Deadline: {plan['deadline']} ({plan['total_days']} days from start)",
            "",
            "Milestones:",
        ]
        for ms in plan["milestones"]:
            lines.append(
                f"  [{ms['date']}] {ms['stage']} ({ms['days_left']} days left)"
            )
        return "\n".join(lines)
