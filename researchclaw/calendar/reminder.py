"""Deadline reminder calculation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from researchclaw.calendar.deadlines import Conference


@dataclass(frozen=True)
class Reminder:
    """A deadline reminder."""

    conference_name: str
    deadline_type: str  # "abstract" or "paper"
    deadline_date: date
    days_until: int
    urgency: str  # "critical" | "warning" | "info"


class ReminderCalculator:
    """Calculate deadline reminders based on configuration."""

    def __init__(
        self,
        reminder_days: tuple[int, ...] = (30, 14, 7, 3, 1),
    ):
        self.reminder_days = sorted(reminder_days, reverse=True)

    def check(
        self,
        conferences: list[Conference],
        check_date: date | None = None,
    ) -> list[Reminder]:
        """Check which conferences need reminders today."""
        today = check_date or date.today()
        reminders: list[Reminder] = []

        for conf in conferences:
            for dl_type, dl_date in [
                ("abstract", conf.abstract_deadline),
                ("paper", conf.paper_deadline),
            ]:
                if dl_date is None:
                    continue
                days_until = (dl_date - today).days
                if days_until < 0:
                    continue
                if days_until in self.reminder_days:
                    urgency = self._classify_urgency(days_until)
                    reminders.append(Reminder(
                        conference_name=conf.name,
                        deadline_type=dl_type,
                        deadline_date=dl_date,
                        days_until=days_until,
                        urgency=urgency,
                    ))

        reminders.sort(key=lambda r: r.days_until)
        return reminders

    def get_active_reminders(
        self,
        conferences: list[Conference],
        check_date: date | None = None,
    ) -> list[Reminder]:
        """Get all reminders for deadlines within the reminder window."""
        today = check_date or date.today()
        max_days = max(self.reminder_days) if self.reminder_days else 30
        reminders: list[Reminder] = []

        for conf in conferences:
            for dl_type, dl_date in [
                ("abstract", conf.abstract_deadline),
                ("paper", conf.paper_deadline),
            ]:
                if dl_date is None:
                    continue
                days_until = (dl_date - today).days
                if 0 <= days_until <= max_days:
                    urgency = self._classify_urgency(days_until)
                    reminders.append(Reminder(
                        conference_name=conf.name,
                        deadline_type=dl_type,
                        deadline_date=dl_date,
                        days_until=days_until,
                        urgency=urgency,
                    ))

        reminders.sort(key=lambda r: r.days_until)
        return reminders

    @staticmethod
    def _classify_urgency(days_until: int) -> str:
        if days_until <= 3:
            return "critical"
        if days_until <= 14:
            return "warning"
        return "info"

    def format_reminders(self, reminders: list[Reminder]) -> str:
        """Format reminders as a readable string."""
        if not reminders:
            return "No upcoming deadline reminders."
        lines = ["Deadline Reminders:", ""]
        for r in reminders:
            icon = {"critical": "!!!", "warning": "!!", "info": "i"}[r.urgency]
            lines.append(
                f"  [{icon}] {r.conference_name} — {r.deadline_type} deadline "
                f"in {r.days_until} days ({r.deadline_date})"
            )
        return "\n".join(lines)
