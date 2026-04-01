"""Conference deadline data management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Conference:
    """A single conference entry."""

    name: str
    full_name: str
    domains: tuple[str, ...]
    tier: int
    url: str = ""
    abstract_deadline: date | None = None
    paper_deadline: date | None = None
    notification: date | None = None
    camera_ready: date | None = None
    conference_date: date | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conference:
        """Parse a conference from a YAML dict."""
        def _parse_date(val: Any) -> date | None:
            if val is None:
                return None
            if isinstance(val, date):
                return val
            return datetime.strptime(str(val), "%Y-%m-%d").date()

        return cls(
            name=str(data["name"]),
            full_name=str(data.get("full_name", data["name"])),
            domains=tuple(data.get("domains") or ()),
            tier=int(data.get("tier", 3)),
            url=str(data.get("url", "")),
            abstract_deadline=_parse_date(data.get("abstract_deadline")),
            paper_deadline=_parse_date(data.get("paper_deadline")),
            notification=_parse_date(data.get("notification")),
            camera_ready=_parse_date(data.get("camera_ready")),
            conference_date=_parse_date(data.get("conference_date")),
        )

    @property
    def next_deadline(self) -> date | None:
        """Return the earliest upcoming deadline (abstract or paper)."""
        today = date.today()
        candidates = []
        if self.abstract_deadline and self.abstract_deadline >= today:
            candidates.append(self.abstract_deadline)
        if self.paper_deadline and self.paper_deadline >= today:
            candidates.append(self.paper_deadline)
        return min(candidates) if candidates else None

    @property
    def days_until_deadline(self) -> int | None:
        """Days until the next deadline, or None if all passed."""
        nd = self.next_deadline
        if nd is None:
            return None
        return (nd - date.today()).days


class ConferenceCalendar:
    """Manage conference deadline data."""

    def __init__(self, conferences: list[Conference] | None = None):
        self._conferences: list[Conference] = conferences or []

    @classmethod
    def load_builtin(cls) -> ConferenceCalendar:
        """Load the built-in conferences.yaml data."""
        yaml_path = _DATA_DIR / "conferences.yaml"
        if not yaml_path.exists():
            logger.warning("Built-in conferences.yaml not found at %s", yaml_path)
            return cls([])
        return cls.load(yaml_path)

    @classmethod
    def load(cls, path: Path | str) -> ConferenceCalendar:
        """Load conferences from a YAML file."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        entries = data.get("conferences", [])
        conferences = []
        for entry in entries:
            try:
                conferences.append(Conference.from_dict(entry))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping invalid conference entry: %s", exc)
        return cls(conferences)

    @property
    def conferences(self) -> list[Conference]:
        return list(self._conferences)

    def get_upcoming(
        self,
        domains: list[str] | None = None,
        days: int = 90,
        tier: int | None = None,
    ) -> list[Conference]:
        """Get conferences with deadlines in the next N days."""
        today = date.today()
        results = []
        for conf in self._conferences:
            nd = conf.next_deadline
            if nd is None:
                continue
            delta = (nd - today).days
            if delta < 0 or delta > days:
                continue
            if domains and not any(d in conf.domains for d in domains):
                continue
            if tier is not None and conf.tier > tier:
                continue
            results.append(conf)
        results.sort(key=lambda c: c.next_deadline or date.max)
        return results

    def get_by_name(self, name: str) -> Conference | None:
        """Find a conference by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for conf in self._conferences:
            if name_lower in conf.name.lower():
                return conf
        return None

    def get_by_domain(self, domain: str) -> list[Conference]:
        """Get all conferences for a domain."""
        return [c for c in self._conferences if domain in c.domains]

    def format_upcoming(
        self,
        domains: list[str] | None = None,
        days: int = 90,
    ) -> str:
        """Format upcoming deadlines as a readable string."""
        upcoming = self.get_upcoming(domains=domains, days=days)
        if not upcoming:
            return "No upcoming deadlines in the next {} days.".format(days)
        lines = [f"Upcoming Conference Deadlines (next {days} days):", ""]
        for conf in upcoming:
            nd = conf.next_deadline
            days_left = conf.days_until_deadline
            dl_type = "abstract" if nd == conf.abstract_deadline else "paper"
            lines.append(
                f"  {conf.name} (Tier {conf.tier})"
            )
            lines.append(
                f"    {dl_type} deadline: {nd} ({days_left} days left)"
            )
            if conf.url:
                lines.append(f"    URL: {conf.url}")
            lines.append("")
        return "\n".join(lines)
