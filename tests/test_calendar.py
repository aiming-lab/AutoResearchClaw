"""Tests for researchclaw.calendar — Conference Deadline Calendar (Agent D4).

15+ tests covering deadlines, planner, and reminder modules.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from researchclaw.calendar.deadlines import Conference, ConferenceCalendar
from researchclaw.calendar.planner import SubmissionPlanner
from researchclaw.calendar.reminder import Reminder, ReminderCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_conference(
    name: str = "TestConf",
    full_name: str = "Test Conference",
    domains: tuple[str, ...] = ("ml",),
    tier: int = 1,
    abstract_deadline: date | None = None,
    paper_deadline: date | None = None,
    **kwargs,
) -> Conference:
    return Conference(
        name=name,
        full_name=full_name,
        domains=domains,
        tier=tier,
        abstract_deadline=abstract_deadline,
        paper_deadline=paper_deadline,
        **kwargs,
    )


def _future(days: int) -> date:
    return date.today() + timedelta(days=days)


def _past(days: int) -> date:
    return date.today() - timedelta(days=days)


# ===================================================================
# Conference dataclass tests
# ===================================================================


class TestConference:
    def test_from_dict_minimal(self):
        data = {"name": "NeurIPS"}
        conf = Conference.from_dict(data)
        assert conf.name == "NeurIPS"
        assert conf.tier == 3  # default
        assert conf.domains == ()

    def test_from_dict_full(self):
        data = {
            "name": "ICML",
            "full_name": "International Conference on Machine Learning",
            "domains": ["ml", "ai"],
            "tier": 1,
            "url": "https://icml.cc",
            "abstract_deadline": "2026-06-01",
            "paper_deadline": "2026-06-08",
        }
        conf = Conference.from_dict(data)
        assert conf.name == "ICML"
        assert conf.full_name == "International Conference on Machine Learning"
        assert conf.domains == ("ml", "ai")
        assert conf.tier == 1
        assert conf.abstract_deadline == date(2026, 6, 1)
        assert conf.paper_deadline == date(2026, 6, 8)

    def test_from_dict_date_passthrough(self):
        """date objects in YAML are already date instances."""
        data = {
            "name": "X",
            "abstract_deadline": date(2026, 12, 1),
        }
        conf = Conference.from_dict(data)
        assert conf.abstract_deadline == date(2026, 12, 1)

    def test_next_deadline_returns_earliest_future(self):
        conf = _make_conference(
            abstract_deadline=_future(10),
            paper_deadline=_future(20),
        )
        assert conf.next_deadline == _future(10)

    def test_next_deadline_skips_past(self):
        conf = _make_conference(
            abstract_deadline=_past(5),
            paper_deadline=_future(15),
        )
        assert conf.next_deadline == _future(15)

    def test_next_deadline_none_when_all_past(self):
        conf = _make_conference(
            abstract_deadline=_past(10),
            paper_deadline=_past(5),
        )
        assert conf.next_deadline is None

    def test_days_until_deadline(self):
        conf = _make_conference(paper_deadline=_future(30))
        assert conf.days_until_deadline == 30

    def test_days_until_deadline_none(self):
        conf = _make_conference()
        assert conf.days_until_deadline is None


# ===================================================================
# ConferenceCalendar tests
# ===================================================================


class TestConferenceCalendar:
    def test_load_from_yaml(self, tmp_path: Path):
        data = {
            "conferences": [
                {
                    "name": "TestConf",
                    "domains": ["ml"],
                    "tier": 1,
                    "paper_deadline": (_future(30)).isoformat(),
                },
                {
                    "name": "TestConf2",
                    "domains": ["cv"],
                    "tier": 2,
                    "paper_deadline": (_future(60)).isoformat(),
                },
            ]
        }
        yaml_path = tmp_path / "conferences.yaml"
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")

        cal = ConferenceCalendar.load(yaml_path)
        assert len(cal.conferences) == 2
        assert cal.conferences[0].name == "TestConf"

    def test_load_skips_invalid_entries(self, tmp_path: Path):
        data = {
            "conferences": [
                {"name": "Valid", "tier": 1},
                {"invalid": "no name field"},
            ]
        }
        yaml_path = tmp_path / "conf.yaml"
        yaml_path.write_text(yaml.dump(data), encoding="utf-8")
        cal = ConferenceCalendar.load(yaml_path)
        assert len(cal.conferences) == 1

    def test_get_upcoming_filters_by_days(self):
        confs = [
            _make_conference(name="Soon", paper_deadline=_future(10)),
            _make_conference(name="Far", paper_deadline=_future(200)),
        ]
        cal = ConferenceCalendar(confs)
        upcoming = cal.get_upcoming(days=90)
        assert len(upcoming) == 1
        assert upcoming[0].name == "Soon"

    def test_get_upcoming_filters_by_domain(self):
        confs = [
            _make_conference(name="ML", domains=("ml",), paper_deadline=_future(10)),
            _make_conference(name="CV", domains=("cv",), paper_deadline=_future(10)),
        ]
        cal = ConferenceCalendar(confs)
        result = cal.get_upcoming(domains=["ml"], days=90)
        assert len(result) == 1
        assert result[0].name == "ML"

    def test_get_upcoming_filters_by_tier(self):
        confs = [
            _make_conference(name="T1", tier=1, paper_deadline=_future(10)),
            _make_conference(name="T3", tier=3, paper_deadline=_future(10)),
        ]
        cal = ConferenceCalendar(confs)
        result = cal.get_upcoming(tier=1, days=90)
        assert len(result) == 1
        assert result[0].name == "T1"

    def test_get_by_name_case_insensitive(self):
        confs = [_make_conference(name="NeurIPS")]
        cal = ConferenceCalendar(confs)
        assert cal.get_by_name("neurips") is not None
        assert cal.get_by_name("NEURIPS") is not None
        assert cal.get_by_name("nonexistent") is None

    def test_get_by_domain(self):
        confs = [
            _make_conference(name="A", domains=("ml", "ai")),
            _make_conference(name="B", domains=("cv",)),
        ]
        cal = ConferenceCalendar(confs)
        assert len(cal.get_by_domain("ml")) == 1
        assert len(cal.get_by_domain("cv")) == 1
        assert len(cal.get_by_domain("nlp")) == 0

    def test_format_upcoming_no_deadlines(self):
        cal = ConferenceCalendar([])
        output = cal.format_upcoming()
        assert "No upcoming deadlines" in output

    def test_format_upcoming_with_deadlines(self):
        confs = [_make_conference(
            name="ICML", paper_deadline=_future(15), url="https://icml.cc"
        )]
        cal = ConferenceCalendar(confs)
        output = cal.format_upcoming(days=90)
        assert "ICML" in output
        assert "15 days left" in output
        assert "https://icml.cc" in output

    def test_load_builtin(self):
        """Built-in conferences.yaml should load without error."""
        cal = ConferenceCalendar.load_builtin()
        assert isinstance(cal.conferences, list)


# ===================================================================
# SubmissionPlanner tests
# ===================================================================


class TestSubmissionPlanner:
    def test_plan_basic(self):
        conf = _make_conference(name="TestConf", paper_deadline=_future(100))
        cal = ConferenceCalendar([conf])
        planner = SubmissionPlanner(cal)
        plan = planner.plan("TestConf", start_date=date.today())
        assert plan["venue"] == "TestConf"
        assert plan["total_days"] == 100
        assert len(plan["milestones"]) == 8  # 8 stages in STAGE_PROPORTIONS

    def test_plan_unknown_venue(self):
        cal = ConferenceCalendar([])
        planner = SubmissionPlanner(cal)
        result = planner.plan("NonExistent")
        assert "error" in result

    def test_plan_past_deadline(self):
        conf = _make_conference(name="Past", paper_deadline=_past(5))
        cal = ConferenceCalendar([conf])
        planner = SubmissionPlanner(cal)
        result = planner.plan("Past", start_date=date.today())
        assert "error" in result
        assert "passed" in result["error"]

    def test_format_plan(self):
        conf = _make_conference(name="ICML", paper_deadline=_future(60))
        cal = ConferenceCalendar([conf])
        planner = SubmissionPlanner(cal)
        output = planner.format_plan("ICML", start_date=date.today())
        assert "Submission Plan for ICML" in output
        assert "Milestones:" in output

    def test_format_plan_error(self):
        cal = ConferenceCalendar([])
        planner = SubmissionPlanner(cal)
        output = planner.format_plan("None")
        assert "Error:" in output


# ===================================================================
# ReminderCalculator tests
# ===================================================================


class TestReminderCalculator:
    def test_check_fires_on_matching_day(self):
        deadline = date.today() + timedelta(days=7)
        conf = _make_conference(name="Conf", paper_deadline=deadline)
        calc = ReminderCalculator(reminder_days=(7,))
        reminders = calc.check([conf])
        assert len(reminders) == 1
        assert reminders[0].days_until == 7

    def test_check_no_fire_on_non_matching_day(self):
        deadline = date.today() + timedelta(days=8)
        conf = _make_conference(name="Conf", paper_deadline=deadline)
        calc = ReminderCalculator(reminder_days=(7,))
        reminders = calc.check([conf])
        assert len(reminders) == 0

    def test_check_skips_past_deadlines(self):
        conf = _make_conference(name="Conf", paper_deadline=_past(3))
        calc = ReminderCalculator(reminder_days=(3,))
        assert len(calc.check([conf])) == 0

    def test_urgency_critical(self):
        assert ReminderCalculator._classify_urgency(1) == "critical"
        assert ReminderCalculator._classify_urgency(3) == "critical"

    def test_urgency_warning(self):
        assert ReminderCalculator._classify_urgency(7) == "warning"
        assert ReminderCalculator._classify_urgency(14) == "warning"

    def test_urgency_info(self):
        assert ReminderCalculator._classify_urgency(30) == "info"

    def test_get_active_reminders(self):
        confs = [
            _make_conference(name="Soon", paper_deadline=_future(5)),
            _make_conference(name="Far", paper_deadline=_future(100)),
        ]
        calc = ReminderCalculator(reminder_days=(30, 14, 7, 3, 1))
        active = calc.get_active_reminders(confs)
        assert len(active) == 1
        assert active[0].conference_name == "Soon"

    def test_format_reminders_empty(self):
        calc = ReminderCalculator()
        assert "No upcoming" in calc.format_reminders([])

    def test_format_reminders_with_data(self):
        r = Reminder(
            conference_name="ICML",
            deadline_type="paper",
            deadline_date=_future(3),
            days_until=3,
            urgency="critical",
        )
        calc = ReminderCalculator()
        output = calc.format_reminders([r])
        assert "ICML" in output
        assert "!!!" in output

    def test_reminder_frozen(self):
        r = Reminder("X", "paper", date.today(), 5, "info")
        with pytest.raises(AttributeError):
            r.days_until = 10  # type: ignore[misc]
