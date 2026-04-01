"""Tests for MetaClaw session management module."""

from researchclaw.metaclaw_bridge.session import MetaClawSession


def test_session_creation():
    session = MetaClawSession("test-run-001")
    assert session.session_id == "arc-test-run-001"
    assert session.is_active is True


def test_session_headers():
    session = MetaClawSession("run-123")
    headers = session.get_headers("hypothesis_gen")
    assert headers["X-Session-Id"] == "arc-run-123"
    assert headers["X-Turn-Type"] == "main"
    assert headers["X-AutoRC-Stage"] == "hypothesis_gen"


def test_session_headers_no_stage():
    session = MetaClawSession("run-123")
    headers = session.get_headers()
    assert "X-AutoRC-Stage" not in headers


def test_session_end():
    session = MetaClawSession("run-456")
    end_headers = session.end()
    assert end_headers["X-Session-Done"] == "true"
    assert end_headers["X-Session-Id"] == "arc-run-456"
    assert session.is_active is False
