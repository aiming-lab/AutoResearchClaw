"""Browser tests for the OrcaRouter settings page.

These drive the real page in Chromium against the real FastAPI app. The
load-bearing case is the back-forward-cache shape: after ``pagehide`` the
busy state and the authorization hint must clear synchronously, and a second
login must be startable *without remounting the page* — a generation guard
alone leaves a bfcache-restored page permanently busy.

The last test produces the screenshot/manifest bundle by running
``scripts/orcarouter_ui_evidence.py`` end to end (own server, own browser)
against the live catalogue and then holds it to the delivery checklist. The
bundle is a build product of that run — it is written to ``orca-evidence/`` in
the working tree, stays git-ignored, and is never carried in a patch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "sk-orca-ui-test-placeholder-000000"
# Captured before the server fixture clears the variable so the evidence run
# can still reach the live catalogue. Never rendered or logged.
LIVE_KEY = os.environ.get("ORCAROUTER_API_KEY", "")


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    state = tmp_path_factory.mktemp("orca-ui-state")
    saved = {
        name: os.environ.get(name)
        for name in ("ORCA_CREDENTIALS_PATH", "ORCA_CATALOG_CACHE_DIR", "ORCAROUTER_API_KEY")
    }
    os.environ["ORCA_CREDENTIALS_PATH"] = str(state / "creds.json")
    os.environ["ORCA_CATALOG_CACHE_DIR"] = str(state / "catalog")
    os.environ.pop("ORCAROUTER_API_KEY", None)

    from researchclaw.config import RCConfig
    from researchclaw.server.app import create_app

    config = RCConfig.load(
        str(REPO_ROOT / "config.researchclaw.example.yaml"), check_paths=False
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(config), host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(120):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        # These are process-global: leaking them makes other modules' default
        # paths (and the credential they resolve) depend on collection order.
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        instance = pw.chromium.launch(
            executable_path="/usr/bin/chromium",
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        yield instance
        instance.close()


@pytest.fixture
def page(browser, live_server):
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(f"{live_server}/providers", wait_until="domcontentloaded")
    page.wait_for_function("() => !!window.rcOrcaProviders", timeout=15000)
    yield page
    page.close()


def test_both_authentication_entries_are_usable_on_one_page(page) -> None:
    assert page.is_visible('[data-testid="api-key-input"]')
    assert page.is_visible('[data-testid="pkce-connect"]')
    assert page.is_enabled('[data-testid="api-key-save"]')
    assert page.is_enabled('[data-testid="pkce-connect"]')
    text = page.inner_text("#providers-app")
    assert "OrcaRouter — API" in text
    assert "OrcaRouter — Auth" in text


def test_the_page_never_receives_the_stored_key(page) -> None:
    page.fill('[data-testid="api-key-input"]', FAKE_KEY)
    page.click('[data-testid="api-key-save"]')
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"secret-masked\"]')"
        ".dataset.masked === 'true'",
        timeout=15000,
    )
    assert FAKE_KEY not in page.content()
    assert "sk-orca" in page.inner_text('[data-testid="secret-masked"]')
    # ...and clearing it is a first-class action.
    page.click('[data-testid="api-key-clear"]')
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"secret-masked\"]')"
        ".dataset.masked === 'false'",
        timeout=15000,
    )


def test_pagehide_clears_busy_state_and_allows_a_second_login(page) -> None:
    """The bfcache shape: no remount, but the page must not stay busy."""
    page.select_option('[data-testid="pkce-flow"]', "oob")
    page.click('[data-testid="pkce-connect"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.attemptId !== null", timeout=15000
    )
    first_attempt = page.evaluate("() => window.rcOrcaProviders.state.attemptId")
    assert page.evaluate("() => window.rcOrcaProviders.state.busy") is True
    assert page.is_visible('[data-testid="pkce-code-row"]')
    assert page.inner_text('[data-testid="pkce-hint"]')
    assert page.is_visible('[data-testid="pkce-authorize-url"]')

    # A real pagehide — the browser may restore this page from bfcache.
    page.evaluate("() => window.dispatchEvent(new Event('pagehide'))")

    assert page.evaluate("() => window.rcOrcaProviders.state.busy") is False
    assert page.inner_text('[data-testid="pkce-hint"]') == ""
    assert page.is_enabled('[data-testid="pkce-connect"]')
    assert page.is_disabled('[data-testid="pkce-cancel"]')
    assert not page.is_visible('[data-testid="pkce-code-row"]')
    assert not page.is_visible('[data-testid="pkce-authorize-url"]')

    # A second login must start without remounting the component.
    page.click('[data-testid="pkce-connect"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.busy === true"
        "  && window.rcOrcaProviders.state.attemptId !== null",
        timeout=15000,
    )
    second_attempt = page.evaluate("() => window.rcOrcaProviders.state.attemptId")
    assert second_attempt, "a second login must be startable after pagehide"
    assert second_attempt != first_attempt
    # No verifier material is reachable from the page.
    assert "code_verifier" not in page.content()
    page.click('[data-testid="pkce-cancel"]')


def test_explicit_cancel_clears_the_lock(page) -> None:
    page.select_option('[data-testid="pkce-flow"]', "oob")
    page.click('[data-testid="pkce-connect"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.busy === true", timeout=15000
    )
    page.click('[data-testid="pkce-cancel"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.busy === false", timeout=15000
    )
    assert page.is_enabled('[data-testid="pkce-connect"]')
    assert "Cancelled." in page.inner_text('[data-testid="pkce-hint"]')

    page.click('[data-testid="pkce-connect"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.busy === true", timeout=15000
    )
    page.click('[data-testid="pkce-cancel"]')


def test_switching_flow_mid_login_releases_and_restarts(page) -> None:
    page.select_option('[data-testid="pkce-flow"]', "loopback")
    page.click('[data-testid="pkce-connect"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.busy === true", timeout=15000
    )
    page.click('[data-testid="pkce-cancel"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.busy === false", timeout=15000
    )
    page.select_option('[data-testid="pkce-flow"]', "oob")
    page.click('[data-testid="pkce-connect"]')
    page.wait_for_function(
        "() => window.rcOrcaProviders.state.attemptId !== null"
        "  && window.rcOrcaProviders.state.busy === true",
        timeout=15000,
    )
    assert page.is_visible('[data-testid="pkce-code-row"]')
    page.click('[data-testid="pkce-cancel"]')


def _png_size(path: Path) -> tuple[int, int]:
    """Width/height straight out of the PNG IHDR chunk."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _bundle_digest(root: Path) -> dict[str, str]:
    """Every file in a bundle directory, keyed by relative path."""
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_gui_evidence_bundle_is_generated_into_the_repository(tmp_path) -> None:
    """Produce the reviewed bundle, then hold it to the delivery checklist.

    Runs the generator the way a maintainer would — real server, real Chromium,
    live catalogue when a credential is present — with its default output, so
    the screenshots are written to ``orca-evidence/`` in this working tree by
    this run. They are a build product and stay git-ignored: the delivery gate
    only accepts evidence produced while the tree under test is being checked,
    so nothing here may be tracked.
    """
    bundle_dir = REPO_ROOT / "orca-evidence"
    shutil.rmtree(bundle_dir, ignore_errors=True)

    env = {k: v for k, v in os.environ.items() if k != "ORCAROUTER_API_KEY"}
    if LIVE_KEY:
        env["ORCAROUTER_API_KEY"] = LIVE_KEY
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "orcarouter_ui_evidence.py"),
            "--state-dir",
            str(tmp_path / "evidence-state"),
        ],
        capture_output=True,
        text=True,
        timeout=480,
        env=env,
    )
    assert proc.returncode == 0, f"evidence run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"

    # The bundle is generated, never carried: a tracked one is stale by
    # construction and the delivery gate refuses a patch that contains it.
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", "orca-evidence"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert tracked.returncode != 0, (
        "the evidence bundle is tracked; it must be generated by this run instead:"
        f" {tracked.stdout.strip()}"
    )
    assert set(_bundle_digest(bundle_dir)) == {
        "manifest.json",
        "auth-methods.png",
        "text-model-dropdown.png",
    }

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    automation = manifest["automation"]
    ui = manifest["ui_assertions"]

    assert automation["framework"] == "playwright"
    assert automation["passed"] is True
    assert automation["catalog_source"] == "https://api.orcarouter.ai/v1/models?capability=chat"
    assert automation["catalog_model_count"] > 0
    assert automation["image_model_count"] >= 0
    assert manifest["catalog_models"], "the dropdown must be backed by a real catalogue"

    # --- both authentication choices are on screen, key stays masked ---
    assert ui["api_key_visible"] is True
    assert ui["pkce_visible"] is True
    assert ui["secret_masked"] is True
    assert ui["controls_enabled"] is True

    # --- the dropdown really opened, anchored to its trigger ---
    assert ui["dropdown_open"] is True
    assert int(ui["item_count"]) == automation["catalog_model_count"]
    assert ui["opaque_background"] is True
    assert ui["visible_border"] is True
    assert float(ui["trigger_panel_right_delta"]) <= 2

    # --- artifacts exist, are real PNGs, and match their recorded digests ---
    by_kind = {a["kind"]: a for a in manifest["artifacts"]}
    assert set(by_kind) == {"auth-methods", "text-model-dropdown"}
    for artifact in manifest["artifacts"]:
        assert artifact["ui"], f"{artifact['kind']} carries no UI assertions"
        path = bundle_dir / artifact["path"]
        assert path.is_file(), f"{artifact['path']} was not written"
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        width, height = _png_size(path)
        assert width >= 800 and height >= 450, f"{artifact['path']} is {width}x{height}"

    # The generator re-checks the gate's checklist before it returns, and this
    # is the same function the generator calls — a bundle that fails it never
    # reaches the delivery run.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import orcarouter_ui_evidence

        orcarouter_ui_evidence.validate_bundle(manifest, bundle_dir)
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))
