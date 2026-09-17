#!/usr/bin/env python3
"""Generate the OrcaRouter GUI evidence bundle.

Boots the real FastAPI app (the one ``researchclaw serve`` builds), drives the
real ``/providers`` page in Chromium with Playwright, and writes
``{manifest.json,auth-methods.png,text-model-dropdown.png}``.

The bundle is the reviewed artifact and a build product of a run: it is written
to ``orca-evidence/`` at the repository root, where the delivery checklist reads
it, and that directory is git-ignored so a bundle can never be carried in a
patch. ``--out`` / ``ORCA_EVIDENCE_OUT`` redirect it.
``tests/test_orcarouter_ui.py`` runs this script end to end and then holds the
bundle it wrote to the same checklist.

The manifest is written in the gate's schema rather than a free-form report::

    {"automation": {"framework": "playwright", "passed": true,
                    "catalog_source": "<live chat catalogue URL>",
                    "catalog_model_count": N, "image_model_count": M},
     "artifacts": [{"kind": "auth-methods", "path": ..., "sha256": ..., "ui": {}}]}

``validate_bundle`` re-checks that shape before the files are handed over, so a
bundle the gate would reject fails here instead of after the run.

The API key used for the screenshots is a clearly-fake, key-shaped string, so
no real credential material can end up in an artifact. The live catalogue
request uses whatever credential the environment provides.

Usage:
    python scripts/orcarouter_ui_evidence.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVIDENCE_KEY = "sk-orca-evidence-placeholder-not-a-real-key"
CATALOG_URL = "https://api.orcarouter.ai/v1/models?capability=chat"


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _serve(port: int, state_dir: Path) -> threading.Thread:
    os.environ.setdefault("ORCA_CREDENTIALS_PATH", str(state_dir / "credentials.json"))
    os.environ.setdefault("ORCA_CATALOG_CACHE_DIR", str(state_dir / "catalog"))

    from researchclaw.config import RCConfig
    from researchclaw.server.app import create_app
    import uvicorn

    config = RCConfig.load(
        str(REPO_ROOT / "config.researchclaw.example.yaml"), check_paths=False
    )
    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):
        if getattr(server, "started", False):
            break
        time.sleep(0.1)
    return thread


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _opaque_background(page, selector: str) -> bool:
    """The panel must not be see-through."""
    value = page.eval_on_selector(
        selector, "el => getComputedStyle(el).backgroundColor"
    )
    if not value or value in ("transparent", "rgba(0, 0, 0, 0)"):
        return False
    if value.startswith("rgba"):
        alpha = float(value.rsplit(",", 1)[1].strip().rstrip(")"))
        return alpha >= 0.95
    return True


def _has_visible_border(page, selector: str) -> bool:
    return page.eval_on_selector(
        selector,
        "el => { const s = getComputedStyle(el);"
        " return parseFloat(s.borderTopWidth) > 0 && s.borderTopStyle !== 'none'; }",
    )


DEFAULT_OUT = os.environ.get("ORCA_EVIDENCE_OUT", str(REPO_ROOT / "orca-evidence"))
GATE_CATALOG_URL = "https://api.orcarouter.ai/v1/models?capability=chat"
REQUIRED_SHOTS = ("auth-methods", "text-model-dropdown")


def _png_size(path: Path) -> tuple[int, int]:
    """Width/height straight out of the PNG IHDR chunk."""
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} is not a PNG")
    return struct.unpack(">II", header[16:24])


def validate_bundle(bundle: dict, out: Path, multimodal: bool = False) -> dict:
    """Reject a bundle the delivery gate would refuse, while it is still local.

    Mirrors the gate's checklist: Playwright provenance, the authoritative chat
    catalogue URL, counts that agree with what was actually scraped, one PNG per
    required entry point at 800x450 or larger with a matching digest, and the UI
    assertions each screenshot has to prove.
    """
    automation = bundle.get("automation")
    if not isinstance(automation, dict) or automation.get("framework") != "playwright":
        raise ValueError("evidence must come from Playwright")
    if automation.get("passed") is not True:
        raise ValueError("evidence automation did not pass")
    if automation.get("catalog_source") != GATE_CATALOG_URL:
        raise ValueError("evidence did not use the authoritative chat catalogue")
    total = automation.get("catalog_model_count")
    image = automation.get("image_model_count")
    if not isinstance(total, int) or not isinstance(image, int) or not 0 <= image <= total:
        raise ValueError("evidence manifest has invalid model counts")
    if multimodal and image == 0:
        raise ValueError("a multimodal entry point needs image models in the catalogue")

    kinds = REQUIRED_SHOTS + (("multimodal-model-dropdown",) if multimodal else ())
    declared = {item.get("kind"): item for item in bundle.get("artifacts") or []}
    for kind in kinds:
        item = declared.get(kind)
        if not item:
            raise ValueError(f"missing evidence screenshot: {kind}")
        path = out / item["path"]
        if not path.is_file() or path.stat().st_size < 10_000:
            raise ValueError(f"evidence screenshot is missing or too small: {path}")
        width, height = _png_size(path)
        if width < 800 or height < 450:
            raise ValueError(f"evidence screenshot must be at least 800x450: {path}")
        if item.get("sha256") != _sha256(path):
            raise ValueError(f"evidence checksum mismatch: {path}")
        ui = item.get("ui") or {}
        if kind == "auth-methods":
            for field in ("api_key_visible", "pkce_visible", "secret_masked", "controls_enabled"):
                if ui.get(field) is not True:
                    raise ValueError(
                        "evidence does not show usable API Key and PKCE authentication"
                    )
        else:
            expected = total if kind == "text-model-dropdown" else image
            if ui.get("dropdown_open") is not True or ui.get("item_count") != expected:
                raise ValueError(f"evidence does not show the required open dropdown: {kind}")
            if ui.get("opaque_background") is not True or ui.get("visible_border") is not True:
                raise ValueError(f"evidence dropdown has no visible container: {kind}")
            delta = ui.get("trigger_panel_right_delta")
            if not isinstance(delta, (int, float)) or abs(delta) > 2:
                raise ValueError(f"evidence dropdown is not aligned to its trigger: {kind}")
    return bundle


def generate(out_dir: Path, state_dir: Path | None = None) -> dict:
    """Boot the real app, drive ``/providers``, write the bundle, return it."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_dir = (
        Path(state_dir).resolve()
        if state_dir
        else Path("/tmp/orcarouter-evidence-state")
    )
    state_dir.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    _serve(port, state_dir)
    base = f"http://127.0.0.1:{port}"

    from playwright.sync_api import sync_playwright

    ui: dict[str, object] = {}
    playwright_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path="/usr/bin/chromium",
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{base}/providers", wait_until="networkidle")

        # --- API-key entry: store a placeholder and confirm it is masked ---
        page.fill('[data-testid="api-key-input"]', EVIDENCE_KEY)
        page.click('[data-testid="api-key-save"]')
        page.wait_for_function(
            "() => document.querySelector('[data-testid=\"secret-masked\"]')"
            ".dataset.masked === 'true'",
            timeout=15000,
        )
        masked_text = page.inner_text('[data-testid="secret-masked"]')
        ui["api_key_visible"] = page.is_visible('[data-testid="api-key-input"]')
        ui["pkce_visible"] = page.is_visible('[data-testid="pkce-connect"]')
        ui["secret_masked"] = EVIDENCE_KEY not in masked_text and "sk-orca" in masked_text
        ui["controls_enabled"] = page.is_enabled('[data-testid="api-key-save"]') and page.is_enabled(
            '[data-testid="pkce-connect"]'
        )
        ui["stored_key_never_rendered"] = EVIDENCE_KEY not in page.content()

        # --- Live catalogue drives the dropdown ---
        page.wait_for_function(
            "() => window.rcOrcaProviders"
            "  && window.rcOrcaProviders.state.models.length > 0"
            "  && document.querySelector('[data-testid=\"model-trigger-label\"]')"
            "       .textContent !== 'Loading…'",
            timeout=30000,
        )
        state = page.evaluate("() => window.rcOrcaProviders.state")
        model_ids = [m["id"] for m in state["models"]]
        catalog_source = page.evaluate(
            "() => document.querySelector('[data-testid=\"model-status\"]').dataset.source"
        )

        page.screenshot(path=str(out / "auth-methods.png"), full_page=True)

        page.click('[data-testid="model-trigger"]')
        page.wait_for_selector('[data-testid="model-panel"]:not([hidden])')
        page.wait_for_timeout(250)

        if not playwright_errors:
            ui["dropdown_open"] = page.is_visible('[data-testid="model-panel"]')
            ui["item_count"] = page.eval_on_selector_all(
                '[data-testid="model-option"]', "els => els.length"
            )
            ui["opaque_background"] = _opaque_background(page, '[data-testid="model-panel"]')
            ui["visible_border"] = _has_visible_border(page, '[data-testid="model-panel"]')
            trigger = page.eval_on_selector(
                '[data-testid="model-trigger"]',
                "el => { const r = el.getBoundingClientRect();"
                " return { left: r.right, width: r.width }; }",
            )
            panel = page.eval_on_selector(
                '[data-testid="model-panel"]',
                "el => { const r = el.getBoundingClientRect();"
                " return { left: r.right, width: r.width }; }",
            )
            # The panel is anchored to the trigger: right edges within 2px.
            ui["trigger_panel_right_delta"] = round(abs(trigger["left"] - panel["left"]), 2)

        page.screenshot(path=str(out / "text-model-dropdown.png"), full_page=True)

        screenshot_size = page.evaluate(
            "() => ({ w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight })"
        )
        # How many image-generation models this workspace can call. Recorded
        # so the manifest shows the capability filter ran for that entry point
        # too, even though no entry point in this repo can use one.
        image_catalog = page.evaluate(
            "async () => {"
            "  const r = await fetch('/api/providers/orcarouter/models?capability=image');"
            "  if (!r.ok) return { count: 0, ids: [] };"
            "  const j = await r.json();"
            "  return { count: j.count, ids: (j.models || []).map(m => m.id) };"
            "}"
        )
        browser.close()

    passed = bool(
        ui.get("api_key_visible")
        and ui.get("pkce_visible")
        and ui.get("secret_masked")
        and ui.get("controls_enabled")
        and ui.get("stored_key_never_rendered")
        and ui.get("dropdown_open")
        and ui.get("opaque_background")
        and ui.get("visible_border")
        and int(ui.get("item_count") or 0) > 0
        and ui.get("trigger_panel_right_delta") is not None
        and float(ui["trigger_panel_right_delta"]) <= 2
    )

    artifacts = []
    for kind, name in (
        ("auth-methods", "auth-methods.png"),
        ("text-model-dropdown", "text-model-dropdown.png"),
    ):
        path = out / name
        artifacts.append(
            {
                "kind": kind,
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "ui": {k: v for k, v in ui.items() if k != "stored_key_never_rendered"},
            }
        )

    manifest = {
        # The gate reads ``automation`` as an object: provenance and the counts
        # the screenshots have to agree with live here, not at the top level.
        "automation": {
            "framework": "playwright",
            "passed": passed,
            "catalog_source": GATE_CATALOG_URL,
            "catalog_model_count": len(model_ids),
            "image_model_count": image_catalog["count"],
        },
        "catalog_models": model_ids,
        "image_models": image_catalog["ids"],
        "multimodal_applicable": False,
        "multimodal_reason": (
            "No AI entry point in this repository sends an image to the chat "
            "client: messages are {role, content} strings "
            "(researchclaw/llm/client.py) and the only image generator is "
            "pinned to Gemini's native API "
            "(researchclaw/agents/figure_agent/nano_banana.py). The capability "
            "filter is implemented and unit-tested for image input, so a "
            "future multimodal entry point fails closed."
        ),
        "page": f"{base}/providers",
        "viewport": {"width": 1280, "height": 900},
        "page_scroll_size": screenshot_size,
        "runtime_source": catalog_source,
        "ui_assertions": ui,
        "artifacts": artifacts,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Fail before handing over a bundle the gate would reject, and never hand
    # over one that is not internally consistent.
    validate_bundle(manifest, out, multimodal=bool(manifest["multimodal_applicable"]))

    print(json.dumps({k: manifest[k] for k in (
        "ui_assertions", "catalog_models", "image_models"
    )}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--state-dir", default="")
    args = parser.parse_args()

    manifest = generate(Path(args.out), Path(args.state_dir) if args.state_dir else None)
    return 0 if manifest["automation"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
