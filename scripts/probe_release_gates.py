#!/usr/bin/env python3
"""Dependency-free probe for the release_check v2 numeric/citation gates.

No pytest required. Builds minimal in-memory run directories exercising the
adversarial cases and asserts release_check reacts correctly. Exit 0 = all
probes behaved as expected, 1 = a gate regressed.

Usage:  python3 scripts/probe_release_gates.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_check as rc  # noqa: E402


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _codes(run_dir: Path, claims: dict) -> set[str]:
    ck = rc.ReleaseChecker(run_dir, quality_threshold=5.0, allow_suspicious=False)
    ck.check_claims_provenance(claims)
    return {f.code for f in ck.findings if f.severity == "error"}


def _claim(evidence: list, values=(0.1234,), text="Our method reaches 0.1234 loss."):
    return {
        "schema_version": "2.0",
        "claims": [
            {
                "id": "clm-0000",
                "text": text,
                "type": "quantitative",
                "values": list(values),
                "evidence": evidence,
                "status": "supported",
            }
        ],
    }


PROBES: list[tuple[str, bool, str]] = []  # (name, expect_fail, error_code)


def run() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # --- BAD: matched_value asserted in claims.json but absent from file ---
        r1 = root / "bad"
        (r1 / "attempts").mkdir(parents=True)
        alog = r1 / "attempts" / "attempt_log.jsonl"
        alog.write_text(json.dumps({"stage": 12, "status": "ok", "elapsed_sec": 3.5}) + "\n")
        codes = _codes(
            r1,
            _claim([{"path": "attempts/attempt_log.jsonl", "sha256": _sha(alog), "matched_value": 0.1234}]),
        )
        ok = "claims_numeric_evidence_value_missing" in codes and "claims_numeric_not_closed" in codes
        print(f"[bad/absent-value]      expect FAIL -> {'OK' if ok else 'REGRESSED'}  {sorted(codes)}")
        failures += 0 if ok else 1

        # --- GOOD: evidence JSON really contains 0.1234 ---
        r2 = root / "good_json"
        (r2 / "stage-14").mkdir(parents=True)
        ev = r2 / "stage-14" / "m.json"
        ev.write_text(json.dumps({"results": {"loss": 0.1234, "acc": 0.9}}))
        codes = _codes(
            r2, _claim([{"path": "stage-14/m.json", "sha256": _sha(ev), "matched_value": 0.1234}])
        )
        ok = not codes
        print(f"[good/json-has-value]   expect PASS -> {'OK' if ok else 'REGRESSED'}  {sorted(codes)}")
        failures += 0 if ok else 1

        # --- GOOD: text artifact really contains 0.1234 ---
        r3 = root / "good_text"
        (r3 / "stage-14").mkdir(parents=True)
        ev = r3 / "stage-14" / "log.txt"
        ev.write_text("Epoch 3: validation loss = 0.1234 (best so far)\n")
        codes = _codes(
            r3, _claim([{"path": "stage-14/log.txt", "sha256": _sha(ev), "matched_value": 0.1234}])
        )
        ok = not codes
        print(f"[good/text-has-value]   expect PASS -> {'OK' if ok else 'REGRESSED'}  {sorted(codes)}")
        failures += 0 if ok else 1

        # --- BAD: correct value in file but WRONG sha256 (tamper) ---
        r4 = root / "bad_sha"
        (r4 / "stage-14").mkdir(parents=True)
        ev = r4 / "stage-14" / "m.json"
        ev.write_text(json.dumps({"loss": 0.1234}))
        codes = _codes(
            r4, _claim([{"path": "stage-14/m.json", "sha256": "deadbeef", "matched_value": 0.1234}])
        )
        ok = "claims_orphan_evidence" in codes
        print(f"[bad/wrong-sha]         expect FAIL -> {'OK' if ok else 'REGRESSED'}  {sorted(codes)}")
        failures += 0 if ok else 1

        # --- BAD: supported claim with empty evidence ---
        r5 = root / "bad_empty"
        r5.mkdir(parents=True)
        codes = _codes(r5, _claim([]))
        ok = "claims_supported_without_evidence" in codes
        print(f"[bad/empty-evidence]    expect FAIL -> {'OK' if ok else 'REGRESSED'}  {sorted(codes)}")
        failures += 0 if ok else 1

    print()
    if failures:
        print(f"PROBE RESULT: {failures} gate(s) REGRESSED")
        return 1
    print("PROBE RESULT: all gates behaved as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
