# Review receipt — R4 P0 (matched_value must exist in evidence artifact)

Status: **ALREADY IMPLEMENTED** in the current working tree. This receipt maps
each of your 6 requirements to the exact code that satisfies it, plus live
proof. Please review the CURRENT `scripts/release_check.py`, not an earlier diff.

Reviewed state:
- git HEAD: `611e1cf` (plus uncommitted working-tree changes to release_check.py)
- `scripts/release_check.py` sha256 prefix: `f35bb294705bb20d`

---

## Requirement → code mapping

**1. Each evidence pointer must have path-exists + sha256 match.**
`scripts/release_check.py` (check_claims_provenance loop):
```python
target = self.run_dir / rel
if not target.is_file():
    orphans += 1
    continue
recorded = str(ev.get("sha256") or "")
# sha256 is mandatory: an unpinned pointer is not verifiable.
if not recorded or sha256_of_file(target) != recorded:
    orphans += 1
    continue
valid_pointers += 1
```
Unchanged — path + mandatory sha256 still enforced.

**2. At least one matched_value must simultaneously (a) match claim.values ∪
claim-text numbers, AND (b) be deterministically extractable from the evidence
artifact's real content.**
```python
mv = ev.get("matched_value")
if isinstance(mv, (int, float)) and not isinstance(mv, bool):
    if rel not in artifact_numbers_cache:
        artifact_numbers_cache[rel] = numbers_in_artifact(target)
    art_numbers = artifact_numbers_cache[rel]
    if any(numbers_close(float(mv), an) for an in art_numbers):
        evidence_backed_values.append(float(mv))   # (b) present in file
    else:
        claimed_but_absent = True
...
claim_numbers  = [float(v) for v in (claim.get("values") or []) ...]
claim_numbers += numbers_from_text(str(claim.get("text", "")))   # values ∪ text
closed = any(any(numbers_close(mv, cn) for cn in claim_numbers)
             for mv in evidence_backed_values)                    # (a) AND (b)
if not closed:
    numeric_unclosed += 1
```
`evidence_backed_values` only contains matched_values that were found in the
file. Closure is computed against `evidence_backed_values`, never against the
raw claims.json matched_value. So a matched_value asserted only in claims.json
cannot close the loop.

**3. Deterministic extraction for JSON and text, no LLM.**
`researchclaw/pipeline/release_artifacts.py::numbers_in_artifact` (also mirrored
as a dependency-free fallback in `release_check.py`):
```python
def numbers_in_artifact(path):
    raw = path.read_text(...)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return extract_numbers(raw)          # text path: regex, no LLM
    nums = _collect_numeric_values(data)     # JSON path: structural walk
    nums.extend(extract_numbers(raw))
    return nums
```
`extract_numbers` is a pure regex (`_NUMBER_RE`). No model calls anywhere.

**4. New error when matched_value is only in claims.json, not in the file.**
```python
if claimed_but_absent and not evidence_backed_values:
    numeric_evidence_value_missing += 1
...
if numeric_evidence_value_missing:
    self.error("claims_numeric_evidence_value_missing", ...)
```

**5. Regression: claim 0.1234 + attempt_log without 0.1234 + correct sha +
matched_value=0.1234 → must FAIL.**
`tests/test_release_check_v2.py::test_matched_value_absent_from_evidence_file_fails`
— asserts BOTH `claims_numeric_evidence_value_missing` and
`claims_numeric_not_closed`. **PASSED.**

**6. Positive: evidence JSON really contains 0.1234 → PASS.**
`test_matched_value_present_in_evidence_json_passes` (JSON) and
`test_matched_value_in_text_artifact_passes` (text) — **both PASSED.**

Not relaxed: excerpt, background whitelist, digest invariance, claim status —
all unchanged from R3.

---

## Live proof (not just unit tests)

Direct invocation of `check_claims_provenance` on a hand-built run dir where
`attempts/attempt_log.jsonl` contains numbers but NOT 0.1234, while claims.json
asserts `matched_value: 0.1234` with the CORRECT sha256:

```
errors: ['claims_numeric_evidence_value_missing', 'claims_numeric_not_closed']
FAIL as required: True
```

Full suite: `tests/test_release_check_v2.py` → **45 passed**.
Adjacent suites (runner/stages/contracts/config/fabrication/hitl) → **489 passed**.

---

## Ask

If R5 still flags this as an open P0, please cite the specific line in the
CURRENT `scripts/release_check.py` (lines 540-600) where claims.json's
matched_value is still trusted without artifact re-extraction. If the finding
is a restatement of the checklist rather than a concrete line, this item should
be closed as resolved.
