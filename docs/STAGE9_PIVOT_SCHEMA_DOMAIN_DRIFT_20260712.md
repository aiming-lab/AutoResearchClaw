# Stage 9 Pivot Schema and Domain-Drift Review

## Status

Design approved; implementation in progress.

Run:

- Run ID: `rc-20260712-173050-45409b`
- Run directory: `runs/hwsec-e0-e9-validation-20260712`
- Resume start: Stage 5 `LITERATURE_SCREEN`
- Pause: Stage 9 `EXPERIMENT_DESIGN`, after the second research-decision pivot

## What Succeeded

Stage 5 screening policy v2 passed its first real DeepSeek execution:

- policy version: 2
- batch size: 8
- batch count: 19
- admitted candidates screened: 150
- selected candidates: 34
- failed batches: 0
- degraded: false

The pipeline then completed Stages 6-15, performed one REFINE rollback, completed
Stages 13-15 again, and performed its second and final PIVOT rollback to Stage 8.

## Observed Pause

The post-pivot Stage 9 response parsed as a mapping but exposed only one root key:

```json
{"received_keys": ["experiment_plan"]}
```

Stage 9 therefore reported `baselines`, `proposed_methods`, and `ablations` as
missing and correctly returned `PAUSED` instead of passing a schema-deficient plan
to Stage 10.

The same Stage 9 attempt also persisted this domain profile:

```json
{"domain_id": "physics_simulation"}
```

Earlier in the same run, before the PIVOT, Stage 9 had correctly persisted
`security_detection` and produced a complete experiment contract.

## Root-Cause Classification

### A. Known-wrapper normalization gap

Stage 9 accepts any YAML mapping as `plan`. Its schema-deficit guard then checks
required fields only at the root. A response shaped as
`{"experiment_plan": { ...required fields... }}` is therefore treated as missing
content even if the nested mapping is complete.

The fail-closed pause is correct. The missing deterministic normalization before
the guard is the usability defect.

### B. Domain detection gives pivot context equal priority to the canonical topic

`detect_domain()` currently concatenates `topic`, `hypotheses`, and `literature`,
then returns the first keyword-rule match. Physics rules precede security rules.
A post-pivot hypothesis containing generic simulation terminology can therefore
override an unambiguously security-focused canonical topic.

This is deterministic but semantically unstable: the domain identity changes as
the pipeline rewrites hypotheses even though the configured research topic does
not change.

## Proposed Minimal Fixes

### Fix 1: unwrap one exact experiment-plan envelope

Before the schema-deficit guard:

1. If the parsed mapping has exactly one root key, `experiment_plan`, and its
   value is a mapping, replace `plan` with a shallow copy of that nested mapping.
2. Do not unwrap arbitrary one-key mappings.
3. Do not merge wrapper and root fields.
4. Run the existing required-content guard unchanged after unwrapping.
5. If the nested value is not a mapping or remains schema-deficient, keep the
   existing `PAUSED` result.

This is deterministic shape normalization, not LLM repair and not content
invention.

### Fix 2: make canonical topic the first domain-detection tier

Change keyword detection order to:

1. forced profile, if configured;
2. keyword detection on `topic` alone;
3. only if topic detection is inconclusive, keyword detection on supplementary
   `hypotheses + literature` context;
4. LLM classification, if configured;
5. generic fallback.

Do not add topic-specific special cases and do not reorder individual domain
profiles to favor this run. The invariant is that mutable downstream context
cannot override an already conclusive canonical-topic classification.

`detect_domain_id()` must use the same tiering so lightweight and full detection
cannot disagree. `detect_domain_async()` is a third consumer and must use the
same shared helper; otherwise synchronous and asynchronous execution can drift.

### Fix 3: make an incomplete direct Stage 9 attempt authoritative

`find_stage09_contract()` must not fall back to a pre-pivot `_vN` contract when
the direct `stage-09/` directory contains markers of a newer PAUSED, FAILED, or
in-progress attempt but lacks a contract. The selector uses this rule:

1. return the direct contract when it exists;
2. if direct attempt markers (`decision.json`, `plan_meta.json`, or
   `stage_health.json`) exist without a direct contract, return `None`;
3. only when no authoritative direct attempt marker exists may legacy/versioned
   selection choose the highest `_vN` contract.

Stage 9 also clears its owned plan, contract, hash, plan metadata, and domain
profile at entry. A successful resume writes a new direct contract; a paused or
failed resume cannot expose the old pre-pivot contract through the selector.

## Rejected Alternatives

### Randomly rerun Stage 9 until the wrapper disappears

Rejected. It hides a deterministic parser gap and leaves domain drift intact.

### Reuse the earlier `stage-09_v2` contract after a PIVOT

Rejected. A PIVOT intentionally changes the hypothesis and experiment design;
reusing the pre-pivot contract would break rollback semantics.

### Flatten arbitrary nested mappings

Rejected. It can silently reinterpret malformed responses and create ambiguous
field precedence.

### Force `security_detection` only in the dry-run config

Rejected as the sole fix. A profile lock is valid for an explicitly governed
domain, but it would mask the general instability for automatic research topics.
The topic-first detector rule fixes the underlying ordering problem without a
run-specific exception.

### Replace PAUSED with a fallback experiment plan

Rejected. The existing guard correctly prevents schema-deficient plans from
reaching code generation. The fix should recover only content already present in
an exact known envelope.

## Required Tests

1. Exact `experiment_plan` mapping wrapper with all required fields reaches DONE.
2. Wrapped mapping still missing all required fields remains PAUSED.
3. `experiment_plan` with a non-mapping value remains PAUSED.
4. Mapping with `experiment_plan` plus any sibling key is not unwrapped and
   remains subject to the root schema guard.
5. Unambiguous security topic plus physics-flavored pivot hypotheses resolves to
   `security_detection`.
6. Topic with no keyword signal may still resolve from hypotheses context.
7. Forced profile still has highest priority.
8. `detect_domain()` and `detect_domain_id()` return the same keyword-derived ID.
9. Existing domain-detector suite remains green across ML, physics, chemistry,
   biology, economics, mathematics, security, and robotics fixtures.
10. Stage 9 resume clears or replaces the paused attempt's owned artifacts and
    does not reuse the earlier pre-pivot contract.
11. Async, sync, and ID-only keyword detection use the same topic-first result.
12. A direct paused attempt without a contract blocks `_vN` fallback, while a
    legacy run with no direct attempt markers may still select the highest `_vN`.

## Validation and Resume Sequence

After approval and implementation:

1. Run Stage 9 guard, domain-detector, executor, and full-suite tests.
2. Commit and push only after read-only diff review.
3. Resume the same run from Stage 9:

   ```bash
   python -m researchclaw run \
     -c config.deepseek.sectional-dry-run.yaml \
     -o runs/hwsec-e0-e9-validation-20260712 \
     --from-stage EXPERIMENT_DESIGN \
     --to-stage DEAI_AUDIT
   ```

4. Treat this as mixed-version functional validation only.
5. Before F0, run a fully fresh Stage 1-25 validation with one code version and
   no resume.

## Review Questions

1. Is exact single-key `experiment_plan` unwrapping sufficiently narrow and
   fail-closed?
2. Should topic-first keyword detection apply to both `detect_domain()` and
   `detect_domain_id()`?
3. Should an explicitly configured profile be added to this dry-run config as
   additional defense, or should the detector fix stand alone for validation?
4. Is resuming from Stage 9 valid for the mixed-version functional check after
   the fix, with a fresh full run still mandatory before F0?
