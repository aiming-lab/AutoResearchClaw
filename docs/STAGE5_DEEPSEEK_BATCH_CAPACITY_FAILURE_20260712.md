# Stage 5 DeepSeek Batch-Capacity Failure Review

## Status

Design approved; implementation in progress.

Run:

- Run ID: `rc-20260712-163837-45409b`
- Run directory: `runs/hwsec-e0-e9-validation-20260712`
- Config: `config.deepseek.sectional-dry-run.yaml`
- Scope: `pipeline_validation`
- First failed stage: Stage 5 `LITERATURE_SCREEN`

## Observed Failure

Stage 4 sealed 1,127 candidates successfully. Stage 5 deterministically admitted
the top 150 candidates and split them into six batches of 25. All six batches
failed the strict response contract, including the one bounded repair per batch.

The failures were not low screening scores:

- Five batches failed because both initial and repair responses contained
  malformed or truncated JSON.
- The sixth initial response was malformed; its repair response parsed but
  contradicted the configured keep/reject score thresholds.
- `screening_partial.jsonl` contains zero rows.
- `screening_report.json` records all 150 admitted candidates as unscreened.

Therefore the user-facing error, "no valid shortlist", is a downstream symptom.
The first causal failure is that the 25-decision response contract is not viable
with the real DeepSeek endpoint used by this run.

## Classification

Primary: contract-capacity mismatch in Stage 5.

Contributing factors:

1. Each decision has five required fields, including an unbounded free-text
   `reason`.
2. A 25-item response therefore contains more than 125 required values plus the
   enclosing JSON structure.
3. The same oversized contract is sent again during repair.
4. Strict parsing correctly rejects truncated JSON; no parser or gate should be
   weakened to make these responses pass.

This is not:

- a Stage 4 literature-collection failure;
- evidence that all 1,127 candidates are irrelevant;
- a reason to restore minimum-count backfill;
- a reason to accept partial JSON or infer omitted decisions.

## Proposed Minimal Fix

Replace the unbounded 25-decision contract with a versioned bounded response
contract:

1. Reduce `SCREEN_BATCH_SIZE` from 25 to 8.
2. Add `MAX_SCREEN_REASON_CHARS = 160`, measured as Python Unicode code points.
3. State the reason limit in the prompt contract.
4. Enforce the same limit in `parse_screening_response` so producer and replay
   share one rule.
5. Keep all existing gates unchanged:
   - exact `batch_id`;
   - complete candidate-ID closure;
   - no duplicate, extra, or missing IDs;
   - exact schema;
   - finite scores in `[0,1]`;
   - keep/reject must agree with thresholds;
   - at most one repair;
   - no backfill;
   - zero valid shortlist remains FAILED.
6. Update the report's deterministic `batch_size` and `batch_count` replay rules
   through the existing shared constants. Do not make batch size runtime- or
   model-selected.

With 150 admitted candidates, this produces 19 deterministic batches. The
additional calls are an explicit reliability/cost tradeoff; each response is
small enough to remain bounded even when every reason reaches its maximum.

## Rejected Alternatives

### Accept or repair truncated JSON locally

Rejected. Missing decisions cannot be reconstructed without inventing model
judgments. Any automatic completion would violate candidate-ID closure.

### Increase output tokens only

Rejected as the primary fix. The observed responses failed at varying character
positions and the current stage prompt does not establish a deterministic output
size. A larger provider budget does not bound free-text reasons or guarantee
valid JSON.

### Keep batches of 25 and remove strict validation

Rejected. Strict validation is the safety property that prevented malformed
screening from becoming citation evidence.

### Restore minimum-count backfill

Rejected. Backfill would promote unscreened candidates and recreate the evidence
fabrication path closed by E1.

### Adaptive recursive splitting

Deferred. It can improve latency when large batches happen to succeed, but it
adds a second batching state machine, more complex report replay, and additional
resume semantics. A fixed versioned batch size is simpler and fully
deterministic for v1.

## Required Tests

1. A 161-code-point reason is rejected; 160 is accepted.
2. The prompt includes the exact reason bound.
3. 150 candidates produce 19 batches with stable IDs and membership.
4. Candidate 8/9 and 144/145 boundaries remain deterministic.
5. Initial malformed JSON followed by valid bounded repair succeeds.
6. Initial and repair malformed responses remain a failed batch.
7. Strict scopes still fail on any failed batch.
8. `pipeline_validation` may retain successful batches only as degraded.
9. Zero selected candidates still produces only `screening_partial.jsonl`, not
   canonical `shortlist.jsonl`.
10. Existing report replay detects any change in batch membership, count, or
    output hash.

## Validation Sequence After Approval

1. Run the Stage 5 contract and executor tests.
2. Run the full suite and `git diff --check`.
3. Review the diff before commit.
4. Resume the same run from Stage 5 only, reusing the sealed Stage 4 artifacts:

   ```bash
   python -m researchclaw run \
     -c config.deepseek.sectional-dry-run.yaml \
     -o runs/hwsec-e0-e9-validation-20260712 \
     --from-stage LITERATURE_SCREEN \
     --to-stage DEAI_AUDIT
   ```

5. If the resumed run reaches Stage 25, run `release_check`. The expected result
   remains nonzero solely because the run is `pipeline_validation`; citation
   evidence replay errors are not expected.

## Review Questions

1. Is fixed batch size 8 preferable to a more complex adaptive split for v1?
2. Is 160 Unicode code points sufficient for an auditable screening reason?
3. Should this contract change increment `screening_policy_version`, and if so,
   should it be version 2 rather than retaining version 1?
4. Is resuming this run from Stage 5 acceptable for the first E0-E9 integration
   validation, followed by a separate fully fresh run before F0 begins?

## Review Decision

All four questions were approved with these bindings:

- fixed batch size 8, never selected at runtime;
- reason limited to 160 Unicode code points and rejected rather than truncated;
- `screening_policy_version` increments to 2 while the response schema remains 1;
- the resumed run is a functional mixed-version check only, followed by a fully
  fresh Stage 1-25 run before F0 begins.
