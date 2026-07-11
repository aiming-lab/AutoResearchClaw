# Stage 19 Phase C: Sectional Revision Release Gates

Status: review draft. This document defines release-check contracts and
adversarial acceptance criteria. It does not authorize implementation, enable
sectional revision in checked-in configuration, or declare any existing run
release-ready.

## 1. Objective

Phase C connects the deterministic Stage 19 sectional audit chain to
`scripts/release_check.py`. A Stage 19 status of DONE, a manifest field saying
`completed: true`, or the existence of `paper_revised.md` is not release
evidence by itself. Release readiness requires an independent, read-only
reconstruction from canonical run artifacts.

The gate must prove all of the following:

1. the run used the sectional Stage 19 path rather than the legacy whole-paper
   revision path;
2. Stage 19 used the same claim scope as the canonical Stage 9 experiment
   contract;
3. every review comment is resolved, with no unresolved or not-actionable
   entries hidden by counts or stale files;
4. every changed section comes from a hash-bound accepted attempt that passed
   deterministic validation and isolated critic assessment;
5. the merged Stage 19 paper is exactly reconstructable from the immutable
   Stage 17 source plus accepted section bodies;
6. manifest, ledger, plan, context, attempt, assessment, validation, and output
   hashes are recomputed from disk rather than trusted as self-assertions.

Phase C does not weaken or replace the existing quality, fabrication, citation,
compile, canonical-source, truth-audit, de-AI, provenance, or reviewer-isolation
gates. It adds another independent release requirement.

## 2. Activation And Migration Rule

After Phase C is implemented, every release candidate must contain a complete
canonical sectional bundle under `stage-19/`. Absence of
`stage-19/section_revision_manifest.json` is an error. A legacy whole-paper
Stage 19 run may remain useful for development, but it cannot pass release
readiness.

This is intentionally fail-closed. Because checked-in configuration currently
keeps `paper_revision.sectional_enabled: false`, Phase C implementation will
make existing or legacy runs fail release readiness until a new sectional run
is executed. That expected failure must not be "fixed" by making the bundle
optional.

Canonical paths are exact. `stage-19_v*` is archival history and is never a
fallback source for release. The current `stage-19/` directory must contain the
final attempt. A release check must not search for a more convenient historical
manifest when the canonical directory is missing or invalid.

## 3. Trust Boundaries

### 3.1 Authoritative inputs

The release audit reads these canonical inputs:

- `stage-09/experiment_contract.yaml`, or the same canonical Stage 9 version
  selected by `find_stage09_contract()` when the direct path is absent;
- `run_manifest.json`, including its dedicated sectional writer/critic model
  identities;
- `stage-17/paper_draft.md`;
- `stage-18/reviews.md`;
- `stage-19/review_comment_ledger.json`;
- `stage-19/revision_plan.json`;
- `stage-19/validation_context.json`;
- `stage-19/section_attempts.jsonl`;
- `stage-19/resolution_assessments.jsonl`;
- `stage-19/unresolved_comments.json`;
- `stage-19/section_revision_manifest.json`;
- `stage-19/paper_revised.md`;
- attempt-specific files under `stage-19/sections/` and
  `stage-19/section_validation/`.

Every required input is loaded fail-closed. Missing files, malformed JSON/YAML,
malformed JSONL, duplicate object keys where detectable, unknown schema fields,
unsafe paths, unsupported schema versions, invalid UTF-8, and empty required
files are errors.

### 3.2 Non-authoritative inputs

The following cannot establish release readiness:

- Stage status or pipeline summary alone;
- `deliverables/` copies;
- `stage-19_v*` archives;
- manifest booleans or counts without recomputation;
- LLM prose claiming that a comment was addressed;
- writer self-attestation without accepted deterministic validation and an
  isolated critic assessment;
- Stage 10 smoke artifacts;
- waiver files for unresolved review comments or manifest/hash mismatches.

There is no waiver for Phase C sectional integrity failures. The existing
synthetic dataset waiver path must be removed before C1 because it permits a
contract state that the runtime `ExperimentContract` validator rejects.

## 4. Required Schema Additions Before Gate Integration

Current B3 artifacts are not yet sufficient for a complete disk-only replay.
The following additions are Phase C prerequisites, not optional enhancements.

### 4.1 Bind the Stage 9 experiment contract

`section_revision_manifest.json` must add:

```json
{
  "experiment_contract_path": "stage-09/experiment_contract.yaml",
  "experiment_contract_sha256": "...",
  "attempts_sha256": "...",
  "writer_model": "...",
  "critic_model": "..."
}
```

The path must equal the canonical contract path selected for the run. The hash
is the exact file-byte SHA-256. Stage 19 records the hash before LLM calls and
rechecks it immediately before finalization, alongside its existing immutable
source checks.

Release check independently verifies:

- the path is safe, run-relative, and points to the canonical Stage 9 contract;
- the current file hash equals `experiment_contract_sha256`;
- the required Stage 10 sealed-candidate manifest carries the same
  `contract_sha256`;
- the contract `claim_scope`, Stage 19 manifest `claim_scope`, and release
  eligibility scope are all exactly `research_release`.

Changing both a free-standing manifest string and the contract string is not a
valid proof. The cross-hash is mandatory.

`attempts_sha256` is the exact UTF-8 text SHA-256 of
`section_attempts.jsonl`. Attempt IDs inside section entries are not a substitute
for binding the complete append-only log: deleting or reordering rejected and
transport-failed attempts must change the manifest.

`writer_model` and `critic_model` are nonempty identities that are each
consistent across the complete sectional bundle and are mutually different.
"Consistent" here does not define a global model-name registry; it means that a
bundle cannot change model identity between attempts or assessments. Every
attempt writer must equal the sectional manifest writer and every assessment
critic must equal the sectional manifest critic. A third per-attempt or
per-assessment model identity is invalid even when it differs from the paired
writer.

The root `run_manifest.json` reviewer block must add two role-specific fields:

```json
{
  "reviewer": {
    "writer_model": "...",
    "critic_model": "...",
    "sectional_writer_model": "...",
    "sectional_critic_model": "..."
  }
}
```

The existing `reviewer.critic_model` remains the Stage 15/run-level critique
identity sourced from `config.llm.critic_model`; it is not required to equal the
Stage 19 critic. `sectional_writer_model` is sourced from
`config.llm.primary_model`, and `sectional_critic_model` is sourced from
`config.paper_revision.critic_model`. For a sectional release,
`sectional_writer_model` must also equal the existing `reviewer.writer_model`.
The Stage 19 manifest pair must equal these two new sectional fields, not the
Stage 15 critic field.

Although runner currently writes `run_manifest.json` best-effort, Phase C treats
the file, reviewer object, and both sectional fields as mandatory. Missing or
malformed run-manifest model binding is a release error.

### 4.2 Persist proposal resolution IDs per attempt

Successful proposal attempt records already contain `candidate_body_sha256` and
must retain its current meaning. Phase C adds `candidate_path` and
`resolution_comment_ids`:

```json
{
  "candidate_path": "stage-19/sections/<section_id>.attempt-<n>.md",
  "resolution_comment_ids": ["rc-..."],
  "candidate_body_sha256": "..."
}
```

Transport-failed attempts use `candidate_path: null`,
`resolution_comment_ids: []`, `candidate_body_sha256: null`, and have no
candidate or validation file. The attempt schema must reject unknown fields and
invalid combinations.

This addition is required because deterministic validation depends on the exact
`resolution_comment_ids` supplied by the writer. Phase C must not infer them
from critic output or assume every assigned comment was claimed addressed.

### 4.3 Strict attempt and assessment loaders

Phase C must introduce versioned strict loaders for both JSONL artifacts.
Current ad hoc dictionaries are not sufficient at a release boundary.

Each attempt loader validates at least:

- exact schema fields and `schema_version == 1`;
- unique `attempt_id` and canonical `section_id + attempt` derivation;
- nonnegative, bounded attempt number;
- writer identity;
- status in `accepted`, `rejected`, or `transport_failed`;
- comment IDs, candidate path/hash, validation path/hash, validator codes, and
  error fields as a valid state combination.

Each assessment loader validates at least:

- exact schema fields and `schema_version == 1`;
- unique assessment identity;
- exact comment/section/attempt linkage;
- critic model identity different from the attempt writer model;
- `context_isolated == true`;
- verdict in `resolved` or `unresolved`;
- a nonempty reason.

After per-record validation, the bundle loader requires exactly one writer model
across all attempts, exactly one critic model across all assessments, and
disjoint writer/critic model sets. These identities must match the Stage 19
manifest and the root run manifest's dedicated sectional fields. They must not
be compared against the root Stage 15 critic identity.

## 5. Release Audit Algorithm

The audit order is normative. Later checks must not hide an earlier malformed
artifact.

### C0. Require a complete sectional bundle

`release_check.py` always invokes `check_sectional_revision()`. Missing manifest
or any required top-level artifact is an error. `manifest.mode` must equal
`sectional` and `manifest.completed` must be exactly `true`.

Required errors:

- `sectional_revision_manifest_missing`
- `sectional_revision_artifact_missing`
- `sectional_revision_artifact_invalid`
- `sectional_revision_incomplete`

### C1. Validate contract and claim-scope closure

Read the canonical Stage 9 contract once through the runtime
`validate_contract_dict` semantics and reuse that parsed contract for the
existing experiment-contract gate and Phase C comparison. The pipeline and
release checker share `find_stage09_contract()` as the single Stage 9 selector;
direct-path priority and numeric `_vN` ordering remain protected by equivalence
tests. This avoids two different path-selection or parsing decisions inside one
release check.

Require:

```text
contract.claim_scope
  == manifest.claim_scope
  == "research_release"
```

Require `stage-10/selected_candidate_manifest.json`, parse it strictly, and
recompute and compare its `contract_sha256`. Deleting this third anchor is an
error, not a reason to fall back to a two-point comparison.

Required errors:

- `sectional_contract_binding_missing`
- `sectional_sealed_candidate_missing`
- `sectional_sealed_candidate_invalid`
- `sectional_contract_hash_mismatch`
- `sectional_claim_scope_mismatch`
- `sectional_non_release_claim_scope`

The current `ExperimentContract` validator rejects
`research_release + synthetic`, while `release_check.py` uses weaker ad hoc YAML
parsing and contains a waiver that can turn that runtime-invalid state into a
warning. The required governance decision is: remove
`synthetic_research_release_waived`, reject the waiver file as ineffective, and
make release checking use the runtime contract validator. This governance change
lands before C1 and includes a negative test proving that a waiver cannot make a
synthetic research-release contract pass. No Phase C integrity gate is waivable.

### C2. Rebuild the ledger and plan from canonical sources

Parse Stage 17 with `parse_manuscript(..., strict=True)`. Extract a fresh review
ledger from Stage 18. Load the stored final ledger and plan through strict
schemas. Then require:

- source paper/review paths and hashes match the current files;
- extracted comment IDs, exact source text, spans, required flags, and counts
  match the stored ledger;
- the plan covers every comment exactly once and references only current
  section IDs;
- final ledger state remains bound to that validated plan;
- the ledger contains at least one review comment for a release run.

Required errors:

- `sectional_source_hash_mismatch`
- `sectional_ledger_recompute_mismatch`
- `sectional_revision_plan_invalid`
- `sectional_review_comments_empty`

### C3. Require zero unresolved comments

Parse `unresolved_comments.json` strictly and rebuild it from the validated final
ledger. Its `ledger_sha256` and full object must match. For release:

- `comments` must be an empty list;
- every ledger comment must have `final_status == "resolved"`;
- manifest counts must satisfy `input > 0`, `resolved == input`,
  `unresolved == 0`, and `not_actionable_with_reason == 0`.

This is deliberately stricter than B2 execution semantics. A general comment or
`not_actionable_with_reason` item may be acceptable for a diagnostic
`pipeline_validation` run, but it is not release-ready.
Operationally, a B3 planner running under `research_release` must assign every
actionable and general comment to existing sections; any terminal planner
disposition other than an evidence-backed resolution makes the run
release-ineligible.

Required errors:

- `sectional_unresolved_artifact_mismatch`
- `sectional_unresolved_comments`
- `sectional_comment_counts_mismatch`

### C4. Verify validation-context provenance

Recompute the exact hash of `validation_context.json` and validate its strict
schema. Every source path/hash is rechecked against the current run. Stage 10
paths remain explicitly forbidden. Citation keys, grounded numeric values,
length ratios, and retry limits must equal the values used to replay each
attempt.

Required errors:

- `sectional_validation_context_invalid`
- `sectional_validation_context_hash_mismatch`
- `sectional_validation_source_missing`
- `sectional_validation_source_hash_mismatch`
- `sectional_validation_source_disallowed`

### C5. Replay attempts, validation, and critic assessments

Load every JSONL line through the strict loaders. The file set is default-deny:

- every non-transport attempt has exactly one canonical candidate file and one
  validation report;
- every recorded candidate/report path exists and has the recorded hash;
- every actual candidate/report file is referenced by exactly one attempt;
- temporary, nested, or otherwise unmanifested files, including `*.tmp`, are
  rejected rather than ignored;
- transport failures have neither file;
- attempt numbers are bounded by the validation-context-bound revision config;
- each accepted attempt is the final accepted attempt for its section;
- accepted attempts have no failed deterministic validator codes;
- `validate_section_candidate()` is rerun using the persisted proposal
  `resolution_comment_ids` and reconstructed context, and its complete result
  equals the stored report;
- every comment linked to an accepted attempt has one matching `resolved`
  isolated critic assessment;
- rejected/transport attempts cannot supply an accepted section or resolved
  ledger state;
- no assessment is orphaned, duplicated, cross-linked, or self-reviewed.
- all attempt writer identities and assessment critic identities satisfy the
  bundle-wide Stage 19 manifest and role-specific root run-manifest binding from
  section 4.1;
- the root run manifest, reviewer object, or required sectional model fields
  cannot be missing or malformed.

Required errors:

- `sectional_attempt_log_invalid`
- `sectional_attempt_reused`
- `sectional_attempt_artifact_missing`
- `sectional_attempt_artifact_unmanifested`
- `sectional_attempt_hash_mismatch`
- `sectional_validation_recompute_mismatch`
- `sectional_assessment_log_invalid`
- `sectional_assessment_identity_mismatch`
- `sectional_critic_isolation_invalid`
- `sectional_run_manifest_binding_missing`
- `sectional_run_manifest_model_mismatch`
- `sectional_resolved_without_evidence`

### C6. Recompute the merged paper

The release audit reconstructs the final paper from the strict Stage 17 document
and only the accepted, revalidated candidate bodies. It must not use
`paper_revised.md` as an input to reconstruction.

Require:

- every source section appears once and in source order;
- unchanged sections remain byte-identical;
- changed sections equal their accepted candidate body exactly;
- heading IDs, paths, levels, titles, and source heading bytes are unchanged;
- deterministic merge output equals `stage-19/paper_revised.md` byte-for-byte;
- exact SHA-256 equals `manifest.merged_paper_sha256`;
- each manifest section entry and changed flag equals the reconstructed result;
- the manifest includes no missing or extra sections.

Required errors:

- `sectional_merge_structure_mismatch`
- `sectional_merge_body_mismatch`
- `sectional_merge_hash_mismatch`
- `sectional_manifest_sections_mismatch`

### C7. Recompute manifest-bound artifact hashes

Recompute according to the producers' actual hash semantics:

- ledger and plan: canonical JSON SHA-256;
- attempts, assessments, unresolved comments, and validation context: exact
  UTF-8 text SHA-256;
- source and merged paper: exact text/file SHA-256 as defined by the sectional
  module;
- contract: exact file-byte SHA-256.

All manifest hashes and counts must match. Do not compare only one hash or trust
`completed: true` after partial success.

Required errors:

- `sectional_manifest_hash_mismatch`
- `sectional_manifest_recompute_mismatch`

## 6. Interaction With Existing Release Gates

Phase C findings are errors, never warnings and never degraded-compatible error
codes. Any Phase C finding forces exit 1, even when a degradation signal also
exists.

The existing gates remain authoritative for their own domains:

- Stage 20 quality/fabrication;
- Stage 22 compile and canonical final source;
- Stage 23 citation verification;
- Stage 24 claim/citation/provenance/truth audit;
- Stage 25 de-AI digest invariance;
- run-level sandbox, environment, reviewer isolation, critique resolution, and
  cost controls.

Stage 19 proves that revision did not lose review comments, invent evidence, or
silently replace manuscript structure. It does not prove that later stages
preserved the paper; the existing Stage 22-25 canonical/digest checks retain
that responsibility.

## 7. Implementation Boundaries

Implementation must use narrow, independently reviewable commits:

1. **C-1 contract-governance commit**: use one canonical Stage 9 selector,
   parse with runtime `ExperimentContract` validation, remove the synthetic
   waiver path, and add selector/waiver regression tests. No sectional bundle
   schema changes.
2. **C0 schema prerequisite commit**: contract path/hash, Stage 19 manifest model
   binding, root run-manifest sectional model fields, `attempts_sha256`,
   `candidate_path`, `resolution_comment_ids`, strict attempt/assessment loaders,
   and producer tests. No `release_check.py` changes.
3. **C1 gate commit**: read-only sectional bundle auditor,
   `ReleaseChecker.check_sectional_revision()`, error codes, and adversarial
   tests. No generated run artifacts and no configuration enablement.
4. **C2 dry-run config commit**: a dedicated non-default
   `pipeline_validation` config only after C0/C1 review. It must not make
   sectional mode the global default or claim release readiness.

Gate changes must obey the repository rule that `release_check.py` changes land
independently from generated outputs. No commit may weaken an existing gate to
make fixtures pass.

## 8. Required Adversarial Tests

At minimum, tests must cover these positive and negative cases:

1. missing Stage 19 sectional manifest;
2. legacy Stage 19 with only `paper_revised.md`;
3. `completed: false` and string/bool type confusion;
4. missing or malformed required artifact;
5. manifest claim scope edited without the Stage 9 contract;
6. contract edited after Stage 19;
7. missing Stage 10 sealed-candidate manifest or contract hash disagreement;
8. nonempty unresolved comments with counts falsely set to zero;
9. unresolved artifact edited while ledger remains unchanged;
10. ledger comment deletion, duplication, reordering, span change, or required
    flag change;
11. plan comment omission, duplicate assignment, or unknown section;
12. candidate file missing, extra, nested, or unreferenced;
13. validation report missing, edited, reused, or linked to another attempt;
14. proposal resolution IDs removed or forged;
15. accepted attempt whose deterministic validation now fails;
16. critic assessment with writer model, wrong identity, missing reason, or
    unresolved verdict;
17. orphan assessment or assessment attached to a rejected attempt;
18. changed section body differing from its accepted candidate;
19. unchanged section modified in `paper_revised.md`;
20. heading/path/order alteration with unchanged body hashes;
21. merged paper hash edited in either file or manifest;
22. extra/missing/reordered manifest section entry;
23. stale `stage-19_v*` bundle used when canonical `stage-19/` is missing;
24. Phase C error combined with a degradation signal still exits 1, not 2;
25. synthetic or `pipeline_validation` run never exits 0.
26. synthetic research-release contract remains blocked even with the legacy
    waiver file present;
27. attempts or assessments introduce a third model identity;
28. `sections/` or `section_validation/` contains an unmanifested `*.tmp` file;
29. pipeline and release contract selectors disagree when `stage-09` and
    `stage-09_v2` coexist;
30. degraded-compatible error-code allowlist remains the exact approved set and
    contains no `sectional_*` code.
31. root `run_manifest.json`, reviewer object, or either sectional model field
    is missing;
32. Stage 15 critic and Stage 19 critic legitimately differ while the dedicated
    sectional fields still match, and changing a per-attempt model to a third
    identity is rejected.

Positive structural tests must include one fully reconstructed synthetic
`pipeline_validation` fixture that passes C0 and C2-C7 but receives the expected
C1 `sectional_non_release_claim_scope` error and therefore fails overall
release. A later `research_release` fixture may pass all Phase C checks only with
a non-synthetic valid contract and all existing release artifacts.

## 9. Acceptance Commands

Before implementation review:

```bash
.venv/bin/python -m pytest tests/test_sectional_revision.py \
  tests/test_sectional_validation.py tests/test_sectional_execution.py \
  tests/test_sectional_llm.py -q
.venv/bin/python -m pytest tests/test_release_check_v2.py -q
.venv/bin/python scripts/probe_release_gates.py
.venv/bin/python -m py_compile scripts/release_check.py \
  researchclaw/pipeline/sectional_revision.py \
  researchclaw/pipeline/sectional_validation.py \
  researchclaw/pipeline/sectional_execution.py
git diff --check
```

After C0/C1 review, run a new directory through Stage 19 with sectional mode
enabled under `pipeline_validation`. The expected result is:

- Stage 19 completes or fails solely according to its deterministic contracts;
- the Phase C sectional audit can validate a complete bundle;
- overall `release_check.py` exits nonzero because the run is not
  `research_release`;
- no historical run directory is reused to manufacture a pass.

## 10. Blocking Review Questions

Claude/Kiro must answer these before C-1/C0 implementation:

1. Are the proposed manifest and attempt schema additions sufficient for a
   disk-only replay, or is another hidden in-memory input still required?
2. Can the auditor reconstruct `SectionValidationContext` exactly from the
   contract, validation context, ledger, plan, and persisted
   `resolution_comment_ids`?
3. Does default-deny file enumeration cover every stale/unmanifested candidate
   and validation artifact without rejecting legitimate transport failures?
4. Is requiring zero unresolved and zero not-actionable comments the intended
   `research_release` policy?
5. Should legacy Stage 19 be permanently release-ineligible once Phase C lands?
6. Is the canonical Stage 9 contract sufficiently anchored by the Stage 19 and
   Stage 10 cross-hashes, or is another run-level binding needed?
7. Does removing the synthetic waiver and sharing the runtime contract validator
   fully close the weak-parser path without changing valid public or
   local-hardware release contracts?
8. Can any Phase C error accidentally enter the degraded-compatible exit-code
   set?
9. Does the merged-paper reconstruction prove byte identity without trusting
   manifest section metadata?
10. Are C-1, C0, C1, and C2 narrow enough to review and revert independently?
11. Do the dedicated sectional model fields preserve Stage 15 critic
    independence while still making every Stage 19 writer/critic identity
    auditable and mandatory?
