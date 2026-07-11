# Stage 19 Phase B: Sectional Revision Contract

Status: review draft. This document specifies interfaces and deterministic
acceptance rules. It does not authorize production integration, release-gate
changes, or LLM calls.

## 1. Objective

Replace whole-manuscript revision with a bounded workflow in which deterministic
code owns manuscript structure, review accounting, validation, merging, and
audit state. An LLM may later propose a body replacement for one existing
section at a time, but it must not own headings, document assembly, evidence
policy, completion state, or release readiness.

Phase A already provides the lossless CommonMark-aware document model in
`researchclaw/pipeline/manuscript_sections.py`. Phase B builds strict contracts
around that model. The existing whole-paper Stage 19 remains the default path
until a later feature-flagged integration is separately reviewed.

## 2. Scope And Non-Goals

Phase B covers:

- deterministic extraction of actionable review comments from Stage 18;
- a complete review comment ledger;
- a strict section assignment plan;
- one-section-at-a-time proposal and validation interfaces;
- deterministic merge and post-merge structural verification;
- unresolved-state semantics for `pipeline_validation` and `research_release`;
- versioned, fail-closed artifacts and tests.

Phase B does not:

- change `scripts/release_check.py`;
- change Stage 20 quality-gate semantics;
- make sectional revision the default;
- allow an LLM to write or reorder headings;
- run new experiments or invent evidence requested by reviewers;
- implement a whole-paper fallback;
- implement cross-section automatic rewriting;
- declare a manuscript release-ready.

Stage 20 and release integration belong to Phase C. Until Phase C lands, a
sectional dry run is diagnostic and cannot establish release readiness.

## 3. Non-Negotiable Invariants

1. The Stage 17 draft and Stage 18 reviews are immutable inputs identified by
   SHA-256.
2. Section IDs, heading paths, source spans, and hashes come only from the Phase
   A parser. The LLM cannot name or create section IDs.
3. Every extracted review comment appears exactly once in the ledger. No comment
   may disappear between extraction, planning, execution, and finalization.
4. A comment has exactly one final status: `resolved`, `unresolved`, or
   `not_actionable_with_reason`. `assigned` is a working status only. The writer
   model cannot set a final status.
5. A `global` target is a routing marker, not permission to rewrite the full
   paper. Before execution it must be decomposed into existing section IDs or
   remain explicitly unresolved/not actionable.
6. Only section bodies may be replaced. Preamble, heading source bytes, heading
   order, levels, titles, and paths remain unchanged.
7. An unchanged section remains byte-for-byte identical.
8. A proposed revision cannot create evidence. Requests for missing experiments,
   datasets, figures, metrics, or statistical analyses are not actionable in
   Stage 19 unless the required run artifact already exists.
9. Unknown fields, unknown enum values, unknown section IDs, duplicate IDs,
   missing hashes, malformed JSON, and count mismatches fail closed.
10. The merged manuscript must reparse with the same heading count, IDs, levels,
    paths, and heading bytes as the source manuscript.
11. No Stage 10 smoke artifact is admissible evidence or a grounded numeric
    source.
12. Sectional mode has one body-writing path: validated replacements passed to
    the deterministic merger. There is no whole-paper LLM fallback.
13. A required comment becomes `resolved` only after deterministic validation
    and an independent critic attestation. The writer cannot self-certify that
    its own revision resolved a comment.

## 4. Configuration Boundary

Phase B should introduce a dedicated immutable config group rather than place
paper policy under `ExperimentConfig`:

```python
@dataclass(frozen=True)
class PaperRevisionConfig:
    sectional_enabled: bool = False
    max_section_retries: int = 1
    min_length_ratio: float = 0.80
    max_length_ratio: float = 1.75
    critic_model: str = ""
```

`RCConfig` receives `paper_revision: PaperRevisionConfig`. Parsing rejects
negative retry counts and ratios outside a documented safe range. B3 requires a
nonempty critic model that differs from the writer model. Critic isolation has
no disable flag: every sectional run uses a fresh critic context, including
`pipeline_validation`. The first implementation must keep `sectional_enabled:
false` in all checked-in configs.

The flag selects the entire Stage 19 strategy at stage start. It must not switch
strategies after a partial failure. If sectional mode starts and fails, Stage 19
uses sectional failure semantics; it never invokes the legacy whole-paper path.

## 5. Input Model

### 5.1 Manuscript

Stage 19 reads the canonical Stage 17 `paper_draft.md` and parses it with
`parse_manuscript(..., strict=True)`. Any structure issue blocks sectional mode.
Permissive parsing is diagnostic only and must never feed replacements.

This is a live precondition, not a theoretical edge case. The current
`runs/hwsec-scaffold-v2/stage-17/paper_draft.md` has 48 parsed sections and four
`duplicate_heading_path` issues (duplicate `3. Method`, `3.1 Problem
Formulation`, `3.2 Counter Selection`, and `4. Experimental Setup` paths).
Therefore that historical draft is valid only for permissive round-trip testing;
it must not be used to demonstrate successful sectional revision. Before B2 can
pass a real run, Stage 17 must emit a structurally unambiguous draft. Stage 19
must fail with a diagnostic artifact rather than rename or merge headings
automatically.

### 5.2 Reviews

The currently generated Stage 18 format is Markdown organized by reviewer
headings and subsections such as `Strengths`, `Weaknesses`, `Actionable
Revisions`, and `General Comments`.

The deterministic extractor uses CommonMark tokens and source maps, not a
free-form LLM summary. Version 1 applies these rules:

- numbered or bulleted items under `Actionable Revisions` become actionable
  comments;
- subsection names are normalized with Unicode normalization, whitespace
  collapse, case-folding, and removal of one trailing parenthetical qualifier;
  therefore `General Comments (Applicable to All Reviewers)` matches `General
  Comments`;
- numbered or bulleted items under normalized `General Comments` become global
  comments with reviewer label exactly `all`, regardless of their CommonMark
  parent heading;
- `Strengths` and `Weaknesses` remain source context but do not automatically
  become ledger comments;
- blank lines, headings, and thematic breaks are structural and are not comments;
- any content-bearing unknown subsection, prose outside a recognized
  reviewer/subsection structure, or list item under an unknown subsection is not
  silently dropped: extraction records an issue such as
  `unknown_review_subsection` and fails closed;
- each comment preserves its exact source text, reviewer label, category,
  one-based line span, and source-text SHA-256;
- nested list continuation paragraphs belong to the preceding list item;
- duplicate textual comments remain distinct entries because source positions
  differ.

`exact_text` is the literal CommonMark source slice for the outer list item,
including its list marker, continuation/nested-list lines, and original line
terminators. `source_line_start` and `source_line_end` are one-based and
inclusive. `source_text_sha256` hashes that exact slice without normalization.

The extractor creates a stable ID from deterministic inputs:

```text
comment_id = "rc-" + zero_padded_ordinal + "-" +
             sha256(reviewer + category + exact_text + line_span)[:12]
```

The number of ledger comments must equal the number of recognized actionable
and general-comment items. Zero extracted comments is an error when
`reviews.md` is nonempty.

The current live `reviews.md` is intentionally a negative fixture: Reviewer C
contains `Additional Rigor Issues`, an unknown content-bearing subsection with
three bullets, so v1 extraction must fail closed. The positive 17-comment fixture
is a normalized copy in which those three bullets are deliberately moved into a
recognized context section rather than silently ignored. The future Stage 18
prompt must constrain output to the recognized subsection vocabulary before B2
integration.

Whether `Weaknesses` should also become ledger inputs is intentionally a review
question. Version 1 chooses the narrower auditable rule above because Stage 18
already emits explicit actionable revisions. Changing this rule requires a
schema-version or extractor-version change and new count tests.

## 6. Artifact Schemas

All JSON files use UTF-8, `schema_version: 1`, stable key ordering when written,
and atomic replacement. Schema validators reject unknown fields at every object
level. Hashes are lowercase 64-character SHA-256 strings.

### 6.1 `review_comment_ledger.json`

This is the authority for review accounting. LLM output never rewrites comment
text or source metadata.

```json
{
  "schema_version": 1,
  "extractor_version": 1,
  "source_reviews_path": "stage-18/reviews.md",
  "source_reviews_sha256": "...",
  "comments": [
    {
      "comment_id": "rc-001-...",
      "reviewer": "Reviewer A (Methodology Expert)",
      "category": "actionable_revision",
      "exact_text": "Unify metrics ...",
      "source_line_start": 18,
      "source_line_end": 19,
      "source_text_sha256": "...",
      "required": true,
      "required_source": "policy-v1:actionable_revision",
      "working_status": "unassigned",
      "target_section_ids": [],
      "final_status": null,
      "resolution_reason": null,
      "attempt_ids": []
    }
  ]
}
```

Allowed `category` values are `actionable_revision` and `general_comment`.
Allowed `working_status` values are `unassigned` and `assigned`. Allowed final
values are `resolved`, `unresolved`, and `not_actionable_with_reason`.
`required` is set deterministically: actionable revisions are required; general
comments are non-required unless a future policy file explicitly promotes them.
`required_source` records the exact policy version and rule that produced the
boolean. Treating general comments as non-required is an intentional delegation
to Stage 20 quality review, not evidence that those comments were resolved.

### 6.2 `revision_plan.json`

The plan references ledger IDs and existing manuscript section IDs. It does not
duplicate or paraphrase review text.

```json
{
  "schema_version": 1,
  "planner_version": 1,
  "source_paper_sha256": "...",
  "source_reviews_sha256": "...",
  "section_model_version": 1,
  "assignments": [
    {
      "comment_id": "rc-001-...",
      "target_section_ids": ["s003-method-..."],
      "disposition": "assigned",
      "reason": null
    }
  ]
}
```

Allowed `disposition` values are `assigned`, `unresolved`, and
`not_actionable_with_reason`. `assigned` requires one or more unique existing
section IDs and forbids `global`. The other dispositions require a nonempty
reason and no section IDs. A planner may initially emit `global`, but the plan
validator must reject it as non-executable until a deterministic or separately
reviewed decomposition step resolves it.

For a required comment, `not_actionable_with_reason` is still a non-resolved
state for release purposes. It cannot be used to convert a missing experiment or
unsupported claim into a successful `research_release` revision.

Planning closure requires exactly one assignment per ledger `comment_id`, no
extras, no duplicates, and equal source hashes.

### 6.3 `section_attempts.jsonl`

Each proposal attempt is append-only:

```json
{
  "schema_version": 1,
  "attempt_id": "sec-s003-a1",
  "section_id": "s003-method-...",
  "source_section_sha256": "...",
  "comment_ids": ["rc-001-..."],
  "writer_model": "...",
  "attempt": 1,
  "status": "rejected",
  "candidate_body_sha256": "...",
  "validation_report_path": "stage-19/section_validation/s003-method-....attempt-1.json",
  "validation_report_sha256": "...",
  "validator_codes": ["unknown_numeric_value"],
  "error_type": null,
  "error": null,
  "timestamp": "..."
}
```

Allowed statuses are `accepted`, `rejected`, and `transport_failed`. Candidate
body text lives in `sections/<section_id>.attempt-<n>.md`; the log stores hashes,
not duplicated manuscript text. `writer_model` is mandatory for every LLM-backed
attempt. The validation path and hash must point to the attempt-specific report;
a transport failure uses null candidate/validation fields.

### 6.4 `section_validation/<section_id>.attempt-<n>.json`

```json
{
  "schema_version": 1,
  "attempt_id": "sec-s003-a1",
  "section_id": "s003-method-...",
  "original_sha256": "...",
  "candidate_sha256": "...",
  "accepted": false,
  "checks": [
    {
      "code": "unknown_numeric_value",
      "status": "failed",
      "details": ["0.990 is not grounded"]
    }
  ]
}
```

Allowed check statuses are `passed` and `failed`. `accepted` is true only when
all mandatory checks pass. Details are diagnostic and cannot override status.
Attempt-specific filenames are mandatory so retries never overwrite prior
validation evidence.

### 6.5 `section_revision_manifest.json`

```json
{
  "schema_version": 1,
  "mode": "sectional",
  "claim_scope": "pipeline_validation",
  "source_paper_path": "stage-17/paper_draft.md",
  "source_paper_sha256": "...",
  "source_reviews_path": "stage-18/reviews.md",
  "source_reviews_sha256": "...",
  "ledger_sha256": "...",
  "plan_sha256": "...",
  "assessments_sha256": "...",
  "unresolved_comments_sha256": "...",
  "sections": [
    {
      "section_id": "s003-method-...",
      "original_sha256": "...",
      "final_body_sha256": "...",
      "changed": true,
      "comment_ids": ["rc-001-..."],
      "attempt_ids": ["sec-s003-a1"],
      "validation_report_sha256": "...",
      "final_status": "accepted"
    }
  ],
  "comment_counts": {
    "input": 10,
    "resolved": 6,
    "unresolved": 3,
    "not_actionable_with_reason": 1
  },
  "merged_paper_sha256": "...",
  "completed": true
}
```

Every source section appears exactly once and in parser order. Allowed section
final states are `unchanged`, `accepted`, and `unresolved_original_preserved`.
The manifest's comment counts must equal the final ledger and sum to input. All
referenced attempt, validation, assessment, ledger, plan, and unresolved hashes
must recompute from current Stage 19 files.

### 6.6 `resolution_assessments.jsonl`

The writer's resolution claim is not authoritative. A separately configured
critic, with no writer conversation context, assesses each attempted comment
against the exact review text, original section, revised section, and validator
summary:

```json
{
  "schema_version": 1,
  "assessment_id": "ra-rc-001-a1",
  "comment_id": "rc-001-...",
  "section_id": "s003-method-...",
  "attempt_id": "sec-s003-a1",
  "critic_model": "...",
  "context_isolated": true,
  "verdict": "resolved",
  "reason": "The metric definition is now explicit in both paragraphs.",
  "timestamp": "..."
}
```

Allowed verdicts are `resolved` and `unresolved`. Missing critic identity,
`context_isolated != true`, missing reason, unknown IDs, or a hard-validator
failure forces `unresolved`. Critic approval cannot override deterministic
validation. In contract-only B0/B1 tests, a deterministic fake assessor may be
injected; production code must not silently substitute the writer as critic.
The critic model must differ from the writer model and receive a fresh context
containing only the bounded assessment inputs.

### 6.7 `unresolved_comments.json`

Contains `schema_version`, `ledger_sha256`, and only ledger entries whose final
status is `unresolved` or `not_actionable_with_reason`, plus their reasons. This
is a derived diagnostic artifact; the ledger remains authoritative. An empty
list is written explicitly rather than omitting the file. A mismatched ledger
hash makes the derived file invalid.

## 7. Deterministic API Boundaries

Phase B should add a module separate from the Stage 19 executor, for example
`researchclaw/pipeline/sectional_revision.py`, with no LLM imports in its
schema, extraction, validation, or merge layers.

Minimum interfaces:

```python
extract_review_ledger(reviews: str, *, source_path: str) -> ReviewLedger
validate_revision_plan(plan: object, ledger: ReviewLedger,
                       document: ManuscriptDocument,
                       *, reviews: str) -> RevisionPlan
validate_section_candidate(context: SectionValidationContext,
                           candidate_body: str) -> SectionValidationResult
finalize_ledger(ledger: ReviewLedger, attempts: Sequence[SectionAttempt],
                plan: RevisionPlan,
                assessments: Sequence[ResolutionAssessment]) -> ReviewLedger
merge_validated_sections(document: ManuscriptDocument,
                         accepted: Mapping[str, str]) -> MergeResult
```

Typed dataclasses own internal state. JSON conversion happens only through
strict `from_dict`/`to_dict` functions. Business logic must not operate on raw
unvalidated dictionaries. `from_dict` proves schema shape but cannot by itself
prove extraction completeness; every loaded ledger must be rebound to the raw
Stage 18 text. Plan validation therefore requires `reviews` and reruns the
deterministic extractor before accepting comment closure.

## 8. Section Candidate Validation

Validation is deterministic and fail-closed. Mandatory check codes are stable
API because tests and later release integration will depend on them.

### 8.1 Structural checks

- `empty_section_body`: candidate is empty or whitespace only.
- `new_heading_introduced`: candidate body contains a CommonMark heading token.
- `html_heading_introduced`: candidate introduces a raw HTML `<h1>` through
  `<h6>` start/end tag, case-insensitively and with attributes allowed.
- `markdown_structure_unbalanced`: fenced code or other tracked block structure
  is not balanced.
- `post_merge_heading_mismatch`: after temporary merge and strict reparse,
  heading count, IDs, levels, paths, titles, or exact heading bytes differ.
- `section_order_mismatch`: reparsed order differs from source order.

The post-merge reparse is mandatory even when the candidate body alone appears
valid. This closes boundary interactions between a replacement body and the next
heading.

### 8.2 Evidence checks

- `unknown_citation_key`: candidate introduces a citation key not present in the
  Stage 18-time canonical `references.bib`.
- `required_citation_removed`: a citation key present in the original section is
  removed. Version 1 has no automatic deletion exemption; legitimate removals
  require a later versioned policy or human path.
- `unknown_numeric_value`: candidate introduces a quantitative value outside the
  grounded metric whitelist and outside numbers already present in the original
  section.
- `required_reference_removed`: a figure/table/equation reference present in the
  original section disappears.
- `unknown_reference_introduced`: candidate introduces a Figure/Table/Equation
  reference whose target label/caption does not exist in the complete source
  document.
- `unparsed_quantitative_expression`: candidate introduces a number-word phrase
  near a quantitative unit that the deterministic normalizer cannot parse.

Citation keys come from deterministic BibTeX parsing. They must not be inferred
from draft prose. Version 1 recognizes LaTeX `\cite{key}` and the pipeline's
bracketed cite-key form. Pandoc `@key` syntax is explicitly unsupported because
a broad `@` matcher would also classify email addresses and handles; adding it
requires a typed parser and separate tests. The grounded numeric whitelist uses
only allowed Stage 12-14 experiment artifacts and excludes Stage 10 diagnostics.

Numeric comparison must normalize equivalent forms. At minimum:

- decimal/fraction/percentage equivalence where context marks a percentage:
  `0.85`, `85%`, and `8.5e-1`;
- thousands separators;
- scientific notation;
- digit-plus-word percentages, so `85 percent` normalizes to `0.85` just as
  `85%` does;
- English number-word sequences adjacent to a unit in the versioned v1 lexicon,
  which explicitly includes `percent`, `trials`, `seeds`, `samples`,
  `counters`, `ms`, and `us`; an unrecognized sequence fails rather than being
  ignored;
- relative tolerance is fixed at no more than `1e-3`, matching the existing
  provenance comparison. It is only for float serialization; `0.475` becoming
  `0.48` must fail as materially different rounding.

In B1 this is implemented as the closed, versioned constant
`QUANTITATIVE_UNIT_LEXICON_V1`; implementations must not silently add units at
runtime. Expanding the lexicon requires a version change and positive/negative
tests. Fraction phrases such as `two thirds` are parsed independently; ambiguous
quantifiers such as `several trials` fail explicitly.

Numbers in typed author-year citations, section labels, figure/table labels, and
bibliography metadata require typed classification so they are not mistaken for
experiment metrics. A naked four-digit number is not exempt merely because it
falls between 1900 and 2099: `2000 samples`, `2048 windows`, and `N=2000` must be
grounded like any other quantity. Plural references such as `Figures 2 and 3`,
`Tables 4 & 5`, and `Eqs. 6, 7` must expand to individual typed targets.

Math spans are not a blanket numeric exemption. Decimal, scientific-notation,
and percentage literals inside inline or display math participate in numeric
grounding, so `$F_1 = 0.97$` cannot bypass the whitelist. Version 1 deliberately
ignores structural integers in expressions such as `$x^2$` and
`$\frac{1}{2}$` to limit false positives; the isolated critic and Stage 24 still
review the semantic claim.

The validator may preserve numbers already present in the original section, but
it must record them as `source_preserved`, not `grounded_metric`.
Preservation does not prove that an old number is true. If a review comment
targets an ungrounded number and the candidate leaves it unchanged, the critic
must keep that comment unresolved; later Stage 20/24 gates remain authoritative
for the paper's full numeric provenance.

### 8.3 Content-preservation checks

- `abnormal_section_shrink`: candidate word count is below
  `min_length_ratio * original_word_count`.
- `abnormal_section_growth`: candidate exceeds the configured maximum ratio,
  guarding against prompt leakage or unrelated generation.
- `unaddressed_required_comment`: the proposal's machine-readable resolution
  record does not account for every required comment assigned to the section.
- `missing_resolution_assessment`: a writer claims a comment was addressed but
  no valid independent critic assessment exists.
- `resolution_conflicts_with_evidence`: a critic marks a comment resolved even
  though the requested evidence artifact is absent or a hard validator failed.

Length is a guardrail, not evidence that revision quality is acceptable.
Version 1 has no semantic exemption for deleting references or shrinking below
the floor. Such exceptions would require a separately versioned policy rather
than trusting a planner's free-text interpretation.

## 9. LLM Boundary For A Later Subphase

The first Phase B commit must implement and test contracts without making LLM
calls. A later, separately reviewed subphase may add three bounded operations:

1. map ledger comment IDs to existing section IDs;
2. propose one replacement body and a resolution record for one section.
3. independently assess whether each attempted comment was actually resolved.

The LLM request may include only:

- one section's heading and body;
- exact assigned review comment text;
- grounded metric whitelist;
- allowed citation keys;
- short deterministic summaries of adjacent sections when necessary.

The LLM response schema contains only:

```json
{
  "schema_version": 1,
  "section_id": "...",
  "revised_body": "...",
  "resolutions": [
    {
      "comment_id": "...",
      "writer_status": "addressed",
      "reason": "..."
    }
  ]
}
```

Allowed writer statuses are `addressed` and `not_addressed`. Unknown fields are
rejected. The writer cannot set validator outcomes, final ledger status, critic
verdicts, retry counts, hashes, or manifest completion. The independent critic
uses a separate call, model identity, and context; its output is validated
against `resolution_assessments.jsonl` and can only attest resolution after all
hard checks pass.

## 10. State Machine And Failure Semantics

```text
extracted -> unassigned
unassigned -> assigned | unresolved | not_actionable_with_reason
assigned -> accepted attempt + independent critic approval -> resolved
assigned -> rejected/transport failures exhausted -> unresolved
assigned -> critic rejection/missing assessment -> unresolved
```

No backward transitions occur in the final artifact set. Attempts remain
append-only even if a later attempt succeeds.

For a comment assigned to multiple sections, final status is `resolved` only if
every assigned section has an accepted attempt and a valid `resolved` assessment
for that comment/section pair. Any missing or rejected pair makes the comment
`unresolved`. `not_actionable_with_reason` is set only by a validated plan
disposition; it is never inferred from a failed attempt.

Stage 19 may return DONE only after ledger closure: every comment has a non-null
final status, terminal counts equal the input count, all assignments have either
terminal attempt evidence or an explicit non-assigned disposition, and the
manifest validates with `completed: true`. An interrupted assigned comment with
no attempt is not silently converted to unresolved; it leaves
`completed: false`, and Stage 19 is not DONE.

For `pipeline_validation`:

- after bounded retries, preserve the exact original body;
- mark the section `unresolved_original_preserved`;
- mark linked comments `unresolved` with a diagnostic reason;
- Stage 19 may return DONE only if manifest and unresolved artifacts are complete
  and internally consistent;
- this status is not release success.

For `research_release` and `exploratory`:

- any required comment whose final status is not `resolved` makes Stage 19
  FAILED, including `not_actionable_with_reason`;
- no `paper_revised.md` is exposed as a successful current artifact;
- diagnostics, ledger, attempts, and incomplete manifest remain available;
- a damaged/missing experiment contract resolves fail-closed to the strict path,
  matching current Stage 19 behavior.

## 11. Resume And Stale-Artifact Rules

At sectional Stage 19 start, remove only Stage 19-owned current-attempt outputs:

```text
paper_revised.md
revision_notes_internal.md
revision_retry_failure.json
revision_plan.json
review_comment_ledger.json
section_attempts.jsonl
resolution_assessments.jsonl
section_revision_manifest.json
unresolved_comments.json
consistency_audit.json
sections/
section_validation/
```

Do not remove archived `stage-19_v*` directories. Current execution must never
read a plan, ledger, attempt, candidate, manifest, or revised paper from a prior
Stage 19 directory. Input discovery is restricted to canonical prior-stage
artifacts; Stage 19 output paths are explicit, not `_read_prior_artifact`
fallbacks.

## 12. Implementation Milestones

### B0: Contracts only

- add strict dataclasses and JSON validators;
- add deterministic review extraction and ledger closure;
- add plan validation against ledger and Phase A section IDs;
- add no LLM imports or executor integration.

### B1: Deterministic validators and merge result

- add citation, numeric, structure, reference, and length checks;
- add mandatory merge-and-reparse structural invariant;
- add manifest construction and internal consistency validation;
- add the CommonMark line-protocol property test requested in Phase A review.

B1 binds assessment and unresolved artifacts by hash and deterministically
rebuilds unresolved comments from the ledger. It does not yet validate critic
semantics; that authority arrives with the isolated assessment contracts in B3.
Therefore a B1 manifest, including `completed: true` in a fixture, is not by
itself evidence of release readiness and is not consumed by production stages.

### Pre-B2: Upstream format contracts

- update Stage 17 generation/validation so a new draft has unique canonical
  heading paths and passes strict parsing;
- constrain the Stage 18 prompt to the recognized subsection vocabulary;
- add the current live Stage 17 draft and Stage 18 reviews as explicit negative
  regression fixtures;
- keep this as a separate upstream-format commit, not an automatic Stage 19
  repair.

The Pre-B2 implementation appends a non-overridable per-call section-output
contract to all Stage 17 writing calls and validates the final post-HITL draft
with the CommonMark section model. Ambiguous output is `FAILED` and writes
`paper_structure_report.json`; it is never auto-renamed or silently repaired.
Stage 18 likewise appends one cross-domain output contract after every active
prompt-bank instruction, then validates the response with the B0 extractor.
Unknown content-bearing subsections write `review_structure_report.json` and
fail Stage 18 rather than being dropped from the ledger.

### B2: Feature-flagged Stage 19 integration

- add `PaperRevisionConfig`, default disabled;
- route sectional mode once at Stage 19 start;
- clean owned artifacts and write complete diagnostics;
- still use a deterministic fake proposal provider in tests.
- require the Pre-B2 upstream-format commit and a new Stage 17 artifact that
  passes strict parsing; do not use permissive parsing or automatic heading
  repair.
- build `SectionValidationContext` only from canonical `references.bib`, allowed
  Stage 12-14 artifacts, and validated `PaperRevisionConfig`; record source paths
  and hashes for the citation and numeric whitelists in the Stage 19 manifest.

### B3: Bounded LLM planner and section proposer

- add strict JSON calls for assignment and one-section revision;
- add an isolated critic call for resolution assessment;
- add per-section retry only;
- preserve all deterministic validators as authority;
- run a new `pipeline_validation` dry run before any default change.

Each milestone is a separate narrow commit and review boundary. No milestone may
combine release-check changes.

Phase C must independently compare the manifest `claim_scope` with the canonical
Stage 9 experiment contract. A matching string inside the manifest is not
self-authenticating.

## 13. Required Tests

### Review extraction and ledger

- a checked-in sanitized fixture derived from the current Stage 18 format
  produces exactly 17 comments (4 + 4 + 5 actionable revisions and 4 general
  comments);
- the unmodified current reviews fixture fails closed on `Additional Rigor
  Issues` rather than dropping its three bullets;
- `General Comments (Applicable to All Reviewers)` normalizes to `General
  Comments`, contributes four comments, and records reviewer `all`;
- thematic breaks such as `---` are not counted as comments;
- continuation lines remain attached to the right list item;
- duplicate text at different spans gets distinct IDs;
- malformed/nonempty reviews with zero recognized comments fail;
- comment count and source hash tampering fail;
- unknown schema fields fail.

### Plan validation

- every ledger ID appears exactly once;
- missing, duplicate, and unknown comment IDs fail;
- unknown/duplicate section IDs fail;
- executable `global` target fails;
- not-actionable/unresolved dispositions require reasons;
- source hash mismatch fails.

### Candidate validation

- empty body and introduced ATX/Setext headings fail;
- introduced raw HTML `<h1>` through `<h6>` headings fail;
- fence-boundary and post-merge heading mutations fail;
- unknown citation and removed original citation fail; preserved known citation
  passes;
- a newly introduced Figure/Table/Equation reference without a declared target
  fails;
- Stage 10 smoke numbers do not enter the whitelist;
- numeric equivalence `0.85 == 85% == 8.5e-1` is tested;
- `85 percent` normalizes to `0.85`;
- `2000 samples` and `2048 windows` are not exempt as citation years;
- an ungrounded decimal metric inside math fails while structural integer math
  remains outside the v1 numeric extractor;
- plural Figure/Table/Equation references expand to individual targets;
- Pandoc `@key` remains explicitly unsupported rather than being guessed by a
  broad regular expression;
- a number-word quantitative phrase is parsed or fails explicitly, and
  `0.475 -> 0.48` is rejected under the `1e-3` tolerance ceiling;
- fabricated metric, removed figure/table ref, shrink, and growth fail;
- required missing-experiment requests cannot be marked resolved.
- writer self-attestation without an isolated critic assessment fails;
- critic approval cannot override a hard-validator failure.

### Merge and manifest

- unchanged sections are byte-identical;
- changed bodies merge in original order;
- merged output reparses to identical heading metadata;
- manifest section/order/hash/count tampering fails;
- attempt-specific validation reports do not overwrite one another and their
  hashes match the attempt log;
- stale unresolved files with the wrong `ledger_sha256` fail;
- unresolved `pipeline_validation` preserves the exact original body;
- unresolved `research_release` fails and exposes no successful revised paper;
- `writer_model == critic_model` invalidates the assessment;
- sectional config cannot disable critic isolation;
- assigned-without-attempt leaves `completed: false` and Stage 19 non-DONE;
- stale current-stage artifacts, including legacy revision diagnostics, cannot
  satisfy or contaminate a resumed attempt.

### Protocol and adversarial tests

- property/fuzz test compares the custom CommonMark line splitter with the
  tokenizer newline protocol;
- line separators include CRLF, CR-only, Unicode separators, form feed, NUL,
  and no final newline;
- fenced headings, nested headings, front matter, tables, equations, HTML, and
  non-ASCII headings retain deterministic boundaries.

## 14. Acceptance Commands

For B0/B1, the implementation must at minimum pass:

```bash
.venv/bin/python -m pytest tests/test_manuscript_sections.py \
  tests/test_sectional_revision.py -q
.venv/bin/python -m pytest tests/test_rc_executor.py \
  tests/test_release_check_v2.py -q
.venv/bin/python -m py_compile \
  researchclaw/pipeline/manuscript_sections.py \
  researchclaw/pipeline/sectional_revision.py \
  researchclaw/pipeline/sectional_validation.py
git diff --check
```

For B2/B3, add a new run directory and stop at Stage 19. A
`pipeline_validation` run with unresolved comments may complete Stage 19 only
with complete diagnostics and must still fail release readiness. A synthetic or
non-release run returning release exit 0 is a P0 false positive.

## 15. External Review Questions

Kiro/Claude should review this document against live code and answer:

1. Does limiting ledger inputs to `Actionable Revisions` plus `General Comments`
   lose any mandatory issue from the current Stage 18 format? If `Weaknesses`
   should also be tracked, define a deterministic deduplication rule.
2. Are all schema fields sufficient to prove no review comment disappeared and
   no section was silently replaced?
3. Can `global` comments be resolved without reintroducing whole-paper
   generation? Specify any missing decomposition invariant.
4. Are numeric normalization and source-preserved-number semantics strict enough
   to prevent fabricated metrics without blocking citation years and labels?
5. Are `pipeline_validation` and `research_release` failure semantics fail-closed
   under retries, resume, malformed contracts, and transport failures?
6. Is the writer/critic separation sufficient to prevent self-certified
   resolution, and which critic failures must remain unresolved?
7. Which validator codes are missing before B1 implementation?
8. Does any proposed artifact become a second authority that can conflict with
   the ledger or manifest?
9. Are B0-B3 commit boundaries narrow enough for independent diff review?

Required response format:

```text
## Verdict: ACCEPT / MODIFY / REJECT
## P0 findings
## P1 findings
## P2 findings
## Schema gaps
## State-machine gaps
## Missing adversarial tests
## Required changes before B0
```
