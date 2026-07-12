# Citation Evidence Pipeline Contract

Status: review draft. This document records the agreed architecture for
citation identity, evidence extraction, citation planning, manuscript closure,
verification, and release audit across Stages 4-24. It does not authorize
implementation, change any current gate, enable full-text acquisition, or
declare any existing run release-ready.

## 1. Objective

The pipeline must distinguish four different propositions:

1. a source exists and its bibliographic identity is known;
2. a source is eligible for use in this run;
3. a retained excerpt supports a specific manuscript claim;
4. every citation and claim in the final manuscript is closed over those
   artifacts.

The presence of a BibTeX key proves none of propositions 2-4. An LLM summary,
a topical title match, a citation count, or a stage status of DONE is not
claim-support evidence.

The immediate objective is a versioned, disk-replayable v1 contract using
abstract excerpts. The later v2 extension adds human-in-the-loop acquisition
of canonical full text without weakening v1 or making PDF acquisition part of
ordinary pipeline-validation runs.

## 2. Observed Failure And Root Cause

The run `runs/hwsec-sectional-dry-run-20260711` reached Stage 19 and failed
because the draft cited `venkatakeerthy2020ir2vec`, while the canonical Stage 4
bibliography contained the same paper under
`venkatakeerthy2020scpecscp`. Stage 19 correctly rejected the missing canonical
key.

The failure exposed a longer chain:

- the current citation-key generator derives a keyword directly from title
  tokens, allowing markup such as `<scp>EC</scp>` to contaminate identity;
- Stage 5 can replace an unparseable model response with fifteen fabricated
  fallback screening rows carrying synthetic relevance and quality scores;
- Stage 6 can replace failed extraction with Markdown cards containing
  `Template method summary`, `Template dataset`, and similar placeholders;
- Stage 17 currently exposes all Stage 4 candidates to the writer rather than
  the screened, evidence-bearing subset;
- Stage 17 has no deterministic `draft keys subset-of allowed keys` gate before
  persistence;
- Stage 17 currently pre-verifies the complete Stage 4 bibliography before
  drafting. In the observed run, that path attempted 843 entries and timed out
  with 629 entries marked SKIPPED;
- before E7, Stage 23 scored paper-level topical relevance from title and topic,
  not claim-specific support;
- before E7, Stage 23 removed citation markers for low-relevance keys from the
  manuscript. That could leave the surrounding claim unsupported and directly
  conflicted with the no-marker-only-repair invariant;
- before E7, Stage 23 verified the full bibliography. E7 bounds verification
  to the final manuscript's actual planned/cited keys.

This is primarily a workflow and gate-placement failure, not a Stage 19 bug.
Stage 19 remains fail-closed and must not be loosened.

## 3. Scope And Non-Goals

### 3.1 In scope

- Stage 4 citation-key identity and collision governance;
- Stage 5 screening provenance and explicit degradation;
- Stage 6 structured evidence cards and citation eligibility;
- Stage 16 citation policy, preliminary and final citation plans, and the v2
  full-text acquisition gate;
- Stage 17 prompt restriction and deterministic citation closure;
- Stage 18 review-policy consistency;
- Stage 20 quality-policy consistency;
- Stage 23 existence, metadata, and topical-relevance verification over actual
  citations only;
- Stage 24 claim-specific support and dataset-origin truth audit;
- read-only release replay over all authoritative artifacts.

### 3.2 Non-goals for v1

- downloading or retaining publisher PDFs;
- OCR;
- automatically rolling back to Stage 3;
- venue-specific citation profiles;
- semantic citation repair by free-form key substitution;
- treating LLM summaries as evidence;
- changing Stage 19 sectional-revision semantics.

### 3.3 Non-goals for v2 first implementation

- OCR or scanned-PDF support;
- bypassing access controls or automating institutional login;
- distributing PDFs or extracted full text in deliverables;
- treating a human-downloaded file as verified without deterministic identity
  checks;
- unbounded pause/resume or acquisition loops;
- supplement bundles or multiple-PDF source packages. F0 initially accepts one
  canonical primary text-native file per source identity; supplements require a
  later versioned bundle schema.

## 4. Non-Negotiable Invariants

1. `metadata` establishes identity or existence only. It never establishes
   claim support.
2. `summary_text` is a secondary LLM extraction used only for search and
   planning. It never enters support closure.
3. Only a byte/hash-bound `evidence_excerpt` from a retained canonical source
   may support a claim.
4. A template or fallback card may remain as a diagnostic artifact but may
   never enter the citation evidence set, including in `pipeline_validation`.
5. `pipeline_validation` is permission to validate engineering flow, not
   permission to fabricate evidence. Zero eligible evidence stops Stage 16/17.
6. Stage 5 is the semantic screening authority. Papers rejected or not
   successfully screened by Stage 5 cannot be reintroduced by citation count,
   ranking score, Stage 6, Stage 16, or writing prompts.
7. Citation identity eligibility is exactly the intersection of valid Stage 5
   shortlist entries, canonical bibliography entries, and valid Stage 6
   evidence cards. There is no top-up from rejected candidates.
8. Once `stage-04/references.bib` is hash-bound by the Stage 4 registry it is
   immutable. Filtered or preverified bibliography files are derived,
   non-authoritative artifacts and cannot be consumed by Stage 17, Stage 22,
   Stage 23, Stage 24, or release replay.
9. Stage 17 may cite only keys preassigned by the final Stage 16 citation plan.
10. Stage 23 verifies only keys actually cited by the manuscript. It does not
   prove claim-specific support.
11. Stage 24 independently evaluates claim-specific support using the claim,
    citation instance, and retained evidence excerpt.
12. Citation existence closure and claim-support closure are independent release
    gates. Passing either one cannot waive the other.
13. No stage may create BibTeX entries or aliases to make an unknown draft key
    appear valid.
14. No stage may remove a citation marker while retaining an unsupported claim
    and call the issue repaired.
15. Stage 19 consumes canonical citation artifacts but remains unchanged by
    this workstream.
16. All policy and provenance artifacts use strict, versioned schemas, reject
    unknown fields, and bind their canonical sources by SHA-256.
17. A non-release scope, degraded screening, fallback extraction, skipped
    verification, unavailable required full text, or incomplete support closure
    can never produce release-check exit 0.

## 5. Terminology And Trust Boundaries

### 5.1 Source identity

A source identity is the canonical bibliographic record selected by Stage 4.
For a scholarly paper, the preferred stable identity is normalized DOI, then
normalized arXiv ID, then provider paper ID, then a deterministic metadata
fingerprint. A citation key is a human-readable handle for that identity, not
the identity itself.

### 5.2 Eligible source

A v1 source is eligible only when all four conditions hold:

1. it appears in the valid Stage 5 shortlist;
2. its canonical key appears in the canonical Stage 4 `references.bib`;
3. Stage 6 produced a non-fallback card that passes its strict schema;
4. the card contains at least one abstract excerpt that can be reproduced from
   the hash-bound Stage 4 candidate record.

An eligible source may still fail to support a particular claim.

### 5.3 Canonical evidence

Canonical evidence is retained source text plus a reproducible locator. v1
permits abstract excerpts only. Metadata fields such as title, author, venue,
year, DOI, and URL remain existence evidence.

v2 adds canonical full text for supported source kinds. The PDF is the
provenance root, while normalized extracted text is the locator surface used by
claim excerpts.

### 5.4 Authoritative inputs

At minimum, release replay reads:

- the canonical Stage 9 experiment contract;
- canonical run/checkpoint state and the run-local active configuration
  snapshot used by Stage 16;
- Stage 4 candidates, bibliography, and cite-key registry;
- Stage 5 screening report and shortlist;
- Stage 6 cards, cards manifest, and citation allowlist;
- Stage 16 effective policy and final citation plan;
- Stage 17 final draft and closure report;
- Stage 18 reviews and review-policy report;
- Stage 20 quality report;
- Stage 23 verification report;
- Stage 24 claims, citations, truth-audit, and support artifacts;
- for v2, the full-text request, ingestion manifest, canonical KB store, and
  full-text evidence manifest.

The Stage 4 bibliography is immutable after the cite-key registry binds it.
Any `references_preverified.bib`, filtered bibliography, verified-only
bibliography, or Stage 22 copy is a derived view. No downstream stage or
release gate may silently choose a derived view in place of the registry-bound
Stage 4 file.

Derived Markdown cards, reading copies, deliverables, prompt prose, LLM
self-attestation, and stage status alone are non-authoritative.

This specification governs citation-backed claims. Uncited factual claims do
not become acceptable merely because they are outside the citation plan; they
remain subject to the existing fabrication, experiment-provenance, truth-audit,
and claims-provenance gates.

## 6. Configuration Contract

Add a top-level `citation_policy` configuration object. It is not nested under
`research`, because it is consumed across literature, writing, review, quality,
verification, truth-audit, and release stages.

```yaml
citation_policy:
  schema_version: 1
  min_unique_sources_research_release: 15
  target_unique_sources: 25
  min_unique_sources_pipeline_validation: 1
  require_fulltext_evidence: false
  max_fulltext_acquisition_rounds: 2
  reading_export_root: ""
```

Rules:

- `schema_version` must equal 1;
- all source counts are integers and exclude booleans;
- `min_unique_sources_research_release` defaults to 15 to preserve the current
  writing, review, and quality bar;
- `target_unique_sources` must be at least the research-release minimum;
- `min_unique_sources_pipeline_validation` must be at least 1;
- `require_fulltext_evidence` is false for v1 and ordinary pipeline validation;
- a future research-release full-text profile sets it to true;
- `max_fulltext_acquisition_rounds` is fixed at 2 for v2 and cannot be raised by
  a run artifact;
- `reading_export_root` is optional, non-authoritative, and never an input.

Effective policy is computed as follows:

```text
research_release:
  eligible_count < min_unique_sources_research_release -> FAILED
  effective_min = min_unique_sources_research_release
  effective_target = min(target_unique_sources, eligible_count)

pipeline_validation:
  eligible_count < min_unique_sources_pipeline_validation -> FAILED
  effective_min = min(min_unique_sources_research_release, eligible_count)
  effective_target = min(target_unique_sources, eligible_count)
```

`exploratory` follows the fail-closed pipeline-validation minimum but remains
non-release. A lower effective target never changes release eligibility.

## 7. Artifact Schemas

Every JSON object below uses a strict loader, rejects duplicate keys and
unknown fields, requires UTF-8, and carries `schema_version: 1` unless stated
otherwise.

### 7.1 Stage 4 cite-key registry

`stage-04/cite_key_registry.json` binds source identities to citation keys:

```json
{
  "schema_version": 1,
  "cite_key_version": 2,
  "candidates_path": "stage-04/candidates.jsonl",
  "candidates_sha256": "...",
  "references_path": "stage-04/references.bib",
  "references_sha256": "...",
  "entries": [
    {
      "source_identity": "doi:10.0000/example",
      "cite_key": "smith2024detection",
      "base_key": "smith2024detection",
      "collision_suffix": null
    }
  ]
}
```

Citation-key v2 rules:

1. normalize title using Unicode NFKC;
2. remove HTML/XML tags before token selection;
3. normalize the first author surname deterministically;
4. select the first non-stopword alphanumeric title token after normalization;
5. derive a source identity from normalized DOI, normalized arXiv ID, provider
   paper ID, or a versioned metadata fingerprint;
6. reuse the same key for the same source identity;
7. when distinct identities share a base key, append a deterministic suffix
   from the identity hash;
8. detect collisions across the complete Stage 4 collection before writing
   candidates or BibTeX;
9. do not migrate or silently reinterpret historical runs.

The strict registry loader requires `source_identity` to be unique across
entries and `cite_key` to be globally unique across distinct identities. Every
Stage 4 candidate row carries the same nonempty `source_identity`; duplicate
candidate identities are rejected or deterministically deduplicated before the
registry is sealed. Neither a JSONL line number nor list position is an
identity.

### 7.2 Stage 5 screening report

`stage-05/screening_report.json` records:

```json
{
  "schema_version": 1,
  "screening_policy_version": 1,
  "candidates_path": "stage-04/candidates.jsonl",
  "candidates_sha256": "...",
  "registry_path": "stage-04/cite_key_registry.json",
  "registry_sha256": "...",
  "references_path": "stage-04/references.bib",
  "references_sha256": "...",
  "screening_output_path": "stage-05/shortlist.jsonl",
  "screening_output_sha256": "...",
  "candidate_count": 200,
  "batch_size": 25,
  "max_screen_candidates": 150,
  "minimum_relevance_score": 0.5,
  "minimum_quality_score": 0.6,
  "claim_scope": "pipeline_validation",
  "prefilter_rejected_candidate_ids": ["..."],
  "selected_candidate_ids": ["..."],
  "screened_candidate_ids": ["..."],
  "semantic_duplicate_candidate_ids": ["..."],
  "unscreened_candidate_ids": ["..."],
  "batch_count": 6,
  "failed_batches": [],
  "screening_complete": true,
  "degraded": false,
  "degradation_codes": []
}
```

The shortlist contains only model decisions that passed strict parsing and
candidate-ID closure. It never contains fabricated scores or rows inserted to
reach a numeric minimum. The prefilter, screened, and unscreened identity sets
form a complete, non-overlapping partition of Stage 4 candidates. Candidates
outside the deterministic top-150 admission bound are prefilter rejections,
not unscreened rows. `selected_candidate_ids` must exactly match the recorded
screening output; the report binds both Stage 4 candidates and that output by
SHA-256.
Before screening, Stage 5 revalidates candidates and the canonical bibliography
against the sealed Stage 4 registry. The report binds all three Stage 4 identity
artifacts, so an edited candidate row or shadow bibliography cannot enter E1.

`shortlist.jsonl` is success-only. Every Stage 5 failure writes decisions, if
any, to `screening_partial.jsonl` and sets `screening_output_path` accordingly;
it never leaves a canonical shortlist that a later `--from-stage` run can
consume. Stage 4, Stage 5, and Stage 6 are evidence-authority stages and cannot
appear in `runtime.skip_stages`.

### 7.3 Stage 6 evidence card

Authoritative cards are JSON:

`stage-06/cards/<card_id>.json`

```json
{
  "schema_version": 1,
  "card_id": "card-001",
  "source_identity": "doi:10.0000/example",
  "cite_key": "smith2024detection",
  "title": "...",
  "extraction_status": "success",
  "fallback_reason": null,
  "summary_text": {
    "problem": "...",
    "method": "...",
    "data": "...",
    "metrics": "...",
    "findings": "...",
    "limitations": "..."
  },
  "evidence_excerpts": [
    {
      "excerpt_id": "ev-...",
      "source_type": "abstract",
      "source_artifact_path": "stage-04/candidates.jsonl",
      "source_artifact_sha256": "...",
      "source_record_id": "doi:10.0000/example",
      "json_pointer": "/abstract",
      "char_start": 120,
      "char_end": 286,
      "excerpt_text": "...",
      "excerpt_sha256": "..."
    }
  ]
}
```

`char_start` and `char_end` are Python Unicode-code-point offsets over the exact
decoded Stage 4 abstract value. The excerpt must equal the indicated slice, and
its UTF-8 SHA-256 must match `excerpt_sha256`. Policy v1 requires every excerpt
to contain at least 25 Unicode code points; shorter strings cannot make a card
eligible.

`source_record_id` equals the candidate row's canonical `source_identity`. It
is never a JSONL line number or ordinal. The Stage 4 candidates loader requires
it to identify exactly one record, so reordering the JSONL cannot move an
excerpt to another source.

`extraction_status` is exactly one of `success`, `fallback`, or `failed`.
Fallback and failed cards have no eligible evidence. The producer writes this
status; consumers do not infer success from text length, entropy, or template
similarity.

`stage-06/cards/<card_id>.md` is a deterministic human-readable rendering of
the JSON card. It is never independently generated and never authoritative.

`stage-06/cards_manifest.json` binds every JSON card, every derived Markdown
view, the shortlist, the validated Stage 5 screening report, and the renderer
version:

```json
{
  "schema_version": 1,
  "shortlist_path": "stage-05/shortlist.jsonl",
  "shortlist_sha256": "...",
  "screening_report_path": "stage-05/screening_report.json",
  "screening_report_sha256": "...",
  "renderer_version": 1,
  "cards": [
    {
      "card_id": "card-001",
      "source_identity": "doi:10.0000/example",
      "cite_key": "smith2024detection",
      "json_path": "stage-06/cards/card-001.json",
      "json_sha256": "...",
      "markdown_path": "stage-06/cards/card-001.md",
      "markdown_sha256": "..."
    }
  ]
}
```

Card IDs, source identities, cite keys, JSON paths, and Markdown paths are each
unique. The renderer recomputes each Markdown view from its JSON card before
hash comparison. Extra, missing, nested, symlinked, or unmanifested files fail
validation.

Stage 6 revalidates the sealed Stage 4 citation collection and the complete
Stage 5 shortlist/report binding before any extraction call. It publishes
`cards/` and `cards_manifest.json` only when at least one card contains a
replayable evidence excerpt. A zero-evidence failure may write
`card_extraction_failures.json` for diagnosis, but it must not leave either
canonical Stage 6 artifact for a later `--from-stage` run to consume.
Stage 7 must replay the manifest, JSON-card, excerpt, and deterministic Markdown
bindings before reading any Markdown view into the synthesis prompt.

A failed Stage 6 extraction batch produces non-evidentiary `failed` cards but
does not by itself fail a strict-scope run when other cards succeed. E3 owns the
scope-specific minimum eligible-evidence count and fails insufficient runs.

### 7.4 Citation allowlist

`stage-06/citation_allowlist.json` contains only eligible sources:

```json
{
  "schema_version": 1,
  "eligibility_policy_version": 1,
  "shortlist_path": "stage-05/shortlist.jsonl",
  "shortlist_sha256": "...",
  "references_path": "stage-04/references.bib",
  "references_sha256": "...",
  "cards_manifest_path": "stage-06/cards_manifest.json",
  "cards_manifest_sha256": "...",
  "eligible_keys": ["smith2024detection"],
  "ineligible": [
    {
      "cite_key": "...",
      "reason_code": "card_fallback"
    }
  ]
}
```

The release audit rebuilds the allowlist from canonical Stage 4-6 artifacts. It
does not trust `eligible_keys` as a self-assertion.

### 7.5 Effective citation policy

`stage-16/citation_policy_effective.json` records the cross-stage target:

```json
{
  "schema_version": 1,
  "policy_version": 1,
  "claim_scope": "pipeline_validation",
  "eligible_count": 9,
  "effective_min_unique_sources": 9,
  "effective_target_unique_sources": 9,
  "citation_allowlist_path": "stage-06/citation_allowlist.json",
  "citation_allowlist_sha256": "...",
  "config_source_path": "config.yaml",
  "config_source_sha256": "..."
}
```

Stage 17, Stage 18, and Stage 20 consume the exact same artifact. Each stage
revalidates its source hashes and recomputes the effective values.

`config_source_path` is always a safe path to the configuration snapshot inside
the run root, never a repository or user-supplied external path. For an
uninterrupted run it is exactly `config.yaml`. On resume, the runner first
persists `config.resumed-<timestamp>.yaml` and records that exact active
snapshot path in canonical run/checkpoint state before re-entering a stage.
Stage 16 binds the snapshot recorded for the attempt that executes Stage 16;
it does not select a file by directory ordering or by trusting this artifact's
path alone. Release replay cross-checks the path against canonical run history
and then re-hashes the run-local snapshot. Later edits to the repository config
are irrelevant, while edits to the bound run snapshot are detected.

The CLI records this selection in `active_config_snapshot.json` and appends one
strict JSONL event to `config_snapshot_history.jsonl`. The active pointer binds
the history path, exact history SHA-256, contiguous final ordinal, snapshot
path, and snapshot SHA-256. If any resumed snapshot or history exists without
the active pointer, policy construction fails rather than selecting by filename
ordering. An uninterrupted programmatic run may use `config.yaml` without a
pointer only when no resume snapshot or history artifact exists.
When `checkpoint.json` exists, the CLI atomically records the active snapshot
path/hash and history hash there before resume, and every later checkpoint
rewrite preserves those fields. Stage 16 requires the checkpoint binding to
match the pointer and history exactly.

### 7.6 Preliminary and final citation plans

Stage 16 writes `citation_plan.preliminary.json` before any v2 acquisition
pause and `citation_plan.json` after evidence requirements are satisfied.

Each planned claim contains:

```json
{
  "claim_id": "planned-claim-001",
  "section_path": ["Introduction"],
  "claim_text": "...",
  "claim_type": "background",
  "planned_citations": [
    {
      "cite_key": "smith2024detection",
      "evidence_excerpt_ids": ["ev-..."],
      "support_status": "abstract_sufficient"
    }
  ]
}
```

Rules:

- every key belongs to the citation allowlist;
- every excerpt ID belongs to that key's authoritative card;
- `claim_text` is planning text, not permission for Stage 17 to invent a
  stronger statement;
- v1 `support_status` is `abstract_sufficient` or `unsupported`;
- a research-release v2 plan marks every scholarly-paper key as
  `fulltext_required` until verified canonical full text exists;
- the final plan cannot contain `unsupported` or unmet `fulltext_required`
  entries;
- no key may be added during Stage 17 writing.

### 7.7 Stage 17 citation-closure report

`stage-17/citation_closure_report.json` binds the final post-HITL draft:

```json
{
  "schema_version": 1,
  "paper_path": "stage-17/paper_draft.md",
  "paper_sha256": "...",
  "citation_plan_path": "stage-16/citation_plan.json",
  "citation_plan_sha256": "...",
  "cited_keys": ["..."],
  "unknown_keys": [],
  "unplanned_keys": [],
  "missing_planned_keys": [],
  "misplaced_planned_keys": [],
  "structure_report_path": "stage-17/paper_structure_report.json",
  "structure_report_sha256": "...",
  "structure_valid": true,
  "experiment_fact_closure_report_path": "stage-17/experiment_fact_closure_report.json",
  "experiment_fact_closure_report_sha256": "...",
  "experiment_fact_closure_valid": true,
  "valid": true
}
```

The two booleans are derived summaries, not self-asserted gates. Structure is
recomputed through the CommonMark-aware Stage 17 structure validator.
Experiment-fact closure is recomputed from a strict report that binds the
Stage 12-14 grounded metric whitelist, Stage 9 dataset origin, final draft, and
the deterministic numeric/dataset-claim checks. E5 owns that report schema and
release replay recomputes both results from their canonical sources.

v1 has no citation auto-repair. Unknown or unplanned keys fail Stage 17.
Planned keys must also occur in the exact planned top-level section; a key
that appears elsewhere in the paper is reported as `misplaced_planned_keys`
and does not satisfy the plan.

A later bounded-regeneration version may write
`stage-17/citation_regeneration_log.json`, but only after the final citation
plan exists. It may regenerate an affected sentence using keys preassigned to
that claim, or delete/rewrite the complete unsupported claim. It may not freely
substitute keys or remove a marker while retaining the claim. It runs once,
records before/after hashes and actions, and then reruns citation closure, plan
consistency, manuscript structure, and experiment-fact closure. A second
failure is final.

### 7.8 Full-text acquisition request and history

v2 writes one immutable request per acquisition round:

`stage-16/fulltext_acquisition_request.round-<n>.json`

```json
{
  "schema_version": 1,
  "run_id": "...",
  "acquisition_round": 1,
  "status": "awaiting_user_download",
  "previous_request_path": null,
  "previous_request_sha256": null,
  "experiment_contract_path": "stage-09/experiment_contract.yaml",
  "experiment_contract_sha256": "...",
  "preliminary_citation_plan_path": "stage-16/citation_plan.preliminary.json",
  "preliminary_citation_plan_sha256": "...",
  "inbox_path": "fulltext_inbox/<run_id>/",
  "items": [
    {
      "source_identity": "doi:10.0000/example",
      "cite_key": "smith2024detection",
      "source_kind": "scholarly_paper",
      "title": "...",
      "doi": "10.0000/example",
      "authors": ["..."],
      "year": 2024,
      "required_for_claim_ids": ["planned-claim-001"],
      "reason": "Research-release scholarly citation requires verified full text",
      "suggested_filename": "smith2024detection.pdf"
    }
  ]
}
```

The inbox path is relative to the configured knowledge-base root, not the run
directory or repository.

Round 2 sets `previous_request_path` to the exact round-1 request and binds its
SHA-256. Round 1 requires both previous fields to be null. Requests are
append-only audit artifacts: Stage 16 never overwrites, renames, or removes a
prior-round request.

`stage-16/fulltext_acquisition_history.jsonl` appends one strict record for
every PAUSED request and every resume/ingestion result. Each record binds the
request path/hash, event type, round number, and previous event hash. The final
Stage 16 plan and run manifest bind the exact history-text SHA-256. Release
replay enumerates request files directly, parses the history, requires rounds
to be consecutive from 1, rejects gaps/duplicates/unreferenced files, and
requires the maximum round to be at most 2. The round count is derived from
retained history, never trusted from the newest request alone.

### 7.9 Full-text ingestion manifest

After each resume, Stage 16 writes an immutable
`fulltext_ingest_manifest.round-<n>.json`. After convergence it writes
`fulltext_ingest_manifest.json` as a strict aggregate that binds every
round-specific request and ingestion manifest:

```json
{
  "schema_version": 1,
  "acquisition_history_path": "stage-16/fulltext_acquisition_history.jsonl",
  "acquisition_history_sha256": "...",
  "round_manifests": [
    {
      "round": 1,
      "request_path": "stage-16/fulltext_acquisition_request.round-1.json",
      "request_sha256": "...",
      "ingest_path": "stage-16/fulltext_ingest_manifest.round-1.json",
      "ingest_sha256": "..."
    }
  ],
  "source_kind_policy_version": 1,
  "extractor_policy_version": 1,
  "items": [
    {
      "source_identity": "doi:10.0000/example",
      "cite_key": "smith2024detection",
      "source_kind": "scholarly_paper",
      "pdf_sha256": "...",
      "canonical_pdf_kb_path": "fulltext_store/sha256/ab/abcdef....pdf",
      "identity_status": "verified",
      "identity_method": "doi_exact",
      "identity_evidence": {
        "identifier": "10.0000/example",
        "identifier_locator": "pdf_metadata:doi",
        "title_locator": "page-1:title-block",
        "author_locator": "page-1:author-block"
      },
      "extractor_name": "pymupdf",
      "extractor_version": "...",
      "normalization_version": 1,
      "canonical_text_kb_path": "fulltext_store/sha256/ab/abcdef....txt",
      "canonical_text_sha256": "...",
      "extraction_status": "success"
    }
  ]
}
```

The manifest records KB-relative paths. Release replay resolves them only under
the configured KB root, rejects traversal and symlink escape, and re-hashes the
canonical files.

`identity_status` is exactly `verified`, `version_mismatch`, or `unverified`.
Only `verified` items with `extraction_status: success` can enter canonical
evidence. The aggregate also records the acquisition-history path/hash and the
ordered round-manifest paths/hashes; it cannot omit an earlier failed or partial
round.

## 8. Stage Algorithms And Failure Semantics

### 8.1 Stage 4: identity before presentation

Stage 4 constructs the full source-identity registry before writing candidate
rows or BibTeX. Candidate and BibTeX keys are projections from that registry.
It does not generate each key independently and discover collisions later.

After the registry, candidates, and `references.bib` are written and mutually
hash-bound, the Stage 4 bibliography is immutable for the run. Verification
results are separate reports; they do not rewrite the canonical bibliography.

Malformed titles, missing authors, missing years, identity collisions, and
duplicate DOI/provider identities are reported explicitly. No historical key
alias is created automatically.

E0 deliberately renders canonical BibTeX from sealed candidate metadata rather
than trusting `Paper._bibtex_override`, because an override carries an
independently generated key and would reopen the dual-source identity problem.
This may omit nonessential CrossRef fields such as pages or publisher in v1.
A later preservation layer may parse an override body, reject structural
injection, replace only its entry key with the sealed key, and prove semantic
field equivalence before retaining extra fields.

The metadata-fingerprint fallback cannot prove that two same-title,
same-first-author, same-year records are distinct works. This is a documented
last-resort identity limitation; such records are not silently separated by
LLM judgment. E1 must also treat DOI-identified and arXiv-only records as
potential semantic duplicates during screening without changing their sealed
Stage 4 identities.

### 8.2 Stage 5: bounded screening without fabricated backfill

Stage 5 must not ask one model response to screen the complete candidate
collection. Screening uses versioned, deterministic pre-ranking, bounded
batches, candidate IDs, strict response schemas, and at most one response
repair per batch. Batch size and maximum screened-candidate count are policy
constants recorded in `screening_report.json`. Policy v1 uses batches of 25
and admits at most 150 candidates after deterministic keyword prefiltering and
ranking. Lower-ranked candidates are rejected by the versioned deterministic
policy and cannot be used for minimum-count supplementation. Policy v1 requires
`relevance_score >= 0.5`; the configured 0-10 research quality threshold is
normalized to 0-1. A `keep`/`reject` decision that contradicts either score
threshold is a schema failure, not an accepted model judgment.

The deterministic merge checks:

- every response ID exists in the requested batch;
- every requested ID has exactly one decision;
- no ID appears in two batches;
- scores and reasons use the strict schema;
- selected IDs preserve the configured deterministic final ordering.
- DOI and arXiv identities with the same normalized title, first author, and
  year are treated as potential semantic duplicates; only the highest-ranked
  screened decision survives, without changing either Stage 4 identity.

There is no minimum-size supplementation from rejected or unparsed candidates.

For `research_release` and `exploratory`, any failed selected batch or
incomplete screening is a stage failure. Only `pipeline_validation` may retain
successfully screened rows as degraded diagnostics, and only their valid
decisions can flow downstream. Zero valid selected rows still fails Stage 5.

Before E9, `screening_report.json` is diagnostic provenance rather than release
authority. E9 must recompute the deterministic prefilter, ordering, top-150
admission, and batch partition from canonical Stage 4 artifacts plus the
run-local config snapshot. No `research_release` candidate run may be started
until that replay gate and its claim-scope cross-check are implemented.

v1 does not automatically roll back to Stage 3. Failure output contains
structured search-expansion recommendations. A later state-machine change may
permit at most two bounded rollbacks to Stage 3, never Stage 4-only reruns with
unchanged queries.

### 8.3 Stage 6: structured extraction and deterministic excerpts

Stage 6 uses one paper or a small fixed batch per model response. The model may
propose summaries and abstract excerpts. Deterministic code validates that
every excerpt is an exact slice of the retained abstract and writes the
provenance locator.

Failed parsing, missing fields, fabricated excerpts, empty abstracts, and
template fallback produce `fallback` or `failed` cards, never eligible cards.
The pipeline does not infer fallback from prose heuristics.

### 8.4 Stage 16: plan, acquire, then finalize

Stage 16 performs these operations in order:

1. load and recompute the citation allowlist;
2. compute and persist effective citation policy;
3. generate and strictly validate the preliminary citation plan;
4. in v1, finalize the plan using abstract evidence only;
5. in v2 research release, classify source kinds and identify required
   canonical full text;
6. check the content-addressed KB store before requesting downloads;
7. when required sources remain absent, write the acquisition request and
   return `StageStatus.PAUSED` with reason
   `fulltext_acquisition_required`;
8. on resume, ingest and verify inbox files, rebuild evidence, and generate the
   final plan;
9. fail if the final plan cannot close over existing verified evidence.

Acquisition is bounded to two pause/resume rounds per run. The second resumed
round must converge using retained evidence. A third request is forbidden and
the stage fails with `fulltext_acquisition_rounds_exhausted`.

F0 includes executor/runner support for a stage implementation returning
`StageStatus.PAUSED`. The runner must persist the request and stage state, count
the stage as paused rather than failed or completed, stop before Stage 17, and
make CLI resume re-enter the same Stage 16 attempt. The existing enum and
transition table are necessary but not sufficient; current HITL timeout pauses
do not by themselves implement this stage-result path.

Stage 5 may produce a non-blocking preview of likely full-text needs, but the
formal request is generated only from the Stage 16 preliminary plan.

### 8.5 Stage 17: closed writing surface

Stage 17 receives only:

- final planned claims;
- allowed keys assigned to each claim;
- bounded summaries for planning context;
- exact evidence excerpts for grounding;
- experiment-fact and dataset-origin constraints.

It does not receive Stage 4 `candidates.jsonl` as a free citation catalog.

The current Stage 17 full-bibliography preverification path is retired in E5.
It is not redirected to produce a filtered authoritative bibliography. E7
replaces it with bounded verification of actual manuscript keys. Stage 17,
Stage 22, and Stage 23 continue to bind the immutable Stage 4 bibliography.
Stage 22 and Stage 23 additionally replay the Stage 6 allowlist and final
Stage 16 plan against the manuscript they actually consume. A key that exists
in the full Stage 4 bibliography but is absent from the evidence-bound
allowlist or final plan is rejected. Direct `--from-stage` entry does not skip
this replay.

After all LLM writing and HITL guidance but before persistence, deterministic
validation requires:

1. every draft key belongs to the allowlist;
2. every draft key is assigned to the relevant planned claim;
3. every required planned citation is accounted for;
4. manuscript structure remains valid;
5. experiment and dataset-origin facts remain closed over canonical artifacts.

Experiment-fact closure uses the promoted root experiment summary when one
exists, otherwise direct `stage-14/experiment_summary.json`, otherwise flat
JSON results under direct `stage-12/runs/`. Version-glob or similarly named
shadow stage directories are not metric authorities. Percent scaling is
allowed only when the manuscript token explicitly carries `%`; arbitrary
value-to-source `x100` matching is forbidden. Decimal, scientific,
percentage, and unit-bearing integer metrics are checked in Abstract,
Results, Experiment/Evaluation/Ablation, Discussion, and Conclusion sections.

v1 fails immediately on unknown or unplanned keys. It has no aliasing,
auto-BibTeX, or semantic repair.

### 8.6 Stage 18: review against the effective policy

Stage 18 reads and revalidates `citation_policy_effective.json`. The prompt
states the effective minimum and target and forbids the reviewer from imposing
a higher citation-count requirement.

After response parsing, a deterministic validator inspects only citation-count
requirements in `Actionable Revisions`. It uses a versioned, closed vocabulary
for citation/reference terms and numeric literals/number words. It does not try
to detect general semantic policy conflicts.

If an actionable citation-count requirement exceeds the effective target, the
review artifact is invalid. Stage 18 may regenerate the review once with the
exact conflict. A second conflict fails Stage 18. The comment never reaches
Stage 19 and is never converted to `not_actionable`.

### 8.7 Stage 20: quality against the same policy

Stage 20 reads the same effective-policy artifact, re-hashes its sources, and
recomputes all counts. It does not retain a separate hard-coded minimum of 15
for pipeline-validation runs.

Before any quality-model call, Stage 20 extracts citations from the actual
revised manuscript and counts only keys that remain in both the validated
Stage 6 allowlist and final Stage 16 plan. Counts below the effective minimum,
or padding with ineligible/unplanned keys, fail deterministically. The target
is guidance rather than an additional hard minimum.

For research release, the effective minimum remains 15 by construction. A
lower target in a non-release scope is diagnostic only and cannot make the run
release eligible.

### 8.8 Stage 23: verify actual citations only

Stage 23 extracts the final manuscript's unique citation keys and verifies that
bounded set. It does not verify every Stage 4 candidate or every unused BibTeX
entry.

Its responsibilities are:

- existence and bibliographic metadata;
- canonical key identity;
- paper-level topical relevance;
- verification completeness.

It does not establish claim support. Unscored, skipped, timed-out, missing, or
malformed results are not assigned a permissive default score in a
research-release run.

Stage 23 reports invalid citations and fails the relevant release path. It must
not silently delete a marker from the manuscript while retaining the claim.
E7 explicitly removes the former low-relevance marker-deletion behavior; a
verification report is not a text-repair authority.

The canonical audit input is exactly `stage-22/paper_final.md`; prior-artifact
search and same-name shadow files are forbidden. Stage 23 cleans its owned
outputs before reading that input. Its verified-paper artifact is a byte-exact
copy of the Stage 22 manuscript, including on failure, so the report cannot hide
an invalid citation by mutating the text.

`pipeline_validation` may complete as degraded when verification is suspicious,
skipped, unscored, topically low-relevance, or the bounded relevance response is
malformed. Hallucinated citations fail every scope. `exploratory` and
`research_release` fail on every incomplete or invalid verification condition.

### 8.9 Stage 24: support and dataset-origin closure

Stage 24 binds each citation instance to:

- a claim ID;
- the exact claim text and manuscript context;
- the canonical cite key;
- one or more retained evidence excerpts;
- an isolated support assessment;
- the source artifact and excerpt hashes.

The semantic assessment may use an isolated critic, but deterministic code
owns identity, excerpt reproduction, path, hash, claim/citation linkage, and
completeness. Unsupported or unassessed citation instances remain unsupported.

Stage 24 also compares claims against the Stage 9 dataset origin. Synthetic
runs may not claim public-dataset or local-hardware measurements. Prompt
instructions in Stage 17 are preventive; Stage 24 is the authoritative audit.

E8 writes `stage-24/citation_support.json` as the support authority. Every
final-paper citation occurrence becomes a separate obligation bound to the
canonical final plan, the planned retained excerpt IDs, the Stage 4 source
artifact hash and span, and the exact sentence containing the citation. The
isolated critic receives only that sentence, its bounded local context, the
planned claim, and the retained excerpts. It may return only
`supported|unsupported` plus a reason; it cannot set IDs, paths, hashes,
existence status, or completeness.

Only Stage 23 status `verified` is eligible for a supported verdict. Missing,
suspicious, hallucinated, or skipped existence status deterministically forces
`unsupported`, irrespective of critic output. No critic, malformed critic
output, any unsupported instance, or any dataset-origin contradiction makes
Stage 24 fail in every claim scope. `citations.json` is derived from this
closure: unsupported instances remain `unmapped` and cannot be relabeled as
background. Stage 24 fixes its paper input to
`stage-23/paper_final_verified.md`, checks byte identity with Stage 22, and
cleans all owned outputs before starting.

E8's hard failure surface is intentionally limited to citation-instance support
and dataset-origin truth. Unsupported non-citation rows in the LLM-extracted
claim ledger remain diagnostic at Stage 24; quantitative experiment facts are
already owned by deterministic Stage 17 closure, while E9 and release checking
independently replay the complete claim-provenance policy. The LLM claim ledger
must not become a new self-authorizing release gate.

## 9. Source-Kind Policy

`source_kind_policy_version: 1` classifies citations deterministically from
canonical bibliography type and normalized identifiers/URLs.

Supported kinds include:

- `scholarly_paper`;
- `standard_or_rfc`;
- `dataset`;
- `software_repository`;
- `book_or_chapter`.

Research-release canonical-source requirements are:

| Source kind | Canonical retained source |
|---|---|
| scholarly paper | verified text-native PDF plus canonical extracted text |
| standard or RFC | official HTML/PDF snapshot plus hash |
| dataset | official data card or dataset documentation snapshot plus hash |
| software repository | repository URL plus immutable commit and retained source/docs |
| book or chapter | lawful retained excerpt with edition and page locator |

Ambiguous or unsupported classifications default to `scholarly_paper`, the
strictest ordinary requirement. Source kind cannot be supplied or overridden
by the writer. Any future manual correction is a separately governed HITL
artifact and cannot waive canonical-source retention.

## 10. Full-Text HITL And Canonical KB Store

### 10.1 Inbox and store

The user downloads requested files to:

```text
<knowledge_base.root>/fulltext_inbox/<run_id>/
```

The inbox is a staging area, not the provenance root. After identity and
extraction checks, the pipeline copies verified content into a content-addressed
store:

```text
<knowledge_base.root>/fulltext_store/sha256/<first-two>/<sha256>.pdf
<knowledge_base.root>/fulltext_store/sha256/<first-two>/<sha256>.txt
```

The store is the only canonical machine-side copy. It is not committed to Git
and is excluded from public deliverables. A later run checks the store by source
identity and hash before requesting another download.

### 10.2 Identity verification

Identity verification is deterministic and ordered:

1. when the candidate has a DOI, the PDF must expose the same normalized DOI in
   a versioned document-identity locator and must also match normalized title
   and first author;
2. when the candidate identity is an arXiv ID, the PDF must expose the same
   normalized arXiv identity under the version policy recorded by the registry
   and must also match normalized title and first author;
3. only when the candidate itself has neither DOI nor arXiv identity may
   normalized-title exact match plus first-author and year agreement establish
   identity;
4. a title/author/year match to a different DOI, arXiv preprint, accepted
   manuscript, version, or version of record is `version_mismatch`, not
   `verified`;
5. all remaining cases are `unverified`.

An identifier found only in an arbitrary body/reference-list occurrence is not
a document identity. The identity extractor records the PDF metadata or bounded
front-matter locator from which each document identifier and title/author value
was obtained. A wrong paper that merely cites the requested DOI cannot pass.

Preprint and version-of-record files are distinct source identities unless the
Stage 4 registry explicitly records a deterministic version relationship. That
relationship does not make their text interchangeable: the requested identity
must still be the identity of the retained evidence file.

An LLM may not assert that a file "looks like" the requested paper. Unverified
files do not enter the store as evidence.

### 10.3 Extraction

The v2 first implementation accepts text-native PDFs only. Extraction runs with
network disabled, resource limits, and a pinned extractor/version. The
normalization algorithm is versioned. The canonical locator surface is the
normalized extracted-text artifact, not PDF byte offsets.

Scanned, malformed, encrypted, or text-empty PDFs produce
`fulltext_extraction_unsupported`. The user must supply a searchable version.
OCR is a later, separately reviewed evidence layer whose output becomes a new
versioned canonical text artifact.

### 10.4 Reading copies

After successful ingestion, an optional deterministic export copies verified
PDFs from the KB store to a configured writing workspace using `<cite-key>.pdf`.
The direction is always KB store to writing directory. The pipeline never reads
the writing copy and never uses it for hashes, identity, excerpts, resume, or
release audit.

Reading copies are disposable human conveniences. Their cleanup is a human
action after paper completion; the pipeline does not delete them automatically.
Deleting them never changes provenance because the KB store remains canonical.

## 11. Pause, Resume, And Stale-Artifact Rules

`fulltext_acquisition_required` is a PAUSED state, not FAILED. A paused run:

- has a complete acquisition request;
- is explicitly not release-ready;
- does not package PDFs or extracted full text;
- does not claim requested sources are verified;
- resumes the same Stage 16 gate rather than skipping to Stage 17.

If generic packaging runs on pause, its deliverables manifest must state
`release_ready: false`, `not_release_ready: true`, and include
`fulltext_acquisition_required` as a blocker. A paused request is never a
successful stage output for downstream scheduling.

Resume is idempotent. Before each request or ingestion attempt, Stage 16 removes
or atomically replaces only its owned derived artifacts. It never deletes user
inbox files or canonical KB-store objects.

The append-only acquisition history, every
`fulltext_acquisition_request.round-<n>.json`, and every corresponding
round-specific ingestion manifest are retained audit records and are explicitly
excluded from stale-output cleanup. Only recomputable final/aggregate views may
be atomically replaced. Missing earlier round files, hash-chain breaks, or a
round-2 request without round 1 are fatal.

The acquisition request binds the preliminary plan and experiment contract.
Changing either while paused invalidates the request. Files not requested by
the bound request are not silently ingested into that run.

When a source remains unavailable:

1. record `fulltext_unavailable`;
2. remove it from the final citation plan;
3. delete or narrow every dependent planned claim;
4. recompute citation sufficiency;
5. fail if the scope's evidence minimum is no longer met.

There is no abstract-only fallback when the active policy requires full text.

## 12. Release Semantics

### 12.1 v1 abstract-only

An abstract-only research-release candidate may support only claims directly
grounded in retained abstract excerpts. It cannot use metadata or card summaries
to support detailed methods, measurements, or conclusions absent from the
abstract.

Passing v1 does not imply that the system has full-paper evidence. Deliverables
and release metadata must identify the evidence policy version.

### 12.2 v2 full text

When `require_fulltext_evidence: true`, every actual scholarly-paper citation
must bind to verified canonical full text. Other source kinds must bind to their
required canonical retained source.

Missing, unverified, unsupported, unavailable, or stale canonical evidence is a
release error. No waiver converts existence-only evidence into claim support.

### 12.3 Exit codes

All new citation/evidence errors remain incompatible with the degraded exit-2
allowlist. Any such error yields exit 1, even when degradation signals also
exist. Pipeline-validation runs remain non-release regardless of citation
closure.

Required error families include:

- `citation_key_registry_*`;
- `literature_screening_*`;
- `evidence_card_*`;
- `citation_allowlist_*`;
- `citation_policy_*`;
- `citation_plan_*`;
- `citation_closure_*`;
- `review_citation_policy_*`;
- `citation_verification_*`;
- `citation_support_*`;
- `fulltext_acquisition_*`;
- `fulltext_identity_*`;
- `fulltext_extraction_*`;
- `canonical_evidence_*`;
- `dataset_origin_claim_*`.

Exact codes are defined in the implementation milestone that owns each schema.
They may not be added to the degraded-compatible set.

The existing `allow_suspicious` citation setting cannot waive any new
citation-identity, evidence, plan, support, full-text, or canonical-source error.
E9 extends the existing `check_citation_support` and
`check_claims_provenance` release gates and their artifact semantics; it does
not create parallel, independently interpreted support/provenance gates.

## 13. Migration And Consumer Survey

Before changing cards, run a repository-wide consumer survey and record every
read of `cards/`, `shortlist.jsonl`, `candidates.jsonl`, `references.bib`, and
hard-coded citation-count thresholds.

Migration uses JSON as the single authority and Markdown as deterministic
derived output:

1. add JSON cards and strict loaders;
2. render existing Markdown from JSON;
3. migrate evidence consumers to JSON first;
4. migrate synthesis/outline consumers one at a time;
5. keep compatibility tests while any Markdown consumer remains;
6. never permit independent JSON and Markdown generation.

Historical runs are not rewritten. Release gates require the new artifacts only
after the corresponding migration commit explicitly activates them.

## 14. Implementation Milestones And Commit Boundaries

Each item is independently reviewable and must not loosen Stage 19 or existing
release gates.

Before E0, perform and record the read-only consumer survey required by
Section 13. It is not an implementation commit, but its result is a prerequisite
for finalizing E2/E3 interfaces and prevents an unobserved Markdown-card or
shadow-bibliography consumer from becoming a second authority.

### E0. Citation-key v2

- NFKC and HTML/XML title cleaning;
- source-identity registry;
- stable candidate `source_identity` / `source_record_id` semantics;
- deterministic collision suffix;
- candidate/BibTeX consistency;
- no historical migration;
- collision and malformed-title tests.

### E1. Stage 5 degradation provenance

- bounded screening protocol;
- strict candidate-ID closure;
- no template shortlist or fabricated scores;
- screening report;
- explicit scope-specific failure semantics.

### E2. Stage 6 evidence cards

- JSON schema and strict loader;
- exact abstract excerpts and locators;
- explicit extraction status;
- cards manifest;
- deterministic Markdown renderer;
- consumer inventory.

### E3. Citation allowlist and effective policy

- top-level config;
- run-local active config snapshot and canonical run-history binding;
- recomputed Stage 4-6 eligibility;
- effective Stage 16 policy artifact;
- Stage 17/18/20 source-hash binding.

### E4. Stage 16 v1 citation plan

- preliminary/final abstract-only plans;
- strict claim/key/excerpt closure;
- deterministic v1 planning: one bounded background claim per selected key,
  using the first retained excerpt verbatim as the wording ceiling; the LLM
  does not choose keys, excerpt IDs, paths, or support status;
- no full-text acquisition;
- controlled synthetic evidence-pack fixture.

### E5. Stage 17 closed writing surface

- remove free Stage 4 candidate catalog from writer input;
- final-plan-only prompt;
- post-HITL deterministic closure;
- retire Stage 17 full-bibliography preverification and every filtered-bib
  authority path;
- require Stage 22/23 to consume the registry-bound Stage 4 bibliography;
- v1 unknown-key hard failure;
- no repair.

### E6. Stage 18 and Stage 20 policy consistency

- deterministic citation-count requirement detection in Actionable Revisions;
- one bounded review regeneration;
- shared effective target;
- no `not_actionable` escape.

### E7. Stage 23 bounded verification

- verify actual cited keys only;
- no permissive unscored default for release;
- no silent citation deletion;
- bind the audit to canonical `stage-22/paper_final.md` and clear stale outputs;
- require exact verifier-result and relevance-response key/count closure;
- permit incomplete verification only as `pipeline_validation` degradation;
- separate existence/topic result from support;
- land verification/gate changes in an independent gate commit.

### E8. Stage 24 support and dataset-origin closure

- citation-instance-to-claim binding;
- retained excerpt verification;
- isolated semantic support assessment;
- unsupported closure failure;
- dataset-origin truth audit.
- strict `citation_support.json` schema with source/span/hash replay fields;
- only `verified` existence status may receive supported closure;
- no background or no-critic escape in any scope.

### E9. Release replay

- strict loaders for every artifact;
- canonical-source hash recomputation;
- independent allowlist/plan/closure reconstruction;
- existence and support gates remain separate;
- extend existing citation-support and claims-provenance gates rather than
  building parallel interpretations;
- prove `allow_suspicious` cannot waive new errors;
- snapshot-test the degraded-compatible error allowlist;
- new errors always exit 1.

E9 uses one package-owned disk auditor, `citation_release_audit.py`, rather
than reimplementing policy inside `release_check.py`. Starting from the active
run-local config and canonical Stage 9 contract, it replays the Stage 4 registry,
Stage 5 deterministic prefilter/ranking/top-150 admission, evidence cards,
allowlist, effective policy, final plan, Stage 17 experiment/citation closure,
final-paper citation closure, Stage 23 result/count/completeness closure, and
Stage 24 support closure. Active config claim scope and dataset origin are
independently compared with the canonical Stage 9 contract. Stored semantic
critic verdicts remain the unavoidable unsigned-run trust boundary, but every
identity, path, hash, span, count, linkage, and validity field around them is
reconstructed.

The release checker converts the first typed replay error into an error finding;
unexpected exceptions also become errors rather than tracebacks. Citation and
evidence errors are structurally excluded from the closed degraded-compatible
code set, so `allow_suspicious` and unrelated degradation signals cannot turn
an E9 failure into exit 0 or exit 2.

E9 is the single release-authoritative interpretation of the citation evidence
chain. The older `check_citations`, `check_citation_support`, and
`check_claims_provenance` checks remain as compatibility diagnostics and may
add stricter findings, but they cannot waive, replace, or contradict a failed
E9 replay. Future citation policy changes must update the package-owned replay
and producer validators first; loosening a legacy diagnostic never changes
release eligibility.

### F0. v2 full-text HITL

- preliminary-plan acquisition request;
- executor/runner handling for stage-level PAUSED results and same-stage resume;
- append-only PAUSED/resume history and two-round bound;
- source-kind classification;
- deterministic identity checks;
- preprint/version mismatch rejection;
- text-native extraction;
- content-addressed KB store;
- optional one-way reading-copy export;
- no public full-text packaging.

E0-E9 complete v1. F0 is implemented only after v1 passes a controlled fixture
and a fresh DeepSeek pipeline-validation run.

E4 and E5 remain separate reviewable commits but must ship in the same release
window. E5 removes the old free candidate catalog; deploying it without E4's
final citation plan would leave Stage 17 with no valid citation input. No real
pipeline run is launched from an intermediate checkout between E4 and E5.

## 15. Required Adversarial Tests

At minimum, implementation must prove:

1. `<scp>` and other title tags cannot enter citation keys;
2. two identities with the same base key receive stable distinct suffixes;
3. the same DOI from two providers receives one key;
4. malformed Stage 5 JSON cannot produce a template shortlist;
5. an omitted candidate ID fails batch closure;
6. rejected candidates cannot return through minimum-count supplementation;
7. a template or explicitly fallback card is ineligible;
8. an LLM summary without an excerpt is ineligible;
9. an excerpt not equal to the source abstract slice is rejected;
10. a changed Stage 4 abstract invalidates its card and allowlist;
11. metadata-only evidence cannot support a claim;
12. zero eligible evidence fails pipeline validation before Stage 17;
13. research release with fourteen eligible sources fails the minimum of 15;
14. pipeline validation with nine eligible sources records target 9 consistently
    in Stages 17, 18, and 20;
15. a Stage 18 actionable requirement for fifteen references when the effective
    target is nine is rejected;
16. a prose quality comment that contains the number fifteen but is not a
    citation-count requirement is not misclassified;
17. Stage 17 cannot cite a valid allowlist key assigned to a different claim;
18. Stage 17 cannot cite any Stage 4 key outside the final plan;
19. post-HITL citation mutation is detected;
20. v1 does not repair or alias an unknown key;
21. Stage 23 verifies only actual manuscript keys;
22. a Stage 23 timeout cannot become a permissive score in research release;
23. Stage 23 cannot delete a marker and leave the claim silently;
24. topical relevance cannot satisfy claim-support closure;
25. Stage 24 rejects an excerpt that is not a source substring;
26. Stage 24 rejects a citation instance with no support assessment;
27. synthetic data claims cannot describe public or local-hardware measurements;
28. a PDF with the wrong DOI is rejected;
29. a DOI-free PDF with title/author/year mismatch is rejected;
30. a scanned or text-empty PDF is rejected as unsupported;
31. an inbox file cannot escape the KB root by symlink or path traversal;
32. an unrequested PDF is not silently added to the run;
33. changing the preliminary plan invalidates an acquisition request;
34. a third full-text acquisition request fails the run;
35. a previously stored matching PDF avoids a repeated request;
36. deleting a reading copy does not affect release replay;
37. changing a reading copy does not affect any canonical hash;
38. canonical KB-store mutation is detected;
39. PDF/full text never appears in public deliverables;
40. any citation/evidence error combined with a degradation signal exits 1.
41. editing `stage-04/references.bib` after registry sealing is rejected;
42. repository-config edits do not affect replay, while mutation of the bound
    run-local config snapshot is rejected;
43. deleting round-1 request/history to disguise a later request as round 1 is
    rejected;
44. a same-title/same-author preprint supplied for a DOI-identified version of
    record is rejected as `version_mismatch`;
45. duplicate Stage 4 `source_identity` / Stage 6 `source_record_id` values are
    rejected rather than resolved by JSONL order;
46. abstract excerpts containing U+2028 and non-BMP characters retain stable
    code-point spans and hashes;
47. a stage-level PAUSED run produces no release-ready deliverables and its
    deliverables manifest is explicitly `release_ready: false`;
48. Stage 22, Stage 23, and release replay reject a filtered shadow bibliography
    as a substitute for canonical Stage 4 `references.bib`;
49. the degraded-compatible allowlist snapshot contains no citation/evidence
    error family, and `allow_suspicious` cannot waive one.
50. a wrong PDF that merely cites the requested DOI in its reference list is
    rejected as an identity mismatch.

## 16. Acceptance Strategy

No real full-chain run is the first validator of a new schema. Each milestone
must pass, in order:

1. strict schema unit tests;
2. negative and mutation tests;
3. a controlled synthetic evidence-pack fixture labelled non-release;
4. existing Stage 19 and release-check regression suites;
5. a fresh DeepSeek pipeline-validation run with a new run directory;
6. read-only release replay whose non-release findings are expected;
7. only after v1 closure, a v2 HITL run that pauses, accepts user-provided PDFs,
   resumes, and proves canonical evidence closure.

The controlled fixture exists to test Stage 16-24 mechanics. It is not an
academic evidence source, is never promoted to research release, and does not
justify weakening Stage 5/6 production gates.

## 17. External Review Questions

The reviewer must answer with file/line evidence where applicable:

1. Does the spec preserve separate existence and support gates?
2. Can any template/fallback Stage 5 or Stage 6 artifact become eligible?
3. Can a rejected candidate re-enter through target-count logic?
4. Are abstract excerpts reproducible without trusting card summaries?
5. Is the effective citation target single-sourced across Stages 17, 18, and
   20?
6. Can Stage 18 create a permanently unresolvable citation-count comment?
7. Can Stage 17 cite any key not assigned by the final plan?
8. Can Stage 23 topical relevance be mistaken for claim support?
9. Can Stage 24 accept a claim without a retained excerpt?
10. Does v2 pause before writing and resume the same Stage 16 gate?
11. Is full-text acquisition bounded to two rounds?
12. Can source-kind classification be manipulated to avoid scholarly full-text
    requirements?
13. Can an incorrect or scanned PDF enter the canonical store?
14. Are the KB store and writing-directory copy unambiguously canonical and
    derived, respectively?
15. Does any milestone weaken Stage 19 or existing release gates?
16. Are the E0-E9 and F0 commit boundaries narrow enough for independent
    review and rollback?

Required review format:

```text
## Verdict: ACCEPT / MODIFY / REJECT
## P0 findings
## P1 findings
## P2 findings
## Trust-boundary assessment
## State-machine assessment
## Schema gaps
## Missing adversarial tests
## Required changes before E0
```
