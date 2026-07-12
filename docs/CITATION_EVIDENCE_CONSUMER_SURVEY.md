# Citation Evidence Consumer Survey

Status: E0 prerequisite, read-only survey. This document records current
consumers before the citation-evidence migration. It does not authorize a
consumer change or make any artifact authoritative.

## Scope

Surveyed repository reads of:

- `candidates.jsonl`;
- `shortlist.jsonl`;
- `cards/`;
- `references.bib` and derived bibliography files;
- hard-coded citation-count thresholds.

## Current Consumers

| Artifact | Consumer | Current use | Migration implication |
|---|---|---|---|
| `stage-04/candidates.jsonl` | Stage 5 `_literature.py` | semantic-screen input | E1 must consume source identities and strict batch IDs |
| `stage-04/candidates.jsonl` | Stage 8 `_synthesis.py` | novelty fallback corpus | keep identity fields; do not treat as citation eligibility |
| `stage-04/candidates.jsonl` | `literature/novelty.py` | overlap search | identity-compatible read only |
| `stage-04/candidates.jsonl` | Stage 17 `_paper_writing.py` | free citation catalog | remove in E5; final plan becomes the only citation input |
| `stage-05/shortlist.jsonl` | Stage 6 `_literature.py` | card extraction input | E2 requires strict screening provenance and source identity |
| `stage-05/shortlist.jsonl` | HITL claim verifier | evidence lookup | migrate to strict shortlist/allowlist semantics |
| `stage-05/shortlist.jsonl` | HITL quality predictor | shortlist-size feature | degraded/fallback entries must not count as eligible evidence |
| `stage-05/shortlist.jsonl` | HITL summarizer | stage summary | diagnostic consumer only |
| `stage-06/cards/*.md` | Stage 7 `_synthesis.py` | synthesis prompt context | E2 replays the JSON/manifest binding before consuming the deterministic view |
| `stage-04/references.bib` | Stage 19 `sectional_execution.py` | canonical draft-key closure | must remain registry-bound and immutable |
| nearest `references.bib` | Stage 17 `_paper_writing.py` | preverification input | retire full-bibliography preverification in E5 |
| nearest `references.bib` | Stage 22 `_review_publish.py` | export and missing-key repair | E5/E7 must pin Stage 4 canonical bib and remove auto-repair authority |
| nearest `references.bib` | Stage 23 `_review_publish.py` | existence/relevance verification | E7 verifies actual cited keys against Stage 4 canonical bib |
| `stage-23/references_verified.bib` | `release_check.py` | citation release input | E7/E9 must keep it derived and hash-bound to canonical bib/report |
| `stage-22/references.bib` | runner/report/deliverables | human/export copy | derived output only; never release provenance root |

## Stage Contracts

After E0-E3, pipeline contracts declare:

- Stage 4 output: candidates, canonical bibliography, and cite-key registry;
- Stage 5 input/output: the sealed Stage 4 collection -> shortlist + screening report;
- Stage 6 input/output: shortlist + screening report -> cards + manifest + citation allowlist;
- Stage 7 input: cards + manifest + citation allowlist, with strict card replay;
- Stage 16 input/output: citation allowlist -> effective citation policy + outline;
- Stages 17, 18, and 20 require and replay the same effective policy.

E0 must add the cite-key registry and canonical bibliography to Stage 4's
declared outputs. E2 later adds structured cards and manifest while preserving
the Markdown directory until Stage 7 migration is complete.

## Hard-Coded Citation Targets

E3 removes independent writing/review thresholds from active prompts and local
draft-quality checks. Remaining citation-count policy comes from:

- top-level `citation_policy` in the bound run-local config snapshot;
- `stage-16/citation_policy_effective.json`, recomputed by Stages 17, 18, and 20.

E5 still removes the free Stage 4 candidate catalog and preverification path;
those are evidence-surface tasks, not citation-count policy sources.

## Migration Risks

1. Changing `Paper.cite_key` alone is insufficient because Stage 4 currently
   generates BibTeX before API, seminal, LLM, and web candidate dictionaries
   have been merged into the final collection.
2. Candidate and bibliography keys must be projected from one collection-level
   registry after deterministic identity deduplication and collision handling.
3. The Stage 7 Markdown cards consumer requires JSON-authority/Markdown-derived
   dual output during E2; two independently generated card formats are not
   allowed.
4. `_read_prior_artifact(..., "references.bib")` is path-ambiguous once
   derived bibliography files exist. E5/E7 must use exact canonical paths.
5. Deliverable/report consumers may retain bibliography copies for humans, but
   no copy may replace Stage 4 as release provenance.

## E0 Boundary

E0 changes citation identity production only:

- normalize and version source identities and base cite keys;
- deduplicate equal identities deterministically;
- resolve base-key collisions over the complete Stage 4 collection;
- write candidates and bibliography from one sealed registry;
- write `cite_key_registry.json` with exact artifact hashes;
- add focused identity, collision, and Stage 4 consistency tests.

E0 does not change Stage 5 screening, Stage 6 cards, citation targets, Stage 17
writing, Stage 23 verification, Stage 24 support, or release gates.
