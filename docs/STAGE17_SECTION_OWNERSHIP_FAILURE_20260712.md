# Stage 17 Section-Ownership Failure Review

## Status

Design approved; implementation complete and awaiting code review.

Run:

- Run ID: `rc-20260712-194920-45409b`
- Run directory: `runs/hwsec-e0-e9-validation-20260712`
- Resume start: Stage 9 `EXPERIMENT_DESIGN`
- First failed stage: Stage 17 `PAPER_DRAFT`

## What Succeeded

The Stage 9 fixes passed their real-run acceptance checks:

- the post-pivot domain returned to `security_detection`;
- a new direct `stage-09/experiment_contract.yaml` was written;
- Stages 10-16 completed;
- Stage 17 reached its deterministic CommonMark structure gate.

## Observed Failure

The final draft contains six duplicate canonical heading paths:

- `Experiments`
- `Experiments / Experimental Setup`
- `Experiments / Results`
- `Discussion`
- `Limitations`
- `Conclusion`

The first LLM call, which owns only title, abstract, introduction, and related
work, emitted an entire paper through conclusion. Calls 2 and 3 then emitted
their requested Method/Experiments and Results/Discussion/Limitations/Conclusion
sections. `_write_paper_sections()` concatenated all three responses without
validating each response's section ownership. The final structure gate correctly
rejected the combined draft.

## Root-Cause Classification

Primary: section ownership exists only as prompt prose, not as an executable
contract at each LLM-call boundary.

Contributing factors:

1. Each later call receives earlier full text as context, increasing the chance
   that the model repeats prior or future sections.
2. The global paper-writing system prompt and outline describe the full paper,
   while the per-call prompt asks for a subset.
3. `_SECTION_OUTPUT_CONTRACT` says not to repeat sections but is not parsed or
   enforced before concatenation.
4. The final structure gate runs only after all three expensive calls.

This is not a parser false positive. The duplicate paths correspond to real
repeated `##` headings in `paper_draft.md`.

## Additional Control-Flow Finding

On strict structure failure, Stage 17 currently returns `FAILED` but leaves
canonical `stage-17/paper_draft.md` and lists it as an artifact. Stage 18's input
contract checks for `paper_draft.md` plus support artifacts. A manual resume from
Stage 18 can therefore consume a draft produced by a failed Stage 17 attempt.

This violates the explicit invariant that canonical success artifacts exist only
after their owning stage succeeds. In the observed run, Stage 18 would also have
been blocked implicitly because the closure artifacts were absent or invalid, so
this is not recorded as a reproduced live Stage 18 bypass. The fix replaces that
fragile downstream dependency with an explicit canonical-artifact invariant.

## Proposed Architecture

### 1. Deterministic per-call section ownership

Define versioned, code-owned section contracts for the ML and HEP paths.

ML:

- Call 1: one paper-title heading, then `Abstract`, `Introduction`, `Related Work`.
- Call 2: `Method`, `Experiments`.
- Call 3: `Results`, `Discussion`, `Limitations`, `Conclusion`.

HEP:

- Call 1: one paper-title heading, then `Abstract`, `Introduction`.
- Call 2: `Model / Theoretical Framework`, `Phenomenology / Computational Setup`.
- Call 3: `Results`, `Discussion`, `Conclusions`.

The first title heading may contain the actual paper title; it must not normalize
to any reserved section name. All other major headings must match the owned
canonical sequence exactly. Optional `###` subsections are allowed only under an
owned `##` parent and must remain unique within that parent.

### 2. Parse each response before appending

After each LLM response:

1. parse it with the existing CommonMark-aware manuscript parser;
2. reject preamble text outside the allowed title behavior;
3. extract the ordered level-2 heading sequence;
4. reject missing, extra, duplicated, renamed, or out-of-order major sections;
5. reject nested-heading and duplicate-path structure issues;
6. append the part only after the contract passes.

Do not silently discard extra sections and do not select the first occurrence.
Either behavior would allow an invalid model response to decide which content is
authoritative.

### 3. One bounded semantic regeneration per part

If a part violates its section contract:

1. make one regeneration call containing the exact deterministic violations and
   the same owned-section contract;
2. validate the complete regenerated part again;
3. if it still fails, fail Stage 17 immediately without making later writing
   calls.

This is separate from transport retries. It must be bounded to one semantic
regeneration and recorded in `section_generation_report.json` with response
hashes, violation codes, and attempt counts. The report is diagnostic, not an
authority for release claims.

### 4. Keep the final whole-manuscript gate

Per-call validation does not replace `_validate_stage17_manuscript_structure()`.
The final assembled draft must still pass strict CommonMark structure validation
after HITL guidance, because HITL rewriting can reintroduce duplicate or nested
headings.

### 5. Never publish a canonical draft on structural failure

Stage 17 owns these failure semantics:

- success: write `paper_draft.md` plus all closure reports;
- structural or per-call failure: write `paper_draft_invalid.md`,
  `paper_structure_report.json`, and `section_generation_report.json` as
  diagnostics; ensure `paper_draft.md` and closure reports are absent;
- Stage 18 continues to require canonical `paper_draft.md`, so manual resume from
  Stage 18 fails input validation.

Stage 17 entry cleanup must include both canonical and invalid draft names plus
the section-generation report.

## Rejected Alternatives

### Add stronger prompt wording only

Rejected. `_SECTION_OUTPUT_CONTRACT` already uses override language, and the real
model ignored it. Prompt prose is guidance, not enforcement.

### Strip duplicate headings after concatenation

Rejected. Deterministic code cannot know whether the first or second Method,
Experiments, or Discussion is semantically authoritative. Deleting one silently
changes scientific content.

### Keep only the headings expected from each response

Rejected for v1. It would silently accept a model response that violated its
contract and could retain subsections whose context depended on discarded text.

### Let the final structure gate remain the only check

Rejected. It wastes later calls, gives poor diagnostics, and leaves the canonical
artifact resume bypass.

### Allow Stage 18 to inspect `paper_structure_report.json` itself

Insufficient as the primary fix. Every consumer would need to remember the extra
gate. Removing the canonical draft on failure makes the existing input contract
structurally fail-closed.

## Required Tests

1. ML Call 1 accepts actual-title + Abstract + Introduction + Related Work.
2. ML Call 1 rejects Method/Experiments/Conclusion as extra sections.
3. ML Call 2 rejects a repeated Method or an extra Discussion.
4. ML Call 3 rejects missing Limitations and duplicate Conclusion.
5. HEP contracts accept their exact owned sequences and reject ML-only sections.
6. Fence-contained `##` text is not treated as a heading.
7. Nested blockquote/list headings are rejected through the CommonMark parser.
8. First invalid response followed by valid semantic regeneration succeeds and
   records two attempts.
9. Two invalid responses fail immediately and do not invoke subsequent calls.
10. Per-call failure leaves no canonical `paper_draft.md` or closure reports.
11. Final post-HITL structure failure renames/persists only
    `paper_draft_invalid.md` and leaves canonical draft absent.
12. Manual Stage 18 resume after Stage 17 failure fails input validation.
13. Stage 17 rerun clears stale canonical and invalid draft artifacts.
14. Existing valid three-part writer fixtures remain byte-stable after assembly.
15. Full Stage 17 structure report still validates the final assembled source.

## Validation and Resume Sequence

After approval and implementation:

1. Run manuscript parser, Stage 17 writer, executor resume, Stage 18, and full
   suites.
2. Perform read-only diff review before commit.
3. Commit and push the bounded fix.
4. Resume the mixed-version run from Stage 17, not Stage 18:

   ```bash
   python -m researchclaw run \
     -c config.deepseek.sectional-dry-run.yaml \
     -o runs/hwsec-e0-e9-validation-20260712 \
     --from-stage PAPER_DRAFT \
     --to-stage DEAI_AUDIT
   ```

5. If it reaches Stage 25, run `release_check`; nonzero is expected only for the
   pipeline-validation identity, with no citation-evidence replay errors.
6. Before F0, run one fully fresh, single-version Stage 1-25 validation without
   resume.

## Review Questions

1. Should v1 reject all unexpected major headings rather than selecting or
   stripping owned sections?
2. Is one semantic regeneration per part the correct reliability bound?
3. Should HEP keep `Conclusions` plural as its canonical heading while ML keeps
   `Conclusion` singular?
4. Is removing canonical `paper_draft.md` on any structural failure sufficient
   to make Stage 18 resume fail-closed, or should Stage 18 additionally validate
   the structure-report hash?
