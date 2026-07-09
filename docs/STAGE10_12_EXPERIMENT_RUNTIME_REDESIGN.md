# Stage 10/12 Experiment Runtime Redesign

Status: draft for architecture review

This document reframes the recurring Stage 10/12 failures as an architecture
problem, not as another prompt or repair-loop problem. It is intended for Codex
and Claude review before implementation.

## Executive Summary

AutoResearchClaw's release hardening gates are now strict enough to prevent
fabricated papers from being marked release-ready. The remaining blocker is
earlier in the pipeline: Stage 10/12 does not reliably produce a runnable,
evidence-bearing experiment.

The observed failures are symptoms of the same design issue:

- Stage 10 asks an LLM to generate a full experiment repository in one open
  step.
- The same stage then validates imports, repairs code, runs a smoke execution,
  repairs metadata, and decides whether the project is ready for Stage 12.
- Stage 12 expects Stage 10 output to behave like a sealed experiment package.
- The boundary between smoke artifacts, real experiment artifacts, and release
  evidence is enforced by later gates instead of being impossible by design.

The fix should not be more ad-hoc repair rules. Stage 10 should become a
bounded experiment compiler:

1. Stage 9 writes an explicit `experiment_contract.yaml`.
2. Stage 10 uses a deterministic scaffold and lets the LLM generate only a
   bounded plugin.
3. Stage 10 evaluates multiple candidates with the same harness and selects one
   sealed candidate.
4. Stage 12 runs only the selected candidate and produces the first evidence
   artifacts that later stages may cite.

## Inputs And References

Local reference material:

- `/Users/sheva/Desktop/GEWU-Deck-v3-2026.07.pdf`
  - Page 7: four-layer research OS stack.
  - Page 11: supervisor scheduling agent plus vertical research agents.
  - Page 16: path from research agents to research OS to physical-digital
    experiment infrastructure.

External systems used as design references:

- AI Scientist-v2: https://github.com/SakanaAI/AI-Scientist-v2
  - Uses an experiment manager and progressive agentic tree search.
  - The README explicitly warns that LLM-written code must run in a controlled
    sandbox.
  - It distinguishes open exploration from higher-success template-like modes.
- Agent Laboratory: https://github.com/SamuelSchmidgall/AgentLaboratory
  - Splits the workflow into literature review, experimentation, and report
    writing, with human feedback as a quality lever.
- ScienceAgentBench: https://github.com/OSU-NLP-Group/ScienceAgentBench
  - Evaluates individual scientific workflow tasks before making end-to-end
    automation claims.
  - Normalizes generated output to self-contained Python programs and evaluates
    code, execution results, and costs.
  - Recommends containerized evaluation where available.
- AlphaEvolve: https://arxiv.org/abs/2506.13131
  - Uses LLM-generated code variants plus objective evaluators and selection,
    rather than trusting one generated program.

## Current Failure Pattern

Recent real DeepSeek runs repeatedly failed at Stage 10/12 with different
surface errors:

- OpenCode timed out or returned server errors.
- DeepSeek long calls failed with `IncompleteRead`.
- Generated code omitted `dataset_origin`.
- Generated code called `ExperimentHarness.elapsed()` as a function even though
  the harness exposes `elapsed` as a property.
- Stage 10 smoke execution returned non-zero.
- Stage 12 refused to proceed because the experiment produced zero real metrics.

The important point is that these failures are not independent. They all come
from one root cause: the system gives the LLM too much unconstrained
responsibility and then tries to recover with repair heuristics.

## GEWU OS Mapping

The GEWU deck describes a research OS with four layers:

| GEWU layer | AutoResearchClaw equivalent | Required redesign |
| --- | --- | --- |
| Research applications | Literature, hypothesis, experiment, data, paper, review agents | Keep stage roles narrow and explicit |
| Runtime platform | Gateway, Brain, Memory, Skills, Heartbeat, GEWU-LAB | Introduce an Experiment Runtime layer |
| MCP/tool access | arXiv, PubMed, OpenAlex, CNKI, local tools | Keep data/tool provenance explicit in contracts |
| Pluggable models | DeepSeek, Kimi, Intern-S1, etc. | Treat models as interchangeable generators, not trusted executors |

The deck's supervisor-agent pattern also implies that Stage 10 should not be a
single monolithic "generate and repair everything" stage. It should delegate to
bounded components:

- contract authoring
- scaffold construction
- plugin generation
- sandbox evaluation
- candidate selection
- artifact sealing

## Design Principles

### 1. Contract Before Code

Stage 10 must not infer the experiment contract from generated code. Stage 9
should emit a machine-readable `experiment_contract.yaml` before any code is
generated.

Minimum fields:

```yaml
schema_version: 1
topic: str
claim_scope: pipeline_validation | exploratory | research_release
dataset_origin: synthetic | public | local_hardware
dataset_name: str | null
primary_metric:
  key: str
  direction: maximize | minimize
  minimum_valid_value: number | null
smoke_budget_sec: int
run_budget_sec: int
allowed_inputs:
  - path: str
    required: bool
allowed_outputs:
  - path: results.json
    schema: stage12_results_v1
evaluator:
  command: python main.py
  owner: scaffold
  timeout_sec: int
  required_result_keys:
    - dataset_origin
    - metrics
    - primary_metric
safety:
  network: none | setup_only | full
  env_policy: allowlist
  evidence_policy: stage12_only
sealing:
  candidate_manifest: selected_candidate_manifest.json
  content_hash_algorithm: sha256
  default_deny_extra_files: true
  include_contract_sha256: true
  include_scaffold_sha256: true
```

Release implication:

- `claim_scope: pipeline_validation` may test the pipeline but must not become a
  release-ready scientific claim.
- `dataset_origin: synthetic` is valid for smoke and method development, but a
  release paper must explicitly state synthetic-data limitations unless a public
  or local-hardware dataset is present.
- `smoke_budget_sec` and `run_budget_sec` are separate. Smoke is a short probe;
  Stage 12 may have a longer legitimate runtime.
- `evaluator.owner` must be `scaffold`. A model-authored evaluator would let a
  candidate grade itself and is not allowed.
- Release-check must also enforce the claim-scope back door: a run with
  `claim_scope: research_release` and `dataset_origin: synthetic` is blocked
  unless an explicit human waiver or synthetic-limitation policy is present.

### 2. Deterministic Scaffold, Bounded Plugin

The LLM should not generate:

- entrypoint structure
- result schema
- artifact paths
- harness API usage
- timeout handling
- provenance fields
- experiment package layout
- evaluator or scoring code

Those should be deterministic scaffold files owned by AutoResearchClaw. The LLM
should generate only a bounded plugin, for example:

```python
class DetectorPlugin:
    name = "candidate_name"

    def fit(self, X_train, y_train):
        ...

    def predict(self, X_test):
        ...

    def describe(self):
        ...
```

This turns the task from "write a whole research repo" into "fill one
scientific method slot that can be evaluated".

The evaluator is scaffold-owned. The plugin may emit predictions or intermediate
artifacts, but the runtime-owned evaluator computes the metric and writes the
normalized result. A candidate that ships its own evaluator must be ignored or
rejected.

For prediction-style experiments, the scaffold owns the train/test split,
ground-truth labels, and scoring function. The plugin may receive training data
and may produce predictions for `X_test`, but it must never receive `y_test` or
the scoring function. This is the concrete anti-self-grading invariant behind
`evaluator.owner: scaffold`.

### 3. Multiple Candidates, One Evaluator

Stage 10 should not rely on one generated project plus repair loops. It should
produce multiple candidates and evaluate them all with the same deterministic
harness.

Candidate attempt record:

```json
{
  "candidate_id": "cand-0003",
  "generated": "2026-07-09T00:00:00Z",
  "model": "deepseek-v4-flash",
  "contract_sha256": "...",
  "path": "stage-10/candidates/cand-0003/",
  "files": ["plugin.py"],
  "validation": {
    "import_ok": true,
    "entrypoint_ok": true,
    "schema_ok": true,
    "unsafe_imports": []
  },
  "execution": {
    "returncode": 0,
    "timed_out": false,
    "elapsed_sec": 12.4
  },
  "metrics": {
    "detection_f1": 0.72
  },
  "dataset_origin": "synthetic",
  "decision": "selected | rejected",
  "blockers": []
}
```

Candidate attempt records must live under `stage-10/candidates/`, not under a
run-root `attempts/` directory. The existing release evidence allowlist has
special handling for other attempt logs; Stage 10 candidate attempts must remain
diagnostic and must never be claim evidence.

Selection must be objective:

- valid schema first
- successful execution second
- finite primary metric third
- runtime under budget fourth
- metric score only after validity gates pass

The Stage 10 metric may be used only for candidate selection. It must not be
copied into Stage 12 summaries, claims provenance, or release evidence. Stage 12
must rerun the sealed candidate and recompute all evidence-bearing metrics from
scratch.

### 4. Stage 10 Smoke Is Not Evidence

Stage 10 may run a smoke evaluator, but its output must remain diagnostic. It
must never appear in:

- Stage 12 experiment summaries
- claims provenance
- citation support
- release-check evidence closure

Stage 10 output should be limited to:

- `experiment_contract.yaml`
- `scaffold_manifest.json`
- `candidates/cand-*/attempt.json`
- `selected_candidate/`
- `selected_candidate_manifest.json`
- `selection_report.json`

Stage 12 should be the first stage allowed to create evidence-bearing
experiment artifacts:

- `stage-12/runs/*.json`
- `stage-12/experiment_manifest.json`
- `stage-12/execution_log.jsonl`

The selected candidate must be content-addressed. Stage 10 writes
`selected_candidate_manifest.json` with selected-candidate file paths,
selected-candidate sha256 hashes, `contract_sha256`, and `scaffold_sha256`.
Stage 12 verifies that manifest before execution and refuses to run if the
selected candidate, scaffold, or contract has drifted.

The selected candidate must be code-only. It must not contain `results.json`,
`smoke_results.json`, metrics files, run directories, candidate attempt records,
or any other numeric result artifact. Stage 12 must reject any selected
candidate containing files not explicitly listed in the manifest; the manifest
is complete and default-deny, not a best-effort hash list.

### 5. Fail Closed, But Fail Early

The release hardening work made late-stage failure safe. The next step is to
make early failure cheap and informative.

Stage 10 should fail before long model calls or deep repairs when:

- no experiment contract exists
- contract fields are unknown or contradictory
- `research_release` is requested with only synthetic data
- requested dataset/tool access is unavailable
- Docker is required but unavailable
- evaluator cannot run in a minimal local fixture

## Proposed Runtime Boundary

Add a small Experiment Runtime layer instead of adding more logic to
`_code_generation.py`.

Suggested modules:

```text
researchclaw/experiment_runtime/
  contract.py
  scaffold.py
  candidate.py
  evaluator.py
  selection.py
  ledger.py
```

Responsibilities:

- `contract.py`: parse and validate `experiment_contract.yaml`.
- `scaffold.py`: materialize deterministic scaffold files.
- `candidate.py`: write plugin candidates and attempt manifests under
  `stage-10/candidates/`.
- `evaluator.py`: run candidates in sandbox/docker with scaffold-owned scoring
  and normalize outputs.
- `selection.py`: choose the sealed candidate using deterministic rules.
- `ledger.py`: record hashes, provenance, and evidence eligibility.

The existing pipeline stages call this runtime, but do not own its internal
state machine.

## Non-Negotiable Runtime Invariants

These decisions are fixed for the redesign:

- The evaluator is scaffold-owned, never model-owned.
- The scaffold owns evaluation ground truth and scoring. Plugins do not receive
  test labels or evaluator internals.
- Stage 10 candidate attempt paths are `stage-10/candidates/cand-*/attempt.json`;
  they are diagnostic and not evidence.
- Stage 10 smoke metrics may select a candidate but must be discarded before
  Stage 12 evidence generation.
- Stage 12 executes only a sha256-verified, code-only `selected_candidate/`.
- A selected candidate that contains stale runtime outputs or candidate
  metadata is invalid.
- The selected-candidate manifest covers candidate files, scaffold, and the
  experiment contract, and rejects unmanifested files by default.
- `smoke_budget_sec` and `run_budget_sec` are separate contract fields.
- Synthetic data defaults to pipeline validation, not release-grade scientific
  evidence.
- The `research_release` plus `synthetic` combination is blocked both at
  contract validation and at release_check. Front-door contract validation alone
  is not sufficient because users can resume from later stages or edit
  artifacts.

## Stage Boundary Redesign

### Stage 9: Experiment Design

New required artifact:

- `experiment_contract.yaml`

Current optional artifacts such as `exp_plan.yaml` may remain, but Stage 10
should not proceed without a valid contract.

### Stage 10: Candidate Generation And Selection

New behavior:

- read `experiment_contract.yaml`
- build deterministic scaffold
- generate N bounded plugin candidates
- run each candidate through the evaluator
- seal exactly one selected candidate with a sha256 manifest or fail with
  diagnostic candidate attempt logs

New artifacts:

- `experiment_contract.yaml` copy or hash reference
- `scaffold_manifest.json`
- `candidates/cand-*/attempt.json`
- `selected_candidate/`
- `selected_candidate_manifest.json`
- `selection_report.json`

Removed or deprecated behavior:

- no free-form full-repo code generation for release-capable runs
- no LLM ownership of entrypoint/result schema
- no smoke output under evidence-like paths
- no repair loop that rewrites the full project after validation

### Stage 11: Resource Planning

Stage 11 should consume the selected candidate metadata and contract, not a
free-form generated experiment folder.

### Stage 12: Experiment Run

New behavior:

- read only `stage-10/selected_candidate/`
- verify `stage-10/selected_candidate_manifest.json`
- execute against the immutable contract
- write the first evidence-eligible outputs
- reject zero metrics, runtime crash signatures, stale outputs, or mismatched
  contract hashes
- reject selected candidates that contain `smoke_results.json`,
  `.smoke_sandbox`, `results.json`, metrics files, `runs/`, `attempts/`, or
  `candidates/`
- reject any file under `selected_candidate/` that is not listed in
  `selected_candidate_manifest.json`

### Stage 13/14: Refinement And Analysis

Stage 13 may create new candidates, but should use the same candidate store and
selection rules. Stage 14 may only merge Stage 12/13 outputs that have a valid
contract hash and an evidence-eligible ledger entry.

## Minimum Viable Implementation Plan

### Phase 0 - Preserve Current Work

Before implementation:

- keep current Stage 10 repair patch uncommitted or commit it separately as an
  interim stabilization patch;
- do not mix architecture redesign with release hardening commits;
- do not delete old run directories during this work.

### Phase 1 - Contract And Fixture, No Behavior Change

Target files:

- `researchclaw/experiment_runtime/contract.py`
- `researchclaw/pipeline/stage_impls/_experiment_design.py`
- `tests/test_experiment_contract.py`

Acceptance:

- Stage 9 can write a valid `experiment_contract.yaml`.
- Invalid or missing required fields fail before Stage 10.
- `synthetic` plus `research_release` is blocked unless an explicit synthetic
  limitation policy is present.
- Stage 10 may still generate code as it does today during this phase, but it
  must read and record the contract hash.

### Phase 2 - Sealed Boundary And Evidence Quarantine

Target files:

- `researchclaw/pipeline/stage_impls/_code_generation.py`
- `researchclaw/pipeline/stage_impls/_execution.py`
- `scripts/release_check.py`
- `tests/test_stage12_selected_candidate.py`
- `tests/test_release_check_v2.py`

Acceptance:

- Stage 10 writes `stage-10/selected_candidate/` and
  `stage-10/selected_candidate_manifest.json`.
- Stage 10 writes a code-only selected candidate. The manifest lists every
  selected-candidate file and also pins `contract_sha256` and
  `scaffold_sha256`.
- Stage 12 reads only the sealed selected candidate and verifies candidate,
  scaffold, and contract sha256 hashes before execution.
- Stage 12 refuses selected candidates containing `smoke_results.json`,
  `.smoke_sandbox`, `results.json`, metrics files, `runs/`, `attempts/`, or
  `candidates/`.
- Stage 12 refuses any unmanifested file under `selected_candidate/`.
- `stage-10/candidates/*` is explicitly rejected as claim evidence.
- Stage 12 reruns the selected candidate and does not read Stage 10 smoke
  metrics.
- release_check blocks `claim_scope: research_release` with
  `dataset_origin: synthetic` unless an explicit human waiver or limitation
  policy is present. This back-door release check lands in Phase 2, not Phase 6.

This phase must land before scaffold/plugin generation. Otherwise the candidate
attempt surface expands before the evidence boundary is sealed.

### Phase 3 - Runtime Extraction, Refactor Only

Target files:

- `researchclaw/experiment_runtime/evaluator.py`
- `researchclaw/experiment_runtime/candidate.py`
- `researchclaw/experiment_runtime/selection.py`
- `researchclaw/experiment_runtime/ledger.py`
- `researchclaw/pipeline/stage_impls/_code_generation.py`
- `tests/test_candidate_selection.py`

Acceptance:

- existing smoke, scrub, result-discovery, candidate-evaluation, and selection
  logic moves into runtime modules;
- behavior remains equivalent to Phase 2;
- `_code_generation.py` becomes orchestration glue instead of the owner of the
  runtime state machine.
- This phase is still a transition state: the free-form project evaluator is
  weaker than the final scaffold-owned evaluator. Do not run or certify
  `research_release` from this phase.

### Phase 4 - Scaffold And Plugin API

Target files:

- `researchclaw/experiment_runtime/scaffold.py`
- `researchclaw/experiment_runtime/candidate.py`
- `researchclaw/pipeline/stage_impls/_code_generation.py`
- `tests/test_experiment_scaffold.py`

Acceptance:

- deterministic scaffold writes `main.py`, `result_schema.py`, and harness glue;
- LLM candidate is limited to `plugin.py` or a small allowlisted file set;
- scaffold owns evaluator/scoring code;
- scaffold owns train/test split, test labels, and metric calculation;
- scaffold can run with a hand-written plugin without using an LLM;
- generated code cannot misuse `ExperimentHarness` APIs because harness calls
  are scaffold-owned.
- existing Stage 12 stdout-to-results fallback logic is deprecated or routed
  through the scaffold's single result writer. There must not be two competing
  sources of `results.json`.

### Phase 5 - Evaluator And Selection Policy

Target files:

- `researchclaw/experiment_runtime/evaluator.py`
- `researchclaw/experiment_runtime/selection.py`
- `tests/test_candidate_selection.py`

Acceptance:

- at least three candidate attempts can be evaluated;
- invalid candidates are rejected with structured blockers;
- best valid candidate is selected deterministically;
- all attempt records include hashes and contract references;
- a plugin-provided evaluator cannot override scaffold-owned scoring;
- plugins cannot access `y_test`, scoring internals, or evaluator command hooks;
- smoke uses `smoke_budget_sec`, while Stage 12 uses `run_budget_sec`.
- Phase 4 and Phase 5 should be treated as a paired implementation milestone:
  multi-candidate selection is not release-grade until the evaluator is
  scaffold-owned.

### Phase 6 - Claim Scope And Release Gate

Target files:

- `researchclaw/experiment_runtime/ledger.py`
- `scripts/release_check.py`
- `tests/test_release_check_v2.py`

Acceptance:

- `claim_scope` and `dataset_origin` propagate from contract to Stage 12 result
  metadata and fabrication flags;
- `pipeline_validation`, `exploratory`, and synthetic-only runs have explicit
  release-check outcomes: block by default, degrade only if a human-reviewed
  waiver policy says so;
- release-check evidence allowlist still excludes Stage 10 smoke and candidate
  artifacts.

The core release-check back door for `research_release` plus `synthetic` was
already added in Phase 2. Phase 6 broadens propagation and reporting rather
than introducing the first blocker.

### Phase 7 - Real DeepSeek Trial

Run a narrow pipeline-validation trial before a research-release trial:

```bash
python -m researchclaw run -c config.deepseek.yaml -o runs/hwsec-runtime-v1 --to-stage EXPERIMENT_RUN
python scripts/release_check.py runs/hwsec-runtime-v1 --json
```

Expected result:

- Stage 10/12 should complete for `pipeline_validation`.
- `release_check.py` should still reject or degrade the run unless evidence and
  claim scope meet release criteria.

## Tests Required

Unit tests:

- contract parser rejects missing or unknown fields;
- scaffold output is deterministic and hashable;
- plugin API rejects extra file writes and forbidden imports;
- evaluator rejects nonzero return code, timeout, missing metric, missing
  `dataset_origin`, stale result files, and schema drift;
- selection picks the best valid candidate and never picks an invalid
  high-metric candidate.
- evaluator is scaffold-owned; plugin-provided evaluator files or commands
  cannot override scoring.
- plugin cannot access test labels, scoring internals, or evaluator command
  hooks.
- a transport failure while generating one candidate is recorded as candidate
  skipped; the stage fails only when no valid candidate remains.

Integration tests:

- Stage 9 to Stage 10 with a fixture contract and fixture LLM response;
- Stage 10 to Stage 12 with a selected candidate;
- Stage 12 fails if selected candidate is tampered after selection;
- Stage 14 ignores outputs without evidence-eligible ledger entries.
- Stage 12 reads only `stage-10/selected_candidate/` and refuses selected
  candidates containing `smoke_results.json`, `.smoke_sandbox`, `runs/`,
  `attempts/`, or `candidates/`.
- Stage 12 refuses selected candidates containing `results.json`, metrics
  files, or any file not listed in `selected_candidate_manifest.json`.
- Stage 12 refuses if candidate, scaffold, or contract sha256 differs from the
  manifest.

Regression tests:

- `ExperimentHarness.elapsed()` misuse cannot occur in scaffold-owned code;
- Stage 10 smoke output cannot be discovered by result collection;
- synthetic pipeline-validation run cannot be marked research-release;
- stale `results.json` in the generated project is scrubbed or ignored.
- `stage-10/candidates/cand-*/attempt.json` is rejected by release_check as
  `claims_disallowed_evidence_path`.
- smoke uses `smoke_budget_sec`; Stage 12 uses `run_budget_sec`.
- Stage 12 zero-metric, suspicious-fast, and crash-signal guards still fire.
- Phase 2-3 transition retains the prompt-level `dataset_origin` honesty
  contract and Stage 10 smoke gate.
- manually edited `research_release` plus `synthetic` artifacts still fail
  release_check even if Stage 9 is not rerun.

## What Not To Loosen

Do not loosen these gates to make a run pass:

- Stage 12 zero-real-metric refusal.
- `dataset_origin` requirement.
- release-check claims provenance closure.
- evidence path allowlist.
- citation support gates.
- Docker/no-network/env-policy hardening.
- generated-code sandbox isolation.
- separation between Stage 10 smoke and Stage 12 evidence.
- scaffold-owned evaluator and scoring.
- selected-candidate sha256 verification.
- `claim_scope` and `dataset_origin` release semantics.
- code-only selected candidates with no result artifacts.
- scaffold/contract/candidate manifest completeness.
- plugin isolation from evaluator ground truth.

Do not solve the problem by:

- increasing model timeout again;
- adding another free-form repair pass;
- letting Stage 10 smoke metrics flow downstream;
- letting Stage 10 candidate attempt records become claim evidence;
- letting the plugin grade itself;
- letting the plugin see test labels or evaluator internals;
- allowing multiple independent writers of `results.json`;
- treating synthetic data as public or local-hardware data;
- writing release-ready deliverables from a failed or degraded run.

## Blocking Questions Before Phase 4

1. What is the first bounded plugin interface?
   - For the current HPC detection topic, `fit/predict` is plausible.
   - For quantum, biology, physics, or non-ML domains, this interface may be too
     narrow. Either define per-domain scaffolds or explicitly limit Phase 4 to
     the `detection_f1`/HPC-security domain.
2. How should scaffold-owned evaluators work for experiments without
   prediction-vs-label metrics?
   - Examples: unsupervised clustering, physical simulation error, generated
     artifact quality, or qualitative analysis.
   - The invariant remains the same: the plugin cannot self-report the final
     score, but the evaluator design must be domain-specific.
3. What are the candidate generation parameters?
   - Define `num_candidates`, retry policy, and minimum valid candidates.
   - Transport failures should skip a candidate; zero valid candidates should
     make Stage 10 `FAILED`.
4. What is the release-check outcome for `claim_scope: exploratory`?
   - Default recommendation: block release by default; allow only a clearly
     labeled non-release/debug package or an explicit human waiver.
5. Is `experiment_contract.yaml` immutable after Stage 9?
   - Recommendation: yes. Stage 10 and Stage 12 read it and verify its hash.
   - If Stage 13 needs a changed contract, it should create a new contract
     version and a new sealed candidate chain.

## Claude Review Prompt

```text
You are an AI for Science system architect and senior code reviewer. Review
`docs/STAGE10_12_EXPERIMENT_RUNTIME_REDESIGN.md` against the current
AutoResearchClaw-v2-clean codebase.

Focus on architecture correctness, not wording.

Please answer:

## Verdict
## Correct mappings from GEWU research OS to AutoResearchClaw
## Incorrect or risky assumptions
## Stage 10/12 boundary risks
## Minimum viable implementation plan changes
## Files/functions to change
## Tests required
## What must not be loosened
## Blocking questions

Constraints:

- Do not suggest weakening release_check gates.
- Do not allow Stage 10 smoke artifacts into claims/evidence.
- Do not treat synthetic data as release-grade real evidence.
- Do not recommend more free-form repair loops as the primary fix.
- Prefer deterministic scaffold + bounded plugin + evaluator-driven selection.
```
