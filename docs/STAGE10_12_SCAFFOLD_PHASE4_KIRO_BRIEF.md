# Stage 10/12 Phase 4 Brief for Kiro

Status: draft for Kiro architecture review

Purpose: explain the latest real DeepSeek failure, separate it from the
already-fixed release gates, and propose the smallest scaffold-owned evaluator
path that can make Stage 10/12 reliable without weakening anti-fabrication
checks.

## Executive Verdict

The latest `runs/hwsec-runtime-v3` failure is not a release-check problem and
not a reviewer-passthrough problem. The new `pipeline_validation` passthrough
worked as intended: the CodeAgent reviewer returned `REVISE`, Stage 10 recorded
`reviewer_critique.json`, and the reviewer critique did not block by itself.

The actual blocker is lower level: Stage 10 generated an invalid experiment
package whose `main.py` is actually config content. The final readiness gate
correctly rejected it because there is no executable entry point.

This confirms the architecture diagnosis in
`docs/STAGE10_12_EXPERIMENT_RUNTIME_REDESIGN.md`: Stage 10 still asks the LLM
to create a complete experiment repository. That is too much authority. The
next fix should move execution ownership into deterministic scaffold code and
let the LLM generate only a bounded detector plugin.

## Latest Failure Recap

Run:

- Output directory: `runs/hwsec-runtime-v3`
- Run ID: `rc-20260710-022114-45409b`
- Command target: `--to-stage EXPERIMENT_RUN`

Observed stage progression:

- Stages 1-9 completed.
- Stage 9 wrote `experiment_contract.yaml` and
  `experiment_contract.sha256`.
- Stage 10 started and reached CodeAgent review.
- Reviewer verdict was `REVISE`, but because the contract was
  `claim_scope: pipeline_validation`, the verdict was recorded as critique
  metadata rather than a blocker.
- Stage 10 then failed final readiness because `main.py` had no executable
  `if __name__ == "__main__"` entry point.
- Stage 12 never ran.

Relevant artifacts:

- `runs/hwsec-runtime-v3/stage-09/experiment_contract.yaml`
- `runs/hwsec-runtime-v3/stage-10/reviewer_critique.json`
- `runs/hwsec-runtime-v3/stage-10/stage10_blockers.json`
- `runs/hwsec-runtime-v3/stage-10/experiment/main.py`
- `runs/hwsec-runtime-v3/stage-10/code_agent_log.json`

Key facts from the artifacts:

```yaml
# stage-09/experiment_contract.yaml
claim_scope: pipeline_validation
dataset_origin: synthetic
evaluator:
  command: python main.py
  owner: scaffold
```

```json
// stage-10/reviewer_critique.json
{
  "claim_scope": "pipeline_validation",
  "verdict": "REVISE",
  "passthrough": true,
  "passthrough_reason": "pipeline_validation validates engineering flow only; not release-grade",
  "critical_issues": [
    "No evaluation code or metric computation ...",
    "Detectors assume batch processing ...",
    "The data loader uses a fixed random seed ...",
    "SPECTECTOR handles inconsistent spectrogram dimensions ..."
  ]
}
```

```json
// stage-10/stage10_blockers.json
{
  "status": "failed",
  "blockers": [
    "main.py has no executable `if __name__ == '__main__'` entry point",
    "main.py has no executable `if __name__ == '__main__' entry point"
  ]
}
```

The packaged `stage-10/experiment/main.py` begins with:

```python
# config.py
import numpy as np

class Config:
    ...
```

There is no main guard and no experiment runner. The readiness gate is correct
to fail.

## What This Failure Means

This failure validates two previous fixes and exposes one remaining architecture
gap.

Validated fixes:

- Reviewer `REVISE` under `pipeline_validation` no longer blocks Stage 10 by
  itself. It is preserved in `reviewer_critique.json`.
- Review repair no longer overwrites a previously executable `main.py` with a
  non-executable replacement.

Remaining gap:

- The initial single-shot generation path can still produce a non-executable
  `main.py`. If the baseline file is already wrong, later "preserve executable
  main" logic has nothing good to preserve.

The CodeAgent log explains the route into this state:

- Blueprint YAML parse failed.
- Sequential generation was abandoned.
- CodeAgent fell back to single-shot generation.
- The single-shot package put config content in `main.py`.
- Exec-fix reported "code runs OK", but that only means the file can execute
  and exit without crashing. It does not mean an experiment ran or wrote valid
  results.
- Hard validation later caught the missing main guard.

This is not an isolated bug. It is the same class of failure seen in earlier
real runs:

- missing `dataset_origin`
- stale or empty metrics
- hardcoded or self-reported metrics
- smoke timeout
- `ExperimentHarness` API misuse
- reviewer semantic rejection
- `main.py` being replaced with config-only content

These are all symptoms of the same design issue: the LLM is still responsible
for entrypoint structure, evaluator logic, result schema, and metric semantics.

## Why More Prompt Repair Is The Wrong Primary Fix

Prompt repair can reduce one failure surface at a time, but it cannot establish
the experiment invariants that matter:

- exactly one entrypoint
- known result schema
- scaffold-owned metric computation
- no self-grading
- deterministic train/test split
- seed sweep controlled by the runtime
- `dataset_origin` and `claim_scope` written by the contract/runtime, not by
  free-form model code
- Stage 10 smoke artifacts never becoming Stage 12 evidence

If Stage 10 keeps asking the LLM to generate the full experiment repository,
the system will keep rotating through new failure modes. The correct boundary
is: scaffold owns execution and evaluation; LLM owns only the method plugin.

## Proposed Phase 4 Minimal Design

Phase 4 should introduce a scaffold-owned evaluator for `pipeline_validation`.
It should not loosen any release gate.

### Ownership Boundary

Scaffold-owned:

- `main.py`
- result writing
- `results.json` schema
- `dataset_origin`
- `claim_scope`
- train/test split
- synthetic data generation for pipeline validation
- labels and scoring function
- primary metric computation
- latency measurement definition
- seed sweep
- timeout handling
- artifact paths

LLM-owned:

- `detector_plugin.py`
- a bounded class/function implementing the detector method
- optional model hyperparameters
- optional `describe()` metadata

Forbidden for plugin:

- writing `results.json`
- reading or modifying the experiment contract
- receiving `y_test` directly
- computing the primary metric
- changing output paths
- creating release evidence artifacts
- importing network or subprocess APIs

Phase 4 must define the banned-import policy as code, not prose. The initial
policy should start from the existing sandbox/security validator and include at
least:

```python
PLUGIN_BANNED_IMPORTS = frozenset({
    "subprocess",
    "os.system",
    "os.popen",
    "shutil.rmtree",
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "multiprocessing",
    "threading",
    "ctypes",
    "cffi",
    "importlib",
})
```

The exact implementation can reuse or extend the existing validator, but the
default must be deny-by-default for process, network, dynamic import, and FFI
escape paths.

### Plugin API

Initial API for HPC anomaly-detection experiments:

```python
class DetectorPlugin:
    name = "candidate_name"

    def fit(self, X_train, y_train):
        ...
        return self

    def predict(self, X_test):
        ...
        return y_pred

    def describe(self):
        return {
            "method": "...",
            "assumptions": [...],
        }
```

Stage 10 validates only this bounded API:

- file is importable
- `DetectorPlugin` exists
- `fit`, `predict`, and `describe` exist
- `predict(X_test)` returns the expected length
- output values are finite and label-like
- no banned imports
- no file writes outside allowed scratch

### Deterministic Scaffold Layout

Proposed Stage 10 selected candidate layout:

```text
stage-10/
  selected_candidate/
    detector_plugin.py
    main.py                  # scaffold-owned
  selected_candidate_manifest.json
  reviewer_critique.json     # optional diagnostic only
  candidates/
    cand-0001/
      detector_plugin.py
      attempt.json
```

`selected_candidate/main.py` must be generated by AutoResearchClaw, not by the
LLM. It imports `detector_plugin.py`, runs the scaffold-owned evaluator, and
writes `results.json`.

`selected_candidate/` must remain flat. It may contain only top-level files
listed in the manifest. Directories under `selected_candidate/` are rejected by
Stage 12. This preserves the existing sealed-candidate boundary and prevents a
plugin from hiding auxiliary code or stale outputs in nested paths.

The selected manifest must seal:

- `detector_plugin.py` hash
- scaffold `main.py` hash
- `experiment_contract.yaml` hash
- scaffold template/version hash
- candidate ID
- model used to generate the plugin

For Phase 4, `scaffold_sha256` must cover the complete scaffold, not just one
harness/template file. Prefer explicit per-file ownership:

```json
{
  "scaffold_files": {
    "main.py": {"sha256": "...", "owner": "scaffold"},
    "evaluator.py": {"sha256": "...", "owner": "scaffold"},
    "synthetic_hpc.py": {"sha256": "...", "owner": "scaffold"}
  },
  "plugin_files": {
    "detector_plugin.py": {"sha256": "...", "owner": "model"}
  }
}
```

If a combined scaffold hash is used instead, it must be computed from all
runtime-owned scaffold files in deterministic sorted order, and Stage 12 must
recompute the same digest before execution.

Stage 12 must continue to read only the sealed candidate manifest and must
recompute results in a fresh sandbox.

### Result Schema

Stage 12 `results.json` minimum:

```json
{
  "schema_version": 1,
  "claim_scope": "pipeline_validation",
  "dataset_origin": "synthetic",
  "dataset_name": "synthetic_hpc_trace_v1",
  "primary_metric": {
    "key": "detection_f1",
    "value": 0.0,
    "direction": "maximize"
  },
  "metrics": {
    "detection_f1": 0.0,
    "tpr": 0.0,
    "tnr": 0.0,
    "fpr": 0.0,
    "latency_ms": 0.0
  },
  "seeds": [42, 123, 256],
  "conditions": [],
  "runtime_sec": 0.0,
  "evaluator_owner": "scaffold"
}
```

For `pipeline_validation`, it is acceptable for the metrics to be weak. It is
not acceptable for metrics to be missing, self-reported by the plugin, stale,
or copied from Stage 10 smoke output.

## Gate Invariants That Must Not Change

Do not weaken these while implementing Phase 4:

- `release_check.py` must reject `claim_scope != research_release`.
- `research_release + synthetic` must remain blocked unless a human waiver
  policy explicitly allows it.
- Stage 12 must not fall back to legacy `experiment/` paths.
- Stage 12 must read only `selected_candidate_manifest.json`.
- `selected_candidate/` must be flat: no subdirectories, and no files outside
  the manifest.
- The manifest must distinguish scaffold-owned files from model-owned plugin
  files and verify every listed hash before execution.
- Stage 10 candidate diagnostics under `stage-10/` must never become claim
  evidence.
- Stage 12 must reject stale `results.json`.
- Stage 12 must reject zero real metrics for research-writing progression.
- The plugin must not grade itself.
- Stage 10 smoke results must not be cited by Stage 13/14/20/22/24.

## Minimal Implementation Plan

### Step 1: Add Scaffold Template

Add a runtime-owned scaffold template, for example:

- `researchclaw/experiment_runtime/scaffold/main_template.py`
- `researchclaw/experiment_runtime/scaffold/synthetic_hpc.py`
- `researchclaw/experiment_runtime/scaffold/evaluator.py`

Keep the first version narrow: HPC anomaly detection only.

The first Phase 4 implementation should support one generated plugin candidate
only. Multi-candidate search can be added after the scaffold-owned evaluator is
itself verified. The manifest layout may remain future-compatible with
`candidates/cand-*`, but Stage 10 should initially select exactly one candidate
or fail.

### Step 2: Change Stage 10 Candidate Sealing

For `claim_scope: pipeline_validation`, Stage 10 should:

1. Ask the LLM for `detector_plugin.py` only.
2. Write scaffold-owned `main.py` into `selected_candidate/`.
3. Run smoke validation through the scaffold.
4. Write `selected_candidate_manifest.json`.
5. Preserve reviewer critique as diagnostic metadata only.

If plugin generation fails, Stage 10 should fail. It should not ask the LLM to
rewrite `main.py`.

For v1, do not implement objective multi-candidate selection yet. The pass/fail
question is whether the single plugin can satisfy the scaffold API and produce
fresh scaffold-computed metrics.

### Step 3: Keep Stage 12 Boundary

Stage 12 should continue to:

1. Load Stage 10 sealed manifest.
2. Verify all hashes.
3. Run selected candidate in a fresh sandbox.
4. Read only the sandbox-produced top-level `results.json`.
5. Verify mtime/result freshness and required schema keys.

### Step 4: Add Tests Before Real DeepSeek Rerun

Required tests:

- Stage 10 selected candidate contains scaffold-owned `main.py`.
- LLM plugin cannot overwrite scaffold `main.py`.
- `selected_candidate/` rejects any subdirectory.
- The manifest rejects unlisted files and scaffold/plugin hash mismatches.
- Stage 10 fails if plugin lacks `DetectorPlugin`.
- Stage 10 fails if plugin imports banned modules.
- Stage 12 rejects a selected candidate whose scaffold hash mismatches.
- Stage 12 rejects a plugin-written/stale `results.json`.
- release_check rejects `pipeline_validation` even when all artifacts exist.

## Kiro Review Questions

Please review these points before implementation:

1. Is it correct to stop repairing the free-form full-project generator and
   move `pipeline_validation` to scaffold-owned evaluator + bounded plugin?
2. Is the proposed plugin API sufficient for the current HPC anomaly-detection
   task?
3. Should Phase 4 initially support only this HPC detector plugin domain, with
   other domains still using the old path but blocked from release?
4. Are the listed non-negotiable gates complete, or is there another evidence
   leak path from Stage 10 diagnostics into Stage 12/claims?
5. Should Stage 10 keep generating multiple plugin candidates immediately, or
   should the first scaffold version support one candidate first and add
   multi-candidate selection after the scaffold boundary is stable?

## Recommended Verdict

Recommended decision: ACCEPT Phase 4 direction, with a narrow first
implementation.

Do not attempt another real DeepSeek rerun before implementing the scaffold
boundary. The latest failure shows that the current free-form project generator
can still produce a structurally invalid `main.py`; another rerun is likely to
surface a different version of the same problem rather than close Stage 10/12.
