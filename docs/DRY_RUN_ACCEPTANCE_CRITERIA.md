# Synthetic Dry-Run Acceptance Criteria

Status: active
Applies to: next `pipeline_validation` + `synthetic` dry-run before TrustHub

## Goal

Run the full 25-stage pipeline on synthetic data. The run must complete
all 25 stages successfully. `release_check` must fail — but **only** for
identity-level reasons (the run is synthetic/pipeline_validation by design),
not for flow or quality reasons.

## Acceptance: release_check error codes

### Must NOT appear (flow/quality failures — fix before proceeding)

| Error code | Meaning | Fix category |
|------------|---------|--------------|
| `incomplete_run` | Pipeline did not reach stage 25 | Stage execution |
| `failed_stages` | One or more stages failed | Stage execution |
| `claims_empty` | Stage 24 extracted zero claims | Claim extraction |
| `claims_unsupported` | Claims exist but none have evidence | Provenance closure |
| `claims_supported_without_evidence` | status=supported with no valid pointer | Provenance closure |
| `claims_numeric_not_closed` | Quant/comparative claim not grounded | Number grounding |
| `claims_numeric_evidence_value_missing` | matched_value not in evidence file | Number grounding |
| `claims_orphan_evidence` | Evidence pointer broken | Artifact integrity |
| `claims_disallowed_evidence_path` | Evidence from wrong stage | Evidence allowlist |
| `citation_support_instances_missing` | Paper has citations but no bindings | Citation mapping |
| `citation_support_unmapped` | Instances not mapped to claims | Citation mapping |
| `citation_support_invalid` | claim_support without excerpt | Citation mapping |
| `citation_support_excerpt_fabricated` | Excerpt not a real substring | Citation mapping |
| `citation_background_not_whitelisted` | Background escape without whitelist | Citation mapping |
| `quality_below_threshold` | Score < 6.0 | Paper quality / grounding |
| `quality_verdict_not_release_ready` | LLM says Reject/Revise | Paper quality / grounding |
| `degradation_signal` | degradation_signal.json exists | Quality gate |
| `degraded_summary` | pipeline_summary.degraded=true | Quality gate |
| `fabrication_suspected` | fabrication_flags says suspected | Experiment integrity |
| `no_real_data` | has_real_data=false without waiver | Experiment integrity |
| `paper_hash_mismatch` | Paper changed between Stage 24 and 25 | Audit ordering |
| `truth_audit_paper_path_missing` | Stage 24 cannot find paper | Artifact routing |
| `hallucinated_citations` | Verification found hallucinated refs | Citation generation |
| `sandbox_metadata_missing` | No sandbox provenance | Metadata emission |
| `deliverables_not_flagged_not_release_ready` | Deliverables mislabeled | Deliverables |
| `missing_artifact` (for required files) | run_manifest.json etc. missing | Artifact emission |
| `compile_status_missing` | Stage 22 did not write compile_status.json | Artifact emission |
| `paper_artifact_placeholder` | Paper is empty/placeholder | Paper generation |
| `canonical_source_metadata_missing` | No source/hash provenance chain | Artifact emission |

### MAY appear (identity-level, expected for synthetic/pipeline_validation)

| Error code | Meaning | Why expected |
|------------|---------|--------------|
| `non_release_claim_scope` | claim_scope ≠ research_release | By design: pipeline_validation cannot release |
| `compile_toolchain_missing` | pdflatex not installed | Environment gap only (not a flow issue); must be resolved before real release |

### If `compile_toolchain_missing` appears

This is acceptable as a temporary environment limitation but must be flagged
as a known gap. Before TrustHub release run, either:
- Install TeX Live / pdflatex on the run host, or
- Run LaTeX compilation in CI and copy `compile_status.json` back

## Verification commands

```bash
# 1. Run the full pipeline
python -m researchclaw run -c config.deepseek.yaml -o runs/hwsec-dry-v1

# 2. Confirm all 25 stages completed
python3 -c "
import json
d = json.load(open('runs/hwsec-dry-v1/pipeline_summary.json'))
assert d['final_stage'] == 25, f'final_stage={d[\"final_stage\"]}'
assert d['final_status'] == 'done', f'final_status={d[\"final_status\"]}'
assert d['stages_failed'] == 0, f'stages_failed={d[\"stages_failed\"]}'
assert not d.get('degraded'), 'pipeline is degraded'
print('OK: 25 stages completed, no failures, no degradation')
"

# 3. Confirm Stage 24 extracted claims
python3 -c "
import json
d = json.load(open('runs/hwsec-dry-v1/stage-24/claims.json'))
assert d['counts']['total'] > 0, f'claims total={d[\"counts\"][\"total\"]}'
print(f'OK: {d[\"counts\"][\"total\"]} claims extracted')
"

# 4. Confirm Stage 25 completed with hash invariant
python3 -c "
import json
d = json.load(open('runs/hwsec-dry-v1/stage-25/deai_audit.json'))
assert d['hash_invariant_ok'], 'hash invariant broken'
print('OK: Stage 25 de-AI audit completed, hash invariant holds')
"

# 5. Run release_check and verify only expected failures
python scripts/release_check.py runs/hwsec-dry-v1 --json > /tmp/rc.json 2>&1
python3 -c "
import json
d = json.load(open('/tmp/rc.json'))
errors = {f['code'] for f in d['findings'] if f['severity'] == 'error'}
expected = {'non_release_claim_scope'}
env_acceptable = {'compile_toolchain_missing'}
unexpected = errors - expected - env_acceptable
assert not unexpected, f'UNEXPECTED ERRORS: {sorted(unexpected)}'
print(f'OK: release_check errors = {sorted(errors)}')
if errors & env_acceptable:
    print(f'NOTE: environment gaps present: {sorted(errors & env_acceptable)}')
"

# 6. Confirm paper numbers are grounded in experiment evidence
python3 -c "
import json, re
paper = open('runs/hwsec-dry-v1/stage-23/paper_final_verified.md').read()
results = json.load(open('runs/hwsec-dry-v1/stage-12/runs/results.json'))
metrics = results.get('metrics', {})
# Extract numbers from paper that look like metrics (0.xxx format)
paper_numbers = {float(m) for m in re.findall(r'0\.\d{2,4}', paper)}
metric_values = {round(v, 4) for v in metrics.values() if isinstance(v, (int, float))}
unsupported = {n for n in paper_numbers if not any(abs(n - m) < 0.002 for m in metric_values)}
# Filter out common non-metric numbers
unsupported -= {0.05, 0.01, 0.001, 0.95, 0.99, 0.10, 0.50}
assert len(unsupported) <= 3, f'Too many unsupported numbers in paper: {sorted(unsupported)}'
print(f'OK: paper numbers grounded (unsupported={sorted(unsupported)})')
"
```

## What must be fixed before this dry-run can pass

### Priority 1: Stage 24 claim extraction

Root cause: DeepSeek returned empty/unparseable JSON for claim extraction.

Fix approaches (try in order):
1. Add JSON repair to `_chat_json` in `_release_audit.py` (strip trailing
   commas, remove `// comments`, handle markdown fences around JSON)
2. If paper text was sanitized with `---` placeholders, ensure extraction
   prompt handles this (look at prose claims, not just table numbers)
3. Add deterministic/heuristic fallback: extract sentences containing numbers
   from Results/Abstract sections using regex, classify as quantitative

Do NOT: make empty claims non-blocking for any claim_scope.

### Priority 2: Paper number grounding

Root cause: Stage 16-19 paper writing LLM hallucinated metrics not present
in `experiment_summary_best.json` or `stage-12/runs/results.json`.

Fix approaches:
1. Stage 16 prompt must inject actual metrics from experiment evidence
   artifacts, with explicit instruction: "Use ONLY these numbers in Results"
2. Stage 22 paper verifier (`paper_verifier.py`) already catches unverified
   numbers and writes `---`. Ensure Stage 20 does not give score=1 for a
   paper whose tables are all `---` — instead, if all result numbers are
   sanitized, the paper needs regeneration (Stage 16 retry with grounded
   metrics), not just sanitization.

Do NOT: weaken Stage 20 quality threshold or remove the sanitizer.

### Priority 3: Stage 24/25 artifact completeness

Ensure Stage 24 writes all four artifacts even on partial success:
- `claims.json` (may have claims with status=unsupported)
- `citations.json` (citation instance bindings)
- `critique_resolution.json` (resolution of Stage 15 critique findings)
- `truth_audit.json` (paper hash + digest freeze)

Stage 25 requires `truth_audit.json` with `paper_sha256`. If Stage 24
completes (even with some unsupported claims), Stage 25 must run.

### Priority 4: Quality gate must pass (score ≥ 6.0)

If paper numbers are grounded (Priority 2 fixed), the LLM should not give
score=1 or verdict="Reject". If it still does, the problem is paper prose
quality — address in Stage 16-19 generation prompts.

Do NOT: lower `quality_threshold` below 6.0.

## What comes AFTER this dry-run passes

1. Install pdflatex (or confirm CI compilation) to resolve `compile_toolchain_missing`
2. Prepare TrustHub small-sample data loader in scaffold
3. New config: `config.trusthub.yaml` with `dataset_origin: public`,
   `claim_scope: research_release`
4. Run `--to-stage EXPERIMENT_RUN` first to validate data loading + metrics
5. Full 25-stage run targeting `release_check status=pass`

## Non-negotiable invariants (do not change to make the dry-run pass)

- Stage 24 must fail on empty claims when LLM is available
- Stage 20 quality threshold remains 6.0
- `non_release_claim_scope` gate stays
- `research_release + synthetic` stays blocked
- Stage 12 reads only sealed selected_candidate
- Stage 10 smoke/candidates never become evidence
- Plugin cannot self-grade
- Paper numbers must be grounded in run-internal evidence
- `graceful_degradation` does not exempt the run from release_check failures
