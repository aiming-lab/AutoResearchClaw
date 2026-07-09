# Release Hardening Checklist

Status: draft for review

This document defines the hardening work required before a generated paper run can be treated as release-ready or submission-ready. It is intentionally stricter than the demo pipeline behavior.

## Scope

The release path must protect three boundaries:

- generated experiment code must not run with more host access than the user explicitly requested;
- paper deliverables must not be packaged or marked successful after degraded, failed, stale, or partially verified runs;
- Markdown, LaTeX, bibliography, and manifest outputs must be produced from the same verified canonical source.

The intended implementation order is:

1. Require Docker or explicit unsafe subprocess opt-in.
2. Pass only a minimal environment to generated experiment code.
3. Disable graceful degradation for release runs.
4. Return a distinct non-success status for degraded runs.
5. Block release on fabrication flags unless explicitly reviewed and resolved.
6. Fail citation verification when the paper cites keys but no bibliography exists.
7. Do not package deliverables after failed or degraded hard gates.
8. Generate Markdown and LaTeX from one verified canonical source.
9. Add one release check command that enforces all release criteria.

## Hardening Items

### H1. Docker Must Not Silently Fall Back To Subprocess

Risk: `experiment.mode: docker` currently falls back to `ExperimentSandbox` when Docker is unavailable. The fallback runs generated code as a local subprocess, losing Docker network and resource isolation.

Target files:

- `researchclaw/experiment/factory.py`
- `researchclaw/config.py`
- `config.researchclaw.example.yaml`
- tests under `tests/test_*sandbox*.py` or a new focused test file

Required behavior:

- Docker mode must fail hard when Docker is unavailable unless a clearly named unsafe opt-in is set.
- The unsafe opt-in must be visible in config and logs.
- Release mode must reject unsafe fallback even if dev mode allows it.

Release check:

- Fail if run metadata indicates requested Docker mode but actual backend is subprocess.
- Fail if unsafe fallback was used.
- Current implementation does not write this backend metadata. Until H1 metadata exists, release check must fail closed when it cannot confirm the actual backend.

Acceptance:

- A Docker-unavailable run with `mode: docker` exits non-zero by default.
- A Docker-unavailable run only falls back when the unsafe opt-in is explicitly set.
- Run metadata records requested backend, actual backend, and whether fallback occurred.

### H2. Generated Code Must Receive A Whitelisted Environment

Risk: `ExperimentSandbox` builds the child environment from `os.environ`, so generated experiment code can inherit API keys and other host secrets.

Target files:

- `researchclaw/experiment/sandbox.py`
- `researchclaw/experiment/docker_sandbox.py` if environment forwarding exists there
- tests covering environment leakage

Required behavior:

- Subprocess sandbox must start from a minimal allowlist.
- Default allowlist should include only execution essentials such as `PATH`, selected Python runtime variables, and explicitly configured experiment variables.
- Secrets such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `S2_API_KEY`, provider keys, tokens, and `.env` values must not be inherited by generated code by default.

Release check:

- Fail if run metadata does not record the sandbox environment policy.
- Fail if environment policy is `inherit_all` or equivalent.
- Current implementation does not write this environment-policy metadata. Until H2 metadata exists, release check must fail closed when it cannot confirm the policy.

Acceptance:

- A generated script that prints known API-key environment variables sees them as unset by default.
- Explicitly configured safe variables still pass through.
- Run metadata records the environment policy and allowlist name used for generated code.

### H3. Release Runs Must Not Degrade Gracefully

Risk: Stage 20 returns `DONE` with `decision="degraded"` when `research.graceful_degradation` is true. This produces a successful stage status while writing `run_dir/degradation_signal.json`.

Target files:

- `researchclaw/config.py`
- `researchclaw/cli.py`
- `researchclaw/pipeline/stage_impls/_review_publish.py`
- `config.researchclaw.example.yaml`

Required behavior:

- Release mode must set `graceful_degradation=false`.
- Degraded runs must not be considered release-ready.
- The degradation signal path is `run_dir/degradation_signal.json`, not `stage-20/` and not `deliverables/`.

Release check:

- Read `run_dir/degradation_signal.json`.
- Read `pipeline_summary.json` and reject `degraded: true`.
- Reject Stage 20 result decision `degraded` if available in stage metadata.
- Do not rely on stage status alone because degraded Stage 20 is still `DONE`.

Acceptance:

- A degraded run exits with a distinct non-success status or fails release check.
- A clean pass has no degradation signal.

### H4. Stale Degradation Signals Must Be Detected

Risk: `degradation_signal.json` is written at run root and may survive clean resume attempts when the same `run_dir` is reused.

Target files:

- future `scripts/release_check.py`
- optional run-start cleanup in `researchclaw/pipeline/runner.py`

Required behavior:

- Release check must treat stale degradation signals as a hard failure or require cleanup plus rerun.
- Staleness should compare `degradation_signal.json.generated` with `stage-20/quality_report.json.generated` when both timestamps exist.
- Release-mode invariant: any `run_dir/degradation_signal.json` is a blocker. Timestamp comparison is only for diagnostics, separating active degraded state from stale residue.

Release check:

- Fail if `degradation_signal.json` exists and is newer than or equal to the active Stage 20 quality report.
- Fail as `stale_artifact` if the signal exists but appears older than the active Stage 20 report.
- Never ignore the signal merely because `deliverables/` lacks a copy.

Acceptance:

- A reused run directory with a leftover degradation signal is not marked release-ready.

### H5. CLI Exit Code Must Distinguish Clean Success From Degraded Success

Risk: CLI currently returns success when no stage has `FAILED`, so degraded Stage 20 can produce exit code 0.

Target files:

- `researchclaw/cli.py`
- `researchclaw/pipeline/runner.py`

Required behavior:

- Clean success returns 0.
- Degraded completion returns a distinct non-zero code, or release mode returns non-zero.
- Paused, failed, rejected, blocked, and degraded states must not all collapse into the same success semantics.

Release check:

- Do not trust only CLI exit code until this is implemented.
- Inspect `pipeline_summary.json`, `degradation_signal.json`, and stage metadata.

Acceptance:

- CI can distinguish clean pass from degraded pass without parsing logs.

### H6. Citation Verification Must Fail On Broken Citation Chains

Risk: Stage 23 currently treats missing `references.bib` as "nothing to verify" with `integrity_score: 1.0`. If the paper body contains citation keys, this is a broken citation chain, not a clean no-reference paper.

Target files:

- `researchclaw/pipeline/stage_impls/_review_publish.py`
- `researchclaw/literature/verify.py` if helper extraction belongs there
- citation verification tests

Required behavior:

- If `paper_final.md` or `paper.tex` contains `\cite{...}` or `[keyYYYY...]` and no bibliography exists, Stage 23 must fail.
- If no citations are present and no bibliography exists, Stage 23 may pass with an explicit no-citation status.
- Verification report must distinguish `no_references` from `broken_citation_chain`.

Release check:

- Extract citation keys from `stage-22/paper_final.md`, `stage-22/paper.tex`, and `deliverables/paper.tex` when present.
- Strip fenced code blocks and display math before extracting Markdown citation keys to reduce false positives from examples.
- Split multi-key LaTeX citations such as `\cite{a,b}`.
- Fail if any cite key lacks a verified bibliography entry.
- Fail if `verification_report.json` is missing.
- Parse `verification_report.json`; do not trust report existence alone.
- Do not use `summary.integrity_score` as a release pass signal because missing-bibliography paths can report `1.0`.
- Fail on `summary.hallucinated > 0`. Define a release policy for `summary.suspicious` before allowing release.

Acceptance:

- A paper with `\cite{smith2024}` and no BibTeX exits non-zero or fails release check.
- A report with hallucinated citations fails release check even if `integrity_score` is high.

### H7. Do Not Package Deliverables After Failed Or Degraded Hard Gates

Risk: deliverables packaging runs after pipeline termination and may package best-available Stage 22 outputs even when Stage 23 failed or a hard gate degraded.

Target files:

- `researchclaw/pipeline/runner.py`
- future `scripts/release_check.py`

Required behavior:

- Release packaging must run only after all hard gates pass.
- If packaging still runs for debugging, its manifest must state `release_ready: false`.
- `deliverables/manifest.json` must not imply release readiness by existence alone.

Release check:

- Release check applies only to complete paper runs. A run intentionally stopped with `--to-stage` is not release-ready.
- Fail if `deliverables/` exists but `pipeline_summary.final_stage != 23` or `pipeline_summary.final_status != "done"`.
- Fail if `verification_report.json` is missing from deliverables for a cited paper.
- Fail if any failed stage appears in `pipeline_summary.json`.

Acceptance:

- Failed Stage 23 does not produce a release-ready deliverables manifest.

### H8. Markdown And LaTeX Must Come From One Verified Canonical Source

Risk: `paper_final.md` and `paper.tex` can be sanitized at different strengths. The LaTeX path may replace numbers that remain in Markdown.

Target files:

- `researchclaw/pipeline/stage_impls/_review_publish.py`
- `researchclaw/pipeline/runner.py`
- template conversion tests

Required behavior:

- Stage 22 or Stage 23 must write a verified canonical source.
- Markdown and LaTeX deliverables must be generated from that same canonical source.
- Sanitization must happen before format split, not independently per output format.

Release check:

- Prefer checking canonical-source provenance over comparing numeric strings.
- Fail if deliverables were generated from mixed sources or if canonical source metadata is missing.

Acceptance:

- The same removed or sanitized claim is absent from both Markdown and LaTeX deliverables.

### H9. Critical Non-Blocking Exceptions Must Become Visible

Risk: broad `except Exception` blocks make delivery and verification failures easy to miss.

Target files:

- `researchclaw/pipeline/runner.py`
- `researchclaw/pipeline/stage_impls/_review_publish.py`
- possibly a shared diagnostics module

Required behavior:

- Critical post-run failures must be recorded in a structured run-level error report.
- Release mode must treat packaging, citation verification, compile verification, and canonical-source generation failures as blocking.
- Demo mode may keep some best-effort behavior, but it must not be indistinguishable from release mode.
- Stage 22 should write a machine-readable compile status artifact. Current compile failure behavior is log/comment based and is not sufficient for release checks.

Release check:

- Read a structured run errors file once implemented.
- Until then, fail on missing expected artifacts rather than trusting logs.
- Until `compile_status.json` exists, require a clearly successful compile artifact for submission packages, or fail closed when compilation success cannot be confirmed.

Acceptance:

- A failed deliverables copy, failed compile verification, or failed citation verification is visible in machine-readable output.

### H10. Fabrication Flags Must Block Release Unless Resolved

Risk: Stage 20 writes `fabrication_flags.json`, and Stage 22 consumes it for downstream sanitization. A run that detected suspected fabrication and then sanitized output can otherwise pass checks that only look at degradation, citation verification, and deliverables.

Target files:

- `researchclaw/pipeline/stage_impls/_review_publish.py`
- `researchclaw/pipeline/runner.py`
- future `scripts/release_check.py`
- tests covering fabricated, sanitized, and clean runs

Required behavior:

- Stage 20 fabrication flags must be part of release readiness.
- Release mode must fail if fabrication is suspected, unless a later explicit review artifact records that the issue was resolved.
- Sanitization is not the same as clean evidence. It may make debug deliverables safer to inspect, but it should not by itself make a run submission-ready.

Release check:

- Read `run_dir/stage-20/fabrication_flags.json`.
- Fail if `fabrication_suspected` is true.
- Fail if the file is missing for a complete release run, because the anti-fabrication gate did not run or did not persist its decision.
- If sanitization occurred, require explicit release policy before passing; do not silently treat `sanitization_report.json` as a clean bill of health.

Acceptance:

- A run with suspected fabrication fails release check even when Stage 22 produced sanitized Markdown or LaTeX.
- A clean run has `fabrication_suspected: false` and passes this gate.

## Release Check Contract

Add a command such as:

```bash
python3 scripts/release_check.py /path/to/run_dir
```

Inputs:

- `run_dir/pipeline_summary.json`
- `run_dir/degradation_signal.json`
- `run_dir/stage-20/quality_report.json`
- `run_dir/stage-20/fabrication_flags.json`
- `run_dir/stage-22/sanitization_report.json`
- `run_dir/stage-22/compilation_quality.json`
- future `run_dir/stage-22/compile_status.json`
- `run_dir/stage-22/paper_final.md`
- `run_dir/stage-22/paper.tex`
- `run_dir/stage-23/verification_report.json`
- `run_dir/stage-23/references_verified.bib`
- `run_dir/deliverables/manifest.json`

Required output:

- exit 0 only for release-ready runs;
- exit 1 for failed or incomplete runs;
- exit 2 for degraded runs, if a distinct degraded code is adopted;
- JSON report option for CI.

Minimum checks:

- release check applies only to complete runs, not `--to-stage` partial runs;
- final stage is `23` and final status is `done`;
- no failed stages;
- no degraded summary state;
- no active or stale `degradation_signal.json`;
- Stage 20 quality report exists and passes threshold;
- Stage 20 fabrication flags exist and do not indicate suspected fabrication;
- required JSON inputs must be read with fail-closed semantics; missing or invalid required artifacts cannot be silently skipped by downstream checks;
- Stage 22 sanitization report is parsed; sanitization is a review flag, not automatic release approval;
- at least one paper artifact (`paper_final.md` or `paper.tex`) exists and is non-empty;
- Stage 23 verification report exists and reports no hallucinated citations;
- suspicious citations have an explicit release policy before pass;
- cited keys in paper have bibliography coverage;
- no missing verified bibliography for cited papers;
- sandbox backend metadata exists, confirms no unsafe Docker fallback, and fails closed when absent;
- sandbox environment metadata exists, confirms no inherited-all environment policy, and fails closed when absent;
- deliverables exist only after hard gates pass;
- Markdown and LaTeX provenance points to the same verified canonical source once implemented;
- compile status is machine-readable once implemented, or release check fails if it cannot confirm successful compilation for a submission package.

Known v1 behavior:

- `release_check.py` is intentionally stricter than the current pipeline.
- Until H1/H2/H8/H9 instrumentation lands, current pipeline outputs are expected to fail `release_check.py` with metadata-missing or compile-status-missing errors, even when the run otherwise looks clean.
- Treat those failures as instrumentation gaps, not checker bugs.
- Do not weaken `release_check.py` to pass those runs; implement the missing pipeline artifacts instead.

## v2 Gates (implemented)

release_check v2 adds machine-readable contracts extracted from the
governance-layer audit skills. New required artifacts for a complete run:

- `run_manifest.json` (run root) — drives `expected_final_stage` (now 25);
  declares writer/critic models. Distinct from `deliverables/manifest.json`.
- `stage-15/critique.json` — Socratic critic findings (recommend-only),
  produced by an isolated critic model (`llm.critic_model`) or an external
  reviewer with its own artifact.
- `stage-24/claims.json` — claim ledger; release-scoped types
  (quantitative / comparative / result / citation) must carry run-internal
  evidence pointers (path + sha256). Orphan pointers fail.
- `stage-24/citations.json` — citation *instances*; each is `claim_support`
  (requires claim id + excerpt) or `background`. Citation existence
  (stage 23) is never accepted as citation support.
- `stage-24/critique_resolution.json` — every P0/P1 finding resolved as
  fixed / rebutted / accepted-risk.
- `stage-24/truth_audit.json` — frozen paper hash + claims digest.
- `stage-25/deai_audit.json` — recommend-only prose audit; paper hash must
  equal the truth-audit hash (truth-before-prose invariant). If de-AI
  suggestions are adopted: claim/citation-touching edits re-run stages
  23+24; style-only edits re-run stage 24.
- `attempts/attempt_log.jsonl` — append-only attempt log incl. failures.
- `cost_log.jsonl` — per-stage cost entries (missing = warning only, by
  design: cost pressure must not incentivize skipping audit rounds).

Behavioral hardening shipped with v2:

- `has_real_data=false` is now a release **error**; a signed
  `waivers/no_real_data.json` (`reason` + `approved_by`) downgrades it to a
  warning. This waiver is an explicit human governance exception: pipeline
  stages, repair scripts, release helpers, and generated run artifacts must not
  auto-create it.
- CLI exits 2 for degraded runs (previously 0).
- Docker-unavailable no longer silently falls back to subprocess
  (`experiment.sandbox.allow_docker_fallback`, default false; fallback is
  recorded and blocks release).
- Subprocess sandbox uses an env allowlist (`experiment.sandbox.env_policy`).
- Stage 22 always writes `compile_status.json`; the
  `compilation_quality.json` inference shortcut was removed.
- Stage 22 fails instead of writing a placeholder final paper; release_check
  additionally rejects placeholder markers.
- Stage 23 fails when the paper cites keys but no bib exists (integrity 0.0,
  not 1.0), and never restores the unverified original bib after heavy
  stripping (`bib_strip_warning.json` records the event).

Gate-change discipline: see "Release-Gate Change Discipline" in
CONTRIBUTING.md — gate changes land in isolated commits, never alongside
run artifacts.

## Suggested Claude Review Prompt

Please review `docs/RELEASE_HARDENING_CHECKLIST.md` as a critical release-safety reviewer. Focus on:

- missing hard gates that could allow an unsafe or unverified paper package;
- incorrect assumptions about actual file paths or artifact fields;
- checks that would produce high false-positive or false-negative rates;
- implementation ordering problems;
- places where demo-mode behavior is being confused with release-mode behavior.

Implementation facts already verified before this review:

- `pipeline_summary.json.final_stage` is an integer stage number, so the release endpoint check should use `final_stage == 23 and final_status == "done"`.
- `deliverables/manifest.json` currently has no release-readiness semantics. Its existence must not be treated as a pass signal.
- In release mode, any `run_dir/degradation_signal.json` should fail the release check. Timestamp comparison only distinguishes active degraded state from stale residue; both are release blockers.
- `stage-20/fabrication_flags.json` is a real artifact and must be included in release readiness checks.
- Sandbox backend and environment-policy metadata are not currently persisted. Release check must fail closed on missing metadata until that instrumentation exists.
- Machine-readable compile status is not currently persisted on compile failure. Release check must fail closed when successful compilation cannot be confirmed for a submission package.
- Release check fail-closed behavior currently depends on required artifacts being loaded as required; reviewers should flag any change that turns required Stage 20/23 inputs into optional reads.
- Paper body artifacts are release-critical: a manifest and JSON reports are not enough if no non-empty `paper_final.md` or `paper.tex` exists.

Do not rewrite the document for style. Return findings ordered by severity, with file/line references when possible.
