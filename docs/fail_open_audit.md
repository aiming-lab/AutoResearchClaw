# Fail-open audit of the experiment path

Scope audited: `execute_pipeline` and helpers that fire between
`Stage.EXPERIMENT_DESIGN` and `Stage.RESULT_ANALYSIS`, plus the stage
implementations for experiment design, code generation, resource planning,
experiment run, iterative refine, result analysis, and sandbox or dataset
helpers called by those stages.

Out of scope per brief: literature, survey, paper writing, export, publishing,
MetaClaw bridge, and the ExperimentSpec v1 gates introduced in the preceding commit (`3724cff`).

Disposition values:

- `converted-surface`: a previously swallowed failure now fails the stage or run.
- `converted-record`: an optional feature now warns once per run and records the
  degradation in `pipeline_summary.json`.
- `removed-dead`: a phantom integration was removed because the module does not
  exist in this fork.
- `kept-with-reason`: the catch is still present and the reason is documented.

| File | Line | What it guards | Failure mode before audit | Disposition | Test | Commit |
|---|---:|---|---|---|---|---|
| `researchclaw/pipeline/runner.py` | 111 | Optional HITL data inside checkpoint write | HITL metadata omitted from checkpoint | kept-with-reason: core checkpoint still writes atomically and the optional HITL adapter must not corrupt checkpoint persistence | N/A | N/A |
| `researchclaw/pipeline/runner.py` | 120 | Checkpoint temp file cleanup | No swallow, temp file is unlinked then exception is re-raised | kept-with-reason: cleanup guard re-raises and is not fail-open | Existing checkpoint tests | N/A |
| `researchclaw/pipeline/runner.py` | 480 | Optional Stage 9 plan load for diagnosis context | Malformed plan omitted from diagnosis context | kept-with-reason: optional context only; diagnosis failures themselves now surface | N/A | N/A |
| `researchclaw/pipeline/runner.py` | 565 | Post-Stage 14 experiment diagnosis | Diagnosis crash logged, pipeline continued | converted-surface | `test_result_analysis_diagnosis_failure_fails_stage` | `e770816` |
| `researchclaw/pipeline/runner.py` | 654 | Post-diagnosis experiment repair | Repair crash logged and printed, pipeline continued | converted-surface | `test_result_analysis_repair_failure_fails_stage` | `0bf2290` |
| `researchclaw/pipeline/runner.py` | 688 | Domain profile setup before stage execution | Adapter/profile setup failure swallowed, run continued with unknown profile state | converted-surface | `test_domain_profile_setup_failure_fails_before_stage` | `3118f97` |
| `researchclaw/pipeline/runner.py` | removed | `researchclaw.pipeline.event_log` phantom import and append hooks | Missing module silently disabled all event logging | removed-dead | Smoke: `test_execute_pipeline_runs_stages_in_sequence` | `53f59a7` |
| `researchclaw/pipeline/runner.py` | 721 | Experiment memory initialization | Wrong constructor call was swallowed, memory never initialized | converted-record | `test_experiment_memory_initialization_failure_is_recorded_once` | `9090fd7` |
| `researchclaw/pipeline/runner.py` | 749 | CLI cost budget enforcer | Missing `researchclaw.cost_tracker` silently disabled configured CLI budgets | converted-surface | `test_cli_cost_budget_without_enforcer_fails_before_stage` | `54bb936` |
| `researchclaw/pipeline/runner.py` | removed | `researchclaw.pipeline.pitfall_detector` phantom import after stages 10 and 12 | Missing module silently disabled pitfall detection | removed-dead | Smoke: `test_execute_pipeline_runs_stages_in_sequence` | `b1587e7` |
| `researchclaw/pipeline/runner.py` | 822 | Experiment memory outcome recording | Missing `ExperimentOutcome` API was swallowed, outcomes were never recorded | converted-record | `test_experiment_memory_records_outcome_after_experiment_stage`; `test_experiment_memory_recording_failure_is_recorded_once` | `6525d02` |
| `researchclaw/pipeline/runner.py` | 861 | Knowledge base stage export | KB export failure was swallowed | converted-record | `test_kb_export_failure_is_recorded_once` | `39f6ee6` |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 117 | Stage 9 domain detection and `domain_profile.json` write | Debug-only fallback to generic domain | kept-with-reason: optional profile enrichment; downstream ExperimentSpec gate still fails invalid experiment plans | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 147 | Domain-specific experiment-design prompt context | Debug-only fallback to generic prompt context | kept-with-reason: optional prompt enrichment, no persisted result data is accepted from this block | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 155 | Optional `dataset_guidance` prompt block | Missing prompt block became empty guidance | kept-with-reason: PromptManager blocks are optional across profiles | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 165 | Optional RL prompt guidance block | Missing block silently omitted extra guidance | kept-with-reason: optional prompt enrichment only | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 190 | Optional framework documentation injection | Failure silently omitted docs | kept-with-reason: optional prompt enrichment; generated plan still passes schema gate | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 397 | BenchmarkAgent fallback domain detection | Debug-only fallback to generic benchmark eligibility | kept-with-reason: fallback only used if the first domain detection did not produce a profile | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 481 | BenchmarkAgent orchestration | Warning logged and Stage 9 continued without BenchmarkAgent selections | kept-with-reason: already visible at warning level and feature is optional | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 491 | Optional `benchmark_plan.json` write | Benchmark plan artifact skipped | kept-with-reason: selected datasets and baselines were already injected into `exp_plan.yaml`; artifact is an optional Stage 10 hint | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 568 | HITL guidance application | Debug-only fallback to original plan | kept-with-reason: optional human guidance file, no stage artifact is corrupted | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_experiment_design.py` | 591 | BaselineNavigator persistence | Navigator UI state skipped | kept-with-reason: HITL workshop state is optional telemetry, not experiment plan data | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 109 | ColliderAgent LLM plan generation | Warning logged, fallback collider plan generated | kept-with-reason: explicit fallback artifact is produced when LLM is unavailable | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 366 | `compute_budget` prompt block | Template fallback used | kept-with-reason: explicit fallback prompt text is generated | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 392 | Offline dataset guidance prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 398 | Full-network dataset guidance prompt blocks | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 404 | Setup-only dataset guidance prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 409 | Docker setup script guidance prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 413 | Hyperparameter reporting prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 418 | Multi-seed enforcement prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment; downstream result gates still validate produced data | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 454 | BenchmarkAgent plan load for Stage 10 | Debug-only omission of extra benchmark prompt block | kept-with-reason: Stage 9 `exp_plan.yaml` remains the authoritative plan | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 479 | Optional RL code-generation prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 494 | Framework API documentation injection | Debug-only omission | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 500 | LLM training guidance prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 504 | LLM eval guidance prompt block | Guidance omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 538 | Domain prompt adapter guidance | Debug-only omission | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 604 | HITL route request for OpenCode | Info logged, OpenCode routing skipped unless configured auto | kept-with-reason: interactive routing is optional and has an explicit auto mode | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 733 | Code search helper | Debug-only omission | kept-with-reason: optional retrieval augmentation | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 735 | Domain detection around code search | Debug-only fallback to no code search | kept-with-reason: optional retrieval augmentation | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 1138 | Deep repair helper | Debug-only omission of advisory repair | kept-with-reason: primary static validation remains active and hard stage failures still return `StageStatus.FAILED` | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 1257 | Review-fix LLM call | Debug-only omission of advisory fix | kept-with-reason: optional review repair after generated code already exists | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 1259 | Code review block | Debug-only omission of advisory review | kept-with-reason: optional review, not the primary code-generation result | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 1466 | Topic-alignment check block | Debug-only omission of advisory alignment check | kept-with-reason: this remains a design risk, but converting it would make optional LLM review availability a hard dependency | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 1532 | Ablation repair LLM call | Debug-only omission of repair | kept-with-reason: advisory repair after a warning artifact is written | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_code_generation.py` | 1534 | Ablation validation block | Debug-only omission of advisory validation | kept-with-reason: optional LLM validation; no metrics are accepted here | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_execution.py` | 63 | Parse persisted `domain_profile.json` | Malformed profile falls back to live detection | kept-with-reason: migration behavior covered by provenance resolver tests | `test_resolver_falls_back_on_malformed_artifact` | N/A |
| `researchclaw/pipeline/stage_impls/_execution.py` | 79 | Live domain detection for provenance policy | Debug-only default to `unspecified` | kept-with-reason: explicit migration fail-open behavior currently asserted by provenance tests; changing it would be a policy change | Provenance resolver tests | N/A |
| `researchclaw/pipeline/stage_impls/_execution.py` | 306 | ColliderAgent structured `results.json` parse and copy | Invalid structured result ignored, run payload still uses sandbox metrics | kept-with-reason: `SandboxResult.metrics` remains the primary Stage 12 result channel for this branch | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_analysis.py` | 726 | FigureAgent chart planning | Warning logged, legacy chart generation attempted | kept-with-reason: already visible and has an explicit fallback | N/A | N/A |
| `researchclaw/pipeline/stage_impls/_analysis.py` | 749 | Legacy early chart generation | Warning logged, analysis still writes core summaries | kept-with-reason: optional visualization artifact, not metric extraction | N/A | N/A |
| `researchclaw/pipeline/_helpers.py` | 119 | Skill registry initialization | Debug-only empty registry fallback | kept-with-reason: optional prompt skills, no experiment data altered | N/A | N/A |
| `researchclaw/pipeline/_helpers.py` | 371 | Sandbox dependency check or install | Warning logged for failed package check/install | kept-with-reason: already visible and sandbox execution later fails if dependency is truly required | N/A | N/A |
| `researchclaw/pipeline/_helpers.py` | 972 | LLM chat retry wrapper | Retries and raises after exhaustion | kept-with-reason: not fail-open because final exception propagates | Existing LLM helper tests | N/A |
| `researchclaw/pipeline/_helpers.py` | 1029 | Evolution overlay prompt section | Overlay omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/_helpers.py` | 1041 | Matched skills prompt section | Skills omitted | kept-with-reason: optional prompt enrichment | N/A | N/A |
| `researchclaw/pipeline/_helpers.py` | 1710 | Framework prompt LLM generation | Debug-only fallback to template prompt | kept-with-reason: explicit fallback prompt is generated | N/A | N/A |
| `researchclaw/pipeline/_helpers.py` | 1907 | One multi-perspective debate role | Warning logged, debate continues with remaining roles | kept-with-reason: already visible and returns partial role outputs; caller can see count | N/A | N/A |
| `researchclaw/pipeline/_domain.py` | 154 | Read `project.profile` | Falls back to empty profile | kept-with-reason: defensive config accessor for non-`RCConfig` tests | N/A | N/A |
| `researchclaw/pipeline/_domain.py` | 169 | Read research topic and domains | Falls back to empty values | kept-with-reason: defensive config accessor for non-`RCConfig` tests | N/A | N/A |
| `researchclaw/pipeline/_domain.py` | 186 | Coarse domain detection | Falls back to ML prompt bank | kept-with-reason: prompt-bank selection fallback, not experiment result data | N/A | N/A |
| `researchclaw/domains/detector.py` | 250 | Loading one YAML domain profile | Warning logged, other profiles load | kept-with-reason: already visible and isolated per profile | N/A | N/A |
| `researchclaw/domains/detector.py` | 493 | Optional LLM domain detection | Warning logged, caller falls back to deterministic detection | kept-with-reason: already visible and deterministic path remains | N/A | N/A |
| `researchclaw/data/__init__.py` | 110 | Seminal papers YAML load | Warning logged, empty list returned | kept-with-reason: literature dataset helper is optional prompt context on this path | N/A | N/A |
| `researchclaw/agents/benchmark_agent/surveyor.py` | 73 | Benchmark knowledge YAML load | Warning logged, empty local KB | kept-with-reason: BenchmarkAgent is optional and Stage 9 still writes `exp_plan.yaml` | N/A | N/A |
| `researchclaw/agents/benchmark_agent/surveyor.py` | 144 | Hugging Face task search item | Debug-only skip for that task | kept-with-reason: optional remote benchmark search | N/A | N/A |
| `researchclaw/agents/benchmark_agent/surveyor.py` | 167 | Hugging Face keyword search item | Debug-only skip for that keyword | kept-with-reason: optional remote benchmark search | N/A | N/A |
| `researchclaw/agents/benchmark_agent/surveyor.py` | 170 | Hugging Face search wrapper | Warning logged, local benchmarks still usable | kept-with-reason: already visible and remote search is optional | N/A | N/A |
| `researchclaw/agents/benchmark_agent/selector.py` | 231 | Benchmark selector knowledge YAML load | Empty injection list returned | kept-with-reason: optional BenchmarkAgent fallback data | N/A | N/A |
| `researchclaw/agents/benchmark_agent/acquirer.py` | 152 | Generated torchvision download script | Generated script prints warning and continues | kept-with-reason: setup script handles optional prefetch failures explicitly | N/A | N/A |
| `researchclaw/agents/benchmark_agent/acquirer.py` | 166 | Generated Hugging Face dataset download script | Generated script prints warning and continues | kept-with-reason: optional prefetch, experiment code can still load cached data | N/A | N/A |
| `researchclaw/agents/benchmark_agent/acquirer.py` | 178 | Generated OGB download script | Generated script prints warning and continues | kept-with-reason: optional prefetch, experiment code can still fail if data is required | N/A | N/A |
| `researchclaw/experiment/agentic_sandbox.py` | 153 | Agentic session run wrapper | Converts exception into failed `AgenticResult` | kept-with-reason: not silent, failure is represented in sandbox result | N/A | N/A |
| `researchclaw/experiment/agentic_sandbox.py` | 263 | Container cleanup | Warning logged | kept-with-reason: cleanup best effort after sandbox result is already decided | N/A | N/A |
| `researchclaw/experiment/collider_agent_sandbox.py` | 125 | ColliderAgent subprocess launch | Converts exception into failed `SandboxResult` | kept-with-reason: not silent, Stage 12 sees returncode `-1` | N/A | N/A |
| `researchclaw/experiment/collider_agent_sandbox.py` | 145 | Incremental result merge | Warning logged, current run result remains | kept-with-reason: merge is best-effort history enrichment | N/A | N/A |
| `researchclaw/experiment/biology_agent_sandbox.py` | 142 | BiologyAgent subprocess launch | Converts exception into failed `SandboxResult` | kept-with-reason: not silent, Stage 12 sees returncode `-1` | N/A | N/A |
| `researchclaw/experiment/stat_agent_sandbox.py` | 114 | StatAgent subprocess launch | Converts exception into failed `SandboxResult` | kept-with-reason: not silent, Stage 12 sees returncode `-1` | N/A | N/A |
| `researchclaw/experiment/docker_sandbox.py` | 319 | Docker sandbox run | Converts exception into failed `SandboxResult` | kept-with-reason: not silent, Stage 12 hard guards inspect failed run output | N/A | N/A |
| `researchclaw/experiment/ssh_sandbox.py` | 399 | SSH sandbox run | Converts exception into failed result object | kept-with-reason: not silent, error is returned in result | N/A | N/A |
| `researchclaw/experiment/colab_sandbox.py` | 104 | Colab subprocess run | Converts exception into result dict with traceback | kept-with-reason: not silent, error is returned in result | N/A | N/A |
| `researchclaw/experiment/sandbox.py` | 343 | Single-file subprocess sandbox run | Converts exception into failed `SandboxResult` | kept-with-reason: not silent, Stage 12 hard guards inspect failed run output | N/A | N/A |
| `researchclaw/experiment/sandbox.py` | 446 | Project subprocess sandbox run | Converts exception into failed `SandboxResult` | kept-with-reason: not silent, Stage 12 hard guards inspect failed run output | N/A | N/A |
| `researchclaw/experiment/sandbox.py` | 543 | Temporary script cleanup | Warning logged | kept-with-reason: cleanup best effort after sandbox result is already captured | N/A | N/A |
| `researchclaw/experiment/metrics.py` | 126 | `results.json` parser | Warning logged, CSV/stdout fallbacks attempted | kept-with-reason: visible parser fallback chain | N/A | N/A |
| `researchclaw/experiment/metrics.py` | 137 | `results.csv` parser | Warning logged, stdout fallback attempted | kept-with-reason: visible parser fallback chain | N/A | N/A |
| `researchclaw/experiment/metrics.py` | 150 | `stdout.log` read | Warning logged, source `none` returned | kept-with-reason: visible parser fallback chain | N/A | N/A |
| `researchclaw/experiment/runner.py` | 271 | LLM code improvement call | Exception logged, current code returned | kept-with-reason: iterative runner keeps last known runnable code | N/A | N/A |
| `researchclaw/experiment/code_agent.py` | 255 | LLM code generation provider | Error logged, failed `CodeAgentResult` returned | kept-with-reason: not silent, caller can inspect failure | N/A | N/A |
| `researchclaw/experiment/code_agent.py` | 322 | LLM code refinement provider | Error logged, failed `CodeAgentResult` returned | kept-with-reason: not silent, caller can inspect failure | N/A | N/A |
| `researchclaw/experiment/code_agent.py` | 372 | LLM code critique provider | Failed `CodeAgentResult` returned | kept-with-reason: not silent, caller can inspect failure | N/A | N/A |
| `researchclaw/experiment/git_manager.py` | 123 | Git command wrapper for experiment worktrees | Warning logged, `None` returned | kept-with-reason: helper is advisory and does not mutate remote state | N/A | N/A |
| `researchclaw/pipeline/experiment_repair.py` | 359 | Repair-loop LLM client creation | Error logged, failed repair result returned | kept-with-reason: not silent, runner now fails Stage 14 if repair wrapper crashes | N/A | N/A |
| `researchclaw/pipeline/experiment_repair.py` | 703 | OpenCode repair attempt | Warning logged, falls back to LLM repair | kept-with-reason: visible fallback between repair providers | N/A | N/A |
| `researchclaw/pipeline/experiment_repair.py` | 732 | LLM repair call | Warning logged, no patch returned for that cycle | kept-with-reason: visible repair-attempt failure inside bounded repair loop | N/A | N/A |
| `researchclaw/pipeline/experiment_repair.py` | 824 | Sandbox execution in repair cycle | Warning logged, cycle returns no run payload | kept-with-reason: visible repair-cycle failure inside bounded repair loop | N/A | N/A |

## Judgment notes

- The two phantom imports called out in the brief, `event_log` and
  `pitfall_detector`, were removed instead of converted to permanent warnings
  because neither module exists in this fork.
- `cost_tracker` is also absent, but it guarded a configured budget. It now
  fails before stage execution when a non-LLM CLI provider has a positive
  budget, while preserving the default `llm` provider behavior.
- Experiment memory was not just made visible. The runner now constructs the
  memory store correctly and `ExperimentMemory` has a persisted outcome API.
- The Stage 12 provenance resolver still has an explicit fail-open migration
  rule. Existing provenance tests assert that behavior, so this audit documents
  it rather than converting it as a drive-by policy change.
