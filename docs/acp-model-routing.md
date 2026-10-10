# Verified ACP model roles

ACP agents normally manage their own model. Opt-in routing selects a model for each pipeline stage and verifies the agent's returned model and reasoning setting before prompts are sent. Unsupported settings fail the stage rather than substituting another model or generating an offline template. This requires an ACP agent/acpx version that exposes model and reasoning configuration; it is tested with acpx 0.19.4 and Codex ACP 2.2.2.

```yaml
llm:
  provider: acp
  primary_model: gpt-6-astra
  fallback_models: []
  acp:
    agent: codex
    model_routing: true
    reasoning_effort: ultra
    execution_model: gpt-6.1-sol
    stage_models:
      4: gpt-6.1-sol   # Literature collection
      6: gpt-6.1-sol   # Knowledge extraction
      10: gpt-6.1-sol  # Code generation and implementation checks
      11: gpt-6.1-sol  # Resource planning
      12: gpt-6.1-sol  # Experiment execution / repair
      13: gpt-6.1-sol  # Iterative refinement
      17: gpt-6.1-sol  # Paper draft
      19: gpt-6.1-sol  # Paper revision
      21: gpt-6.1-sol  # Knowledge archive
      22: gpt-6.1-sol  # Export
```

The example uses the default model for research scoping, search strategy, screening, synthesis, hypotheses, experiment design, result analysis, research decisions, paper outline, peer review, quality gate and citation verification. Available model IDs and reasoning levels depend on the signed-in agent account; this example is not a promise that an arbitrary account supports these IDs. Both settings must be advertised and selected by the live session. `fallback_models` is forbidden with routing enabled.

Stage 14 keeps the research model for chart decisions, planning and critique. Its plotting code generator and its repairs use `execution_model` in a separate session. The post-analysis experiment repair loop also uses the execution model. Co-Pilot discussions use the default research model. Stage 10's internal implementation checks follow the code model; the split is by stage responsibility, not an attempt to classify every sentence as reasoning or execution.

Sessions are separated by run, stage and purpose. Stages exchange saved artifacts; they do not rely on a shared conversation whose model changes between callers. Each routed stage writes `llm_selection.json` with requested/selected model and reasoning, verification status and session identity. Figure code, repair and Co-Pilot selections are saved separately. Review provenance distinguishes the paper author from the judge, preferring saved draft/revision selection records.

This covers the ACP pipeline clients. External CLI experiment providers, OpenCode, Gemini image generation and MetaClaw use independent settings and are not redirected. Actual experiment code still runs on the configured compute backend; these model roles do not move a language model onto the experiment GPU. Existing configurations with `model_routing: false` keep their original ACP behaviour.

Verification records describe the agent-reported session configuration. They do not prove the provider's internal serving implementation. Full research quality and citation reliability still require experiment and human review.
