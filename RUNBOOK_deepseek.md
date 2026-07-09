# DeepSeek Runbook — hardware-security paper, end to end

Turnkey steps to run AutoResearchClaw with DeepSeek V4 on your machine.
Assumes you already have Python deps installed (`pip install -e .`).

## 0. Set your key (never commit it)

```bash
export DEEPSEEK_API_KEY="sk-...your key..."
```

## 1. Verify connectivity (cheap, ~1 call)

```bash
python3 -m researchclaw doctor -c config.deepseek.yaml
```

Expect: LLM check PASS. If it fails on the model name, DeepSeek V4 uses
`deepseek-v4-flash` / `deepseek-v4-pro` (legacy `deepseek-chat` /
`deepseek-reasoner` are deprecated 2026-07-24).

## 2. Smoke test — first half only (scoping → experiment design)

Cheap sanity run that stops before the expensive experiment/writing loop:

```bash
python3 -m researchclaw run -c config.deepseek.yaml -o runs/hwsec \
  --to-stage EXPERIMENT_DESIGN
```

Check `runs/hwsec/stage-0*/` for goal, problem tree, real downloaded
literature (Semantic Scholar / arXiv / OpenAlex must be reachable from your
network), synthesis, hypotheses. Look at `cost_log.jsonl` for spend so far.

## 3a. Full autonomous run (Route A: v4-flash writes, v4-pro critiques)

```bash
python3 -m researchclaw run -c config.deepseek.yaml -o runs/hwsec
```

Runs all 25 stages. The Stage 15 Socratic critic uses `deepseek-v4-pro`
(distinct from the `deepseek-v4-flash` writer) so the reviewer_isolation gate
is satisfied automatically.

## 3b. "Experiment already done" — start from writing

If you have real experiment results, inject them and start at PAPER_OUTLINE.
Put your results where the pipeline expects them, then:

```bash
# config already lists runtime.inject_artifacts in config.hwsec-inject.yaml;
# point those entries at YOUR real files, then:
python3 -m researchclaw run -c config.deepseek.yaml -o runs/hwsec \
  --from-stage PAPER_OUTLINE
```

Minimum injected artifacts (see config.hwsec-inject.yaml for the exact map):
- `experiment_summary_best.json` (aggregate metrics)
- `stage-14/analysis.md`, `stage-15/decision.md`
- `stage-12/runs/run_seed*.json` (per-seed real runs, status != "simulated")
- `stage-13/experiment_final/*.py` (the code that produced the results)

## 4. Release gate (authoritative)

```bash
python3 scripts/release_check.py runs/hwsec --json
# exit 0 = release-ready, 2 = degraded, 1 = fail
```

A real run should clear: claims_empty (v4 extracts claims), reviewer_isolation
(v4-pro critic), quality/degraded (real score), sandbox/env metadata (real
stage-12 execution).

## 5. Optional — external second review (Route B, Claude/Kiro)

For important papers, add a human/agent reviewer on top of Route A:
1. Read the run, write your review to `runs/hwsec/reviews_external/review.md`.
2. Add P0/P1 findings to `runs/hwsec/stage-15/critique.json`.
3. In config set `llm.critic_source: "external"` and
   `llm.external_review_path: "reviews_external/review.md"`.
4. Re-run `--from-stage TRUTH_AUDIT`, then release_check.

## Cost expectations

- Smoke test (steps 1-2): a handful of calls, negligible.
- Full run (step 3): dozens of calls; writing stages dominate. v4-flash is
  the cheap tier — watch `cost_log.jsonl` (per-stage $) as you go.
- v4-pro critic adds a few reasoning calls at Stage 15 only.
