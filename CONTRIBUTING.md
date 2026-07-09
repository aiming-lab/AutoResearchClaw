# Contributing to AutoResearchClaw

## Setup

1. Fork and clone the repo
2. Create a venv and install with dev extras:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Generate your local config:
   ```
   researchclaw init
   ```
4. Edit `config.arc.yaml` with your LLM settings

## Config Convention

- `config.researchclaw.example.yaml` — tracked template (do not add secrets)
- `config.arc.yaml` — your local config (gitignored, created by `researchclaw init`)
- `config.yaml` — also gitignored, supported as fallback

## Running Tests

```
pytest tests/
```

## Checking Your Environment

```
researchclaw doctor
```

## PR Guidelines

- Branch from main
- One concern per PR
- Ensure `pytest tests/` passes
- Include tests for new functionality

## Release-Gate Change Discipline

`scripts/release_check.py` and the artifact contracts in
`researchclaw/pipeline/release_artifacts.py` are release gates. Gate changes
follow stricter rules than ordinary code:

1. **Isolated commits.** Any change to gate rules, thresholds, or required
   artifacts must land in its own commit. Never combine a gate change with
   generated run artifacts, paper output, or a fix that makes a specific run
   pass — that is the classic overfit-to-your-own-benchmark failure mode.
2. **Stated rationale.** The commit message must say *why* the rule changed,
   independent of any particular run.
3. **No weakening to pass.** If a run fails a gate, fix the run (or file a
   machine-readable waiver where the gate supports one, e.g.
   `waivers/no_real_data.json`). Do not loosen the gate.
4. **Digest functions stay in sync.** `paper_sha256` / `claims_digest` are
   duplicated in `scripts/release_check.py` for standalone use; change both
   sides in the same (isolated) commit.
5. **Invariant ordering.** The truth audit (stage 24) always precedes the
   de-AI prose audit (stage 25), and the de-AI audit is recommend-only.
   PRs that reorder these stages or make the de-AI audit write to the paper
   will be rejected.
