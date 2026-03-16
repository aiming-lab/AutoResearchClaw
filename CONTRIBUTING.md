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
