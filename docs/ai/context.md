# AutoResearchClaw — Project Context

## What This Is
A fully autonomous 22-stage research pipeline: topic → literature review → experiment design/execution → statistical analysis → multi-agent peer review → conference-ready LaTeX paper. MIT license, v0.5.0.

## Owner
Ariel Tolome — forked from `aiming-lab/AutoResearchClaw` to `ArielleTolome/AutoResearchClaw`.

## Stack
- **Language:** Python 3.12+ (stdlib-first, minimal deps)
- **LLM Client:** `researchclaw/llm/client.py` — currently OpenAI-compatible only (`/chat/completions`, Bearer auth)
- **Config:** `researchclaw/config.py` + YAML files (`config.arc.yaml`)
- **CLI:** `researchclaw/cli.py` — `researchclaw run --config ... --topic ...`
- **Experiment runner:** `researchclaw/experiment/` — sandbox + GPU execution
- **Literature:** `researchclaw/literature/` — arXiv + Semantic Scholar
- **Pipeline:** `researchclaw/pipeline/` — 22-stage runner

## Key Files
- `researchclaw/llm/client.py` — LLM client (main focus of current work)
- `researchclaw/config.py` — config dataclasses
- `config.arc.yaml` — active config (Bill's GPU server)
- `config.researchclaw.example.yaml` — example config template
- `researchclaw/health.py` — `researchclaw doctor` check

## Active Work: Multi-Provider LLM Support
Adding native support for 3 providers:
1. **OpenAI** (default, already works) — `api.openai.com/v1`, Bearer auth, `/chat/completions`, supports GPT-5.4/5.3-Codex/o3/o4-mini/gpt-4o
2. **Anthropic** — `api.anthropic.com/v1`, `x-api-key` header + `anthropic-version: 2023-06-01`, `/messages` endpoint, different request/response format, system message is top-level param
3. **OpenRouter** — `openrouter.ai/api/v1`, Bearer auth, OpenAI-compatible format, extra headers (`HTTP-Referer`, `X-Title`), any model via `provider/model` format

## Deployment
- **Local (Mac mini):** `/Users/arieltolome/.openclaw/workspace/AutoResearchClaw/`
- **GPU server (Bill):** `/root/AutoResearchClaw/` on RTX PRO 6000 (96GB VRAM)
- **Discord channel:** `#research` under `🤖 Agents` — Bill listens here

## Coding Standards
- stdlib only for HTTP (urllib) — no new HTTP library deps
- Dataclasses for config (not Pydantic)
- Type hints throughout
- Graceful error messages (not raw stack traces to user)
- Keep OpenAI as default — existing configs must not break
