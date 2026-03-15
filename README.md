<p align="center">
  <img src="image/logo.png" width="700" alt="AutoResearchClaw Logo">
</p>

<h2 align="center"><b>Chat an Idea. Get a Paper. Fully Autonomous.</b></h2>



<p align="center">
  <b><i><font size="5">Just chat with <a href="#openclaw-integration">OpenClaw</a>: "Research X" → done.</font></i></b>
</p>

<p align="center">
  <img src="image/framework.png" width="100%" alt="AutoResearchClaw Framework">
</p>


<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="#testing"><img src="https://img.shields.io/badge/Tests-1039%20passed-brightgreen?logo=pytest&logoColor=white" alt="1039 Tests Passed"></a>
  <a href="https://github.com/Jiaaqiliu/AutoResearchClaw"><img src="https://img.shields.io/badge/GitHub-AutoResearchClaw-181717?logo=github" alt="GitHub"></a>
  <a href="#openclaw-integration"><img src="https://img.shields.io/badge/OpenClaw-Compatible-ff4444?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6IiBmaWxsPSJ3aGl0ZSIvPjwvc3ZnPg==" alt="OpenClaw Compatible"></a>
</p>

<p align="center">
  <a href="docs/README_CN.md">🇨🇳 中文</a> ·
  <a href="docs/README_JA.md">🇯🇵 日本語</a> ·
  <a href="docs/README_KO.md">🇰🇷 한국어</a> ·
  <a href="docs/README_FR.md">🇫🇷 Français</a> ·
  <a href="docs/README_DE.md">🇩🇪 Deutsch</a> ·
  <a href="docs/README_ES.md">🇪🇸 Español</a> ·
  <a href="docs/README_PT.md">🇧🇷 Português</a> ·
  <a href="docs/README_RU.md">🇷🇺 Русский</a> ·
  <a href="docs/README_AR.md">🇸🇦 العربية</a>
</p>

<p align="center">
  <a href="docs/integration-guide.md">📖 Integration Guide</a>
</p>

---

## ⚡ One Command. One Paper.

```bash
pip install -e . && researchclaw run --topic "Your research idea here" --auto-approve
```


---

## 🤔 What Is This?

**You think it. AutoResearchClaw writes it.**

Drop a research topic — get back a full academic paper with real literature from arXiv & Semantic Scholar, hardware-aware sandbox experiments (GPU/MPS/CPU auto-detected), statistical analysis, multi-agent peer review, and conference-ready LaTeX targeting NeurIPS/ICML/ICLR. No babysitting. No copy-pasting. No hallucinated references.

<table>
<tr><td>📄</td><td><code>paper_draft.md</code></td><td>Full academic paper (Introduction, Related Work, Method, Experiments, Results, Conclusion)</td></tr>
<tr><td>📐</td><td><code>paper.tex</code></td><td>Conference-ready LaTeX (NeurIPS / ICLR / ICML templates)</td></tr>
<tr><td>📚</td><td><code>references.bib</code></td><td>Real BibTeX references from Semantic Scholar and arXiv — auto-pruned to match inline citations</td></tr>
<tr><td>🔍</td><td><code>verification_report.json</code></td><td>4-layer citation integrity + relevance verification (arXiv, CrossRef, DataCite, LLM)</td></tr>
<tr><td>🧪</td><td><code>experiment runs/</code></td><td>Generated code + sandbox results + structured JSON metrics</td></tr>
<tr><td>📊</td><td><code>charts/</code></td><td>Auto-generated condition comparison charts with error bars and confidence intervals</td></tr>
<tr><td>📝</td><td><code>reviews.md</code></td><td>Multi-agent peer review with methodology-evidence consistency checks</td></tr>
<tr><td>🧬</td><td><code>evolution/</code></td><td>Self-learning lessons extracted from each run</td></tr>
<tr><td>📦</td><td><code>deliverables/</code></td><td>All final outputs in one folder — compile-ready for Overleaf</td></tr>
</table>

The pipeline runs **end-to-end without human intervention**. When experiments fail, it self-heals. When hypotheses don't hold, it pivots. When citations are fake, it kills them.

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Jiaaqiliu/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Configure
cp config.researchclaw.example.yaml config.arc.yaml
# Edit config.arc.yaml — set your LLM API endpoint and key

# 3. Run
export OPENAI_API_KEY="sk-..."
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

Output → `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/` — compile-ready LaTeX, BibTeX, experiment code, charts.

<details>
<summary>📝 Minimum required config</summary>

```yaml
project:
  name: "my-research"

research:
  topic: "Your research topic here"

llm:
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-4o"
  fallback_models: ["gpt-4o-mini"]

experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python"
```

</details>

---

## 💳 Run With Your Existing AI Subscriptions (No API Key Required)

Already paying for **ChatGPT Pro**, **Claude Max**, or **Google One AI Premium**?  
You don't need to pay for separate API credits. AutoResearchClaw supports all three via their official CLIs.

> **TL;DR** — install the CLI, log in once, point at the right config file, run research.

---

### 🚀 Zero-to-Running in 5 Minutes

#### Step 1 — Clone & Install AutoResearchClaw

```bash
git clone https://github.com/ArielleTolome/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

#### Step 2 — Install your CLI (pick one or all three)

<details>
<summary>💬 ChatGPT Pro — Codex CLI (gpt-5.3-codex-spark, gpt-4.1, o3, o4-mini)</summary>

**Requirements:** ChatGPT Pro or Plus subscription

```bash
# Install
brew install codex
# OR: npm install -g @openai/codex

# Log in (browser opens, one-time)
codex login

# Verify
codex exec - <<< "ping"
# Expected: pong (or similar one-word reply)
```

**What's happening:** The Codex CLI authenticates via the same OAuth session as ChatGPT. Credentials are stored at `~/.codex/`. No API key — your Pro subscription covers the usage.

**Available models:**

| Model | Notes |
|---|---|
| `gpt-5.3-codex-spark` | Default (set in `~/.codex/config.toml`) — recommended for research |
| `gpt-4.1` | Faster fallback |
| `o3` | Best reasoning quality, slower |
| `o4-mini` | Fast + cheap fallback |

</details>

<details>
<summary>🤖 Claude Max — Anthropic OAuth (claude-opus-4-6, claude-sonnet-4-6)</summary>

**Requirements:** Claude Max subscription (Pro also works, rate limits apply)

The OAuth token lives inside your [Claude Code](https://docs.anthropic.com/claude-code) installation. If you have Claude Code installed, you're already authenticated.

```bash
# Install Claude Code (if not already installed)
npm install -g @anthropic-ai/claude-code
# OR: brew install claude-code

# Log in (browser opens, one-time)
claude login

# Verify — token is at:
cat ~/.openclaw/agents/main/agent/auth-profiles.json | python3 -m json.tool | grep -A2 anthropic

# Quick test
echo "ping" | claude --print --permission-mode bypassPermissions - 
# Expected: pong
```

**What's happening:** AutoResearchClaw reads your OAuth token directly from Claude Code's auth profile and uses it with the Anthropic REST API (`Bearer` auth + `anthropic-beta: oauth-2025-04-20`). No API key. Your Claude Max subscription covers usage.

**Available models:**

| Model | Notes |
|---|---|
| `claude-opus-4-6` | Best quality — recommended for research writing |
| `claude-sonnet-4-6` | Faster, good quality — good fallback |
| `claude-haiku-3-5` | Fast, lightweight fallback |

**Note:** OAuth tokens expire every ~1 hour. Claude Code auto-refreshes them. If you get a 401, run `claude login` to refresh manually.

</details>

<details>
<summary>♊ Google One AI Premium — Gemini CLI (gemini-2.5-pro, gemini-2.5-flash)</summary>

**Requirements:** Google One AI Premium or Gemini free tier (free tier has lower rate limits)

```bash
# Install
brew install gemini-cli
# OR: npm install -g @google/gemini-cli

# Log in (browser opens, one-time)
gemini auth login

# Verify
gemini --output-format json -p "ping"
# Expected: JSON with response field containing "pong"

# Check token freshness (tokens expire ~1hr, auto-refreshed on use)
cat ~/.gemini/oauth_creds.json | python3 -c "
import json, sys, time
d = json.load(sys.stdin)
exp = d.get('expiry_date', 0) / 1000
remaining = exp - time.time()
print(f'Token expires in: {remaining/60:.0f} min' if remaining > 0 else 'EXPIRED — run: gemini auth login')
"
```

**What's happening:** Gemini CLI authenticates via Google OAuth and stores credentials at `~/.gemini/oauth_creds.json`. AutoResearchClaw shells out to the `gemini` binary, which handles token refresh automatically.

**Available models:**

| Model | Notes |
|---|---|
| `gemini-2.5-pro` | Best quality — recommended |
| `gemini-2.5-flash` | 3–5× faster, slightly lower quality |
| `gemini-2.5-flash-lite` | Fastest, good for fallback |

**Note:** Doctor check may show ⚠️ for gemini-cli even when working (the 15s smoke-test sometimes times out). `Result: PASS` still means it's ready — the live research run will work fine.

</details>

---

#### Step 3 — Run a doctor check

```bash
# Codex CLI (ChatGPT Pro)
./research.sh "test" --config config.arc.codex-cli.yaml --doctor

# Claude Max (OAuth — recommended)
./research.sh "test" --config config.arc.anthropic-oauth.yaml --doctor

# Gemini CLI
./research.sh "test" --config config.arc.gemini-cli.yaml --doctor
```

Look for `Result: PASS` — all checks green (⚠️ on gemini smoke-test is normal).

#### Step 4 — Run research

```bash
# ChatGPT Pro — gpt-5.3-codex-spark
./research.sh "attention mechanisms in transformer models" --config config.arc.codex-cli.yaml

# Claude Max — claude-opus-4-6
./research.sh "attention mechanisms in transformer models" --config config.arc.anthropic-oauth.yaml

# Gemini 2.5 Pro — free via Google One AI Premium
./research.sh "attention mechanisms in transformer models" --config config.arc.gemini-cli.yaml
```

Output lands in: `artifacts/rc-<timestamp>/deliverables/` — compile-ready LaTeX, BibTeX, charts.

---

### 📊 Provider Comparison

| Provider | Config | Subscription | Best For | Speed |
|---|---|---|---|---|
| **Claude Max** (OAuth) | `config.arc.anthropic-oauth.yaml` | Claude Max / Pro | Research writing quality | Medium |
| **ChatGPT Pro** (Codex CLI) | `config.arc.codex-cli.yaml` | ChatGPT Pro / Plus | Code generation stages | Medium |
| **Gemini** (CLI) | `config.arc.gemini-cli.yaml` | Google One AI Premium / Free | Long context, cost-free | Fast |
| **Claude Max** (CLI) | `config.arc.claude-cli.yaml` | Claude Max | Fallback for oauth | Medium |
| **OpenAI API** | `config.arc.yaml` | Pay-as-you-go | Standard deployments | Fast |
| **Anthropic API** | `config.arc.anthropic.yaml` | Pay-as-you-go | Standard deployments | Medium |

> **Recommended:** `config.arc.anthropic-oauth.yaml` (Claude Max) for research quality, or `config.arc.codex-cli.yaml` (ChatGPT Pro) for coding-heavy experiments.

---

### 🔧 Customizing Models

Edit your chosen config file and change `primary_model` + `fallback_models`:

```yaml
# config.arc.codex-cli.yaml
llm:
  provider: "codex-cli"
  primary_model: "o3"              # Override the ~/.codex/config.toml default
  fallback_models:
    - "gpt-4.1"
    - "o4-mini"
```

```yaml
# config.arc.anthropic-oauth.yaml
llm:
  provider: "anthropic-oauth"
  primary_model: "claude-opus-4-6"
  fallback_models:
    - "claude-sonnet-4-6"
    - "claude-haiku-3-5"
```

---

### 🚨 Troubleshooting

<details>
<summary>Codex CLI: "model not supported when using Codex with a ChatGPT account"</summary>

The model you specified isn't available on your subscription tier. Use `gpt-5.3-codex-spark` (the Codex CLI default) or `gpt-4.1`. Set `primary_model: "gpt-5.3-codex-spark"` in the config.

Verify your default:
```bash
cat ~/.codex/config.toml | grep model
```
</details>

<details>
<summary>Anthropic OAuth: 401 Unauthorized</summary>

Your OAuth token has expired. Claude Code auto-refreshes it during normal use. Manually refresh:

```bash
claude login
```

Also check you're using `config.arc.anthropic-oauth.yaml` (not `config.arc.anthropic.yaml` which needs a console API key).
</details>

<details>
<summary>Gemini CLI: smoke-test fails / ⚠️ warning in doctor</summary>

The 15-second smoke-test window is tight for Gemini. The actual research run (which has a much longer timeout) usually works fine. Run doctor and look for `Result: PASS` — that's the signal that matters.

If the token is truly expired:
```bash
gemini auth login
```
</details>

<details>
<summary>General: "CLI not found in PATH"</summary>

```bash
# Codex
brew install codex

# Gemini
brew install gemini-cli
# or: npm install -g @google/gemini-cli

# Claude
npm install -g @anthropic-ai/claude-code
```

Make sure `~/.nvm/versions/node/*/bin` or `/opt/homebrew/bin` is in your `$PATH`.
</details>

---

### 🎓 Full Example: End-to-End Research Run

```bash
# Clone
git clone https://github.com/ArielleTolome/AutoResearchClaw.git
cd AutoResearchClaw

# Set up Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Install and log in to Codex CLI (ChatGPT Pro)
brew install codex && codex login

# Doctor check
./research.sh "test" --config config.arc.codex-cli.yaml --doctor
# → Result: PASS

# Run a full research pipeline (takes 20–60 min)
./research.sh "sparse attention mechanisms for long-context transformers" \
  --config config.arc.codex-cli.yaml \
  --auto-approve

# Check output
ls artifacts/rc-*/deliverables/
# paper_draft.md   paper.tex   references.bib   reviews.md   charts/
```

---

## 🧠 What Makes It Different

| Capability | How It Works |
|-----------|-------------|
| **🔄 PIVOT / REFINE Loop** | Stage 15 autonomously decides: PROCEED, REFINE (tweak params), or PIVOT (new direction). Artifacts auto-versioned. |
| **🤖 Multi-Agent Debate** | Hypothesis generation, result analysis, and peer review each use structured multi-perspective debate. |
| **🧬 Self-Learning** | Lessons extracted per run (decision rationale, runtime warnings, metric anomalies) with 30-day time-decay. Future runs learn from past mistakes. |
| **📚 Knowledge Base** | Every run builds structured KB across 6 categories (decisions, experiments, findings, literature, questions, reviews). |
| **🛡️ Sentinel Watchdog** | Background quality monitor: NaN/Inf detection, paper-evidence consistency, citation relevance scoring, anti-fabrication guard. |

---

## 🦞 OpenClaw Integration

<table>
<tr>

**AutoResearchClaw is an [OpenClaw](https://github.com/openclaw/openclaw)-compatible service.** Install it in OpenClaw and launch autonomous research with a single message — or use it standalone via CLI, Claude Code, or any AI coding assistant.

</tr>
</table>

### 🚀 Use with OpenClaw (Recommended)

If you already use [OpenClaw](https://github.com/openclaw/openclaw) as your AI assistant:

```
1️⃣  Share the GitHub repo URL with OpenClaw
2️⃣  OpenClaw auto-reads RESEARCHCLAW_AGENTS.md → understands the pipeline
3️⃣  Say: "Research [your topic]"
4️⃣  Done — OpenClaw clones, installs, configures, runs, and returns results
```

**That's it.** OpenClaw handles `git clone`, `pip install`, config setup, and pipeline execution automatically. You just chat.

<details>
<summary>💡 What happens under the hood</summary>

1. OpenClaw reads `RESEARCHCLAW_AGENTS.md` → learns the research orchestrator role
2. OpenClaw reads `README.md` → understands installation and pipeline structure
3. OpenClaw copies `config.researchclaw.example.yaml` → `config.yaml`
4. Asks for your LLM API key (or uses your environment variable)
5. Runs `pip install -e .` + `researchclaw run --topic "..." --auto-approve`
6. Returns the paper, LaTeX, experiments, and citations

</details>

### 🔌 OpenClaw Bridge (Advanced)

For deeper integration, AutoResearchClaw includes a **bridge adapter system** with 6 optional capabilities:

```yaml
# config.arc.yaml
openclaw_bridge:
  use_cron: true              # ⏰ Scheduled research runs
  use_message: true           # 💬 Progress notifications (Discord/Slack/Telegram)
  use_memory: true            # 🧠 Cross-session knowledge persistence
  use_sessions_spawn: true    # 🔀 Spawn parallel sub-sessions for concurrent stages
  use_web_fetch: true         # 🌐 Live web search during literature review
  use_browser: false          # 🖥️ Browser-based paper collection
```

Each flag activates a typed adapter protocol. When OpenClaw provides these capabilities, the adapters consume them without code changes. See [`docs/integration-guide.md`](docs/integration-guide.md) for full details.

### 🛠️ Other Ways to Run

| Method | How |
|--------|-----|
| **Standalone CLI** | `researchclaw run --topic "..." --auto-approve` |
| **Python API** | `from researchclaw.pipeline import Runner; Runner(config).run()` |
| **Claude Code** | Reads `RESEARCHCLAW_CLAUDE.md` — just say *"Run research on [topic]"* |
| **OpenCode** | Reads `.claude/skills/` — same natural language interface |
| **Any AI CLI** | Provide `RESEARCHCLAW_AGENTS.md` as context → agent auto-bootstraps |

---

## 🔬 Pipeline: 23 Stages, 8 Phases

```
Phase A: Research Scoping          Phase E: Experiment Execution
  1. TOPIC_INIT                      12. EXPERIMENT_RUN
  2. PROBLEM_DECOMPOSE               13. ITERATIVE_REFINE  ← self-healing

Phase B: Literature Discovery      Phase F: Analysis & Decision
  3. SEARCH_STRATEGY                 14. RESULT_ANALYSIS    ← multi-agent
  4. LITERATURE_COLLECT  ← real API  15. RESEARCH_DECISION  ← PIVOT/REFINE
  5. LITERATURE_SCREEN   [gate]
  6. KNOWLEDGE_EXTRACT               Phase G: Paper Writing
                                     16. PAPER_OUTLINE
Phase C: Knowledge Synthesis         17. PAPER_DRAFT
  7. SYNTHESIS                       18. PEER_REVIEW        ← evidence check
  8. HYPOTHESIS_GEN    ← debate      19. PAPER_REVISION

Phase D: Experiment Design         Phase H: Finalization
  9. EXPERIMENT_DESIGN   [gate]      20. QUALITY_GATE      [gate]
 10. CODE_GENERATION                 21. KNOWLEDGE_ARCHIVE
 11. RESOURCE_PLANNING               22. EXPORT_PUBLISH     ← LaTeX
                                     23. CITATION_VERIFY    ← relevance check
```

> **Gate stages** (5, 9, 20) pause for human approval or auto-approve with `--auto-approve`. On rejection, the pipeline rolls back.

> **Decision loops**: Stage 15 can trigger REFINE (→ Stage 13) or PIVOT (→ Stage 8), with automatic artifact versioning.

<details>
<summary>📋 What Each Phase Does</summary>

| Phase | What Happens |
|-------|-------------|
| **A: Scoping** | LLM decomposes the topic into a structured problem tree with research questions |
| **A+: Hardware** | Auto-detects GPU (NVIDIA CUDA / Apple MPS / CPU-only), warns if local hardware is limited, adapts code generation accordingly |
| **B: Literature** | Multi-source search (arXiv-first, then Semantic Scholar) for real papers, screens by relevance, extracts knowledge cards |
| **C: Synthesis** | Clusters findings, identifies research gaps, generates testable hypotheses via multi-agent debate |
| **D: Design** | Designs experiment plan, generates hardware-aware runnable Python (GPU tier → package selection), estimates resource needs |
| **E: Execution** | Runs experiments in sandbox, detects NaN/Inf and runtime bugs, self-heals code via targeted LLM repair |
| **F: Analysis** | Multi-agent analysis of results; autonomous PROCEED / REFINE / PIVOT decision with rationale |
| **G: Writing** | Outlines → section-by-section drafting (5,000-6,500 words) → peer reviews (with methodology-evidence consistency) → revises with length guard |
| **H: Finalization** | Quality gate, knowledge archival, LaTeX export with conference template, citation integrity + relevance verification |

</details>

---

## ✨ Key Features

| Feature | Description |
|---------|------------|
| **📚 Multi-Source Literature** | Real papers from arXiv (primary) + Semantic Scholar — query expansion, deduplication, circuit breaker with graceful degradation |
| **🔍 4-Layer Citation Verification** | arXiv ID check → CrossRef/DataCite DOI → Semantic Scholar title match → LLM relevance scoring. Hallucinated refs auto-removed. |
| **🖥️ Hardware-Aware Execution** | Auto-detects GPU (NVIDIA CUDA / Apple MPS / CPU-only) and adapts code generation, imports, and experiment scale accordingly |
| **🧪 Sandbox Experiments** | AST-validated code, immutable harness, NaN/Inf fast-fail, self-healing repair, iterative refinement (up to 10 rounds), partial result capture |
| **📝 Conference-Grade Writing** | NeurIPS/ICML/ICLR templates, section-by-section drafting (5,000-6,500 words), anti-fabrication guard, revision length guard, anti-disclaimer enforcement |
| **📐 Template Switching** | `neurips_2025`, `iclr_2026`, `icml_2026` — Markdown → LaTeX with math, tables, figures, cross-refs, `\cite{}` |
| **🚦 Quality Gates** | 3 human-in-the-loop gates (Stages 5, 9, 20) with rollback. Skip with `--auto-approve`. |

---

## ⚙️ Configuration Reference

<details>
<summary>Click to expand full configuration reference</summary>

```yaml
# === Project ===
project:
  name: "my-research"              # Project identifier
  mode: "docs-first"               # docs-first | semi-auto | full-auto

# === Research ===
research:
  topic: "..."                     # Research topic (required)
  domains: ["ml", "nlp"]           # Research domains for literature search
  daily_paper_count: 8             # Target papers per search query
  quality_threshold: 4.0           # Minimum quality score for papers

# === Runtime ===
runtime:
  timezone: "America/New_York"     # For timestamps
  max_parallel_tasks: 3            # Concurrent experiment limit
  approval_timeout_hours: 12       # Gate stage timeout
  retry_limit: 2                   # Retry count on stage failure

# === LLM ===
llm:
  provider: "openai-compatible"    # Provider type
  base_url: "https://..."          # API endpoint (required)
  api_key_env: "OPENAI_API_KEY"    # Env var for API key (required)
  api_key: ""                      # Or hardcode key here
  primary_model: "gpt-4o"          # Primary model
  fallback_models: ["gpt-4o-mini"] # Fallback chain
  s2_api_key: ""                   # Semantic Scholar API key (optional, higher rate limits)

# === Experiment ===
experiment:
  mode: "sandbox"                  # simulated | sandbox | ssh_remote
  time_budget_sec: 600             # Max execution time per run (default: 600s)
  max_iterations: 10               # Max optimization iterations
  metric_key: "val_loss"           # Primary metric name
  metric_direction: "minimize"     # minimize | maximize
  sandbox:
    python_path: ".venv/bin/python"
    gpu_required: false
    allowed_imports: [math, random, json, csv, numpy, torch, sklearn]
    max_memory_mb: 4096
  ssh_remote:
    host: ""                       # GPU server hostname
    gpu_ids: []                    # Available GPU IDs
    remote_workdir: "/tmp/researchclaw_experiments"

# === Export ===
export:
  target_conference: "neurips_2025"  # neurips_2025 | iclr_2026 | icml_2026
  authors: "Anonymous"
  bib_file: "references"

# === Prompts ===
prompts:
  custom_file: ""                  # Path to custom prompts YAML (empty = defaults)

# === Security ===
security:
  hitl_required_stages: [5, 9, 20] # Stages requiring human approval
  allow_publish_without_approval: false
  redact_sensitive_logs: true

# === Knowledge Base ===
knowledge_base:
  backend: "markdown"              # markdown | obsidian
  root: "docs/kb"

# === Notifications ===
notifications:
  channel: "console"               # console | discord | slack
  target: ""

# === OpenClaw Bridge ===
openclaw_bridge:
  use_cron: false                  # Scheduled research runs
  use_message: false               # Progress notifications
  use_memory: false                # Cross-session knowledge persistence
  use_sessions_spawn: false        # Spawn parallel sub-sessions
  use_web_fetch: false             # Live web search
  use_browser: false               # Browser-based paper collection
```

</details>

---

## 🙏 Acknowledgments

Inspired by:

- 🔬 [AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Sakana AI) — Automated research pioneer
- 🧠 [AutoResearch](https://github.com/karpathy/autoresearch) (Andrej Karpathy) — End-to-end research automation
- 🌐 [FARS](https://analemma.ai/blog/introducing-fars/) (Analemma) — Fully Automated Research System

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 📌 Citation

If you find AutoResearchClaw useful, please cite:

```bibtex
@misc{liu2026autoresearchclaw,
  author       = {Liu, Jiaqi and Xia, Peng and Han, Siwei and Qiu, Shi and Zhang, Letian and Chen, Guiming  and Tu, Haoqin and Yang, Xinyu and and Zhou, Jiawei and Zhu, Hongtu and Li, Yun and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  title        = {AutoResearchClaw: Fully Autonomous Research from Idea to Paper},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/aiming-lab/AutoResearchClaw},
}
```

<p align="center">
  <sub>Built with 🦞 by the AutoResearchClaw team</sub>
</p>
