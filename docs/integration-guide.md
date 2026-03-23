# AutoResearchClaw Integration Guide

> **The simplest way to use AutoResearchClaw**: give the repo URL to [OpenClaw](https://github.com/openclaw/openclaw) and say *"Research [your topic]."* That's it — OpenClaw handles cloning, installing, configuring, and running the entire 23-stage pipeline for you.

This guide is for humans who want to understand what's happening under the hood, or who prefer to set things up manually.

---

## Table of Contents

1. [The Easy Way: OpenClaw](#1-the-easy-way-openclaw)
2. [Manual Setup](#2-manual-setup)
3. [Configuration Walkthrough](#3-configuration-walkthrough)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Understanding the 23 Stages](#5-understanding-the-23-stages)
6. [Output Artifacts](#6-output-artifacts)
7. [Experiment Modes](#7-experiment-modes)
8. [Conference Templates](#8-conference-templates)
9. [OpenClaw Bridge (Advanced)](#9-openclaw-bridge-advanced)
10. [MetaClaw Integration (Cross-Run Learning)](#10-metaclaw-integration-cross-run-learning)
11. [Other AI Platforms](#11-other-ai-platforms)
12. [Python API](#12-python-api)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)

---

## 1. The Easy Way: OpenClaw

If you use [OpenClaw](https://github.com/openclaw/openclaw) as your AI assistant, you don't need to read the rest of this guide.

### Steps

1. Share the GitHub repo URL with OpenClaw:
   ```
   https://github.com/aiming-lab/AutoResearchClaw
   ```
2. OpenClaw reads `RESEARCHCLAW_AGENTS.md` and `README.md` — it now understands the entire system.
   > **Note:** `RESEARCHCLAW_AGENTS.md` is generated locally and listed in `.gitignore`. If it doesn't exist, OpenClaw can bootstrap from `README.md` and the project structure.
3. Say something like:
   ```
   Research the application of graph neural networks in drug discovery
   ```
4. OpenClaw will:
   - Clone the repo
   - Create a virtual environment and install dependencies (`pip install -e .`)
   - Copy `config.researchclaw.example.yaml` → `config.yaml`
   - Ask you for an OpenAI API key (or use your environment variable)
   - Run the full 23-stage pipeline
   - Return the paper, experiment code, charts, and citations

**That's the whole process.** OpenClaw is designed to read agent definition files and bootstrap itself. AutoResearchClaw ships with these files specifically so that any OpenClaw-compatible AI assistant can pick it up and run.

### What if I want to tweak settings?

Tell OpenClaw in natural language:

- *"Use GPT-5.2 instead of GPT-4o"*
- *"Run experiments in sandbox mode, not simulated"*
- *"Target ICLR 2025 format instead of NeurIPS"*
- *"Skip the quality gate, just auto-approve everything"*

OpenClaw will modify `config.yaml` accordingly before running the pipeline.

---

## 2. Manual Setup

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.11 or newer |
| **LLM API** | Any OpenAI-compatible endpoint (OpenAI, Azure, local proxy, etc.) |
| **Disk space** | ~100 MB for the repo + artifacts per run |
| **Network** | Required for LLM API calls and literature search (Semantic Scholar, arXiv) |

### Installation

```bash
# Clone the repository
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install
pip install -e .
```

### Verify Installation

```bash
# Check the CLI is available
researchclaw --help

# Validate your configuration
researchclaw validate --config config.yaml
```

---

## 3. Configuration Walkthrough

Start from the provided template:

```bash
cp config.researchclaw.example.yaml config.yaml
```

Open `config.yaml` in your editor. Here's what each section does:

### LLM Settings (Required)

This is the only section you **must** configure. Everything else has sensible defaults.

```yaml
llm:
  base_url: "https://api.openai.com/v1"     # Your LLM API endpoint
  api_key_env: "OPENAI_API_KEY"              # Environment variable name...
  api_key: ""                                # ...or paste the key directly here
  primary_model: "gpt-4o"                    # Model to use (gpt-4o, gpt-5.2, etc.)
  fallback_models:                           # Tried in order if primary fails
    - "gpt-4.1"
    - "gpt-4o-mini"
  s2_api_key: ""                             # Optional: Semantic Scholar API key for higher rate limits
```

**Using an environment variable** (recommended for security):
```bash
export OPENAI_API_KEY="sk-..."
```

**Using a direct key** (simpler, less secure):
```yaml
llm:
  api_key: "sk-your-key-here"
```

**Using a proxy or alternative provider**:
```yaml
llm:
  base_url: "https://your-proxy.example.com/v1"
  api_key: "your-proxy-key"
  primary_model: "gpt-4o"    # Must be supported by your endpoint
```

### Research Settings

```yaml
research:
  topic: "Your research topic here"    # Can also be set via CLI --topic flag
  domains:
    - "machine-learning"               # Guides literature search scope
  daily_paper_count: 10                # Target papers to collect
  quality_threshold: 4.0               # Minimum paper quality score (1-5)
```

### Experiment Settings

```yaml
experiment:
  mode: "sandbox"              # How experiments run (see Section 7)
  time_budget_sec: 300         # Max seconds per experiment run
  max_iterations: 10           # Max refinement loops in Stage 13
  metric_key: "primary_metric" # What metric to optimize
  metric_direction: "minimize" # "minimize" or "maximize"
  sandbox:
    python_path: ".venv/bin/python3"   # Python binary for sandbox execution
    gpu_required: false
    max_memory_mb: 4096
  code_agent:                        # CodeAgent v2 (multi-phase code generation)
    enabled: true                    # Architecture planning + sequential file gen + hard validation
  benchmark_agent:                   # Automated dataset & baseline selection
    enabled: true                    # 4-agent pipeline: Surveyor→Selector→Acquirer→Validator
  figure_agent:                      # Academic figure generation
    enabled: true                    # 5-agent pipeline: Planner→CodeGen→Renderer→Critic→Integrator
  repair:                            # Anti-fabrication experiment repair
    enabled: true                    # Diagnose and fix failed experiments before paper writing
    max_cycles: 3                    # Repair retry loops
  opencode:                          # OpenCode Beast Mode (see README for details)
    enabled: true
```

### Export Settings

```yaml
export:
  target_conference: "neurips_2025"   # See Section 8 for all available templates
  authors: "Anonymous"                 # Author line in the paper
  bib_file: "references"              # BibTeX file name (without .bib)
```

### Everything Else (Optional)

These have reasonable defaults. Change them only if you need to:

```yaml
project:
  name: "my-research"      # Just an identifier for your run
  mode: "full-auto"         # "docs-first", "semi-auto", or "full-auto"

runtime:
  timezone: "America/New_York"
  max_parallel_tasks: 3
  approval_timeout_hours: 12
  retry_limit: 2

security:
  hitl_required_stages: [5, 9, 20]     # Stages that pause for human approval
  allow_publish_without_approval: false

notifications:
  channel: "console"        # "console", "discord", or "slack"

knowledge_base:
  backend: "markdown"
  root: "docs/kb"
```

---

## 4. Running the Pipeline

### Basic Run

```bash
# Run with topic from config.yaml
researchclaw run --config config.yaml --auto-approve

# Override topic from command line
researchclaw run --config config.yaml --topic "Transformer attention for time series" --auto-approve
```

### CLI Commands

| Command | What It Does |
|---------|-------------|
| `researchclaw setup` | Interactive first-time setup (installs OpenCode Beast Mode, checks Docker/LaTeX) |
| `researchclaw init` | Interactive config creation (choose LLM provider, creates `config.arc.yaml`) |
| `researchclaw run` | Run the full 23-stage pipeline |
| `researchclaw validate` | Check your config file for errors |
| `researchclaw doctor` | Diagnose environment issues (Python, dependencies, API connectivity) |
| `researchclaw report --run-dir <path>` | Generate a human-readable summary of a completed run |

### Run Flags

| Flag | Effect |
|------|--------|
| `--topic "..."` | Override the topic in config.yaml |
| `--config path` | Config file path (default: `config.yaml`) |
| `--output path` | Output directory (default: `artifacts/<run-id>/`) |
| `--auto-approve` | Skip manual approval at gate stages (5, 9, 20) |
| `--from-stage STAGE_NAME` | Start from a specific stage (e.g., `PAPER_OUTLINE`) |
| `--resume` | Resume from the last checkpoint (auto-detects the most recent run matching your topic) |
| `--skip-preflight` | Skip LLM connectivity check before starting |
| `--skip-noncritical-stage` | Skip non-critical stages on failure instead of aborting |
| `--no-graceful-degradation` | Fail pipeline on quality gate failure instead of degrading gracefully |

### Examples

```bash
# Full autonomous run — no human intervention
researchclaw run -c config.yaml -t "Graph neural networks for protein folding" --auto-approve

# Resume a failed run from where it stopped
researchclaw run -c config.yaml --resume --auto-approve

# Re-run just the paper writing stages
researchclaw run -c config.yaml --from-stage PAPER_OUTLINE --auto-approve

# Check your setup before running
researchclaw doctor -c config.yaml
```

---

## 5. Understanding the 23 Stages

The pipeline runs in 8 phases. Each stage reads artifacts from previous stages and produces new ones.

### Phase A: Research Scoping

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 1 | TOPIC_INIT | LLM formulates a SMART research goal; auto-detects GPU hardware (NVIDIA/MPS/CPU) | `goal.md`, `hardware_profile.json` |
| 2 | PROBLEM_DECOMPOSE | Breaks the goal into prioritized sub-questions | `problem_tree.md` |

### Phase B: Literature Discovery

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 3 | SEARCH_STRATEGY | Plans search queries and data sources | `search_plan.yaml`, `sources.json` |
| 4 | LITERATURE_COLLECT | Queries **real APIs** (arXiv-first, then Semantic Scholar) with expanded queries for broad coverage | `candidates.jsonl` |
| 5 | LITERATURE_SCREEN | **[Gate]** Filters by relevance and quality | `shortlist.jsonl` |
| 6 | KNOWLEDGE_EXTRACT | Extracts structured knowledge cards from each paper | `cards/` |

### Phase C: Knowledge Synthesis

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 7 | SYNTHESIS | Clusters findings, identifies research gaps | `synthesis.md` |
| 8 | HYPOTHESIS_GEN | Generates falsifiable hypotheses | `hypotheses.md` |

### Phase D: Experiment Design

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 9 | EXPERIMENT_DESIGN | **[Gate]** Designs experiment plan with baselines and metrics | `exp_plan.yaml` |
| 10 | CODE_GENERATION | LLM writes hardware-aware experiment code (adapts packages/constraints to GPU tier) | `experiment.py`, `experiment_spec.md` |
| 11 | RESOURCE_PLANNING | Estimates GPU/time requirements | `schedule.json` |

### Phase E: Experiment Execution

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 12 | EXPERIMENT_RUN | Runs the experiment code (sandbox or simulated); immutable harness injected for time guard and metric validation; partial results captured on timeout | `runs/` |
| 13 | ITERATIVE_REFINE | LLM analyzes results, improves code, re-runs (up to 10 iterations); timeout-aware prompts; NaN/divergence fast-fail; stdout truncated for context efficiency | `refinement_log.json`, `experiment_final.py` |

### Phase F: Analysis & Decision

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 14 | RESULT_ANALYSIS | Statistical analysis of experiment results | `analysis.md` |
| 15 | RESEARCH_DECISION | PROCEED / PIVOT decision with evidence | `decision.md` |

### Phase G: Paper Writing

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 16 | PAPER_OUTLINE | Creates section-level paper outline | `outline.md` |
| 17 | PAPER_DRAFT | Writes paper section-by-section (3 LLM calls, 5,000-6,500 words); **hard-blocked when no experiment metrics** (anti-fabrication); conference-grade title guidelines and abstract structure injected | `paper_draft.md` |
| 18 | PEER_REVIEW | Simulates 2+ reviewer perspectives with NeurIPS/ICML rubric (1-10 scoring); checks baselines, ablations, claims vs evidence | `reviews.md` |
| 19 | PAPER_REVISION | Addresses review comments with length guard (auto-retries if revised paper is shorter than draft) | `paper_revised.md` |

### Phase H: Finalization

| # | Stage | What Happens | Produces |
|---|-------|-------------|----------|
| 20 | QUALITY_GATE | **[Gate]** Checks paper quality score | `quality_report.json` |
| 21 | KNOWLEDGE_ARCHIVE | Saves retrospective + reproducibility bundle | `archive.md`, `bundle_index.json` |
| 22 | EXPORT_PUBLISH | Generates LaTeX, charts, and code package | `paper_final.md`, `paper.tex`, `code/` |
| 23 | CITATION_VERIFY | Fact-checks all references against real APIs | `verification_report.json`, `references_verified.bib` |

### Gate Stages

Three stages pause for human review (unless `--auto-approve` is set):

| Gate | What's Being Reviewed | On Reject, Rolls Back To |
|------|-----------------------|--------------------------|
| Stage 5 | Are the collected papers relevant and sufficient? | Stage 4 (re-collect literature) |
| Stage 9 | Is the experiment design sound? | Stage 8 (re-generate hypotheses) |
| Stage 20 | Does the paper meet quality standards? | Stage 16 (re-write from outline) |

For fully autonomous operation, always use `--auto-approve`.

---

## 6. Output Artifacts

Each run creates a timestamped directory under `artifacts/`:

```
artifacts/rc-20260310-143200-a1b2c3/
├── stage-1/goal.md                        # Research goal
├── stage-2/problem_tree.md                # Problem decomposition
├── stage-3/search_plan.yaml               # Search strategy
├── stage-4/candidates.jsonl               # Raw literature results
├── stage-5/shortlist.jsonl                # Screened papers
├── stage-6/cards/                         # Knowledge cards (one per paper)
├── stage-7/synthesis.md                   # Research gap analysis
├── stage-8/hypotheses.md                  # Research hypotheses
├── stage-9/exp_plan.yaml                  # Experiment plan
├── stage-10/experiment.py                 # Generated experiment code
├── stage-10/experiment_spec.md            # Experiment specification
├── stage-11/schedule.json                 # Resource schedule
├── stage-12/runs/run-1.json               # Experiment results
├── stage-13/experiment_final.py           # Refined experiment code
├── stage-13/experiment_v1.py              # Iteration 1 snapshot
├── stage-13/refinement_log.json           # Refinement history
├── stage-14/analysis.md                   # Statistical analysis
├── stage-14/experiment_summary.json       # Metrics summary
├── stage-15/decision.md                   # Proceed/Pivot decision
├── stage-16/outline.md                    # Paper outline
├── stage-17/paper_draft.md                # Full paper draft
├── stage-18/reviews.md                    # Simulated peer reviews
├── stage-19/paper_revised.md              # Revised paper
├── stage-20/quality_report.json           # Quality assessment
├── stage-21/archive.md                    # Knowledge retrospective
├── stage-22/
│   ├── paper_final.md                     # Final paper (Markdown)
│   ├── paper.tex                          # Conference-ready LaTeX
│   ├── references.bib                     # BibTeX references
│   ├── charts/                            # Result visualizations
│   └── code/                              # Open-source code package
│       ├── experiment.py
│       ├── requirements.txt
│       └── README.md
├── stage-23/
│   ├── verification_report.json           # Citation fact-check results
│   └── references_verified.bib            # Cleaned bibliography
└── pipeline_summary.json                  # Overall execution summary
```

### Key Output Files

| File | What You'll Use It For |
|------|----------------------|
| `stage-22/paper.tex` | Submit to a conference (compile with `pdflatex` or `tectonic`) |
| `stage-22/paper_final.md` | Read or further edit the paper |
| `stage-22/references.bib` | Bibliography for LaTeX compilation |
| `stage-22/code/` | Share experiment code alongside the paper |
| `stage-23/verification_report.json` | Check which citations are real vs. hallucinated |
| `stage-13/experiment_final.py` | The best-performing experiment code |
| `stage-22/charts/` | Figures for the paper |

---

## 7. Experiment Modes

AutoResearchClaw supports four modes for running experiments:

### Simulated (Default)

```yaml
experiment:
  mode: "simulated"
```

The LLM **generates synthetic experiment results** without executing any code. This is fast and requires no special setup, but the results are not real.

**Best for**: Quick prototyping, testing the pipeline end-to-end, environments without Python scientific packages.

### Sandbox

```yaml
experiment:
  mode: "sandbox"
  sandbox:
    python_path: ".venv/bin/python3"
    gpu_required: false
    max_memory_mb: 4096
```

The pipeline **generates Python code and actually runs it** in a subprocess. The code is validated before execution (AST parsing, import whitelist, no file I/O outside sandbox). **Hardware-aware**: Stage 1 auto-detects your GPU (NVIDIA CUDA / Apple MPS / CPU-only) and adapts the generated code accordingly — high-tier GPUs get full PyTorch code, limited GPUs get lightweight experiments, CPU-only gets NumPy/sklearn only.

**Best for**: Real experiments on your local machine. Supports numpy and stdlib; deep learning frameworks (torch, tensorflow) are available if installed in your environment and GPU is detected.

**Safety features**:
- Code validation blocks dangerous operations (subprocess, eval, exec, network calls)
- Configurable memory limit and execution timeout
- Auto-repair: if generated code has validation errors, the LLM fixes them (up to 3 attempts)

### Docker

```yaml
experiment:
  mode: "docker"
  docker:
    image: "researchclaw/experiment:latest"
    gpu_enabled: true
    memory_limit_mb: 8192
    network_policy: "setup_only"   # none | setup_only | pip_only | full
    auto_install_deps: true
    shm_size_mb: 2048
```

The pipeline runs generated code inside a **Docker container** with GPU passthrough, dependency auto-installation, and network isolation. Execution follows a **three-phase model** within a single container:

1. **Phase 0 (pip install)**: Installs auto-detected dependencies from `requirements.txt` (network enabled)
2. **Phase 1 (setup.py)**: Runs `setup.py` for dataset downloads and environment preparation (network enabled)
3. **Phase 2 (experiment)**: Executes the experiment code (network disabled by default via iptables)

**Network policies**:
- `none` — No network at all (all phases offline). Requires all deps pre-installed in image.
- `setup_only` (default) — Network during Phase 0+1, disabled before Phase 2 via iptables (`--cap-add=NET_ADMIN`).
- `pip_only` — Network only during Phase 0 (pip install), disabled for Phase 1+2.
- `full` — Network available throughout all phases.

**Pre-cached datasets**: The Docker image includes CIFAR-10/100, MNIST, FashionMNIST, STL-10, and SVHN at `/opt/datasets`, mounted read-only as `/workspace/data`. No download needed for these standard benchmarks.

**Best for**: Reproducible experiments with full dependency isolation. Supports GPU passthrough (NVIDIA) and configurable network policies.

**Setup**: Build the image first:
```bash
docker build -t researchclaw/experiment:latest researchclaw/docker/
```

### SSH Remote

```yaml
experiment:
  mode: "ssh_remote"
  ssh_remote:
    host: "gpu-server.example.com"
    gpu_ids: [0, 1]
    remote_workdir: "/tmp/researchclaw_experiments"
```

The pipeline sends generated code to a remote GPU server for execution.

**Best for**: Experiments that require GPU hardware you don't have locally.

---

## 8. Conference Templates

AutoResearchClaw generates LaTeX files formatted for specific conferences:

```yaml
export:
  target_conference: "neurips_2025"
```

| Conference | Config Value | Layout |
|------------|-------------|--------|
| NeurIPS 2025 | `neurips_2025` (default) | Single-column, `neurips_2025` style |
| NeurIPS 2024 | `neurips_2024` | Single-column, `neurips_2024` style |
| ICLR 2026 | `iclr_2026` | Single-column, `iclr2026_conference` style |
| ICLR 2025 | `iclr_2025` | Single-column, `iclr2025_conference` style |
| ICML 2026 | `icml_2026` | Double-column, `icml2026` style |
| ICML 2025 | `icml_2025` | Double-column, `icml2025` style |

Short aliases are also accepted: `neurips` (→ 2025), `iclr` (→ 2026), `icml` (→ 2026).

The Markdown-to-LaTeX converter handles:
- Section headings (`#`, `##`, `###`)
- Inline and display math (`$...$`, `$$...$$`)
- Bold and italic text
- Ordered and unordered lists
- Tables
- Code blocks
- Citation references (`[cite_key]` → `\cite{cite_key}`)

### Compiling the LaTeX

```bash
# Using tectonic (recommended)
tectonic artifacts/<run-id>/stage-22/paper.tex

# Using pdflatex
cd artifacts/<run-id>/stage-22/
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## 9. OpenClaw Bridge (Advanced)

For deeper integration with OpenClaw, AutoResearchClaw includes a bridge adapter system. Each flag in the config activates a typed protocol interface:

```yaml
openclaw_bridge:
  use_cron: true              # Scheduled research runs
  use_message: true           # Progress notifications (Discord/Slack/Telegram)
  use_memory: true            # Cross-session knowledge persistence
  use_sessions_spawn: true    # Spawn parallel sub-sessions for concurrent stages
  use_web_fetch: true         # Live web search during literature review
  use_browser: false          # Browser-based paper collection
```

### What Each Adapter Does

| Adapter | Protocol | Use Case |
|---------|----------|----------|
| **Cron** | `CronAdapter.schedule_resume(run_id, stage_id, reason)` | Schedule pipeline resumption (e.g., daily re-runs) |
| **Message** | `MessageAdapter.notify(channel, subject, body)` | Send progress updates to chat platforms |
| **Memory** | `MemoryAdapter.append(namespace, content)` | Persist knowledge across sessions |
| **Sessions** | `SessionsAdapter.spawn(name, command)` | Run pipeline stages in parallel sub-sessions |
| **WebFetch** | `WebFetchAdapter.fetch(url)` | Fetch web pages during literature search |
| **Browser** | `BrowserAdapter.open(url)` | Open and interact with web pages |

When OpenClaw provides a capability (e.g., message sending), the adapter consumes it automatically. When running standalone, recording stubs capture all calls for debugging without side effects.

This is an **extension point** — you don't need to configure it for basic usage.

---

## 10. MetaClaw Integration (Cross-Run Learning)

[MetaClaw](https://github.com/aiming-lab/MetaClaw) adds **cross-run knowledge transfer** to AutoResearchClaw. When enabled, the pipeline automatically captures lessons from failures and converts them into reusable skills that improve subsequent runs.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│              AutoResearchClaw Pipeline                │
│  Stage 1 → 2 → ... → 23                             │
│                                                      │
│  ┌─────────────┐    ┌──────────────────────────────┐ │
│  │ LLMClient   │───▶│ MetaClaw Integration Layer   │ │
│  │             │    │ (metaclaw_bridge module)      │ │
│  └─────────────┘    └──────────┬───────────────────┘ │
│                                │                     │
│  ┌─────────────┐    ┌──────────▼───────────────────┐ │
│  │ Evolution   │◀──▶│ Lesson ↔ Skill Bridge        │ │
│  │ Store       │    └─────────────────────────────┘ │
│  └─────────────┘                                     │
└──────────────────────────┬───────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │     MetaClaw Proxy Server    │
            │     (optional, :30000)       │
            │  ┌────────────────────────┐  │
            │  │ SkillManager (40+ skills)│ │
            │  │ + arc-* learned skills   │ │
            │  └────────────────────────┘  │
            └─────────────────────────────┘
```

### How It Works

1. **Lesson Capture**: During each pipeline run, the `EvolutionStore` automatically records failures, warnings, and anomalies as structured lessons in `evolution/lessons.jsonl`.

2. **Lesson → Skill Conversion**: After a run completes, lessons above a configurable severity threshold are converted into `arc-*` skill files stored in `~/.metaclaw/skills/`. Each skill contains: trigger conditions, failure root cause, and actionable guidance.

3. **Skill Injection**: On the next run, `build_overlay()` reads all `arc-*` skills and injects them into the LLM prompt for every stage via the `evolution_overlay` parameter. The LLM receives explicit instructions to avoid previously encountered pitfalls.

4. **Proxy Routing (Optional)**: When the MetaClaw proxy is running, LLM requests are routed through it for additional skill matching and session tracking. If the proxy is unavailable, requests automatically fall back to the direct LLM endpoint.

### Setup

#### Step 1: Install MetaClaw

```bash
pip install metaclaw
# Or clone from source:
git clone https://github.com/aiming-lab/MetaClaw.git
cd metaclaw && pip install -e .
```

#### Step 2: Configure

Add the `metaclaw_bridge` section to your `config.arc.yaml`:

```yaml
metaclaw_bridge:
  enabled: true
  proxy_url: "http://localhost:30000/v1"    # MetaClaw proxy (optional)
  skills_dir: "~/.metaclaw/skills"          # Skill storage directory
  fallback_url: "https://api.openai.com/v1" # Direct LLM fallback
  fallback_api_key_env: "OPENAI_API_KEY"
  lesson_to_skill:
    enabled: true
    min_severity: "warning"                 # Convert warnings + errors
    max_skills_per_run: 5                   # Max new skills per run
```

#### Step 3: Run

```bash
# First run — captures lessons, generates initial skills
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve

# Check generated skills
ls ~/.metaclaw/skills/arc-*/SKILL.md

# Second run — skills from Run 1 are automatically injected
researchclaw run --config config.arc.yaml --topic "Your idea" --auto-approve
```

#### Optional: Start MetaClaw Proxy

For full skill matching and session tracking:

```bash
metaclaw start --mode skills_only --port 30000
# Or use the provided script:
bash scripts/metaclaw_start.sh
```

The proxy is optional — without it, the pipeline still benefits from skill injection via `build_overlay()` and falls back to your configured LLM endpoint.

### Experiment Results

In controlled A/B experiments (same topic, same LLM, same configuration):

| Metric | Baseline | With MetaClaw | Improvement |
|--------|----------|---------------|-------------|
| Stage retry rate | 10.5% | 7.9% | **-24.8%** |
| Refine cycle count | 2.0 | 1.2 | **-40.0%** |
| Pipeline stage completion | 18/19 | 19/19 | **+5.3%** |
| Overall robustness score (composite) | 0.714 | 0.845 | **+18.3%** |

> Composite robustness score is a weighted average of stage completion rate (40%), retry reduction (30%), and refine cycle efficiency (30%).

### Key Files

| File | Purpose |
|------|---------|
| `researchclaw/metaclaw_bridge/` | Integration module (config, session, lesson_to_skill, prm_gate, skill_feedback) |
| `researchclaw/evolution.py` | `build_overlay()` — reads intra-run lessons + cross-run arc-* skills |
| `researchclaw/llm/client.py` | Proxy routing with automatic fallback |
| `~/.metaclaw/skills/arc-*/SKILL.md` | Learned skill files (auto-generated) |
| `scripts/metaclaw_start.sh` | Helper script to launch MetaClaw proxy |

### Backward Compatibility

- **Default: OFF.** Without `metaclaw_bridge.enabled: true`, the pipeline is completely unchanged.
- **No new required dependencies.** MetaClaw is optional.
- **All 1,823 existing tests pass** with the integration code.

---

## 11. Other AI Platforms

AutoResearchClaw works with any AI coding assistant that can read project context files.

### Claude Code

Claude Code automatically reads `RESEARCHCLAW_CLAUDE.md` (if present) when you open the project. It also loads the skill definition from `.claude/skills/researchclaw/SKILL.md`.

> **Note:** `RESEARCHCLAW_CLAUDE.md` is generated locally and listed in `.gitignore`. The `.claude/skills/researchclaw/SKILL.md` file is always available in the repo.

```
You: Research the impact of attention mechanisms on speech recognition
Claude: [Reads project context, runs the pipeline, returns results]
```

### Copilot CLI (GitHub)

GitHub Copilot can be used as an ACP agent via the `gh` CLI command (GitHub CLI with Copilot extension). Set the ACP agent to `gh` in your config:

```yaml
llm:
  provider: "acp"
  acp:
    agent: "gh"
    cwd: "."
```

Prerequisites:
1. Install [GitHub CLI](https://cli.github.com/) (`gh`)
2. Install the Copilot extension: `gh extension install github/gh-copilot`
3. Authenticate: `gh auth login`

### OpenCode

OpenCode loads skills from `.claude/skills/`. The `researchclaw` skill activates on research-related queries and guides the agent through the pipeline.

### Any AI CLI

Provide `RESEARCHCLAW_AGENTS.md` (if generated locally) or `README.md` as context to any AI assistant. `RESEARCHCLAW_AGENTS.md` contains:
- The agent role definition (research orchestrator)
- Quick setup instructions
- Pipeline stage reference
- Decision guide for common scenarios

The agent reads this file and knows how to install, configure, and run the pipeline. If the file is not present, the `README.md` and `.claude/skills/researchclaw/SKILL.md` provide sufficient context for any AI assistant to operate the pipeline.

---

## 12. Python API

For programmatic use or custom integrations:

```python
from researchclaw.pipeline.runner import execute_pipeline
from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from pathlib import Path

# Load configuration
config = RCConfig.load("config.yaml", check_paths=False)

# Run the full pipeline
results = execute_pipeline(
    run_dir=Path("artifacts/my-run"),
    run_id="run-001",
    config=config,
    adapters=AdapterBundle(),
    auto_approve_gates=True,
)

# Check results
for result in results:
    print(f"Stage {result.stage.name}: {result.status.value}")
```

### Iterative Pipeline (Multiple Paper Revisions)

```python
from researchclaw.pipeline.runner import execute_iterative_pipeline

results = execute_iterative_pipeline(
    run_dir=Path("artifacts/my-run"),
    run_id="run-001",
    config=config,
    adapters=AdapterBundle(),
    max_iterations=3,       # Re-run paper writing up to 3 times
    convergence_rounds=2,   # Stop if quality stabilizes for 2 rounds
)
```

### Literature Search Only

```python
from researchclaw.literature.search import search_papers

papers = search_papers("transformer attention mechanisms", limit=20)
for p in papers:
    print(f"{p.title} ({p.year}) — cited {p.citation_count}x")
    print(p.to_bibtex())
```

---

## 13. Troubleshooting

### Pre-Run Diagnostics

```bash
# Check everything: Python version, dependencies, API connectivity, config validity
researchclaw doctor --config config.yaml
```

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `Missing required field: llm.base_url` | Config incomplete | Set `llm.base_url` and `llm.api_key` (or `api_key_env`) |
| `Config validation FAILED` | Invalid YAML or missing fields | Run `researchclaw validate -c config.yaml` for details |
| `Preflight check... FAILED` | LLM API unreachable | Check `base_url`, API key, and network connectivity |
| Sandbox execution fails | Python path wrong or missing packages | Verify `experiment.sandbox.python_path` exists; ensure numpy is installed |
| Code validation rejects all attempts | LLM generates unsafe code | Switch to `simulated` mode, or try a more capable model |
| Gate stage blocks pipeline | Manual approval required | Use `--auto-approve` for autonomous mode |
| Pipeline fails mid-run | Transient API error | Run with `--resume` to continue from the last checkpoint |
| Citations marked HALLUCINATED | LLM invented fake references | This is expected — Stage 23 catches these. Use `references_verified.bib` instead |
| LaTeX won't compile | Missing style packages | Install the conference style files, or use `tectonic` which auto-downloads them |

### Resuming a Failed Run

```bash
# Resume from the exact point of failure
researchclaw run -c config.yaml --resume --auto-approve

# Or restart from a specific stage
researchclaw run -c config.yaml --from-stage EXPERIMENT_RUN --auto-approve --output artifacts/<run-id>
```

### Reading a Run Report

```bash
researchclaw report --run-dir artifacts/rc-20260310-143200-a1b2c3
```

This prints a human-readable summary: which stages passed, which failed, key metrics, and paper quality scores.

---

## 14. FAQ

**Q: How much does a full pipeline run cost in API credits?**
A: Depends on your model and topic complexity. A typical run with GPT-4o makes ~35-60 API calls across all 23 stages (paper drafting now uses 3 sequential calls for section-by-section writing). Expect roughly $3-12 per run. Simulated mode uses slightly fewer tokens since it doesn't generate real experiment code.

**Q: Can I use a local LLM (Ollama, vLLM, etc.)?**
A: Yes — any OpenAI-compatible endpoint works. Set `llm.base_url` to your local server (e.g., `http://localhost:11434/v1` for Ollama). Quality depends heavily on the model's capabilities.

**Q: Can I run only part of the pipeline?**
A: Yes. Use `--from-stage STAGE_NAME` to start from any stage. The stage reads its inputs from previously generated artifacts, so the earlier stages must have completed at least once.

**Q: Are the literature references real?**
A: Yes. Stage 4 uses a multi-source strategy (arXiv-first, then Semantic Scholar) with query expansion to find real papers with real titles, DOIs, and citation counts. The pipeline typically collects 100-200 candidates and aims for 30-60 references in the final paper. Stage 23 then verifies every reference to catch any that the LLM might have hallucinated during paper writing.

**Q: Can I use this for a real paper submission?**
A: AutoResearchClaw is a research tool, not a paper mill. The output is a strong first draft that should be reviewed, improved, and validated by a human researcher before submission. Think of it as an extremely thorough research assistant.

**Q: What happens if the LLM API goes down mid-run?**
A: The pipeline checkpoints after every stage. Use `--resume` to pick up where it left off. Failed stages are retried according to the `max_retries` setting in each stage's contract.

**Q: Can I change the research topic mid-run?**
A: Not recommended — the pipeline builds on prior stages' outputs. Start a new run with the new topic instead.

---

*Last updated: March 2026 · AutoResearchClaw v0.3.1+*
