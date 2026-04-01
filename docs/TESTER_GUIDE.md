<p align="center">
  <img src="../image/logo.png" width="500" alt="AutoResearchClaw Logo">
</p>

<h2 align="center">🧪 Community Testing Guide</h2>

<p align="center">
  <b>Help us stress-test the world's first fully autonomous research pipeline — across every domain.</b>
</p>

<p align="center">
  <a href="https://github.com/aiming-lab/AutoResearchClaw">⭐ Star the Repo</a> ·
  <a href="#-quick-start">🚀 Quick Start</a> ·
  <a href="#-feedback-template">📋 Feedback Template</a> ·
  <a href="TESTER_GUIDE_CN.md">🇨🇳 中文测试指南</a> ·
  <a href="TESTER_GUIDE_JA.md">🇯🇵 日本語テストガイド</a>
</p>

---

## 👋 Welcome, Tester!

**AutoResearchClaw** is a fully autonomous academic paper generation pipeline. You give it a research idea — it handles everything else: literature search, experiment design, code generation, experiment execution, paper writing, peer review, and final delivery. **23 stages, zero human intervention.**

We're looking for testers from **all disciplines and backgrounds** — machine learning, NLP, computer vision, reinforcement learning, bioinformatics, physics, social sciences, and beyond. The more diverse the testing, the better the pipeline becomes.

**Your mission:** Run the pipeline with your own research idea, inspect the output, and submit a detailed feedback report. That's it. Every piece of feedback directly shapes the next version.

---

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Installation & Setup](#-installation--setup)
3. [Running the Pipeline](#-running-the-pipeline)
4. [Inspecting the Output](#-inspecting-the-output)
5. [Feedback Report Requirements](#-feedback-report-requirements)
6. [Feedback Template](#-feedback-template)
7. [FAQ](#-faq)

---

## 📦 Prerequisites

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | macOS / Linux / WSL2 | Linux (Ubuntu 22.04+) |
| Python | 3.11+ | 3.11 or 3.12 |
| Disk | 500 MB | 2 GB+ |
| RAM | 8 GB | 16 GB+ |
| GPU | Not required (sandbox mode) | NVIDIA GPU + CUDA 12.x (docker mode) |
| Network | Required (LLM API + literature search) | Stable connection |
| LLM API Key | **Required** | OpenAI or Anthropic |

### 🔑 About API Keys

The pipeline calls a large language model (LLM) at every stage — writing, coding, reviewing, and more. You'll need an API key from **OpenAI** or **Anthropic**.

> **We strongly recommend using the most capable models available for the best results:**
>
> | Provider | Recommended Model | Fallback |
> |----------|------------------|----------|
> | **OpenAI** | **GPT-5.4** (best) | GPT-5.1 or GPT-4.1 |
> | **Anthropic** | **Claude Opus 4.6** (best) | Claude Sonnet 4.6 |
>
> Using a top-tier model significantly improves paper quality, code correctness, and experiment design. Older models (e.g., GPT-4o) may produce noticeably weaker output.

---

## 🛠 Installation & Setup

### ⚠️ Always Use the Latest Version

> **This project is under active development.** The codebase is updated frequently, and different versions can produce significantly different results.
>
> **Before every test run, always pull the latest code:**
>
> ```bash
> cd AutoResearchClaw
> git pull origin main
> pip install -e .    # Re-install to pick up changes
> ```
>
> Record your version for the feedback report:
> ```bash
> git log --oneline -1
> ```

---

### Option A: Claude Code (Fastest — Recommended ⚡)

If you have [Claude Code](https://claude.ai/claude-code) (Anthropic's CLI tool), just paste this:

```
Please clone and install AutoResearchClaw:
https://github.com/aiming-lab/AutoResearchClaw.git

If already cloned, run git pull origin main to update to the latest version first.

Then create a config file with:
- LLM: OpenAI with gpt-5.4 (or Anthropic Claude Opus 4.6)
- Experiment mode: sandbox (local execution)
- Research topic: "<YOUR RESEARCH IDEA HERE>"
- Auto-approve all gate stages

My API key is: sk-xxxx (set it as an environment variable, don't hardcode it)
```

Claude Code will handle cloning, dependencies, configuration, and execution automatically.

### Option B: Manual Installation

```bash
# 1. Clone the repo
git clone https://github.com/aiming-lab/AutoResearchClaw.git
cd AutoResearchClaw

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows (prefer WSL2)

# 3. Install
pip install -e .

# 4. Verify
researchclaw --help
```

### ⚙️ Configuration

```bash
cp config.researchclaw.example.yaml config.yaml
```

Edit `config.yaml` — here are the key fields:

```yaml
# === Project ===
project:
  name: "my-test"
  mode: "full-auto"

# === Research Topic — describe your idea in English ===
research:
  topic: "Your research idea in 1-2 sentences"
  domains:
    - "machine-learning"     # Options: nlp, cv, rl, graph-learning, etc.

# === LLM — use the strongest model you have access to! ===
#
# Option 1: OpenAI (GPT-5.4 recommended)
llm:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  primary_model: "gpt-5.4"              # Best available
  fallback_models:
    - "gpt-5.1"
    - "gpt-4.1"

# Option 2: Anthropic Claude (Claude Opus 4.6 recommended)
# llm:
#   provider: "openai-compatible"
#   base_url: "https://api.anthropic.com/v1"
#   api_key_env: "ANTHROPIC_API_KEY"
#   primary_model: "claude-opus-4-6"
#   fallback_models:
#     - "claude-sonnet-4-6"

# === Experiment ===
experiment:
  mode: "sandbox"                # sandbox = local execution (recommended)
  time_budget_sec: 600           # Max seconds per experiment run
  max_iterations: 10
  metric_key: "primary_metric"
  metric_direction: "minimize"   # or "maximize"
```

### 🔐 Set Your API Key

```bash
# OpenAI users:
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Anthropic users:
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx"

# Optional: Semantic Scholar API key (speeds up literature search)
export S2_API_KEY="your-s2-key"
```

> **🔒 Security:** Never hardcode API keys in files. Use `api_key_env` in the config to reference an environment variable.

---

## 🚀 Running the Pipeline

### Quick Start

```bash
source .venv/bin/activate
export OPENAI_API_KEY="sk-xxxx"       # or ANTHROPIC_API_KEY

researchclaw run --config config.yaml --auto-approve
```

### With a Specific Topic

```bash
researchclaw run \
  --config config.yaml \
  --topic "Investigating the effect of curriculum learning on image classification with adaptive difficulty scheduling" \
  --auto-approve
```

### ⏱ Expected Runtime

| Mode | Estimated Time | Notes |
|------|---------------|-------|
| sandbox | 30 min – 2 hours | Depends on experiment complexity & API speed |
| docker (GPU) | 1 – 4 hours | For heavier deep learning experiments |

The terminal shows real-time progress. **No manual intervention needed** — sit back and let it run.

### ✅ How to Know It's Done

You'll see output like:

```
[Stage 23/23] ✓ Deliverables packaged
Pipeline complete — deliverables at: artifacts/rc-20260315-XXXXXX-YYYY/deliverables/
```

### 🔄 If It Gets Interrupted

The pipeline supports checkpointing — just resume:

```bash
researchclaw run --config config.yaml --resume
```

---

## 🔍 Inspecting the Output

After completion, find your results in `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/deliverables/`.

### 📂 Deliverables

| File / Directory | Description |
|-----------------|-------------|
| `paper_final.md` | Final paper in Markdown (5,000–6,500 words) |
| `paper.tex` | Conference-ready LaTeX source (directly compilable) |
| `references.bib` | BibTeX bibliography (verified citations) |
| `code/main.py` | Auto-generated experiment code |
| `code/requirements.txt` | Python dependencies for experiments |
| `charts/` | Result visualization charts (PNG) |
| `verification_report.json` | Citation integrity verification report |
| `manifest.json` | Deliverable manifest with metadata |

### 🔎 What to Check

1. **Paper Content** (`paper_final.md` or `paper.tex`)
   - Is the title relevant to the topic?
   - Does the abstract clearly state problem, method, and results?
   - Does Related Work cite key papers in the field?
   - Is the method description technically correct?
   - Is the experiment design sound (datasets, baselines, metrics)?
   - Are results meaningful (not all zeros, not NaN)?
   - Are conclusions consistent with experimental findings?

2. **Experiment Code** (`code/main.py`)
   - Can it run independently?
   - Does it use real datasets (not randomly generated fake data)?
   - Does it implement what the paper describes?
   - Are hyperparameters reasonable?

3. **Charts** (`charts/`)
   - Are they readable and clean?
   - Are axis labels correct?
   - Does the data match the paper's claims?

4. **References** (`references.bib`)
   - Do the cited papers actually exist?
   - Are citations relevant to the discussion?

### 📊 Auto-Generated Quality Report

The pipeline produces a quality assessment at `stage-20/quality_report.json` containing:

- `score_1_to_10` — automated quality score
- `verdict` — accept / reject recommendation
- `strengths` — what went well
- `weaknesses` — identified issues
- `required_actions` — suggested improvements

Please reference this in your feedback, and add your own expert judgment.

---

## 📝 Feedback Report Requirements

**Your feedback is the single most important input for improving this project.** Please be thorough and honest — critical feedback is just as valuable as praise.

### What to Submit

| # | Item | Details |
|---|------|---------|
| F1 | **Feedback Report** (use template below) | Markdown format, named `feedback_<your-name>.md` |
| F2 | **Full Output Directory** | Zip the entire `artifacts/rc-XXXXXX/` directory |
| F3 | **Config File** | Your `config.yaml` (**remove API keys first!**) |
| F4 | **Terminal Log** (optional but helpful) | Copy of the terminal output during the run |

### The Four Dimensions of Feedback

#### 🎯 (a) Quality Assessment

From your domain expertise:

- If this were a paper in your field, what level would it reach? (top venue / mid-tier / workshop / unpublishable)
- How does the writing compare to papers you normally read?
- Is the method technically correct? Any obvious errors?
- Is the experiment design reasonable?

#### 💡 (b) Improvement Suggestions

- Which stage produced the weakest output? (literature search / experiment design / code generation / paper writing)
- Any obvious code errors or poor design choices?
- Specific suggestions for improving the paper structure or writing?

#### ⚖️ (c) Pipeline Design Assessment

- Are the 23 stages well-designed? Any redundant or missing steps?
- Is the iterative experiment refinement effective?
- Is the LLM guidance at each stage appropriate?

#### 🐛 (d) Bug Reports

Please report any issues you find, as specifically as possible:

- **Writing bugs:** grammar errors, repeated paragraphs, contradictions, references to non-existent figures
- **Code bugs:** runtime errors, logic errors, data handling issues
- **Result bugs:** all-zero results, NaN values, unreasonable metrics
- **Pipeline bugs:** stages getting stuck, unexpected crashes, resource exhaustion

---

## 📋 Feedback Template

Copy the template below, fill it out, and save as `feedback_<your-name>.md`:

````markdown
# AutoResearchClaw — Test Feedback Report

## Basic Information

- **Tester Name:**
- **Domain / Field:** (e.g., Computer Vision / NLP / Reinforcement Learning / Bioinformatics / ...)
- **Test Date:**
- **Code Version:** (output of `git log --oneline -1`, e.g., `44151b1 fix: Phase 3 regression test findings`)
- **Research Topic (English):**
- **LLM Model Used:** (e.g., gpt-5.4 / gpt-5.1 / claude-opus-4-6 / claude-sonnet-4-6)
- **Experiment Mode:** (sandbox / docker)
- **Total Runtime:** (~X minutes)
- **Completed All 23 Stages?:** Yes / No (if No, which stage failed?)

---

## 1. Quality Assessment (Score: 1–10)

**My Score:** X / 10

### 1.1 Overall Paper Quality
- What level paper does this correspond to? (top venue / mid-tier / workshop / unpublishable)
- Reason for score:

### 1.2 Section-by-Section Assessment

| Section | Score (1-10) | Comments |
|---------|-------------|----------|
| Title | | |
| Abstract | | |
| Introduction | | |
| Related Work | | |
| Method | | |
| Experiment Design | | |
| Results & Analysis | | |
| Conclusion | | |
| References | | |
| Charts / Figures | | |
| Code Quality | | |

### 1.3 Comparison with Human-Written Papers
- Compared to papers you normally read/write, where are the gaps?
- Anything surprisingly good?

---

## 2. Improvement Suggestions

### 2.1 Top Issues (list 3-5, in priority order)

1.
2.
3.

### 2.2 Code Issues
- Can the code run independently?
- Does it use real datasets and baselines?
- Specific code issues (if any):

### 2.3 Writing Issues
- Is the paper structure reasonable?
- Is the technical description accurate?
- Specific writing issues (if any):

---

## 3. Pipeline Design Assessment

### 3.1 Pipeline Flow
- Is the 23-stage design reasonable?
- Any redundant or missing steps?

### 3.2 Experiment Execution
- Is the experiment design sound? (dataset choices, comparison methods, metrics)
- Is the iterative refinement effective?

### 3.3 LLM Usage
- How well did the LLM perform at each stage?
- Any obvious "hallucinations" or unreasonable outputs?

---

## 4. Bug Reports

### 4.1 Writing Bugs
| # | Location (section/paragraph) | Description | Severity (High/Med/Low) |
|---|------------------------------|-------------|------------------------|
| W1 | | | |
| W2 | | | |

### 4.2 Code Bugs
| # | File / Line | Description | Severity (High/Med/Low) |
|---|-------------|-------------|------------------------|
| C1 | | | |
| C2 | | | |

### 4.3 Result Bugs
| # | Description | Affected Metrics/Charts | Severity (High/Med/Low) |
|---|-------------|------------------------|------------------------|
| R1 | | | |
| R2 | | | |

### 4.4 Pipeline Bugs
| # | Stage | Description | Severity (High/Med/Low) |
|---|-------|-------------|------------------------|
| P1 | | | |
| P2 | | | |

---

## 5. Additional Comments

(Free-form: any observations, ideas, or suggestions you think would be valuable)

---

## Attachments Checklist

- [ ] Feedback report (`feedback_<name>.md`)
- [ ] Full output directory (`artifacts/rc-XXXXXX.zip`)
- [ ] Config file (`config.yaml`, API keys removed)
- [ ] Terminal log (optional)
````

---

## ❓ FAQ

### Q1: Can I test without a GPU?

**Yes!** Use `experiment.mode: "sandbox"` — the pipeline runs experiments on your CPU. The experiments will be simpler, but still enough for a full end-to-end test.

### Q2: How much does an API call cost?

A full pipeline run costs roughly **$5–15** in API fees, depending on the model, number of revision iterations, and experiment complexity. Top-tier models (GPT-5.4, Claude Opus 4.6) cost a bit more but produce significantly better results.

### Q3: What if the pipeline crashes mid-run?

Resume from the checkpoint:

```bash
researchclaw run --config config.yaml --resume
```

### Q4: Can I use a non-English research topic?

We recommend describing your topic in **English**. The pipeline's prompts, literature search, and paper generation are all English-based. If your idea is originally in another language, please translate it first.

### Q5: What kind of research topic should I pick?

Choose a **specific research question in a field you know well** — that way you can meaningfully assess whether the output is technically correct. Tips:

- ✅ Pick topics with clear experimental validation (classification, regression, RL tasks, etc.)
- ❌ Avoid overly broad or abstract topics (e.g., "AGI", "general intelligence")
- ✅ Be specific: *"Investigating the effect of data augmentation strategies on few-shot learning for medical image classification"*

### Q6: How do I use Docker mode? (Advanced)

If you have an NVIDIA GPU with Docker + NVIDIA Container Toolkit:

```bash
# 1. Build the experiment image
docker build -t researchclaw/experiment:latest researchclaw/docker/

# 2. Update config.yaml:
#   experiment:
#     mode: "docker"
#     docker:
#       gpu_enabled: true
#       memory_limit_mb: 8192
#       network_policy: "setup_only"  # recommended default

# 3. Run
researchclaw run --config config.yaml --auto-approve
```

Docker mode uses a three-phase execution model: pip install (network on) → setup.py (network on) → experiment (network off). The image includes pre-cached datasets (CIFAR-10/100, MNIST, FashionMNIST, STL-10, SVHN) so standard benchmarks work without network access.

### Q7: I tested before — what should I do for a re-test?

**Always pull the latest code** before each test:

```bash
cd AutoResearchClaw
git pull origin main
pip install -e .
```

Then verify your version:

```bash
git log --oneline -1
```

Different versions can produce very different results. Always note the commit hash in your feedback report.

### Q8: Where do I submit my feedback?

Submit your feedback report and attachments through one of these channels:

- **GitHub Issues:** [Open an issue](https://github.com/aiming-lab/AutoResearchClaw/issues) with the label `feedback`
- **Pull Request:** Submit your `feedback_<name>.md` to the `community-feedback/` directory
- **Email:** Contact the project maintainers (see repo for details)

---

## 🌍 We Need Testers from Every Field

The pipeline has been tested primarily on ML topics so far. We especially welcome testers from:

- 🧬 **Bioinformatics & Computational Biology**
- 🧪 **Chemistry & Materials Science**
- 📊 **Statistics & Applied Mathematics**
- 🤖 **Robotics & Control Systems**
- 🗣️ **NLP & Computational Linguistics**
- 👁️ **Computer Vision & Graphics**
- 🎮 **Reinforcement Learning & Game Theory**
- 🏥 **Medical AI & Healthcare**
- 🌐 **Graph Learning & Network Science**
- 💹 **Financial ML & Econometrics**
- 🛰️ **Remote Sensing & Geospatial AI**

...and any other field where computational experiments are involved!

---

## 🙏 Thank You

Every piece of feedback — big or small — directly improves AutoResearchClaw. Thank you for being part of this journey.

<p align="center">
  <b>⭐ If you find this project interesting, please give us a star on <a href="https://github.com/aiming-lab/AutoResearchClaw">GitHub</a>!</b>
</p>
