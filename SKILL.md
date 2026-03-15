---
name: auto-research-claw
description: >
  Run AutoResearchClaw — a fully autonomous research pipeline that goes from a
  topic prompt to a structured research paper. Use when someone says "research X",
  "write a paper on X", "run AutoResearchClaw on X", or needs a deep autonomous
  research run with literature search, experiments, and citation verification.
  Covers any domain: marketing, ML, finance, science, business, etc.
metadata:
  openclaw:
    requires:
      bins: []
      python: "3.11"
    install:
      - id: pip
        kind: shell
        label: "Install AutoResearchClaw dependencies"
        command: |
          cd ~/.openclaw/workspace/AutoResearchClaw
          python3 -m venv .venv
          .venv/bin/pip install -e . -q
---

# AutoResearchClaw

Fully autonomous research: **Chat an idea → Get a paper.**

Runs a 23-stage pipeline: topic scoping → literature search → experiment design → code execution → paper writing → citation verification.

Uses **Codex CLI** (authenticated via your ChatGPT Pro/Plus account — no separate OpenAI API key needed).

---

## Setup (first time only)

```bash
# Clone into your workspace
git clone https://github.com/ArielleTolome/AutoResearchClaw.git \
  ~/.openclaw/workspace/AutoResearchClaw

cd ~/.openclaw/workspace/AutoResearchClaw

# Create virtualenv and install
python3 -m venv .venv
.venv/bin/pip install -e .

# Copy the codex-cli config
cp config.arc.codex-cli.yaml config.arc.yaml
```

Codex CLI auth uses `~/.codex/` — run `codex login` once if not already authenticated.

---

## Running Research

### Via shell script (recommended)

```bash
cd ~/.openclaw/workspace/AutoResearchClaw
./research.sh "Your research topic here"
```

For manual gate review (stops at approval checkpoints):
```bash
./research.sh "Your topic" --no-auto-approve
```

With a custom config:
```bash
./research.sh "Your topic" --config config.arc.codex-cli.yaml
```

### Via Python CLI

```bash
cd ~/.openclaw/workspace/AutoResearchClaw
.venv/bin/researchclaw run \
  --config config.arc.codex-cli.yaml \
  --topic "Your research topic here" \
  --auto-approve
```

---

## Codex CLI — Correct Invocation

The pipeline internally uses `codex exec` with these flags:

```bash
codex exec --json --skip-git-repo-check -o <outfile> -
```

> **Do NOT use these — they don't exist in Codex CLI:**
> - `--approval-policy never` ❌
> - `-q "prompt"` ❌
> - `-m o4-mini` ❌ (use `-c model=o4-mini` instead)

For quick one-off research prompts (outside the full pipeline):

```bash
/path/to/codex exec -s danger-full-access "Your research prompt here"
```

---

## Configuration

Edit `config.arc.yaml` before running. Key fields:

```yaml
research:
  topic: "Override the topic here (or pass via --topic flag)"
  daily_paper_count: 10        # papers fetched from arXiv/Semantic Scholar
  quality_threshold: 4.0       # 0-5 gate score to advance stages

llm:
  provider: "codex-cli"        # uses ChatGPT Pro via Codex CLI
  primary_model: "gpt-5.3-codex-spark"  # default model

experiment:
  mode: "sandbox"              # sandbox | ssh_remote
  # For GPU jobs on a remote machine:
  ssh_remote:
    host: "100.86.239.1"       # Bill's GPU server
    remote_workdir: "/tmp/researchclaw_experiments"
```

---

## Output Artifacts

After a run, results land in `outputs/<run-id>/`:

| File | Description |
|------|-------------|
| `paper.md` / `paper.pdf` | Final research paper |
| `experiments/` | Code, results, charts |
| `literature/` | Fetched papers + summaries |
| `citations.bib` | BibTeX references |
| `pipeline.log` | Full stage-by-stage log |

---

## Tips

- Keep topics **specific** — "loyalty discount mechanics in auto insurance" beats "auto insurance"
- Use `--no-auto-approve` for sensitive domains where you want to review at gates
- For GPU experiments, set `experiment.mode: ssh_remote` and point to Bill's server
- Research outputs should feed into your next creative brief or hook batch
- Tag runs in Discord `#research` so Marcus and Christina can see findings
