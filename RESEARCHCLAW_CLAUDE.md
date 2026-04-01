# ResearchClaw

## What This Is

ResearchClaw is a **fully autonomous academic research pipeline**. Given a research topic, it automatically completes literature review, hypothesis generation, experiment design, code generation & execution, result analysis, paper writing, peer review simulation, and final export — all through a 23-stage state machine driven by LLM calls.

## Quick Start

```bash
# 1. Copy and edit config
cp config.researchclaw.example.yaml config.yaml
# Fill in your LLM API key and base URL

# 2. Install
pip install -e .

# 3. Run
researchclaw run --topic "Your research topic" --auto-approve
```

Or programmatically:

```python
from researchclaw.pipeline.runner import execute_pipeline
from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from pathlib import Path

config = RCConfig.load("config.yaml", check_paths=False)
results = execute_pipeline(
    run_dir=Path("artifacts/my-run"),
    run_id="test-001",
    config=config,
    adapters=AdapterBundle(),
    auto_approve_gates=True,
)
```

## Project Structure

```
researchclaw/
├── __init__.py              # Version (0.3.2)
├── config.py                # RCConfig dataclass, validation, YAML loading
├── adapters.py              # AdapterBundle (recording stubs for notifications, OpenClaw bridge)
├── cli.py                   # CLI: `researchclaw run` and `researchclaw validate`
├── pipeline/
│   ├── stages.py            # 23-stage IntEnum, transitions, gate logic, rollback rules
│   ├── contracts.py         # StageContract for each stage (required_keys, produced_keys, gate flag)
│   ├── executor.py          # 23 stage executor functions + dispatch table (_STAGE_EXECUTORS)
│   └── runner.py            # execute_pipeline(), execute_iterative_pipeline()
├── llm/
│   └── client.py            # LLMClient (OpenAI-compatible), from_rc_config() factory
├── experiment/
│   ├── sandbox.py           # ExperimentSandbox (local subprocess execution)
│   ├── runner.py            # ExperimentRunner (run management)
│   ├── git_manager.py       # ExperimentGitManager
│   ├── validator.py         # AST syntax check, security scan, import check, auto-repair
│   └── visualize.py         # matplotlib charts (trajectory, comparison, timeline, iteration)
└── knowledge/
    └── base.py              # KnowledgeBase (markdown file write, 23-stage category map)
```

## 23-Stage Pipeline

```
Phase A: Research Scoping
  1: TOPIC_INIT           — Define research question, scope, constraints
  2: PROBLEM_DECOMPOSE    — Break into sub-problems, identify variables

Phase B: Literature Discovery
  3: SEARCH_STRATEGY      — Search strategy + data source verification
  4: LITERATURE_COLLECT   — Execute search, collect candidate papers
  5: LITERATURE_SCREEN    — Relevance + quality screening [GATE]
  6: KNOWLEDGE_EXTRACT    — Structured knowledge card extraction

Phase C: Knowledge Synthesis
  7: SYNTHESIS            — Topic clustering + research gap analysis
  8: HYPOTHESIS_GEN       — Generate falsifiable research hypotheses

Phase D: Experiment Design
  9: EXPERIMENT_DESIGN    — Design experiment protocol [GATE]
  10: CODE_GENERATION     — Generate executable experiment code (with validation)
  11: RESOURCE_PLANNING   — Resource scheduling, dependency ordering

Phase E: Experiment Execution
  12: EXPERIMENT_RUN      — Execute experiments (sandbox/remote/simulated)
  13: ITERATIVE_REFINE    — Edit→Run→Evaluate improvement loop

Phase F: Analysis & Decision
  14: RESULT_ANALYSIS     — Statistical analysis, generate experiment_summary.json + results_table.tex
  15: RESEARCH_DECISION   — PROCEED/PIVOT/ITERATE decision

Phase G: Paper Writing
  16: PAPER_OUTLINE       — Generate paper outline
  17: PAPER_DRAFT         — Write full draft (data-driven: uses real experiment metrics)
  18: PEER_REVIEW         — Simulated peer review
  19: PAPER_REVISION      — Revise based on review feedback

Phase H: Finalization
  20: QUALITY_GATE        — Automated quality scoring [GATE]
  21: KNOWLEDGE_ARCHIVE   — Archive findings and lessons learned
  22: EXPORT_PUBLISH      — Generate charts, export final artifacts
  23: CITATION_VERIFY     — Cross-check all citations against source data
```

## Gate Stages

Stages 5, 9, and 20 are **gate stages** requiring approval (or `--auto-approve`):
- Stage 5 (LITERATURE_SCREEN): reject → rollback to Stage 4
- Stage 9 (EXPERIMENT_DESIGN): reject → rollback to Stage 8
- Stage 20 (QUALITY_GATE): reject → rollback to Stage 16

## Configuration

Config file: `config.yaml` (or `config.researchclaw.example.yaml` as template).

Key sections:
- `project.name` / `project.mode` — Project identity
- `research.topic` — The research question
- `llm.base_url` / `llm.api_key` / `llm.primary_model` — LLM provider
- `experiment.mode` — `simulated`, `sandbox`, `docker`, `ssh_remote`, or `colab_drive`
- `experiment.sandbox.python_path` — Python interpreter for sandbox mode
- `security.hitl_required_stages` — Gate stage numbers (default: [5, 9, 20])
- `knowledge_base.root` — Directory for knowledge base files

## Important Constraints

- **Python 3.11+** required
- **Dependencies**: `pyyaml`, `rich`, `matplotlib` (for visualization)
- **LLM**: Any OpenAI-compatible API (tested with GPT-4o, GPT-5.x)
- Sandbox mode executes generated code locally — ensure `experiment.sandbox.python_path` points to a safe environment
- Code validation (AST + security scan) runs automatically before execution in Stage 10

## Testing

```bash
# Run all unit tests (2400+ tests)
python -m pytest tests/test_rc_*.py -q --tb=short

# Run real LLM E2E test (requires API key in config)
python tests/e2e_real_llm.py

# Validate config
researchclaw validate --config config.yaml
```

## Key APIs

```python
# Main pipeline entry
from researchclaw.pipeline.runner import execute_pipeline, execute_iterative_pipeline

# Single stage execution
from researchclaw.pipeline.executor import execute_stage

# Code validation
from researchclaw.experiment.validator import validate_code, security_scan, check_imports

# Chart generation
from researchclaw.experiment.visualize import generate_all_charts

# Config loading
from researchclaw.config import RCConfig, load_config
```
