"""Prompt externalization for the ResearchClaw pipeline.

All 23 stage prompts are defined here as defaults and can be overridden
via a user-provided YAML file.  Users customize prompts without touching
Python source code.

Architecture
------------
* ``_DEFAULT_STAGES`` — every LLM-facing prompt, keyed by stage name.
* ``_DEFAULT_BLOCKS`` — reusable prompt fragments (topic constraint, etc.).
* ``_DEFAULT_SUB_PROMPTS`` — secondary prompts (code repair, etc.).
* ``PromptManager`` — loads defaults → merges user overrides → renders templates.
* ``_render()`` — safe ``{variable}`` substitution that leaves unmatched
  patterns (JSON schemas, curly-brace literals) untouched.

Usage
-----
::

    from researchclaw.prompts import PromptManager

    pm = PromptManager()                           # defaults only
    pm = PromptManager("my_prompts.yaml")          # with user overrides

    sp = pm.for_stage("topic_init", topic="RL for drug discovery", domains="ml, bio")
    resp = llm.chat(
        [{"role": "user", "content": sp.user}],
        system=sp.system,
        json_mode=sp.json_mode,
        max_tokens=sp.max_tokens,
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _render(template: str, variables: dict[str, str]) -> str:
    """Replace ``{var_name}`` placeholders with *variables* values.

    Only bare ``{word_chars}`` tokens are substituted — JSON schema
    examples like ``{candidates:[...]}`` or ``{score_1_to_10:number}``
    are left untouched because the regex requires the closing ``}``
    immediately after the identifier.
    """

    def _replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(variables[key]) if key in variables else match.group(0)

    return re.sub(r"\{(\w+)\}", _replacer, template)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPrompt:
    """Fully rendered prompt ready for ``llm.chat()``."""

    system: str
    user: str
    json_mode: bool = False
    max_tokens: int | None = None


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------


class PromptManager:
    """Central registry for pipeline prompts with optional YAML overrides."""

    def __init__(self, overrides_path: str | Path | None = None) -> None:
        # Deep-copy defaults so mutations don't leak across instances
        self._stages: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in _DEFAULT_STAGES.items()
        }
        self._blocks: dict[str, str] = dict(_DEFAULT_BLOCKS)
        self._sub_prompts: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in _DEFAULT_SUB_PROMPTS.items()
        }
        if overrides_path:
            self._load_overrides(Path(overrides_path))

    # -- loading ----------------------------------------------------------

    def _load_overrides(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Prompts file not found: %s — using defaults", path)
            return
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            logger.warning("Bad prompts YAML %s: %s — using defaults", path, exc)
            return

        for stage_name, stage_data in (data.get("stages") or {}).items():
            if stage_name in self._stages and isinstance(stage_data, dict):
                self._stages[stage_name].update(stage_data)
            else:
                logger.warning("Unknown stage in prompts file: %s", stage_name)

        for block_name, block_text in (data.get("blocks") or {}).items():
            if isinstance(block_text, str):
                self._blocks[block_name] = block_text

        for sub_name, sub_data in (data.get("sub_prompts") or {}).items():
            if sub_name in self._sub_prompts and isinstance(sub_data, dict):
                self._sub_prompts[sub_name].update(sub_data)

        logger.info("Loaded prompt overrides from %s", path)

    # -- primary API ------------------------------------------------------

    def for_stage(
        self,
        stage: str,
        *,
        evolution_overlay: str = "",
        **kwargs: Any,
    ) -> RenderedPrompt:
        """Return a fully rendered prompt for *stage* with variables filled.

        If *evolution_overlay* is provided, it is appended to the user prompt
        so the LLM can learn from prior run lessons.
        """
        entry = self._stages[stage]
        kw = {k: str(v) for k, v in kwargs.items()}
        user_text = _render(entry["user"], kw)
        if evolution_overlay:
            user_text = f"{user_text}\n\n{evolution_overlay}"
        return RenderedPrompt(
            system=_render(entry["system"], kw),
            user=user_text,
            json_mode=entry.get("json_mode", False),
            max_tokens=entry.get("max_tokens"),
        )

    def system(self, stage: str) -> str:
        """Return the raw system prompt template for *stage*."""
        return self._stages[stage]["system"]

    def user(self, stage: str, **kwargs: Any) -> str:
        """Return the rendered user prompt for *stage*."""
        return _render(
            self._stages[stage]["user"],
            {k: str(v) for k, v in kwargs.items()},
        )

    def json_mode(self, stage: str) -> bool:
        return self._stages[stage].get("json_mode", False)

    def max_tokens(self, stage: str) -> int | None:
        return self._stages[stage].get("max_tokens")

    # -- blocks -----------------------------------------------------------

    def block(self, name: str, **kwargs: Any) -> str:
        """Render a reusable prompt block."""
        return _render(
            self._blocks[name],
            {k: str(v) for k, v in kwargs.items()},
        )

    # -- sub-prompts (code repair, etc.) ----------------------------------

    def sub_prompt(self, name: str, **kwargs: Any) -> RenderedPrompt:
        """Return a rendered sub-prompt (e.g. code_repair)."""
        entry = self._sub_prompts[name]
        kw = {k: str(v) for k, v in kwargs.items()}
        return RenderedPrompt(
            system=_render(entry["system"], kw),
            user=_render(entry["user"], kw),
        )

    # -- introspection ----------------------------------------------------

    def stage_names(self) -> list[str]:
        return list(self._stages.keys())

    def has_stage(self, stage: str) -> bool:
        return stage in self._stages

    def export_yaml(self, path: Path) -> None:
        """Write current prompts (defaults + overrides) to a YAML file."""
        data: dict[str, Any] = {
            "version": "1.0",
            "blocks": dict(self._blocks),
            "stages": {k: dict(v) for k, v in self._stages.items()},
            "sub_prompts": {k: dict(v) for k, v in self._sub_prompts.items()},
        }
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )


# ========================================================================
# DEFAULT PROMPTS — edit prompts.yaml to override; do NOT edit these.
# ========================================================================

# -- Canonical section word-count targets ----------------------------------
# Single source of truth for per-section word-count ranges.
# Used by executor._validate_draft_quality() and converter.check_paper_completeness().
SECTION_WORD_TARGETS: dict[str, tuple[int, int]] = {
    "abstract": (180, 220),
    "introduction": (800, 1000),
    "related work": (600, 800),
    "method": (1000, 1500),
    "experiments": (800, 1200),
    "results": (600, 800),
    "discussion": (400, 600),
    "limitations": (200, 300),
    "conclusion": (200, 300),
    "broader impact": (200, 400),
}

# Aliases mapping heading variants to canonical names in SECTION_WORD_TARGETS.
_SECTION_TARGET_ALIASES: dict[str, str] = {
    "methods": "method",
    "methodology": "method",
    "proposed method": "method",
    "approach": "method",
    "experimental setup": "experiments",
    "experimental results": "results",
    "results and discussion": "results",
    "results and analysis": "results",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "summary": "conclusion",
    "background": "related work",
    "literature review": "related work",
    "prior work": "related work",
    "limitation": "limitations",
    "limitations and future work": "limitations",
    "broader impacts": "broader impact",
    "societal impact": "broader impact",
    "ethical considerations": "broader impact",
}

# -- Reusable blocks -----------------------------------------------------

_DEFAULT_BLOCKS: dict[str, str] = {
    "title_guidelines": (
        "\n## TITLE RULES (Hard Constraints)\n"
        "1. MAXIMUM 14 words. Ideal: 8-12 words. NEVER exceed 14.\n"
        "2. Preferred structure: 'MethodName: Descriptive Phrase' (colon format)\n"
        "   - Create a catchy 1-3 word method name (acronym, portmanteau, or evocative word)\n"
        "   - Subtitle explains what it does: 'for X' / 'via Y' / 'in Z'\n"
        "   - Examples: 'AlphaEdit: Null-Space Knowledge Editing for LMs' (8 words)\n"
        "   - Examples: 'VAR: Visual Autoregressive Modeling via Next-Scale Prediction' (8 words)\n"
        "3. Alternative: Bold declarative claim that surprises the reader\n"
        "   - 'Not All Tokens Are What You Need for Pretraining' (9 words)\n"
        "   - 'Vision Transformers Need Registers' (4 words)\n"
        "4. FORBIDDEN patterns:\n"
        "   - 'Investigating...', 'An Empirical Study of...', 'Towards...'\n"
        "   - 'A Novel Approach to...', 'On the...' (generic academic filler)\n"
        "   - Repeating the full method description as title\n"
        "   - Weakness qualifiers: 'in Two Runs', 'Under Limited Data'\n"
        "5. MUST define a short method name (2-5 chars) that serves as memorable handle.\n"
        "   The reader should be able to say 'Have you read the X paper?'\n"
        "6. No abbreviations unless universally known (LLM, RL, GAN, NLP are OK).\n"
    ),
    "abstract_structure": (
        "\n## ABSTRACT (Hard Rules — 180-220 words, 5-7 sentences)\n"
        "STRUCTURE (PMR+ format):\n"
        "S1-S2: PROBLEM — What gap exists? Why does it matter? (NO method names yet)\n"
        "S3-S4: METHOD — Name your system. One-sentence description of key insight.\n"
        "S5-S6: RESULTS — At most 3 specific numbers. Use relative improvements\n"
        "  ('X% over baseline') not raw values ('0.7667'). Bold the single most\n"
        "  important result.\n"
        "S7 (optional): IMPACT — What does this enable?\n\n"
        "HARD CONSTRAINTS:\n"
        "- NO \\texttt{{}} in abstract\n"
        "- NO more than 3 numeric values in the entire abstract\n"
        "- NO per-seed breakdowns or confidence intervals\n"
        "- NO method names longer than 3 words (use the short system name)\n"
        "- The abstract must be readable by a researcher who skimmed only the title\n"
        "- First sentence must NOT start with 'We' or 'This paper'\n"
    ),
    "compute_budget": (
        "\n## Compute Budget Constraint\n"
        "- Total execution time limit: {time_budget_sec} seconds\n"
        "- You MUST design experiments that complete within this budget\n"
        "- Estimate: a simple numpy loop runs ~10M iterations/sec; a nested loop over\n"
        "  conditions runs proportionally slower\n"
        "- SCALING RULES (mandatory):\n"
        "  - If total conditions > 100: reduce seeds to 3-5 (not 20)\n"
        "  - If total conditions > 500: reduce to 2-3 representative conditions per factor\n"
        "  - If time_budget < 300s: limit total optimization steps to ≤5,000 per run\n"
        "  - If time_budget < 120s: limit total optimization steps to ≤1,000 per run\n"
        "  - Always print intermediate results so partial data is captured on timeout\n"
        "- MANDATORY: print a 'TIME_ESTIMATE: Xs' line before the main loop,\n"
        "  estimating total runtime based on a small pilot (run 1 condition, extrapolate)\n"
        "- MANDATORY: implement a time guard — check elapsed time periodically and\n"
        "  stop gracefully if approaching 80% of budget, saving all results collected so far\n"
        "- MANDATORY: add NaN/divergence fast-fail guard:\n"
        "  - After each optimization step, check if loss is NaN or > 100\n"
        "  - If detected, print 'FAIL: NaN/divergence detected', save partial results, and exit\n"
        "  - Do NOT waste compute on a diverging run\n"
        "- MINIMUM TRAINING EPOCHS (CRITICAL for meaningful results):\n"
        "  - CIFAR-10/100 with ResNet/CNN: minimum 50 epochs (200 recommended)\n"
        "  - FashionMNIST with small CNN: minimum 20 epochs\n"
        "  - RL environments: follow the RL STEP BUDGET below (CRITICAL)\n"
        "  - If time_budget is too short for minimum epochs, REDUCE model complexity\n"
        "    or dataset size INSTEAD of reducing epochs. 8 epochs on CIFAR-10 will\n"
        "    produce random-chance accuracy (~10%), making all comparisons meaningless.\n"
        "  - Use a SMALL model (simple CNN, few layers) to fit enough epochs into the budget.\n"
        "  - A converged small model is worth infinitely more than a diverged large model.\n"
        "- MANDATORY: use the experiment_harness module (pre-installed in sandbox):\n"
        "  ```\n"
        "  from experiment_harness import ExperimentHarness\n"
        "  harness = ExperimentHarness(time_budget={time_budget_sec})\n"
        "  # In your experiment loop:\n"
        "  if harness.should_stop():\n"
        "      break  # graceful stop at 80% of budget\n"
        "  if not harness.check_value(value, 'metric_name'):\n"
        "      print('SKIP: NaN/Inf detected')  # skip invalid values\n"
        "      continue\n"
        "  harness.report_metric('metric_name', value)  # validated output\n"
        "  # At the end of ALL experiments:\n"
        "  harness.finalize()  # writes results.json — MUST be called\n"
        "  ```\n"
        "  The harness provides: time budget enforcement, NaN/Inf detection,\n"
        "  validated metric reporting, and results.json output. NOT using it\n"
        "  means your metrics may be lost or malformed.\n"
    ),
    "topic_constraint": (
        "\n\n=== HARD TOPIC CONSTRAINT ===\n"
        "The paper MUST be about: {topic}\n"
        "PROHIBITED content (unless user explicitly specifies case-study mode):\n"
        "- Do NOT treat environment setup, dependency installation, or infrastructure "
        "failures as a research contribution.\n"
        "- Do NOT present debugging logs, system errors, or configuration issues "
        "as experimental findings.\n"
        "- Do NOT drift to tangential topics not directly related to the stated topic.\n"
        "- Every section MUST connect back to the core research question.\n"
        "- The Abstract and Introduction MUST clearly state the research problem "
        "derived from: {topic}\n"
        "- The Method section MUST describe a technical approach, not a workflow.\n"
        "- The Results section MUST report quantitative outcomes of experiments, "
        "not environment status.\n"
        "=== END CONSTRAINT ===\n"
    ),
    "pkg_hint_sandbox": (
        "\nAVAILABLE PACKAGES (sandbox mode): Python stdlib, numpy, math, random, "
        "statistics, json.\n"
        "Do NOT use: torch, tensorflow, jax, sklearn, pandas, scipy, matplotlib, "
        "or any deep learning framework.\n"
        "Write the experiment using ONLY numpy and stdlib.\n"
    ),
    "dataset_guidance": (
        "\n## Standard Datasets & Real Baselines (MANDATORY when applicable)\n"
        "You MUST use real benchmark datasets — NEVER synthetic torch.randn() data.\n\n"
        "### Tier 1: Pre-cached (ALWAYS available, use download=False)\n"
        "These datasets are already in the Docker image. Use download=False:\n"
        "- `torchvision.datasets.CIFAR10(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.CIFAR100(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.MNIST(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.FashionMNIST(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.STL10(root='/opt/datasets', split='train'/'test', download=False)`\n"
        "- `torchvision.datasets.SVHN(root='/opt/datasets', split='train'/'test', download=False)`\n\n"
        "### Tier 2: Downloadable (use setup.py to download before main.py runs)\n"
        "For any dataset NOT in Tier 1, create a `setup.py` file that downloads it.\n"
        "setup.py runs WITH network access; main.py runs WITHOUT network.\n"
        "- Any torchvision dataset (Caltech-101, Flowers102, etc.)\n"
        "- HuggingFace datasets: `from datasets import load_dataset`\n"
        "  Examples: IMDB, AG News, WikiText, SST-2, SQuAD, MMLU\n"
        "- OGB benchmarks: ogbg-molhiv, ogbn-arxiv, etc.\n"
        "- Tiny-ImageNet (237MB, 200 classes) — good ImageNet proxy\n\n"
        "### Tier 3: Too large for download (use alternatives)\n"
        "These datasets are TOO LARGE to download within experiment time limits:\n"
        "- ImageNet-1K (168GB) → use Tiny-ImageNet or CIFAR-100 as proxy\n"
        "- LAION (>1TB) → use smaller HuggingFace image-text datasets\n"
        "- Common Crawl, The Pile → use WikiText-103 or pre-tokenized subsets\n"
        "NEVER generate 'ImageNet-like' synthetic data — always use a real alternative.\n\n"
        "### ANTI-PATTERNS (NEVER DO THESE):\n"
        "- `torch.randn(N, 3, 224, 224)` as dataset → use real datasets\n"
        "- `download=True` in main.py → put downloads in setup.py\n"
        "- `download=False` for non-cached datasets → will FileNotFoundError\n"
        "- Random train/test splits → use official splits from dataset\n"
        "- `os.makedirs('/opt/datasets/...')` → /opt/datasets is READ-ONLY\n\n"
        "DATA PATH: For Tier 1 pre-cached datasets, use `/opt/datasets` as root.\n"
        "For Tier 2 datasets downloaded by setup.py, use `/workspace/data` as root.\n"
        "WARNING: `/opt/datasets` is READ-ONLY. NEVER call os.makedirs() on it.\n"
        "Just pass `root='/opt/datasets'` directly to torchvision dataset constructors.\n\n"
        "DISTRIBUTION SHIFT — use torchvision corruption transforms:\n"
        "- Gaussian noise: `transforms.Lambda(lambda x: x + torch.randn_like(x) * sigma)`\n"
        "- Brightness shift: `transforms.ColorJitter(brightness=0.5)`\n"
        "- Contrast shift: `transforms.ColorJitter(contrast=0.5)`\n"
        "- Blur: `transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))`\n"
        "- For CIFAR-10-C style corruptions, apply transforms to test set only.\n\n"
        "REAL BASELINES & MODERN BENCHMARKS (CRITICAL):\n"
        "- Use proper train/test splits from the dataset (never split randomly in code)\n"
        "- Use standard architectures (ResNet-18/50, ViT, ConvNeXt) — not toy 2-layer MLPs\n"
        "- CIFAR INPUT SIZE (IMPORTANT): CIFAR images are 32×32. Two valid approaches:\n"
        "  1. PRETRAINED models (ImageNet weights): Use `transforms.Resize(224)` — "
        "pretrained models require 224×224 inputs.\n"
        "  2. TRAINING FROM SCRATCH (most experiments): Modify the model for 32×32 "
        "inputs instead of resizing. For ResNet: use `nn.Conv2d(3,64,3,1,1)` as "
        "first conv (not 7×7/stride-2) and REMOVE the initial MaxPool. This is 49× "
        "more memory-efficient and trains faster than Resize(224). Use the `timm` "
        "library's CIFAR variants or build a custom `get_resnet18_cifar()` helper.\n"
        "- Report standard metrics (top-1 accuracy for classification tasks)\n"
        "- Compare against published baselines where available\n"
        "- BASELINES MUST BE CURRENT: Use baselines from recent top-venue papers "
        "(2023-2026). Do NOT use outdated methods as the primary comparison.\n"
        "  * AlexNet, VGG-16 → use ResNet-50, ViT, ConvNeXt instead\n"
        "  * Vanilla SGD → use AdamW, SGD+momentum+cosine LR\n"
        "  * Simple RNN/LSTM for NLP → use Transformer-based models\n"
        "- Include at LEAST one strong, modern baseline (near-SOTA).\n"
        "- BENCHMARKS MUST BE STANDARD and actively used in the community.\n\n"
        "WHEN TO USE SYNTHETIC DATA (required for these domains):\n"
        "- **PDE / Scientific computing**: Generate synthetic PDE data (Burgers "
        "equation, Darcy flow, heat equation, Navier-Stokes). Use numerical solvers "
        "(scipy.integrate, finite differences) to create ground truth.\n"
        "- **Combinatorial optimization** (TSP, graph coloring, scheduling): Generate "
        "random problem instances (random TSP cities, Erdos-Renyi graphs).\n"
        "- **Theoretical analysis**: Synthetic optimization landscapes, toy problems.\n"
        "- **Domain with no standard dataset**: Novel combinatorial or mathematical domains.\n"
        "For these domains, do NOT use CIFAR/MNIST/ImageNet — they are irrelevant. "
        "Generate problem-specific synthetic data in main.py.\n\n"
        "DOMAIN-DATASET MATCHING (CRITICAL):\n"
        "- Image classification → CIFAR-10/100, MNIST, ImageNet variants\n"
        "- NLP → IMDB, AG News, SST-2, WikiText\n"
        "- Graph learning → Cora, CiteSeer, ogbn-arxiv\n"
        "- PDE/Physics → SYNTHETIC (Burgers, Darcy, Navier-Stokes)\n"
        "- Combinatorial optimization → SYNTHETIC (random TSP, graph instances)\n"
        "- RL → Gymnasium environments (CartPole, LunarLander, HalfCheetah)\n"
        "NEVER use image datasets for non-image problems.\n"
    ),
    "setup_script_guidance": (
        "\n## Setup Script (setup.py) — Dataset Download & Preparation\n"
        "If your experiment needs datasets NOT in the pre-cached list, generate "
        "a SEPARATE file called `setup.py` that downloads and prepares them.\n"
        "The setup.py runs WITH NETWORK ACCESS before main.py (which runs WITHOUT network).\n\n"
        "IMPORTANT: All download logic MUST be in setup.py, NOT in main.py.\n"
        "main.py should only load pre-cached data from /opt/datasets (download=False) "
        "or downloaded data from /workspace/data.\n\n"
        "Example setup.py:\n"
        "```python\n"
        "import os\n"
        "DATA_DIR = '/workspace/data'\n"
        "os.makedirs(DATA_DIR, exist_ok=True)\n\n"
        "# Download torchvision datasets\n"
        "import torchvision\n"
        "torchvision.datasets.Caltech101(root=DATA_DIR, download=True)\n\n"
        "# Download HuggingFace datasets\n"
        "from datasets import load_dataset\n"
        "ds = load_dataset('imdb', cache_dir=os.path.join(DATA_DIR, 'hf'))\n\n"
        "# Download OGB benchmarks\n"
        "# from ogb.graphproppred import PygGraphPropPredDataset\n"
        "# dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=DATA_DIR)\n\n"
        "print('[setup] Dataset download complete.')\n"
        "```\n\n"
        "IMPORT ANTI-PATTERN (NEVER DO THIS):\n"
        "```python\n"
        "from datasets import load_dataset\n"
        "datasets.load_dataset('imdb', ...)  # WRONG — NameError!\n"
        "```\n"
        "If you write `from datasets import load_dataset`, call `load_dataset(...)` directly.\n"
        "If you write `import datasets`, call `datasets.load_dataset(...)` with module prefix.\n"
        "NEVER mix the two styles.\n\n"
        "If ALL your datasets are pre-cached (CIFAR-10/100, MNIST, FashionMNIST, "
        "STL-10, SVHN), you do NOT need setup.py — just use download=False in main.py.\n\n"
        "You may also include a `requirements.txt` file listing any additional "
        "pip packages your experiment needs beyond the pre-installed set.\n"
    ),
    "network_disabled_guidance": (
        "\n## ⚠️ NO NETWORK ACCESS — CRITICAL CONSTRAINT ⚠️\n"
        "This experiment runs with network_policy='none'. There is NO network access\n"
        "at ANY phase (no pip install, no dataset downloads, no HTTP requests).\n\n"
        "### ONLY these pre-cached datasets are available:\n"
        "- `torchvision.datasets.CIFAR10(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.CIFAR100(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.MNIST(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.FashionMNIST(root='/opt/datasets', train=True/False, download=False)`\n"
        "- `torchvision.datasets.STL10(root='/opt/datasets', split='train'/'test', download=False)`\n"
        "- `torchvision.datasets.SVHN(root='/opt/datasets', split='train'/'test', download=False)`\n\n"
        "### FORBIDDEN (will cause runtime failure):\n"
        "- Do NOT create setup.py (it cannot run without network)\n"
        "- Do NOT create requirements.txt (pip install is unavailable)\n"
        "- Do NOT use `download=True` on any dataset\n"
        "- Do NOT use `urllib`, `requests`, `httpx`, or any HTTP library\n"
        "- Do NOT use `datasets.load_dataset()` from HuggingFace (requires download)\n"
        "- Do NOT import packages not pre-installed in the Docker image\n\n"
        "### Available pre-installed packages:\n"
        "torch, torchvision, torchaudio, numpy, scipy, sklearn, matplotlib, seaborn,\n"
        "pandas, tqdm, gymnasium, networkx, PyYAML, Pillow, timm, einops, torchmetrics,\n"
        "h5py, transformers, datasets, accelerate, peft, bitsandbytes.\n\n"
        "If your research topic requires a dataset NOT in the pre-cached list,\n"
        "you MUST adapt to use one of the 6 pre-cached datasets instead.\n"
    ),
    "network_full_guidance": (
        "\n## Network Access: Full\n"
        "This experiment runs with network_policy='full'. Network access is available\n"
        "throughout ALL execution phases (setup, pip install, and main experiment).\n"
        "You may download datasets, install packages, and make HTTP requests at any time.\n"
    ),
    "hp_reporting": (
        "\n## Hyperparameter Reporting (MANDATORY)\n"
        "At the TOP of main.py, define a HYPERPARAMETERS dictionary containing ALL "
        "tunable hyperparameters used in your experiment:\n"
        "```python\n"
        "HYPERPARAMETERS = {\n"
        "    'learning_rate': 0.001,\n"
        "    'batch_size': 64,\n"
        "    'num_epochs': 50,\n"
        "    'hidden_dim': 256,\n"
        "    # ... all other hyperparameters\n"
        "}\n"
        "```\n"
        "At the end of main.py, save hyperparameters to results.json:\n"
        "```python\n"
        "import json\n"
        "results = {'hyperparameters': HYPERPARAMETERS, 'metrics': collected_metrics}\n"
        "with open('results.json', 'w') as f:\n"
        "    json.dump(results, f, indent=2)\n"
        "```\n"
        "EVERY hyperparameter must be used in the code — no dead parameters.\n"
        "The paper MUST include a hyperparameter table — this data feeds into it.\n"
    ),
    "rl_step_guidance": (
        "\n## RL Training Step Budget (MANDATORY for RL experiments)\n"
        "Reinforcement learning requires MANY more training steps than supervised learning.\n"
        "Under-trained RL agents produce random-chance performance, making ALL comparisons\n"
        "meaningless and the paper unpublishable.\n\n"
        "### Environment Availability:\n"
        "#### Always available (classic control — no extra dependencies):\n"
        "- CartPole-v1, Pendulum-v1, MountainCar-v0, MountainCarContinuous-v0,\n"
        "  Acrobot-v1, LunarLander-v3\n"
        "- These are lightweight and fast — PREFER these unless MuJoCo is specifically required.\n\n"
        "#### MuJoCo environments (pre-installed in Docker image):\n"
        "- HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5,\n"
        "  Swimmer-v5, Reacher-v5, InvertedPendulum-v5, InvertedDoublePendulum-v5\n"
        "- Require MuJoCo runtime — available in Docker but NOT in basic sandbox mode.\n\n"
        "#### RULE: If the research topic says 'MuJoCo-free', 'without MuJoCo',\n"
        "  or 'classic control only' → you MUST use classic control environments ONLY.\n"
        "  Do NOT import or reference MuJoCo in any way.\n\n"
        "#### DEFAULT RECOMMENDATION: Prefer classic control environments unless the\n"
        "  research topic specifically requires MuJoCo locomotion tasks.\n\n"
        "### ALGORITHM-ENVIRONMENT COMPATIBILITY (HARD RULE — violation = crash):\n"
        "- DQN is ONLY for DISCRETE action spaces (CartPole, LunarLander, Acrobot, Atari).\n"
        "  DQN will CRASH on Pendulum, HalfCheetah, Hopper, Walker2d, etc.\n"
        "- For CONTINUOUS action spaces: use SAC, TD3, or PPO.\n"
        "- PPO works for both discrete and continuous.\n"
        "- NEVER combine DQN + any continuous environment.\n\n"
        "### TIME BUDGET RULES FOR RL:\n"
        "- If time_budget ≤ 3600s → ONLY classic control "
        "(CartPole, Pendulum, MountainCar, Acrobot, LunarLander)\n"
        "- If time_budget ≤ 1800s → ONLY CartPole or Pendulum (simplest)\n"
        "- MuJoCo requires >5000s for meaningful results.\n\n"
        "### Minimum Steps by Algorithm Family:\n"
        "| Algorithm | Environment | Min Steps | Recommended |\n"
        "|-----------|-------------|-----------|-------------|\n"
        "| PPO       | MuJoCo (Ant, HalfCheetah, Humanoid) | 500K | 1M-3M |\n"
        "| PPO       | Simple control (CartPole, Pendulum) | 100K | 500K |\n"
        "| SAC/TD3   | MuJoCo locomotion | 300K | 1M |\n"
        "| SAC/TD3   | Simple control | 50K  | 200K |\n"
        "| DQN/Rainbow | Atari | 1M | 10M |\n"
        "| A2C/A3C   | Any continuous | 500K | 2M |\n"
        "| REINFORCE | Any | 200K | 1M |\n\n"
        "### Step Budget Allocation Strategy:\n"
        "1. Compute pilot_time = time for 1000 steps of 1 condition.\n"
        "2. steps_per_sec = 1000 / pilot_time.\n"
        "3. max_steps_per_condition = (time_budget * 0.7) / num_conditions * steps_per_sec.\n"
        "4. If max_steps < min_steps for the algorithm, REDUCE num_seeds to 3 (not steps).\n"
        "5. If STILL under min_steps, use a simpler environment (e.g., Pendulum instead of Ant).\n"
        "6. NEVER reduce steps below the minimum — it wastes compute on meaningless results.\n\n"
        "### Evaluation Protocol for RL:\n"
        "- Evaluate every N_eval steps (e.g., every 10K steps) using deterministic policy.\n"
        "- Run 10 evaluation episodes per checkpoint.\n"
        "- Report: mean return, std return, success rate (if applicable).\n"
        "- Plot learning curves (return vs steps) — this is EXPECTED by reviewers.\n"
        "- Final metric = mean over last 10 evaluation checkpoints (NOT last episode).\n\n"
        "### Gymnasium Environment Version (CRITICAL):\n"
        "- Use v5 environments (NOT v4): `gym.make('HalfCheetah-v5')`, `gym.make('Hopper-v5')`\n"
        "- v4 environments are deprecated and will produce warnings.\n"
        "- Available MuJoCo v5 envs: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5,\n"
        "  Humanoid-v5, Swimmer-v5, Reacher-v5, InvertedPendulum-v5, InvertedDoublePendulum-v5\n"
        "- For simple/fast experiments: use Pendulum-v1, CartPole-v1, MountainCarContinuous-v0\n\n"
        "### Gymnasium API (CRITICAL — common crash source):\n"
        "- `env.reset()` returns `(obs, info)` — ALWAYS unpack both:\n"
        "  `obs, info = env.reset(seed=seed)`\n"
        "- `env.step(action)` returns `(obs, reward, terminated, truncated, info)` — 5 values:\n"
        "  `obs, reward, terminated, truncated, info = env.step(action)`\n"
        "  `done = terminated or truncated`\n"
        "- DO NOT use old `done = env.step(action)[2]` — this is the Gym (v0.26-) API.\n"
        "- `reward` is a scalar float, NOT an array. Do NOT index it: use `reward` directly.\n"
        "- `obs` shape depends on env: discrete envs give 1D array, image envs give 3D.\n"
        "  Always check `env.observation_space.shape` and handle accordingly.\n\n"
        "### Learning Curve Logging (MANDATORY for RL papers):\n"
        "- Print evaluation metrics at regular intervals: every N_eval steps\n"
        "  `EVAL: step=<S> condition=<C> seed=<seed> return=<R>`\n"
        "- This enables plotting learning curves (return vs training steps)\n"
        "- Learning curves are EXPECTED by RL reviewers — a paper without them\n"
        "  will be rejected regardless of final performance.\n"
        "- At the end, print the full curve:\n"
        "  `LEARNING_CURVE: condition=<C> seed=<seed> steps=[...] returns=[...]`\n"
    ),
    "multi_seed_enforcement": (
        "\n## Multi-Seed Experiment Requirement (MANDATORY — NO EXCEPTIONS)\n"
        "Running each condition with only 1 seed is NEVER acceptable. Results from\n"
        "a single seed cannot distinguish signal from noise and reviewers will reject.\n\n"
        "### HARD REQUIREMENT:\n"
        "- You MUST use exactly seeds = [0, 1, 2] (3 seeds minimum).\n"
        "- Each condition MUST loop over ALL seeds.\n"
        "- Print per-seed: `condition=X seed=S {metric_key}: V`\n"
        "- Print aggregated: `condition=X {metric_key}_mean: M {metric_key}_std: S`\n"
        "- Tables MUST show mean ± std, NEVER single-run values.\n\n"
        "### Implementation Pattern (copy this structure):\n"
        "```python\n"
        "SEEDS = [0, 1, 2]  # EXACTLY 3 seeds — mandatory minimum\n"
        "all_results = {}  # {condition_name: {seed: metric_value}}\n\n"
        "for condition_name, ConditionClass in conditions.items():\n"
        "    all_results[condition_name] = {}\n"
        "    for seed in SEEDS:\n"
        "        set_all_seeds(seed)  # torch, numpy, random\n"
        "        result = run_single(ConditionClass, seed=seed)\n"
        "        all_results[condition_name][seed] = result\n"
        "        print(f'condition={condition_name} seed={seed} metric: {result}')\n"
        "    values = list(all_results[condition_name].values())\n"
        "    print(f'condition={condition_name} metric_mean: {np.mean(values):.4f} '\n"
        "          f'metric_std: {np.std(values):.4f}')\n"
        "```\n\n"
        "### Reporting Requirements:\n"
        "- Print per-seed results: `condition=X seed=S metric: V`\n"
        "- Print aggregated: `condition=X metric_mean: M metric_std: S`\n"
        "- Tables in the paper MUST show mean ± std, NEVER single-run values.\n"
        "- If time budget forces < 5 seeds, use EXACTLY 3 seeds (minimum).\n"
        "  Print: `SEED_WARNING: only 3 seeds used due to time budget`.\n"
    ),
    "writing_structure": (
        "\n## Paper Section Writing Rules\n"
        "MARKDOWN FORMATTING (CRITICAL):\n"
        "- Use `# Title` (H1) for the paper title\n"
        "- Use `# Abstract`, `# Introduction`, `# Method`, etc. (H1) for MAIN sections\n"
        "- Use `## Subsection Name` (H2) for subsections WITHIN a main section\n"
        "- NEVER use `##` for main sections — that produces wrong LaTeX heading levels\n"
        "- Each main section (H1) MUST contain subsections (H2) when it exceeds 3 paragraphs\n"
        "- NEVER place sub-topics (e.g., 'Knowledge Distillation for Compact Models') "
        "at the same heading level as main sections (e.g., 'Related Work')\n"
        "- NEVER wrap the paper in ```markdown fences\n"
        "- NEVER use raw variable names (e.g., `method_name/metric_key = 0.85`) — "
        "always use human-readable text\n\n"
        "ABSTRACT (150-200 words, 5-sentence structure):\n"
        "- (1) Problem and significance (2) Prior approaches and gaps\n"
        "- (3) Your approach and novelty (4) Key results with 2-3 specific numbers\n"
        "- (5) Implication/takeaway\n"
        "- Do NOT list per-seed ranges (e.g., '0.71-0.73 across seeds') — use mean +/- std\n"
        "- Do NOT repeat numbers that appear in the Results section — pick the 2-3 most impactful\n\n"
        "INTRODUCTION (4 paragraphs, 800-1000 words, cite 8-12 references):\n"
        "Paragraph 1: Problem motivation (why this matters). "
        "Paragraph 2: What exists and why it falls short. "
        "Paragraph 3: Your approach and key insight. "
        "Paragraph 4: Contributions (2-3 bullet points allowed here ONLY).\n\n"
        "RELATED WORK:\n"
        "Organize by sub-topic, not chronologically. "
        "End each paragraph with how YOUR work differs from the cited work. "
        "Cite at least 15 references, all directly relevant.\n\n"
        "METHOD:\n"
        "Write as flowing narrative prose (NOT bullet points). "
        "Include full algorithm description with pseudocode or step-by-step. "
        "State all hyperparameters with values and justification. "
        "Provide architecture details sufficient for reproduction.\n\n"
        "RESULTS:\n"
        "- Do NOT repeat the same number more than twice across the paper\n"
        "- Each number in a table should be discussed AT MOST once in text\n"
        "- Tables: mean +/- std with 95% CI in parentheses\n"
        "- Bold the best result in each column\n"
        "- Every comparison claim must cite a p-value or note multiple seeds\n"
        "- Report the number of random seeds/runs used\n\n"
        "FIGURES AND TABLES:\n"
        "- Every figure MUST be referenced in the text (e.g., 'As shown in Figure 1')\n"
        "- Every table MUST be referenced in the text (e.g., 'Table 2 summarizes')\n"
        "- Figure captions: 1-2 descriptive sentences (not just 'Results comparison')\n"
        "- Table captions go ABOVE the table; figure captions go BELOW the figure\n"
        "- Axis labels must include units where applicable\n"
        "- Use consistent font sizes across all figures\n\n"
        "DISCUSSION (if applicable, can be merged into Results):\n"
        "- Paragraph 1: Summarize key findings and their significance\n"
        "- Paragraph 2: Compare with prior work — explain WHY results differ\n"
        "- Paragraph 3: Discuss unexpected or negative results honestly\n"
        "- Paragraph 4: Broader implications and practical applications\n\n"
        "LIMITATIONS (3-5 points):\n"
        "- State each limitation ONCE, here only — not scattered throughout\n"
        "- No disclaimers like 'due to computational constraints'\n"
        "- Include compute resources used (GPU type, training time)\n\n"
        "CONCLUSION:\n"
        "- Summarize findings (match actual results, no aspirational claims)\n"
        "- 2-3 sentences of future work\n\n"
        "PROSE QUALITY (CRITICAL — violation = desk reject):\n"
        "- Write FLOWING ACADEMIC PARAGRAPHS, not bullet-point lists.\n"
        "- Each paragraph must have 4-8 sentences with smooth transitions.\n"
        "- Introduction, Related Work, and Method must each be >=3 paragraphs.\n"
        "- FORBIDDEN: starting 3+ consecutive paragraphs with the same word.\n"
        "- FORBIDDEN: bullet-point lists in Introduction or Related Work sections.\n"
        "- Use varied sentence structures: mix simple, compound, and complex sentences.\n"
        "- Connect paragraphs with transition phrases: 'Building on this insight...', "
        "'In contrast to prior work...', 'To address this limitation...'.\n"
        "- Each Related Work paragraph must COMPARE your approach to cited work, "
        "not merely summarize what each paper does.\n"
        "- FORBIDDEN AI-BOILERPLATE phrases (instant credibility loss):\n"
        "  'delves into', 'it is worth noting', 'plays a crucial role',\n"
        "  'leverages the power of', 'paves the way', 'a myriad of',\n"
        "  'paradigm shift', 'groundbreaking', 'in the realm of',\n"
        "  'holistic approach', 'multifaceted', 'navigate the complexities'.\n"
        "  Replace ALL such phrases with precise, specific academic language.\n"
    ),
    "llm_training_guidance": (
        "\n## LLM Fine-Tuning Guidance (when topic involves language model training)\n"
        "AVAILABLE FRAMEWORKS (pre-installed in Docker):\n"
        "- transformers (AutoModelForCausalLM, AutoTokenizer, Trainer)\n"
        "- peft (LoraConfig, get_peft_model, PeftModel)\n"
        "- trl (SFTTrainer, DPOTrainer, GRPOTrainer)\n"
        "- datasets (load_dataset, Dataset)\n"
        "- accelerate (Accelerator)\n"
        "- bitsandbytes (4-bit/8-bit quantization)\n\n"
        "GPU MEMORY GUIDELINES (RTX 6000 Ada, 49GB VRAM):\n"
        "- Full fine-tune: <=3B parameters\n"
        "- LoRA (16-bit): <=14B parameters\n"
        "- QLoRA (4-bit): <=72B parameters (practical limit ~14B for training)\n"
        "- Optimal: 7B-14B model with QLoRA (rank 16-64)\n\n"
        "RECOMMENDED TRAINING PATTERN:\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
        "from peft import LoraConfig, get_peft_model, TaskType\n"
        "from trl import SFTTrainer, SFTConfig\n"
        "from datasets import load_dataset\n\n"
        "# 4-bit quantization for memory efficiency\n"
        "bnb_config = BitsAndBytesConfig(\n"
        "    load_in_4bit=True,\n"
        "    bnb_4bit_quant_type='nf4',\n"
        "    bnb_4bit_compute_dtype=torch.bfloat16,\n"
        ")\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    model_name, quantization_config=bnb_config, device_map='auto'\n"
        ")\n"
        "lora_config = LoraConfig(\n"
        "    r=16, lora_alpha=32, target_modules='all-linear',\n"
        "    lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,\n"
        ")\n"
        "model = get_peft_model(model, lora_config)\n"
        "```\n\n"
        "KEY HYPERPARAMETERS:\n"
        "- learning_rate: 1e-4 to 2e-4 (LoRA), 5e-5 to 1e-4 (full FT)\n"
        "- lora_r: 8 (minimal) to 64 (high-capacity)\n"
        "- lora_alpha: typically 2x lora_r\n"
        "- batch_size: 1-4 per device (use gradient_accumulation_steps for effective batch)\n"
        "- gradient_accumulation_steps: 4-16 (effective_batch = per_device * accum)\n"
        "- max_seq_length: 512 (short), 1024-2048 (standard), 4096 (long)\n"
        "- warmup_ratio: 0.03-0.1\n"
        "- weight_decay: 0.01-0.1\n\n"
        "DATA FORMAT (use datasets library):\n"
        "- Instruction tuning: {'instruction': '...', 'output': '...'}\n"
        "- Chat format: {'messages': [{'role': 'user', 'content': '...'}, ...]}\n"
        "- DPO: {'prompt': '...', 'chosen': '...', 'rejected': '...'}\n"
        "- Use load_dataset('json', data_files='train.json') for local data\n"
        "- Use load_dataset('HuggingFace/dataset_name') for HF Hub datasets\n\n"
        "EVALUATION:\n"
        "- Use evaluate library for standard metrics\n"
        "- Common: perplexity, ROUGE (summarization), BLEU (translation), accuracy\n"
        "- LLM benchmarks: MMLU, ARC, HellaSwag, TruthfulQA\n"
        "- Generate sample outputs for qualitative comparison\n\n"
        "MODEL DOWNLOAD:\n"
        "- Models will be downloaded from HuggingFace Hub at runtime\n"
        "- Use 'trust_remote_code=True' for custom model architectures\n"
        "- Cache directory: default HF cache (~/.cache/huggingface)\n"
        "- Common models: Qwen/Qwen2.5-7B, meta-llama/Llama-3.1-8B, "
        "microsoft/Phi-4, google/gemma-2-9b\n\n"
        "CRITICAL — NO SIMULATION:\n"
        "- You MUST load and train a REAL model from HuggingFace Hub.\n"
        "- NEVER simulate training with synthetic utility functions or random scores.\n"
        "- NEVER replace model training with np.random/torch.randn mock results.\n"
        "- A real experiment loads a model, tokenizes data, runs optimizer steps, "
        "and measures real loss/perplexity/accuracy on held-out data.\n"
        "- If compute budget is tight, use a SMALLER model (Qwen2.5-0.5B or 1.5B) "
        "with fewer training steps rather than simulating.\n"
    ),
    "llm_eval_guidance": (
        "\n## LLM Evaluation Guidance\n"
        "STANDARD BENCHMARKS:\n"
        "- Reasoning: MMLU, ARC-Challenge, HellaSwag, WinoGrande\n"
        "- Math: GSM8K, MATH, MathVista\n"
        "- Coding: HumanEval, MBPP, LiveCodeBench\n"
        "- Safety: TruthfulQA, BBQ, CrowS-Pairs\n"
        "- Instruction following: MT-Bench, AlpacaEval, IFEval\n"
        "- Multimodal: MMBench, POPE, MathVista, MMMU\n\n"
        "EVALUATION FRAMEWORKS:\n"
        "- lm-eval-harness: Standard eval framework, run via CLI or Python API\n"
        "- vllm: Fast inference engine for throughput-focused evaluation\n"
        "- lighteval: HuggingFace's lightweight eval framework\n\n"
        "EVALUATION PROTOCOL:\n"
        "- Report on at least 3 benchmarks relevant to the task\n"
        "- Compare with published baselines from model cards/leaderboards\n"
        "- Report both zero-shot and few-shot results where applicable\n"
        "- Include perplexity on held-out test set\n"
    ),
    # IMP-20: Academic writing style guide (from NeurIPS/ICLR/ICML 2024-2025 best papers)
    "academic_style_guide": (
        "\n## ACADEMIC WRITING STANDARDS (from NeurIPS/ICLR/ICML 2024-2025 best papers)\n\n"
        "### Title Standards\n"
        "- Target 8-14 words. Median of award-winning papers: ~10 words.\n"
        "- Preferred format: 'SystemName: Descriptive Subtitle' (35% of best papers)\n"
        "  e.g., 'AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models'\n"
        "- Alternative: Declarative statement that surprises\n"
        "  e.g., 'Not All Tokens Are What You Need for Pretraining'\n"
        "- Give your method a memorable, catchy name (VAR, Genie, PRISM, SEDD).\n"
        "- NEVER exceed 18 words. NEVER use 'A Novel Approach to...' or 'Investigating...'\n\n"
        "### Abstract Standards (PMR+ Structure, 180-220 words)\n"
        "S1-S2: PROBLEM — State the gap. Open with a challenge or status-quo critique.\n"
        "S3-S4: METHOD — Name your system by sentence 3. Describe the key insight.\n"
        "S5-S6: RESULTS — At least 2-3 concrete quantitative claims:\n"
        "  - One relative improvement ('36.7% boost over baseline')\n"
        "  - One absolute benchmark score ('FID of 1.01 on ImageNet')\n"
        "AVOID: Per-seed ranges, excessive texttt, defensive hedging.\n\n"
        "### Section Writing Standards\n"
        "INTRODUCTION (800-1000 words, 4 paragraphs):\n"
        "  - Para 1: Motivation; Para 2: Gap (cite 3-5 papers); Para 3: Your approach;\n"
        "    Para 4: Contributions (bullet list of 3-4 specific contributions)\n"
        "  - MUST cite 8-12 references throughout Introduction\n\n"
        "RELATED WORK (600-800 words):\n"
        "  - Organize by sub-topic (2-3 subsections), NOT as a flat list\n"
        "  - End each subsection with how YOUR work differs\n"
        "  - Target >= 15 unique references in this section alone\n\n"
        "METHOD (1000-1500 words):\n"
        "  - Start with problem formulation (notation, objective function)\n"
        "  - Use algorithm environment for pseudocode (not verbatim)\n"
        "  - Write as a flowing narrative, NOT bullet points\n\n"
        "EXPERIMENTS (800-1200 words):\n"
        "  - Experimental setup as subsection (datasets, baselines, metrics, hardware)\n"
        "  - Hyperparameter table (Table 1 always)\n"
        "  - MUST reference figures: 'As shown in Figure 1, our method...'\n"
        "  - MUST cite baseline method papers (not just name them)\n\n"
        "RESULTS (600-800 words):\n"
        "  - Main results table with descriptive caption\n"
        "  - Ablation study table\n"
        "  - Analysis paragraphs connecting numbers to insights\n"
        "  - DO NOT repeat the same numbers from Experiments section\n"
        "  - Reference figures for visual evidence\n\n"
        "DISCUSSION (400-600 words):\n"
        "  - Compare findings with prior work (cite papers here!)\n"
        "  - Explain surprising results; broader implications\n\n"
        "LIMITATIONS (200-300 words): 3-5 specific, concrete limitations. ALL caveats go HERE.\n\n"
        "CONCLUSION: Summarize in 2-3 sentences, future work in 2-3 sentences.\n\n"
        "### Writing Quality Rules\n"
        "- Write as FLOWING PROSE, not bullet points or enumerated lists\n"
        "- Each paragraph: topic sentence, evidence, analysis, transition\n"
        "- Use transitions: 'Building on this insight...', 'In contrast to...'\n"
        "- Academic tone: confident but precise\n"
        "- Vary sentence structure: mix short declarative with longer analytical\n"
        "- AVOID: Starting 3+ consecutive sentences with 'We', 'The', 'Our'\n"
        "- AVOID: 'It is worth noting that', 'It should be mentioned that' (filler)\n"
        "- Citations belong in EVERY section, not just Introduction and Related Work\n"
    ),
    # IMP-25: Narrative writing requirements
    "narrative_writing_rules": (
        "\n## NARRATIVE WRITING REQUIREMENTS\n\n"
        "You are writing a paper for human reviewers at a top AI conference. The paper\n"
        "must read like a cohesive academic story, NOT a technical report or bullet list.\n\n"
        "### Structure of Each Paragraph\n"
        "Every paragraph MUST follow this pattern:\n"
        "1. TOPIC SENTENCE — states the main claim or finding\n"
        "2. EVIDENCE — data, citations, or reasoning that supports the claim\n"
        "3. ANALYSIS — what the evidence means, why it matters\n"
        "4. TRANSITION — connects to the next paragraph's topic\n\n"
        "### FORBIDDEN Writing Patterns\n"
        "- Bullet-point lists in the main body (ONLY allowed in Contributions paragraph\n"
        "  of Introduction and Limitations section)\n"
        "- Numbered lists of findings or results\n"
        "- Starting a paragraph with 'Table X shows...' without context first\n"
        "- Consecutive short sentences without analysis between them\n"
        "- Repeating the same sentence structure 3+ times in a row\n\n"
        "### REQUIRED Writing Patterns\n"
        "- Transition phrases: 'Building on this observation...', 'In contrast to prior work...'\n"
        "- Vary sentence length: alternate between short impactful and longer analytical\n"
        "- Ground every claim in evidence: '[Result] because [mechanism] (cite)'\n"
        "- Discuss implications: 'This X% improvement indicates that [mechanism Y]\n"
        "  is more effective than [mechanism Z] for [context]'\n"
        "- For temporal data: describe trends in prose rather than bullet-point lists\n\n"
        "### Example: BAD vs GOOD Method Description\n"
        "BAD (bullet-list style):\n"
        "  'Our method has three components:\n"
        "   - Component A\n"
        "   - Component B\n"
        "   - Component C'\n\n"
        "GOOD (narrative style):\n"
        "  'Our method builds on the insight that [core problem] stems from\n"
        "   [root cause identified in Section 2]. To address this, we introduce\n"
        "   [MethodName], a [N]-stage framework. First, [Stage 1] maps inputs\n"
        "   to [representation]. These representations feed into [Stage 2],\n"
        "   enabling [benefit] without [drawback of prior approaches].\n"
        "   Crucially, we augment this with [Stage 3] based on [technical\n"
        "   foundation] (cite original paper), triggering [mechanism] when\n"
        "   [condition is met].'\n"
        "  NOTE: Replace all [placeholders] with YOUR actual method details.\n"
        "  Do NOT copy this template verbatim.\n"
    ),
    # IMP-31: Anti-hedging rules
    "anti_hedging_rules": (
        "\n## ANTI-HEDGING RULES (MANDATORY)\n"
        "1. The following phrases are BANNED from the paper body:\n"
        "   - 'we do not claim' / 'we cannot claim'\n"
        "   - 'we intentionally frame this conservatively'\n"
        "   - 'the evidence does not support' (unless followed by what it DOES support)\n"
        "   - 'only N seeds/runs' (belongs ONLY in Limitations, stated ONCE)\n"
        "   - 'this paper is not' / 'we do not' as paragraph openers\n"
        "2. Limitations and caveats MUST be consolidated in the Limitations section.\n"
        "   They may NOT appear in Introduction, Method, Results, or Conclusion.\n"
        "3. Confidence framing: Instead of 'we cannot prove X', write 'our results\n"
        "   provide evidence for X' or 'X is supported by [metrics]'.\n"
        "4. If you have a negative result, frame it as an INSIGHT:\n"
        "   BAD: 'Our method failed to outperform the baseline, we do not claim...'\n"
        "   GOOD: 'Surprisingly, the standard baseline proved competitive, suggesting\n"
        "   that [insight about why] — an observation with practical implications for...'\n"
    ),
    # IMP-24: Anti-repetition rules
    "anti_repetition_rules": (
        "\n## ANTI-REPETITION RULE\n"
        "Each specific number (e.g., '0.7667', '36.7%') may appear in AT MOST 2 sections:\n"
        "  - Once in Results/Experiments (where it is first reported)\n"
        "  - Once in Abstract (as a summary highlight)\n"
        "The Introduction, Discussion, and Conclusion MUST refer to results qualitatively\n"
        "('significantly outperformed', 'X% improvement') WITHOUT repeating exact numbers\n"
        "from the Results section. Violation of this rule will result in desk rejection.\n"
    ),
}

# -- Debate role prompts (multi-perspective generation) -------------------

DEBATE_ROLES_HYPOTHESIS: dict[str, dict[str, str]] = {
    "innovator": {
        "system": (
            "You are a bold, creative researcher who thinks outside the box. "
            "You pursue high-risk high-reward ideas, draw cross-domain analogies, "
            "and propose counter-intuitive hypotheses that challenge mainstream thinking."
        ),
        "user": (
            "Generate at least 2 novel, unconventional hypotheses from the synthesis below.\n"
            "CRITICAL REQUIREMENTS for EVERY hypothesis:\n"
            "1. NOVELTY: Must go beyond incremental combination of existing methods.\n"
            "2. FEASIBILITY: Must be testable within 30 minutes of compute on a single GPU.\n"
            "3. FALSIFIABILITY: Must define a specific metric threshold that would reject it.\n"
            "For each hypothesis provide:\n"
            "- A bold claim that pushes boundaries\n"
            "- Cross-domain inspiration (if applicable)\n"
            "- Rationale grounded in the literature gaps\n"
            "- Measurable prediction and failure condition\n"
            "- Estimated risk level (low/medium/high)\n\n"
            "Topic: {topic}\n"
            "Synthesis:\n{synthesis}"
        ),
    },
    "pragmatist": {
        "system": (
            "You are a practical ML engineer focused on what actually works. "
            "You prioritize computational feasibility, engineering simplicity, "
            "reliable baselines, and incremental but solid improvements."
        ),
        "user": (
            "Generate at least 2 feasible, well-grounded hypotheses from the synthesis below.\n"
            "For each hypothesis provide:\n"
            "- A concrete, testable claim with clear methodology\n"
            "- Why this is achievable with limited compute\n"
            "- Rationale based on proven techniques\n"
            "- Measurable prediction and failure condition\n"
            "- Resource requirements estimate\n\n"
            "Topic: {topic}\n"
            "Synthesis:\n{synthesis}"
        ),
    },
    "contrarian": {
        "system": (
            "You are a rigorous devil's advocate who challenges assumptions. "
            "You find blind spots, hidden failure modes, and counter-evidence. "
            "Your value is in finding problems others ignore. Be provocative "
            "but always grounded in evidence."
        ),
        "user": (
            "Critically examine the synthesis and generate at least 2 contrarian hypotheses.\n"
            "For each hypothesis provide:\n"
            "- A challenge to a widely-held assumption in this area\n"
            "- Evidence or reasoning for why the mainstream view may be wrong\n"
            "- An alternative hypothesis that accounts for overlooked factors\n"
            "- Measurable prediction and failure condition\n"
            "- Potential negative results that would be informative\n\n"
            "Topic: {topic}\n"
            "Synthesis:\n{synthesis}"
        ),
    },
}

DEBATE_ROLES_ANALYSIS: dict[str, dict[str, str]] = {
    "optimist": {
        "system": (
            "You highlight positive findings, promising extensions, and silver linings "
            "in experimental results. You identify what worked well and why, "
            "and suggest how to build on successes."
        ),
        "user": (
            "Analyze the experiment results from an optimistic perspective.\n"
            "Cover:\n"
            "- What worked well and why\n"
            "- Unexpected positive findings\n"
            "- Promising extensions and next steps\n"
            "- Silver linings in any negative results\n\n"
            "{preamble}\n{data_context}\n"
            "Run context:\n{context}"
        ),
    },
    "skeptic": {
        "system": (
            "You question the significance of results with maximum rigor. "
            "You check statistical validity, identify confounds, and demand "
            "stronger evidence. Every claim must earn its place."
        ),
        "user": (
            "Critically scrutinize the experiment results.\n"
            "Cover:\n"
            "- Statistical concerns (significance, sample size, multiple comparisons)\n"
            "- Potential confounds and alternative explanations\n"
            "- Missing evidence or controls\n"
            "- Whether metrics truly capture the intended phenomenon\n\n"
            "{preamble}\n{data_context}\n"
            "Run context:\n{context}"
        ),
    },
    "methodologist": {
        "system": (
            "You scrutinize HOW experiments were conducted. You audit "
            "internal/external validity, reproducibility, baseline fairness, "
            "and evaluation protocols."
        ),
        "user": (
            "Audit the experimental methodology.\n"
            "Cover:\n"
            "- Baseline fairness and completeness\n"
            "- Metric appropriateness for the research question\n"
            "- Evaluation protocol (data leakage, contamination risks)\n"
            "- Ablation completeness\n"
            "- Reproducibility assessment\n"
            "- Specific methodology improvements needed\n\n"
            "{preamble}\n{data_context}\n"
            "Run context:\n{context}"
        ),
    },
}

# -- Sub-prompts (secondary LLM calls within a stage) --------------------

_DEFAULT_SUB_PROMPTS: dict[str, dict[str, Any]] = {
    "hypothesis_synthesize": {
        "system": (
            "You are a senior research director synthesizing multiple perspectives "
            "into a decisive research proposal. The best synthesis is not a "
            "compromise but takes the strongest elements from each viewpoint. "
            "Preserve genuine disagreements — do not flatten controversy."
        ),
        "user": (
            "Below are hypotheses generated from three different research perspectives.\n"
            "Synthesize them into a final set of 2-4 hypotheses that:\n"
            "1. Take the strongest, most novel ideas\n"
            "2. Address critical concerns raised by the contrarian\n"
            "3. Ensure feasibility (pragmatist's input)\n"
            "4. Note unresolved disagreements between perspectives\n"
            "5. For each final hypothesis: rationale, measurable prediction, "
            "failure condition\n\n"
            "{perspectives}"
        ),
    },
    "analysis_synthesize": {
        "system": (
            "You are a senior research director synthesizing multiple analytical "
            "perspectives into a comprehensive assessment. Find the truth — if "
            "the skeptic or methodologist raise valid concerns, acknowledge them. "
            "Do not suppress criticism."
        ),
        "user": (
            "Below are analyses from three different perspectives (optimist, "
            "skeptic, methodologist).\n"
            "Produce a unified analysis that:\n"
            "1. Identifies consensus points (high-confidence conclusions)\n"
            "2. Resolves conflicts with evidence-based judgment\n"
            "3. Rates result quality (1-10 with justification)\n"
            "4. Lists 3-5 key findings\n"
            "5. Notes methodology gaps that need addressing\n"
            "6. Gives a clear PROCEED/PIVOT/REFINE recommendation\n\n"
            "Required sections: Metrics Summary, Consensus Findings, "
            "Contested Points, Statistical Checks, Methodology Audit, "
            "Limitations, Conclusion.\n\n"
            "{perspectives}"
        ),
        "max_tokens": 8192,
    },
    "code_repair": {
        "system": "You fix Python code validation errors while preserving functionality.",
        "user": (
            "The file `{fname}` in the experiment project has validation errors. "
            "Fix ALL issues and return ONLY the corrected file.\n\n"
            "## Validation Issues in {fname}\n{issues_text}\n\n"
            "## All Project Files\n{all_files_ctx}\n\n"
            "IMPORTANT: Do NOT use subprocess, os.system, eval, exec, or any "
            "network/shell calls.\n"
            "NUMPY 2.x: np.trapz→np.trapezoid, np.erfinv→scipy.special.erfinv, "
            "np.bool/int/float→Python builtins.\n"
            "Return ONLY the corrected code for `{fname}`."
        ),
    },
    "iterative_improve": {
        "system": (
            "You improve experiment projects and return valid executable Python code. "
            "Use ```filename:xxx.py format for each file."
        ),
        "user": (
            "Improve the experiment code based on prior run results.\n"
            "Return the improved files using ```filename:xxx.py format for each file.\n"
            "Primary metric key: {metric_key}\n"
            "Metric direction: {metric_direction}\n"
            "Do not use subprocess, os.system, eval, exec, or any network/shell calls.\n"
            "NUMPY 2.x: np.trapz→np.trapezoid, np.erfinv→scipy.special.erfinv, "
            "np.bool/int/float→Python builtins, np.math→math.\n\n"
            "EXPERIMENT PLAN ANCHOR (CRITICAL — read before making changes):\n"
            "The research topic is: {topic}\n"
            "{exp_plan_anchor}"
            "RULES FOR REFINEMENT:\n"
            "- NEVER rename, remove, or replace existing condition names. "
            "The condition names in the code MUST match the experiment plan.\n"
            "- NEVER add new conditions that are not in the experiment plan.\n"
            "- ONLY improve the IMPLEMENTATION of existing conditions "
            "(fix bugs, tune hyperparameters, improve training loops).\n"
            "- If the code has fundamental issues (wrong algorithm, missing "
            "components), fix the implementation but keep the same condition "
            "names and class hierarchy.\n\n"
            "{condition_coverage_hint}"
            "SEED ENFORCEMENT (MANDATORY — BUG-183):\n"
            "- You MUST use exactly seeds = [0, 1, 2] (3 seeds minimum).\n"
            "- Each condition MUST loop over ALL seeds.\n"
            "- Print per-seed: condition=X seed=S {metric_key}: V\n"
            "- Print aggregated: condition=X {metric_key}_mean: M {metric_key}_std: S\n"
            "- If 3 seeds × all conditions exceeds the time budget, REDUCE training "
            "epochs or conditions — NEVER reduce seed count below 3.\n\n"
            "CONDITION COUNT LIMIT (HARD RULE):\n"
            "- MAXIMUM 8 total conditions (baselines + methods + ablations).\n"
            "- If the previous code had >8 conditions, consolidate ablations to 2-3 values.\n\n"
            "DOCKER MOUNT TOPOLOGY (for fixing PermissionError/path issues):\n"
            "- WRITABLE: /workspace/ (project files), /tmp/, /workspace/data/\n"
            "- READ-ONLY: /opt/datasets/ (pre-cached CIFAR-10/100, MNIST, etc)\n"
            "- If you see PermissionError on /opt/datasets, do NOT call "
            "os.makedirs() there. Use root='/opt/datasets' with download=False.\n"
            "- For new data downloads, use /workspace/data/ as root.\n\n"
            "Current project files:\n{files_context}\n"
            "Run summaries (JSON):\n{run_summaries}"
        ),
        "max_tokens": 8192,
    },
    "iterative_repair": {
        "system": "You fix Python validation issues without adding unsafe behavior.",
        "user": (
            "Fix all validation issues in main.py and return corrected Python code only.\n\n"
            "## Validation Issues\n{issue_text}\n\n"
            "## Common RL Stability Fixes (apply if NaN/divergence detected):\n"
            "- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(params, 1.0)`\n"
            "- Lower learning rate to 1e-4 or 3e-4\n"
            "- Add reward normalization/clipping: `reward = np.clip(reward, -10, 10)`\n"
            "- Add NaN guard: `if torch.isnan(loss): continue`\n"
            "- Use float32 (not float16) for RL value functions\n"
            "- NUMPY 2.x: np.trapz→np.trapezoid, np.erfinv→scipy.special.erfinv, "
            "np.bool/int/float→Python builtins\n\n"
            "## All Project Files\n{all_files_ctx}"
        ),
    },
    # ── Advanced Code Agent sub-prompts ──────────────────────────────────
    "architecture_planning": {
        "system": (
            "You are a senior software architect who designs implementation "
            "blueprints for scientific experiment codebases. You produce detailed, "
            "directly-implementable specifications with pseudocode for every "
            "class method and explicit tensor shape annotations. You emphasize "
            "separation of concerns: data loading, model definition, training "
            "loop, and evaluation are distinct components. You understand ML "
            "training deeply and design for correctness: proper .detach(), "
            "consistent tensor shapes, and correct gradient flow.\n\n"
            "NUMPY 2.x COMPATIBILITY (CRITICAL):\n"
            "- np.trapz is REMOVED → use np.trapezoid\n"
            "- np.erfinv does NOT exist → use scipy.special.erfinv\n"
            "- np.bool, np.int, np.float, np.complex are REMOVED → use Python builtins\n"
            "- np.str, np.object are REMOVED → use str, object\n"
            "- np.math is REMOVED → use math module"
        ),
        "user": (
            "Create a detailed IMPLEMENTATION BLUEPRINT for an experiment codebase.\n\n"
            "## Research Context\n"
            "TOPIC: {topic}\n"
            "PRIMARY METRIC: {metric}\n\n"
            "## Experiment Plan\n{exp_plan}\n\n"
            "## Requirements\n"
            "1. `main.py` MUST be the entry point — runs ALL conditions sequentially.\n"
            "2. Each condition MUST be a SEPARATE class with DISTINCT implementation.\n"
            "3. Data loading and model definitions in separate modules.\n"
            "4. No more than 5 Python files total.\n"
            "5. Every class must have at least 20 lines of effective code.\n"
            "6. Child classes MUST override at least one core method with DIFFERENT logic.\n"
            "7. NEVER override nn.Module.train/eval with different signatures.\n"
            "8. Design child classes as STRATEGY variants, not PARAMETER variants.\n\n"
            "## Blueprint Format (YAML)\n"
            "The blueprint MUST include ALL of the following for EACH file:\n"
            "- `generation_order`: integer (1=first to generate, higher=later)\n"
            "- `dependencies`: list of other files this file imports from\n"
            "- `classes` or `functions`: with pseudocode for each method\n"
            "- For neural network classes: input/output tensor shapes\n\n"
            "```yaml\n"
            "files:\n"
            "  - name: config.py\n"
            "    generation_order: 1\n"
            "    dependencies: []\n"
            "    purpose: Hyperparameter configuration\n"
            "    classes:\n"
            "      - name: Config\n"
            "        fields:\n"
            "          - lr: 0.01\n"
            "          - batch_size: 128\n"
            "          - epochs: 20\n"
            "          - hidden_dim: 128\n\n"
            "  - name: data.py\n"
            "    generation_order: 2\n"
            "    dependencies: [config.py]\n"
            "    purpose: Dataset loading and preprocessing\n"
            "    functions:\n"
            "      - name: get_dataloaders\n"
            "        signature: (config) -> (train_loader, val_loader, test_loader)\n"
            "        pseudocode: |\n"
            "          1. Load dataset from torchvision/disk\n"
            "          2. Apply standard transforms (normalize, augment)\n"
            "          3. Split train into train/val (90/10)\n"
            "          4. Return DataLoaders with config.batch_size\n\n"
            "  - name: models.py\n"
            "    generation_order: 3\n"
            "    dependencies: [config.py]\n"
            "    purpose: All model implementations\n"
            "    classes:\n"
            "      - name: BaseModel(nn.Module)\n"
            "        input_shape: [B, 3, 32, 32]\n"
            "        output_shape: [B, 10]\n"
            "        methods:\n"
            "          - name: __init__\n"
            "            pseudocode: Define layers (conv/linear/attention)\n"
            "          - name: forward\n"
            "            pseudocode: |\n"
            "              1. x = self.encoder(x)  # [B,3,32,32] -> [B, hidden]\n"
            "              2. logits = self.classifier(x)  # [B, hidden] -> [B, 10]\n"
            "              3. return logits\n"
            "      - name: ProposedMethod(BaseModel)\n"
            "        differentiator: Uses novel component X\n"
            "        overrides: [forward]\n"
            "        methods:\n"
            "          - name: forward\n"
            "            pseudocode: |\n"
            "              1. x = self.encoder(x)\n"
            "              2. x = self.novel_component(x)  # KEY DIFFERENCE\n"
            "              3. logits = self.classifier(x)\n"
            "              4. return logits\n"
            "          - name: compute_special_loss\n"
            "            pseudocode: |\n"
            "              1. Compute task loss: CE(logits, labels)\n"
            "              2. Compute novel regularizer\n"
            "              3. return task_loss + lambda * reg\n\n"
            "  - name: training.py\n"
            "    generation_order: 4\n"
            "    dependencies: [config.py, data.py, models.py]\n"
            "    purpose: Training loop and evaluation\n"
            "    functions:\n"
            "      - name: train_one_epoch\n"
            "        signature: (model, loader, optimizer, device) -> float\n"
            "        pseudocode: |\n"
            "          1. model.train()\n"
            "          2. For each batch: forward, loss, backward, step\n"
            "          3. Return average loss\n"
            "      - name: evaluate\n"
            "        signature: (model, loader, device) -> dict\n"
            "        pseudocode: |\n"
            "          1. model.eval() with torch.no_grad()\n"
            "          2. For each batch: forward, argmax predictions\n"
            "          3. Return {accuracy, loss}\n\n"
            "  - name: main.py\n"
            "    generation_order: 5\n"
            "    dependencies: [config.py, data.py, models.py, training.py]\n"
            "    purpose: Entry point — runs ALL conditions\n"
            "    contract:\n"
            "      prints_metric_def: true\n"
            "      prints_registered_conditions: true\n"
            "      runs_all_conditions: true\n"
            "      per_seed_reporting: true\n"
            "      time_budget_guard: true\n"
            "    functions:\n"
            "      - name: main\n"
            "        pseudocode: |\n"
            "          1. Print METRIC_DEF line\n"
            "          2. Print REGISTERED_CONDITIONS\n"
            "          3. Setup time budget guard\n"
            "          4. For each condition:\n"
            "             a. Create model instance\n"
            "             b. For each seed:\n"
            "                - Set random seed\n"
            "                - Train model\n"
            "                - Evaluate and print per-seed metrics\n"
            "             c. Print mean/std across seeds\n"
            "          5. Print SUMMARY comparison\n\n"
            "verification_criteria:\n"
            "  - All condition classes have DIFFERENT forward/step implementations\n"
            "  - Input/output tensor shapes are consistent across data->model->loss\n"
            "  - Time budget guard exists in main training loop\n"
            "  - Per-seed random state isolation\n"
            "  - All .detach() calls present for values used across iterations\n\n"
            "conditions:\n"
            "  - name: ConditionName\n"
            "    class: ClassName\n"
            "    description: What makes it different\n"
            "```\n\n"
            "Output ONLY the YAML specification wrapped in ```yaml``` fences.\n"
            "Be SPECIFIC in pseudocode — include tensor shapes, loss formulas, "
            "and algorithmic details from the experiment plan.\n"
            "Every class must have detailed pseudocode showing HOW it differs "
            "from others, not just THAT it differs."
        ),
        "max_tokens": 8192,
    },
    "generate_single_file": {
        "system": (
            "You are an expert ML engineer who writes production-quality Python code "
            "for scientific experiments. You follow implementation blueprints precisely, "
            "ensuring tensor shapes match, gradients flow correctly, and all imports "
            "resolve. You write complete, runnable code — never stubs or placeholders."
        ),
        "user": (
            "Generate the Python file `{file_name}` for an ML experiment project.\n\n"
            "## File Specification\n{file_spec}\n\n"
            "## Full Project Blueprint\n{blueprint}\n\n"
            "## Already Generated Files (summaries)\n{dependency_summaries}\n\n"
            "## Already Generated Files (full code of direct dependencies)\n"
            "{dependency_code}\n\n"
            "## Research Topic\n{topic}\n\n"
            "## Experiment Plan\n{exp_plan}\n\n"
            "## Environment\n{pkg_hint}\n\n"
            "## CRITICAL Rules\n"
            "1. Follow the blueprint specification EXACTLY — implement every class "
            "and function listed for this file.\n"
            "2. Tensor shapes MUST match the blueprint annotations.\n"
            "3. Imports from dependency files MUST use the exact class/function names "
            "from the already-generated code.\n"
            "4. Every method must have a REAL implementation — no `pass`, no `...`, "
            "no `raise NotImplementedError`.\n"
            "5. NEVER use random numbers as fake metrics.\n"
            "6. For RL code: .detach() ALL values from previous iterations before "
            "using in current loss.\n"
            "7. For neural networks: create layers in __init__, not in forward().\n"
            "8. METHOD RICHNESS: Every non-trivial method should be >=5 lines of "
            "real logic. If a method only calls super() or returns a constant, "
            "add the actual computation it should perform. Training methods should "
            "include proper gradient handling, metric logging, and error checks.\n"
            "9. ABLATION DIFFERENTIATION: If this file contains ablation/variant "
            "classes, each MUST differ in actual algorithm logic — not just in "
            "parameter values or by removing a line. Ablations should clearly "
            "implement a different computational path.\n"
            "10. NO CLI CONDITION ARGS: If this is main.py, NEVER add argparse "
            "arguments like --condition or --method. All conditions must be "
            "iterated inside main.py with a for-loop. The harness runs "
            "`python main.py` with no arguments.\n"
            "11. NUMPY 2.x COMPATIBILITY: np.trapz→np.trapezoid, "
            "np.erfinv→scipy.special.erfinv, np.bool/np.int/np.float→Python builtins, "
            "np.str/np.object→str/object, np.math→math.\n\n"
            "Output ONLY the Python code for `{file_name}` — no markdown fences, "
            "no explanations, just the code."
        ),
        "max_tokens": 8192,
    },
    "code_exec_fix": {
        "system": (
            "You are a debugging expert who fixes runtime errors in Python "
            "experiment code. You preserve the original experiment design and "
            "scientific methodology while fixing the specific error. You fix "
            "the ROOT CAUSE, not just the symptom."
        ),
        "user": (
            "The following experiment code crashed during execution.\n\n"
            "## Error Output (stderr, last 3000 chars)\n"
            "```\n{stderr}\n```\n\n"
            "## Standard Output (last 50 lines)\n"
            "```\n{stdout_tail}\n```\n\n"
            "## Return Code: {returncode}\n\n"
            "## Current Code Files\n{files_context}\n\n"
            "## Instructions\n"
            "1. Identify the ROOT CAUSE of the error.\n"
            "2. Fix it while preserving the experiment design.\n"
            "3. Check for similar potential issues in ALL files.\n"
            "4. Do NOT simplify or remove experiment logic — fix the bug.\n"
            "5. Do NOT add subprocess, os.system, eval, exec, or network calls.\n"
            "6. COMMON BUG: If error is about `train()` missing arguments, it means "
            "a class overrode nn.Module.train() with a custom signature. Fix by "
            "renaming the custom method to `fit()` or `run_training()` and updating "
            "all callers. Never override nn.Module.train/eval with extra args.\n"
            "7. NUMPY 2.x: np.trapz→np.trapezoid, np.erfinv→scipy.special.erfinv, "
            "np.bool/int/float/complex→Python builtins, np.str/object→str/object.\n\n"
            "Output ALL files in ```filename:xxx.py``` format, including files "
            "that don't need changes."
        ),
        "max_tokens": 16384,
    },
    "code_reviewer": {
        "system": (
            "You are a meticulous experiment code reviewer focused on "
            "scientific correctness, statistical rigor, and code quality. "
            "You catch bugs that static analysis cannot: incorrect algorithm "
            "implementations, missing controls, wrong metric computation, "
            "and experimental design flaws."
        ),
        "user": (
            "Review this experiment code for correctness and quality.\n\n"
            "## Research Context\n"
            "TOPIC: {topic}\n"
            "PRIMARY METRIC: {metric}\n\n"
            "## Experiment Plan\n{exp_plan}\n\n"
            "## Code Files\n{files_context}\n\n"
            "## Review Criteria\n"
            "1. **CORRECTNESS**: Does the code correctly implement the "
            "experiment plan? Are algorithms implemented properly?\n"
            "2. **COMPLETENESS**: Are all conditions/ablations implemented "
            "with DISTINCT logic? (Not just renamed copies of baseline.)\n"
            "3. **STATISTICAL RIGOR**: Multiple seeds? Results averaged and "
            "reported with std? Paired comparisons?\n"
            "4. **METRIC REPORTING**: Is {metric} correctly computed and "
            "printed in the required format?\n"
            "5. **ROBUSTNESS**: Shape mismatches? Missing imports? Type "
            "errors? Division by zero? GPU/CPU device conflicts?\n"
            "6. **CLASS DEPTH**: Each experimental condition class must have "
            "at least 20 lines of effective code with distinct logic. Classes "
            "that only override __init__ to change parameters are CRITICAL "
            "issues — they indicate the condition is not truly different.\n\n"
            "## Output Format (JSON)\n"
            "```json\n"
            '{{\n'
            '  "verdict": "APPROVE or REVISE",\n'
            '  "score": 1-10,\n'
            '  "critical_issues": ["issue1", "issue2"],\n'
            '  "suggestions": ["suggestion1", "suggestion2"]\n'
            '}}\n'
            "```\n\n"
            "Only use verdict REVISE if there are critical issues that would "
            "cause the code to crash or produce scientifically invalid results."
        ),
        "json_mode": True,
        "max_tokens": 4096,
    },
}

# -- Stage prompts (one entry per LLM-calling stage) ---------------------

_DEFAULT_STAGES: dict[str, dict[str, Any]] = {
    # ── Phase A: Research Scoping ────────────────────────────────────────
    "topic_init": {
        "system": (
            "You are a rigorous research planner who identifies NOVEL, TIMELY "
            "research angles. You follow recent trends from top venues in the "
            "relevant domain and propose research that advances "
            "the frontier rather than repeating known results.\n\n"
            "NOVELTY PRINCIPLES:\n"
            "- A good research angle addresses a GAP not yet covered by existing work.\n"
            "- Avoid pure benchmark/comparison studies unless the methodology is novel.\n"
            "- Prefer angles that combine existing techniques in new ways, apply methods "
            "to underexplored domains, or challenge common assumptions.\n"
            "- The research must be FEASIBLE with limited compute (single GPU, hours not days).\n"
            "- Check: would a reviewer say 'this is already well-known'? If so, find a sharper angle."
        ),
        "user": (
            "Create a SMART research goal in markdown.\n"
            "Topic: {topic}\n"
            "Domains: {domains}\n"
            "Project: {project_name}\n"
            "Quality threshold: {quality_threshold}\n\n"
            "Required sections:\n"
            "- **Topic**: The broad area\n"
            "- **Novel Angle**: What specific aspect has NOT been well-studied? "
            "Why is this timely NOW (2024-2026)? What recent development creates "
            "an opportunity? How does this differ from standard approaches?\n"
            "- **Scope**: Focused enough for a single paper\n"
            "- **SMART Goal**: Specific, Measurable, Achievable, Relevant, Time-bound\n"
            "- **Constraints**: Compute budget, available tools, data access\n"
            "- **Success Criteria**: What results would make this publishable?\n"
            "- **Generated**: Timestamp\n\n"
            "IMPORTANT: The 'Novel Angle' section must convincingly argue why this "
            "specific research direction is NOT already covered by existing work. "
            "If the topic is well-studied (e.g., 'comparing optimizers'), you MUST "
            "find a specific unexplored aspect (e.g., 'under distribution shift with "
            "noisy gradients', 'in the few-shot regime', 'with modern architectures').\n\n"
            "TREND VALIDATION (MANDATORY):\n"
            "- Identify 2-3 recent papers (2024-2026) that establish the relevance "
            "of this research direction.\n"
            "- Name the specific benchmark/dataset that will be used for evaluation.\n"
            "- If no standard benchmark exists, explain how results will be measured.\n"
            "- State whether SOTA results exist on this benchmark and what they are.\n"
            "- Add a 'Benchmark' subsection listing: name, source, metrics, "
            "current SOTA (if known)."
        ),
    },
    "problem_decompose": {
        "system": "You are a senior research strategist.",
        "user": (
            "Decompose this research problem into at least 4 prioritized "
            "sub-questions.\n"
            "Topic: {topic}\n"
            "Output markdown with sections: Source, Sub-questions, Priority "
            "Ranking, Risks.\n"
            "Goal context:\n{goal_text}"
        ),
    },
    # ── Phase B: Literature Discovery ────────────────────────────────────
    "search_strategy": {
        "system": (
            "You design literature retrieval strategies and source verification plans."
        ),
        "user": (
            "Create a merged search strategy package.\n"
            "Return a JSON object with keys: search_plan_yaml, sources.\n"
            "search_plan_yaml must be valid YAML text.\n"
            "sources must include id,name,type,url,status,query,verified_at.\n"
            "Topic: {topic}\n"
            "Problem tree:\n{problem_tree}"
        ),
        "json_mode": True,
    },
    "literature_collect": {
        "system": "You are a literature mining assistant.",
        "user": (
            "Generate candidate papers from the search plan.\n"
            "Return JSON: {candidates:[...]} with >=8 rows.\n"
            "Each candidate must include id,title,source,url,year,abstract,"
            "collected_at.\n"
            "Topic: {topic}\n"
            "Search plan:\n{plan_text}"
        ),
        "json_mode": True,
    },
    "literature_screen": {
        "system": (
            "You are a strict domain-aware reviewer with zero tolerance for "
            "cross-domain false positives. You MUST reject papers that are "
            "from unrelated fields, even if they share superficial keyword "
            "overlap. A paper about 'normalization in database systems' is "
            "NOT relevant to 'normalization in deep learning'. A paper about "
            "'graph theory in social networks' is NOT relevant to 'graph "
            "neural networks for molecular property prediction'."
        ),
        "user": (
            "Perform merged relevance+quality screening and return shortlist.\n"
            "Return JSON: {shortlist:[...]} each with title, cite_key "
            "(if present), relevance_score (0-1), quality_score (0-1), "
            "keep_reason.\n"
            "Preserve all original fields (paper_id, doi, arxiv_id, cite_key, "
            "etc.) from the input.\n"
            "Topic: {topic}\n"
            "Domains: {domains}\n"
            "Threshold: {quality_threshold}\n\n"
            "SCREENING RULES (apply strictly):\n"
            "1. DOMAIN MATCH: The paper's actual research domain must match "
            "the topic's domain. Shared keywords across domains do NOT count.\n"
            "2. METHOD RELEVANCE: The paper must discuss methods, benchmarks, "
            "or findings directly applicable to the research topic.\n"
            "3. CROSS-DOMAIN REJECTION: Reject papers from unrelated fields "
            "(e.g., wireless communications, database systems, social science) "
            "even if they use similar terminology.\n"
            "4. RECENCY PREFERENCE: Prefer papers from 2020+ for methodology, "
            "but accept foundational papers (pre-2020) if they introduced key "
            "techniques still in use today.\n"
            "5. SEMINAL PAPERS: Papers marked as source='seminal_library' are "
            "pre-vetted foundational references — keep them if their keywords "
            "match the topic (relevance_score >= 0.7).\n"
            "6. QUALITY FLOOR: Reject papers with no abstract, no venue, and "
            "no citation count (likely not real papers).\n"
            "Candidates JSONL:\n{candidates_text}"
        ),
        "json_mode": True,
    },
    "knowledge_extract": {
        "system": "You extract high-signal evidence cards from papers.",
        "user": (
            "Extract structured knowledge cards from shortlist.\n"
            "Return JSON: {cards:[{card_id,title,cite_key,problem,method,"
            "data,metrics,findings,limitations,citation}]}.\n"
            "IMPORTANT: If the input contains cite_key fields, preserve them "
            "exactly in the output.\n"
            "Shortlist:\n{shortlist}"
        ),
        "json_mode": True,
    },
    # ── Phase C: Knowledge Synthesis ─────────────────────────────────────
    "synthesis": {
        "system": "You are a synthesis specialist for literature reviews.",
        "user": (
            "Produce merged synthesis output (topic clusters + research gaps).\n"
            "Output markdown with sections: Cluster Overview, Cluster 1..N, "
            "Gap 1..N, Prioritized Opportunities.\n"
            "Topic: {topic}\n"
            "Cards context:\n{cards_context}"
        ),
        "max_tokens": 8192,
    },
    "hypothesis_gen": {
        "system": (
            "You formulate testable scientific hypotheses that address gaps "
            "NOT covered by existing literature. Your hypotheses must be:\n"
            "1. NOVEL: Not simply replicating known results or testing obvious things.\n"
            "2. GAP-FILLING: Address specific weaknesses or blind spots identified "
            "in the literature synthesis.\n"
            "3. FEASIBLE: Testable with limited compute (single GPU, <1 day runtime).\n"
            "4. FALSIFIABLE: Have clear failure conditions that would definitively "
            "reject the hypothesis.\n"
            "5. SURPRISING: At least one hypothesis should challenge conventional "
            "wisdom or test a counter-intuitive prediction."
        ),
        "user": (
            "Generate at least 2 falsifiable hypotheses from the synthesis below.\n"
            "For each hypothesis provide:\n"
            "- **Hypothesis statement**: A clear, testable claim\n"
            "- **Novelty argument**: Why this has NOT been tested before, citing "
            "specific gaps from the synthesis\n"
            "- **Rationale**: Theoretical or empirical basis for expecting this result\n"
            "- **Measurable prediction**: Specific quantitative outcome expected\n"
            "- **Failure condition**: What result would reject this hypothesis?\n"
            "- **Required baselines**: What modern, state-of-the-art methods must be "
            "compared against to make the finding meaningful?\n\n"
            "AVOID:\n"
            "- Hypotheses that are trivially obvious (e.g., 'more data improves accuracy')\n"
            "- Hypotheses that replicate well-known results already in the literature\n"
            "- Hypotheses that cannot be tested within the compute budget\n\n"
            "Synthesis:\n{synthesis}"
        ),
    },
    # ── Phase D: Experiment Design ───────────────────────────────────────
    "experiment_design": {
        "system": "You are a principal investigator designing rigorous research experiments.",
        "user": (
            "{preamble}\n\n"
            "Design an experiment plan as YAML.\n"
            "Required keys: objectives,datasets,baselines,proposed_methods,"
            "ablations,metrics,risks,compute_budget.\n\n"
            "NAMING REQUIREMENT (CRITICAL for paper quality):\n"
            "- Every condition name in baselines, proposed_methods, and ablations MUST be "
            "a DESCRIPTIVE algorithm name DERIVED FROM THE HYPOTHESES ABOVE, NOT a generic label.\n"
            "- WRONG: baseline_1, baseline_2, method_variant_1, method_variant_2\n"
            "- WRONG: random_search, bayesian_optimization, ppo_policy, curiosity_driven_rl "
            "(these are generic defaults — NEVER use them unless they are actually what "
            "the hypotheses call for)\n"
            "- RIGHT: names that reflect the specific methods/architectures/algorithms in "
            "the hypotheses (e.g., rim_agent, monolithic_gru, ewc_baseline, sleep_consolidation, "
            "no_sleep_ablation, coarse_routing, fine_routing)\n"
            "- The name should immediately tell a reader WHAT algorithm or strategy is used.\n"
            "- This is critical because these names appear directly in the paper.\n\n"
            "BASELINE & BENCHMARK MODERNITY (CRITICAL for acceptance):\n"
            "- Baselines MUST be modern, widely-adopted methods from recent top-venue "
            "papers (2023-2026). Beating only outdated or weak baselines is NOT a valid "
            "contribution and will result in desk rejection.\n"
            "- Include at LEAST one strong baseline that represents current SOTA or "
            "near-SOTA in the specific sub-area. Check recent NeurIPS/ICML/ICLR papers "
            "to identify appropriate baselines.\n"
            "- Benchmarks MUST be standard and actively used. If a benchmark has been "
            "superseded, use the newer version.\n"
            "- For each baseline, cite the original paper and note why it is a fair "
            "and competitive comparison.\n\n"
            "HYPOTHESIS ALIGNMENT (CRITICAL — most common failure mode):\n"
            "- Your experiment plan MUST directly test the hypotheses listed above.\n"
            "- Each hypothesis should map to at least one comparison between conditions.\n"
            "- Baselines must be the specific alternatives named in the hypotheses, NOT "
            "generic optimization methods like random_search or bayesian_optimization.\n"
            "- If a hypothesis says 'X outperforms Y', then X must be a proposed_method "
            "and Y must be a baseline.\n"
            "- Ablations must isolate the specific components claimed to matter in the "
            "hypotheses (e.g., if hypothesis claims routing helps, ablate routing).\n\n"
            "STABILITY & REPRODUCIBILITY (CRITICAL for RL-based methods):\n"
            "- Under `proposed_methods`, specify key hyperparameters (learning rate, "
            "gradient clip threshold, entropy coefficient, etc.).\n"
            "- Under `risks`, explicitly list numerical stability concerns "
            "(NaN/divergence, reward explosion, policy collapse) and mitigations "
            "(gradient clipping, reward normalization, early stopping on NaN).\n"
            "- Under `metrics`, include:\n"
            "  * Primary metric: `{metric_key}` with direction: `{metric_direction}` "
            "and units\n"
            "  * IMPORTANT: The metric direction MUST be `{metric_direction}` — do "
            "NOT use a different direction. If {metric_direction}=='minimize', lower "
            "is better. If {metric_direction}=='maximize', higher is better.\n"
            "  * `success_rate`: fraction of seeds that complete without NaN/crash\n"
            "  * At least ONE discovery-aligned endpoint (e.g., identification "
            "accuracy, time-to-discovery, final posterior mass on true hypothesis) "
            "in addition to any proxy metric\n"
            "{dataset_guidance}\n\n"
            "- Under `datasets`, specify AT LEAST 2 regime factors to stratify by "
            "(e.g., noise_level: [low, high], hypothesis_space_size: [small, large]). "
            "Results MUST be reported per-regime. A single-regime experiment cannot "
            "support generality claims and will be rejected by reviewers.\n"
            "- FACTORIAL DESIGN PREFERRED: If you vary multiple factors (e.g., scale AND "
            "noise), design a factorial grid (e.g., small+low, small+high, large+low, "
            "large+high) so each factor's effect can be isolated. Bundling factors "
            "(e.g., easy=small+low, hard=large+high) is a confounder and reviewers will "
            "flag it. If computational budget limits the grid, at minimum acknowledge "
            "that factors are bundled and limit claims accordingly.\n"
            "- Under `compute_budget`, plan for minimum 10 seeds per condition to "
            "ensure valid statistical comparisons.\n\n"
            "STATISTICAL POWER REQUIREMENTS (CRITICAL for publishability):\n"
            "- Use AT LEAST 5 random seeds per condition (10 preferred)\n"
            "- Use AT LEAST 30 episodes per seed for RL methods\n"
            "- Report: mean ± std, 95% bootstrap CI, per-seed raw values\n"
            "- For method comparisons: use paired bootstrap or Wilcoxon signed-rank test "
            "(NOT paired t-test with n < 10)\n"
            "- Report effect sizes (Cohen's d or rank-biserial correlation)\n"
            "- 3 seeds is INSUFFICIENT — reviewers will reject papers with n=3\n\n"
            "HARDWARE ENVIRONMENT (your experiments run on THIS exact machine):\n"
            "{hardware_profile}\n"
            "- You have exactly ONE GPU. No distributed training. No multi-GPU. No multi-node.\n"
            "- Design experiments that fit this single GPU.\n\n"
            "COMPUTE BUDGET CONSTRAINT (CRITICAL — experiments MUST fit time budget):\n"
            "- Total experiment time budget: {time_budget_sec} seconds.\n"
            "- Per-condition budget: ~{per_condition_budget_sec} seconds "
            "(= time_budget × 0.7 / 6 conditions).\n"
            "- Pre-cached datasets (instant, no download): {available_tier1_datasets}\n"
            "- DO NOT plan experiments requiring multiple GPUs or more than "
            "{time_budget_sec}s.\n"
            "- HARD CONDITION LIMIT: The total number of conditions (baselines + "
            "proposed_methods + ablations) MUST NOT exceed 8 for budgets ≤ 3600s.\n"
            "  * Recommended: 2-3 baselines + 1-2 proposed methods + 2-3 ablations = 5-8 total.\n"
            "  * Generating 10+ conditions guarantees most will time out and data will be wasted.\n"
            "  * Quality over quantity: 6 well-run conditions with 5 seeds each >> 20 conditions "
            "with 1 seed each.\n"
            "- Each run needs AT LEAST 60 seconds for RL (environment setup + "
            "training + evaluation). For deep learning with GPU, at least 120 seconds.\n"
            "- HARD CAP: total_conditions × num_seeds × seconds_per_run MUST be < "
            "{time_budget_sec} × 0.8 (leave 20% margin for overhead).\n"
            "- If total would exceed the budget, you MUST reduce by:\n"
            "  1. First: reduce conditions (merge similar ablations, keep strongest baselines)\n"
            "  2. Then: reduce seeds to 5 (minimum for statistical validity)\n"
            "  3. Then: reduce regimes/environments to 1\n"
            "- Example: {time_budget_sec}s budget with 120s/condition/seed, 5 seeds → "
            "max {time_budget_sec} / (120 * 5) ≈ 4 conditions.\n\n"
            "IMPLEMENTATION SPECIFICATION (CRITICAL for code generation):\n"
            "For each proposed method AND each baseline, you MUST include an "
            "'implementation_spec' key with:\n"
            "  - class_name: the Python class name for this method\n"
            "  - key_methods: list of methods the class must implement "
            "(e.g., [__init__, forward, train_step, predict])\n"
            "  - algorithm_steps: pseudocode-level description of the core algorithm "
            "(3-10 steps), e.g.:\n"
            "    1. Encode input via encoder network (MLP: input_dim -> hidden_dim)\n"
            "    2. Compute attention weights over memory buffer\n"
            "    3. Aggregate attended features with learned gate\n"
            "    4. Decode to output via decoder network\n"
            "  - loss_function: the mathematical formula for the training loss "
            "(e.g., 'L = CE(y_pred, y_true) + lambda * KL(q||p)')\n"
            "  - key_hyperparameters: dict of hyperparameter name -> default value\n"
            "  - differentiator: what makes THIS method different from others "
            "(must be an algorithmic difference, not just a hyperparameter change)\n\n"
            "For each ablation, you MUST specify:\n"
            "  - what_is_removed: the specific component being ablated\n"
            "  - how_it_differs: concrete code-level description of the change "
            "(e.g., 'replace attention layer with mean pooling', 'set routing "
            "weight to uniform 1/N', 'remove skip connection in block 3')\n"
            "  - expected_effect: why removing this should change results\n\n"
            "This specification is MANDATORY — without it, the code generation "
            "stage cannot produce correct implementations.\n\n"
            "Hypotheses:\n{hypotheses}"
        ),
    },
    "code_generation": {
        "system": (
            "You are a computational scientist who writes real, runnable "
            "experiments. Your code implements actual algorithms with real "
            "mathematical operations. You NEVER fake results with random number "
            "generators. Always use the ```filename:xxx.py format for each file. "
            "Use numpy for numerical computation. Keep code self-contained "
            "and deterministic."
        ),
        "user": (
            "Generate a Python experiment project for the following research "
            "topic:\n"
            "TOPIC: {topic}\n\n"
            "CRITICAL REQUIREMENTS — your code MUST satisfy ALL of these:\n"
            "1. Implement the ACTUAL experiment described in the topic and "
            "plan below.\n"
            "   If the topic is about simulation (e.g., multi-agent systems, "
            "network dynamics),\n"
            "   write simulation code. If about optimization, write "
            "optimization code.\n"
            "   Match the code to the topic — do NOT default to generic "
            "gradient descent.\n"
            "2. Use proper mathematical models appropriate to the research "
            "question.\n"
            "   Examples: agent-based simulation, graph algorithms, "
            "statistical analysis,\n"
            "   optimization, Monte Carlo methods — whatever fits the topic.\n"
            "3. Run REAL computational experiments with meaningful "
            "parameters.\n"
            "4. Collect REAL metrics that directly answer the research "
            "question.\n"
            "5. The code must be scientifically meaningful — a reviewer should "
            "see\n"
            "   actual implementations relevant to the TOPIC, not a generic "
            "optimizer.\n\n"
            "OUTPUT FORMAT — return multiple files using this exact format:\n"
            "```filename:main.py\n"
            "# entry point code\n"
            "```\n\n"
            "```filename:models.py\n"
            "# model/algorithm implementations\n"
            "```\n\n"
            "Only create additional files (optimizers.py, data_utils.py, etc.) "
            "if they contain substantial logic (>20 lines). Do NOT create stub "
            "files with only imports or pass statements.\n\n"
            "CODE STRUCTURE:\n"
            "- main.py: entry point that runs experiments and prints metrics\n"
            "- main.py MUST begin with a docstring specifying:\n"
            "  (a) Dataset used and how it is loaded\n"
            "  (b) Distribution shift / corruption definition (if applicable)\n"
            "  (c) Model architecture (layers, dimensions, activation)\n"
            "  (d) Training protocol (optimizer, epochs, batch size, LR schedule)\n"
            "  (e) Evaluation protocol (train/test split, metrics computed)\n"
            "- Additional modules for algorithms, objective functions, "
            "utilities\n"
            "- Primary metric key: {metric}\n"
            "- main.py must print metric lines as `name: value` (one per "
            "line)\n"
            "- Use deterministic seeds (numpy.random.seed or random.seed)\n"
            "- No external data files, no network calls, no GPU required\n"
            "- FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket\n"
            "{pkg_hint}\n"
            "ANTI-PATTERNS (do NOT do these):\n"
            "- Do NOT generate random numbers and pretend they are experiment "
            "results\n"
            "- Do NOT use `random.uniform()` to simulate a decreasing loss "
            "curve\n"
            "- Do NOT hardcode metric values or use trivial arithmetic as "
            "metrics\n\n"
            "MULTI-CONDITION REQUIREMENT (CRITICAL):\n"
            "The experiment plan below specifies multiple conditions, treatments, "
            "or strategies to compare. Your code MUST:\n"
            "1. Implement ALL conditions/treatments listed in the experiment plan "
            "— not just one baseline.\n"
            "2. Run each condition independently with the same controlled setup "
            "(same seeds, same initialization, same budget).\n"
            "   IMPORTANT: All conditions MUST be iterated INSIDE main.py using a "
            "for-loop or dispatch table. NEVER use argparse --condition or any CLI "
            "argument to select a condition. The harness calls `python main.py` "
            "with NO arguments — if you add a required --condition arg it will crash.\n"
            "3. Print metrics with condition labels: "
            "`condition=<name> {metric}: <value>` for EACH condition.\n"
            "4. After all conditions, print a summary comparison line: "
            "`SUMMARY: condition1=<val>, condition2=<val>, ...`\n"
            "5. If the plan has N conditions, the output MUST contain N separate "
            "labeled metric streams. Running only one condition is NOT acceptable.\n"
            "6. BREADTH-FIRST ORDERING: Run ONE representative configuration per "
            "condition FIRST (e.g., default parameters), so that ALL conditions "
            "produce at least one result. Only AFTER all conditions have results, "
            "run additional parameter sweeps if time remains. This prevents the "
            "time budget from being exhausted on condition 1's parameter sweep "
            "while conditions 2..N never execute.\n"
            "7. CONDITION COMPLETENESS: After code generation, mentally verify that "
            "EVERY condition in the experiment plan below has a corresponding code "
            "path. If the plan lists conditions A, B, C, D — your code must handle "
            "all four, not just A, B, C. Missing conditions invalidate the experiment.\n"
            "8. CRASH RESILIENCE: Wrap each condition's execution in a try/except "
            "block so that if one condition crashes (e.g., NaN, timeout, config error), "
            "the remaining conditions still execute. Print `CONDITION_FAILED: <name> "
            "<error>` on failure and continue to the next condition. A partial result "
            "set is far more valuable than a complete crash.\n"
            "9. CONDITION REGISTRY VALIDATION: At startup (before running experiments), "
            "enumerate all condition names and verify each has a valid code path. Print "
            "`REGISTERED_CONDITIONS: <name1>, <name2>, ...` at the top of output. If "
            "any condition is unrecognized, print `MISSING_CONDITION: <name>` and skip "
            "it gracefully rather than raising an exception.\n"
            "10. TOTAL CONDITIONS LIMIT (HARD RULE): Your code MUST NOT register more "
            "than 8 total conditions. If the experiment plan lists ablations with many "
            "parameter values (e.g., 'test decay rates 0.9, 0.99, 0.995, 0.999, 0.9999'), "
            "pick the 2-3 most informative values — do NOT create a separate condition for "
            "each value. 8 conditions × 3 seeds × budget ÷ conditions = tight timing. "
            "Quality of each condition matters more than quantity.\n\n"
            "METRIC DEFINITION REQUIREMENT (CRITICAL):\n"
            "- At the top of main.py, include a docstring or comment block that defines:\n"
            "  * METRIC NAME: the exact key printed as `{metric}: <value>`\n"
            "  * DIRECTION: {metric_direction_hint}\n"
            "  * UNITS/SCALE: what the number represents (e.g., MSE in log scale, "
            "accuracy 0-1, discovery rate per episode)\n"
            "  * FORMULA: how the metric is computed from raw experiment outputs\n"
            "  * AGGREGATION: how per-step/per-episode values are reduced to a scalar\n"
            "- Print this definition at runtime: `METRIC_DEF: {metric} | direction=<higher/lower> "
            "| desc=<one-line description>`\n"
            "- Without this definition, the metric is UNINTERPRETABLE and the paper cannot "
            "make any claims about which method is better.\n\n"
            "STATISTICAL RIGOR REQUIREMENT:\n"
            "- Run each condition with at least 5 different random seeds (10+ preferred "
            "if time budget allows). Minimum 3 seeds is MANDATORY.\n"
            "- Print per-seed results: `condition=<name> seed=<s> {metric}: <value>`\n"
            "- Print mean and std across seeds: "
            "`condition=<name> {metric}_mean: <val> {metric}_std: <val>`\n"
            "- If time budget is tight, reduce per-seed iterations rather than "
            "reducing seed count. Minimum 3 seeds is non-negotiable.\n"
            "- SEED COUNT IS FIXED AT 3 MINIMUM. Do NOT compute seed count dynamically.\n"
            "  Hardcode `SEEDS = [0, 1, 2]`. If 3 seeds × all conditions exceeds the time "
            "budget, REDUCE the number of conditions or training epochs — NEVER reduce seeds.\n"
            "  Print: `SEED_COUNT: 3 (fixed minimum, budget={time_budget}s, conditions=N)`.\n"
            "- Report bootstrap 95% confidence intervals when n >= 5.\n\n"
            "FAILURE-AWARE REPORTING (CRITICAL for RL/unstable methods):\n"
            "- Track how many seeds succeed vs fail (NaN, divergence, crash) per "
            "condition. Print: `condition=<name> success_rate: <succeeded>/<total>`\n"
            "- Compute UNCONDITIONAL metrics: treat failed seeds as worst-case "
            "(e.g., metric=0 or metric=worst_baseline). Print: "
            "`condition=<name> unconditional_{metric}_mean: <val>`\n"
            "- This prevents survivorship bias where a method looks good only "
            "because failed runs are excluded.\n"
            "- For RL methods, add STABILITY SAFEGUARDS in the code:\n"
            "  * Gradient clipping (max norm 1.0)\n"
            "  * Reward normalization/clipping to [-10, 10]\n"
            "  * NaN checks on loss/gradients with graceful early stop (not crash)\n"
            "  * Learning rate warmup or conservative initial learning rate\n"
            "  These safeguards should PREVENT most NaN/divergence, not just catch "
            "them after the fact.\n\n"
            "PYTORCH RL IMPLEMENTATION BUGS (CRITICAL — these cause 100% crash rate):\n"
            "- 'Trying to backward through the graph a second time' is the #1 crash.\n"
            "  CAUSE: reusing a computed tensor across multiple backward() calls.\n"
            "  FIX: Always .detach() values used in the next iteration:\n"
            "  ```\n"
            "  # WRONG:\n"
            "  old_log_prob = policy.log_prob(action)  # still attached to graph\n"
            "  # ... later in update loop:\n"
            "  ratio = new_log_prob / old_log_prob  # backward crashes\n"
            "  \n"
            "  # CORRECT:\n"
            "  old_log_prob = policy.log_prob(action).detach()  # detach!\n"
            "  # ... later in update loop:\n"
            "  ratio = new_log_prob / old_log_prob.detach()  # safe\n"
            "  ```\n"
            "- For PPO: old_log_probs MUST be .detach()ed when stored for later ratio computation.\n"
            "- For value functions: target values MUST be .detach()ed (don't backprop through targets).\n"
            "- For curiosity/intrinsic reward: prediction errors used as reward MUST be .detach()ed.\n"
            "- General rule: any tensor from a PREVIOUS forward pass that is used in the CURRENT "
            "loss computation MUST be .detach()ed.\n"
            "- When in doubt, add .detach() — it never causes crashes, but missing it always does.\n\n"
            "NEURAL NETWORK DIMENSION CONSISTENCY (CRITICAL — #2 crash cause):\n"
            "- 'input and weight.T shapes cannot be multiplied' means obs_dim != network input_dim.\n"
            "- When the environment observation size VARIES across regimes (e.g., easy=6, hard=8), "
            "the neural network's input layer MUST match EACH regime's obs_dim.\n"
            "- FIX: Create the network INSIDE the per-regime loop, or parameterize input_dim:\n"
            "  ```\n"
            "  # WRONG: fixed input_dim for all regimes\n"
            "  policy = PolicyNet(input_dim=10)  # breaks if obs_dim != 10\n"
            "  for regime in regimes:\n"
            "      obs = env.reset()  # obs.shape may vary!\n"
            "  \n"
            "  # CORRECT: dynamic input_dim per regime\n"
            "  for regime in regimes:\n"
            "      obs = env.reset()\n"
            "      obs_dim = obs.shape[-1]  # or len(obs)\n"
            "      policy = PolicyNet(input_dim=obs_dim)  # fresh network per regime\n"
            "  ```\n"
            "- ALWAYS initialize neural networks AFTER knowing the observation dimension.\n\n"
            "KNOWLEDGE DISTILLATION (KD) STABILITY (if applicable):\n"
            "- Teacher network MUST be frozen: `teacher.eval()` and "
            "`for p in teacher.parameters(): p.requires_grad = False`\n"
            "- Temperature parameter T: typical range 1-20. Use T=4 as default. "
            "NEVER use T<1 (causes sharp distributions → NaN gradients).\n"
            "- Loss balance: `loss = alpha * kd_loss + (1-alpha) * task_loss` — "
            "set alpha=0.5-0.9. If kd_loss scale >> task_loss, val_loss becomes NaN.\n"
            "- PROJECTION LAYERS: If teacher and student have different intermediate "
            "dimensions (e.g., teacher_dim=768, student_dim=256), you MUST add "
            "`nn.Linear(student_dim, teacher_dim)` to align features before computing "
            "distillation loss. Without projection layers, tensor shape mismatch WILL crash.\n"
            "- Common KD NaN causes: (1) no temperature scaling on logits, "
            "(2) missing gradient clipping, (3) learning rate too high (use ≤1e-3), "
            "(4) teacher not frozen → unstable targets.\n\n"
            "PAIRED STATISTICAL ANALYSIS (CRITICAL for publishable results):\n"
            "- Use the SAME random seeds across all conditions so results are paired.\n"
            "- After collecting per-seed results for all conditions, compute paired "
            "differences: for each seed s, diff(s) = method(s) - baseline(s).\n"
            "- Print paired analysis: "
            "`PAIRED: <method> vs <baseline> mean_diff=<val> std_diff=<val> "
            "t_stat=<val> p_value=<val>`\n"
            "- Also print bootstrap 95% CI of the paired difference.\n"
            "- This is FAR more powerful than independent comparisons because it "
            "controls for seed-to-seed variance.\n\n"
            "MULTI-REGIME REQUIREMENT (CRITICAL for generality claims):\n"
            "- The experiment MUST test at least 2 different difficulty/noise regimes "
            "(e.g., low noise vs high noise, small hypothesis space vs large).\n"
            "- Report results per-regime, not just aggregated across regimes.\n"
            "- Print regime labels: "
            "`condition=<name> regime=<regime_name> {metric}: <value>`\n"
            "- This prevents conclusions that only hold in one setting from being "
            "presented as general findings.\n\n"
            "DIMENSION CONSISTENCY CHECK (CRITICAL for RL/neural methods):\n"
            "- Before passing observations/states to neural networks or policy "
            "parameters, VERIFY that dimensions match. Common bug: environment "
            "state has dimension D1 but network expects D2.\n"
            "- At the start of each condition, print the state/observation "
            "dimension and the network input dimension. If they mismatch, "
            "reshape or adjust the network before proceeding.\n"
            "- Test EVERY condition with a single dry-run step before the full "
            "loop to catch shape mismatches early.\n\n"
            "TIME-TO-EVENT METRIC BUG PREVENTION (CRITICAL — common silent bug):\n"
            "- If the primary metric is a 'time-to-X' measure (e.g., time-to-discovery, "
            "steps-to-convergence, episodes-to-threshold), you MUST check the success "
            "criterion at EVERY step inside the loop, not only at the end.\n"
            "- WRONG pattern (produces degenerate ceiling data):\n"
            "  ```\n"
            "  for t in range(horizon):\n"
            "      obs, r, done, info = env.step(a)\n"
            "  success = check(info)  # only checked ONCE at end\n"
            "  time_to_X = horizon if not success else t + 1  # t+1 = horizon always!\n"
            "  ```\n"
            "- CORRECT pattern (captures actual first-success time):\n"
            "  ```\n"
            "  time_to_X = horizon  # default: never succeeded\n"
            "  for t in range(horizon):\n"
            "      obs, r, done, info = env.step(a)\n"
            "      if check(info) and time_to_X == horizon:  # first success\n"
            "          time_to_X = t + 1\n"
            "      if done: break\n"
            "  ```\n"
            "- This bug causes ALL methods to return the same ceiling value, making "
            "the entire experiment useless. Every method looks identical at the cap.\n"
            "- APPLY THIS TO ALL CONDITIONS: RandomSearch, BO, RL — every single "
            "condition must check at every step. If even one condition uses the wrong "
            "pattern, the comparison is invalid.\n\n"
            "METRIC DISCRIMINATION VALIDATION (CRITICAL):\n"
            "- After running all conditions, check if all conditions produce the SAME "
            "mean metric value. If they do, the metric is NOT discriminative and the "
            "experiment is scientifically useless.\n"
            "- Common causes: ceiling/floor effects, too-easy or too-hard tasks, "
            "time-to-event bug above, metric that doesn't capture real differences.\n"
            "- If all conditions have identical means, print "
            "`WARNING: DEGENERATE_METRICS all conditions have same mean=<val>` "
            "and you MUST take corrective action:\n"
            "  (a) If all means = 1.0 or max: increase task difficulty (reduce budget, "
            "increase noise, enlarge hypothesis space)\n"
            "  (b) If all means = 0.0: decrease difficulty\n"
            "  (c) Re-run after adjustment and verify means now differ\n"
            "  (d) If adjustments don't help, switch to a different primary metric\n"
            "- A degenerate experiment CANNOT produce a publishable paper. Fix it.\n\n"
            "DIFFICULTY CALIBRATION (CRITICAL for meaningful results):\n"
            "- After running a pilot (3-5 seeds, 2 conditions: random_search + one RL), "
            "check BOTH success rate AND metric discrimination.\n"
            "- TWO things must be true for the experiment to be informative:\n"
            "  1. Success rate between 30-80% (not too hard, not too easy)\n"
            "  2. Primary metric varies across conditions (not all methods score the same)\n"
            "- CEILING DETECTION (CRITICAL): If primary_metric is 1.0 (or max possible) "
            "for ALL pilot seeds in ALL pilot conditions, the task is TRIVIALLY EASY. "
            "You MUST increase difficulty until the metric varies. Options:\n"
            "  * Reduce experiment budget/horizon (fewer steps to find solution)\n"
            "  * Increase hypothesis space size\n"
            "  * Increase observation noise\n"
            "  * Tighten the success criterion (e.g., require closer match)\n"
            "  * Reduce the number of allowed experiments per episode\n"
            "- FLOOR DETECTION: If primary_metric is 0.0 for all conditions, task is "
            "too hard. Reduce noise, enlarge budget, simplify.\n"
            "- Print `CALIBRATION: regime=<name> pilot_success_rate=<val> "
            "pilot_primary_metric_std=<val>` after calibration.\n"
            "- If std=0, the metric is NOT discriminative — adjust until std > 0.\n"
            "- Run a calibration loop: pilot → check → adjust → re-pilot (max 3 iterations).\n\n"
            "ALGORITHM IMPLEMENTATION INTEGRITY (CRITICAL — mismatch = academic fraud):\n"
            "1. If you name a method 'Bayesian Optimization', you MUST implement:\n"
            "   - A surrogate model (e.g., Gaussian Process or random forest)\n"
            "   - An acquisition function (e.g., Expected Improvement, UCB)\n"
            "   - Surrogate model updates after each observation\n"
            "   DO NOT implement UCB1 bandit and call it 'Bayesian Optimization'.\n"
            "2. If you name a method 'PPO', you MUST implement:\n"
            "   - A clipped surrogate objective: min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)\n"
            "   - A learned value function baseline\n"
            "   - The clip_eps parameter MUST be used in the policy update\n"
            "   DO NOT implement vanilla REINFORCE and call it 'PPO'.\n"
            "3. Every declared hyperparameter MUST be used in the algorithm:\n"
            "   - If you declare clip_eps, it must appear in the loss computation\n"
            "   - If you declare entropy_coef, it must be added to the policy loss\n"
            "   - Dead parameters (declared but never used) are strictly forbidden\n"
            "4. Ablation conditions MUST produce different behavior:\n"
            "   - Two conditions that differ only in a parameter that is never read are IDENTICAL\n"
            "   - Verify: if two conditions produce identical outputs on the same seed, "
            "the ablation is broken and MUST be fixed\n"
            "   ABLATION DESIGN PATTERN (CRITICAL — #1 cause of broken ablations):\n"
            "   - 'no_key_component': Must REMOVE a core algorithmic component "
            "(e.g., disable the graph structure by zeroing the adjacency, or remove "
            "the contrastive loss, or disable the RL policy and use random actions). "
            "The removal MUST change the forward() / step() computation.\n"
            "   - 'reduced_capacity': Must REDUCE model capacity by at least 2x "
            "(e.g., halve hidden dimensions, reduce layers, shrink embedding size). "
            "This MUST create a new model with different architecture, NOT just "
            "rename a parameter with the same value.\n"
            "   - SELF-TEST: After implementing ablations, add a startup check that "
            "runs one forward pass per condition on the SAME input and asserts outputs "
            "differ. Print: `ABLATION_CHECK: <name1> vs <name2> outputs_differ=True`.\n"
            "   - If outputs are identical, the ablation is BROKEN — do not proceed.\n\n"
            "CODE IMPLEMENTATION DEPTH (CRITICAL — shallow code = reject):\n"
            "- Each algorithm/method MUST be a separate Python class with genuine logic.\n"
            "- Each class MUST have at least: __init__(), and one core method "
            "(forward/predict/train_step/step) with non-trivial implementation.\n"
            "- The core method of the MAIN proposed method MUST be at least 20 lines "
            "of effective code (excluding comments, blanks, imports).\n"
            "- FORBIDDEN patterns that will be detected and rejected:\n"
            "  * `class MethodB(MethodA): pass` — empty subclass\n"
            "  * Two classes with identical method bodies but different names\n"
            "  * nn.Linear/nn.Conv2d created inside forward() instead of __init__()\n"
            "  * Variables defined only inside an if-branch but used after the branch\n"
            "  * Using np.erf() (doesn't exist — use scipy.special.erf or math.erf)\n"
            "  * Using ndarray.ptp() (removed in NumPy 2.0 — use np.ptp(arr) or arr.max()-arr.min())\n"
            "  * Using np.bool, np.int, np.float, np.complex (removed in NumPy 2.0 — use np.bool_, np.int64, etc.)\n"
            "  * Replacing real model training with synthetic utility functions or random scores\n"
            "  * Using dict[key] without ensuring key exists — use dict.get(key, default) "
            "or verify key is in dict before access\n"
            "- If the experiment plan includes 'implementation_spec', you MUST follow "
            "the pseudocode steps exactly. Each algorithm_step should correspond to "
            "1-3 lines of code in the class.\n"
            "- Ablation variants MUST modify the forward() or step() logic, not just "
            "change a hyperparameter value.\n\n"
            "MINIMUM SEED COUNT (CRITICAL — 3 seeds = unpublishable):\n"
            "- Use AT LEAST 5 random seeds per condition (10 preferred if time permits)\n"
            "- Use AT LEAST 30 episodes per seed for RL methods\n"
            "- When computing bootstrap CIs, use at least 1000 bootstrap samples\n"
            "- For method comparisons: use paired bootstrap or Wilcoxon signed-rank test\n"
            "- Report effect sizes (Cohen's d) alongside p-values\n\n"
            "Experiment plan:\n{exp_plan}"
        ),
        "max_tokens": 8192,
    },
    "resource_planning": {
        "system": "You are an experiment scheduler.",
        "user": (
            "Create schedule JSON with GPU/time estimates.\n"
            "Schema: {tasks:[{id,name,depends_on,gpu_count,estimated_minutes,"
            "priority}], total_gpu_budget, generated}.\n"
            "Experiment plan:\n{exp_plan}"
        ),
        "json_mode": True,
    },
    # ── Phase F: Analysis & Decision ─────────────────────────────────────
    "result_analysis": {
        "system": (
            "You are a quantitative research analyst. Always cite exact numbers "
            "from the provided data."
        ),
        "user": (
            "{preamble}\n\n"
            "{data_context}\n\n"
            "Analyze run metrics and produce markdown report with statistical "
            "interpretation.\n"
            "Use the ACTUAL quantitative values provided above — do NOT invent "
            "numbers.\n\n"
            "SANITY CHECKS (perform BEFORE interpreting results):\n"
            "1. MONOTONICITY: If a condition scales a parameter (e.g., N agents, "
            "model size), check whether metrics move in the expected direction. "
            "If accuracy *decreases* when adding more agents under majority voting, "
            "flag this as a likely implementation bug (vote parsing, normalization, "
            "or aggregation issue).\n"
            "2. BASELINE PLAUSIBILITY: Random-chance baselines should match "
            "theoretical expectations (e.g., 1/K for K-class classification).\n"
            "3. CROSS-CONDITION CONSISTENCY: Results across datasets or conditions "
            "should be internally coherent — wildly different patterns may indicate "
            "confounds or bugs.\n"
            "4. REPLICATION: If results are from a single seed (n=1), explicitly "
            "note that no statistical significance claims can be made.\n"
            "5. ABLATION ISOLATION: Compare per-seed values across conditions. If "
            "two conditions produce IDENTICAL values for the same seed, this is a "
            "RED FLAG — the ablation/variant may not have actually changed the code "
            "path (e.g., config not applied, caching, shared state). Flag this "
            "explicitly and recommend a config/registry audit.\n"
            "6. METRIC DEFINITION CHECK: Look for a `METRIC_DEF:` line in the output. "
            "If absent, flag that the primary metric is UNDEFINED — direction, units, "
            "and formula are unknown, making all comparisons uninterpretable. This is "
            "a critical methodology gap.\n"
            "7. CONDITION COMPLETENESS CHECK: Look for `REGISTERED_CONDITIONS:` in "
            "the output. Compare against the experiment plan. If conditions are missing "
            "or failed (look for `CONDITION_FAILED:`), list them explicitly and assess "
            "whether the remaining conditions can still answer the research question.\n"
            "8. DEGENERATE METRICS CHECK: If ALL conditions (or all but one) produce "
            "the SAME mean primary metric value, flag this as DEGENERATE — the metric "
            "is NOT discriminative. Common causes: (a) time-to-event metric that only "
            "checks success at the final step (returns horizon for all methods), "
            "(b) ceiling/floor effects from too-easy or too-hard tasks, "
            "(c) metric capped at a budget value. This makes the experiment "
            "scientifically useless — recommend REFINE with a note to fix the metric "
            "computation or task difficulty. Look for `WARNING: DEGENERATE_METRICS` "
            "in stdout. Even if not printed, check the numbers yourself.\n\n"
            "Required sections: Metrics Summary (with real values), "
            "Consensus Findings (high confidence), "
            "Contested Points (with evidence-based resolution), "
            "Statistical Checks, Methodology Audit, Limitations, Conclusion.\n"
            "In the Conclusion, include:\n"
            "- Result quality rating (1-10)\n"
            "- Key findings (3-5)\n"
            "- Methodology gaps to address next\n"
            "- Recommendation: PROCEED / REFINE / PIVOT\n\n"
            "Run context:\n{context}"
        ),
        "max_tokens": 8192,
    },
    "research_decision": {
        "system": "You are a research program lead making go/no-go decisions.",
        "user": (
            "Based on the analysis, make one of three decisions:\n"
            "- **PROCEED** — results are sufficient, move to paper writing\n"
            "- **PIVOT** — hypotheses are fundamentally flawed, generate new ones\n"
            "- **REFINE** — hypotheses are sound but experiments need re-tuning\n\n"
            "MINIMUM QUALITY CRITERIA for PROCEED (ALL must be met):\n"
            "1. At least 2 baselines AND the proposed method have results\n"
            "2. The primary metric is defined (direction, units known)\n"
            "3. Each condition has results from ≥3 seeds\n"
            "4. No identical per-seed values across different conditions (ablation integrity)\n"
            "5. The analysis quality rating is ≥4/10\n"
            "If ANY criterion is not met, you MUST choose REFINE (not PROCEED).\n\n"
            "Output markdown with sections:\n"
            "## Decision\n"
            "State exactly one of: PROCEED, PIVOT, or REFINE\n\n"
            "## Justification\n"
            "Why this decision is warranted based on evidence.\n\n"
            "## Evidence\n"
            "Key data points supporting the decision.\n\n"
            "## Next Actions\n"
            "Concrete steps for the chosen path.\n\n"
            "Analysis:\n{analysis}"
        ),
    },
    # ── Phase G: Paper Writing ───────────────────────────────────────────
    "paper_outline": {
        "system": "You are an academic writing planner for top-tier AI conferences.",
        "user": (
            "{preamble}\n\n"
            "{academic_style_guide}\n\n"
            "Create a detailed paper outline in markdown.\n"
            "Include per-section goals, word count targets, and evidence links.\n"
            "The outline MUST include a catchy method name (2-5 chars) for the paper title.\n"
            "Propose 3 candidate titles following the 'MethodName: Subtitle' format "
            "(each <= 14 words). Rate each on memorability (1-5), specificity (1-5), "
            "and novelty signal (1-5).\n"
            "{topic_constraint}"
            "{feedback}"
            "Analysis:\n{analysis}\n\nDecision:\n{decision}"
        ),
        "max_tokens": 8192,
    },
    "paper_draft": {
        "system": (
            "You are a top-tier academic paper author writing for leading venues.\n\n"
            "KEY PRINCIPLES (from accepted paper analyses):\n"
            "1. NOVELTY: A good paper has 1-2 key ideas and keeps the rest simple.\n"
            "2. NARRATIVE: A short, rigorous, evidence-based technical story with a takeaway.\n"
            "3. STRONG BASELINES: Invest real effort in making baselines competitive.\n"
            "4. ABLATIONS: Remove one component at a time and measure the effect.\n"
            "5. HONESTY: Acknowledge limitations explicitly.\n"
            "6. REPRODUCIBILITY: Include all details needed to reproduce results.\n\n"
            "EVIDENCE-BOUNDING RULES (CRITICAL — violation = reject):\n"
            "7. EVERY claim in the title, abstract, and conclusion MUST be directly "
            "supported by specific experimental metrics provided below.\n"
            "8. If the experiment only covers partial conditions, the title MUST NOT "
            "make global causal claims. Use 'Toward...', 'Investigating...', or "
            "'An Empirical Study of...' instead of 'X Dominates Y'.\n"
            "9. BEFORE writing the title, list the conditions actually tested and "
            "their metric values. The title must only claim what those numbers show.\n"
            "10. If a metric is a single scalar without condition labels, do NOT "
            "claim comparative results between strategies/methods.\n"
            "11. Distinguish between 'we propose and validate' (has full results) vs "
            "'we propose and present preliminary evidence' (partial results).\n\n"
            "You ONLY use real experimental data — never fabricate or approximate numbers.\n\n"
            "METHOD SECTION REQUIREMENTS:\n"
            "12. The Method section MUST include ALL implementation details needed "
            "for reproduction: algorithm pseudocode or step-by-step description, "
            "hyperparameters (learning rate, clipping, discount factor, etc.), "
            "state/observation representation, reward definition, and baseline "
            "configurations.\n"
            "13. For learning-based methods: specify model architecture, training procedure "
            "(iterations, epochs, batch handling), and any stability "
            "mechanisms (regularization, normalization).\n"
            "14. For baselines: specify the exact algorithm/method configuration "
            "and any tuning performed to make baselines competitive.\n\n"
            "FAILURE-AWARE REPORTING REQUIREMENTS:\n"
            "15. If any method has a success rate < 100%, the Results section "
            "MUST report success rates per method and explain inclusion/exclusion "
            "criteria.\n"
            "16. Report BOTH conditional metrics (successful runs only) AND "
            "unconditional metrics (treating failures as worst-case). Without "
            "both, comparative claims are biased by survivorship.\n"
            "17. The Limitations section MUST discuss stability/reliability "
            "if any method showed NaN/divergence/crashes.\n\n"
            "BENCHMARK & ENVIRONMENT SPECIFICATION:\n"
            "18. The Experiments section MUST fully specify the evaluation "
            "environment: state/observation space, action space, hypothesis space, "
            "noise model, episode length, and any randomization procedures.\n"
            "19. Report results PER REGIME (e.g., per noise level, per problem "
            "size) with separate tables or sub-sections. Aggregated-only results "
            "cannot support claims about robustness or generality.\n"
            "20. Include a table comparing all methods across all regimes with "
            "paired statistical tests (bootstrap CI of paired differences, or "
            "paired t-test p-values). Without this, comparative claims lack "
            "statistical grounding.\n\n"
            "METHOD NAMING RULES:\n"
            "21. NEVER use generic labels like 'baseline_1', 'method_variant_1', "
            "'method_variant_2' in the paper. Use descriptive algorithm/method names "
            "that reflect what the method actually does. Generic labels make the paper "
            "scientifically uninterpretable.\n"
            "22. Each method MUST have a full description: architecture, "
            "training procedure, key hyperparameters, and implementation details. "
            "A reader should be able to reimplement every method from the paper alone.\n\n"
            "STATISTICAL REPORTING (MANDATORY for acceptance):\n"
            "23. EVERY result table MUST include 95% confidence intervals "
            "(mean +/- CI or [low, high]).\n"
            "24. EVERY comparison claim ('A outperforms B') MUST cite p-value. "
            "If p >= 0.05, write: 'The difference is not statistically significant.'\n"
            "25. If the proposed method does NOT statistically significantly "
            "outperform a baseline, do NOT claim superiority. Reframe as "
            "'comparable', 'competitive', or 'negative result'.\n\n"
            "WRITING STYLE RULES:\n"
            "26. DO NOT repeat disclaimers like 'due to computational constraints, "
            "this analysis was not conducted' more than once. State each limitation "
            "ONCE in the Limitations section.\n"
            "27. The Limitations section should be concise (200-400 words) listing "
            "3-5 key limitations. Do NOT scatter limitation disclaimers throughout "
            "every section.\n"
            "28. Focus 80% of the paper on WHAT YOU DID and WHAT YOU FOUND, not "
            "on what you could not do. Positive scientific contribution should "
            "dominate the paper.\n"
            "29. Cite 25-40 unique references in the paper body. The Related Work "
            "section alone should cite at least 15 references. Cite only directly "
            "relevant work — do NOT pad with tangentially related papers.\n"
            "30. CITE ORIGINAL PAPERS: When discussing a technique (e.g., Batch "
            "Normalization, ResNet, Adam, PPO), ALWAYS cite the original paper that "
            "introduced it. Do NOT cite a survey or follow-up instead of the original. "
            "The available references list includes foundational papers — use them.\n"
            "31. BASELINE MODERNITY: When discussing baselines and comparisons, ensure "
            "the paper acknowledges whether the baselines represent current practice. "
            "If baselines are older methods, explicitly discuss why they were chosen "
            "and acknowledge stronger modern alternatives exist."
        ),
        "user": (
            "{preamble}\n\n"
            "{academic_style_guide}\n"
            "{narrative_writing_rules}\n"
            "{anti_hedging_rules}\n"
            "{anti_repetition_rules}\n"
            "Write a full paper draft section by section in markdown.\n"
            "Required sections: Title, Abstract, Introduction, Related Work, "
            "Method, Experiments, Results, Discussion, Limitations, Broader Impact, "
            "Conclusion, References.\n"
            "The Broader Impact section (2-3 paragraphs) MUST discuss: "
            "(1) potential positive societal impacts of this work, "
            "(2) potential negative societal impacts or risks, "
            "(3) ethical considerations specific to this research area. "
            "This section is MANDATORY for top ML venues and recommended for all research papers.\n"
            "{writing_structure}\n"
            "{topic_constraint}"
            "{exp_metrics_instruction}"
            "{citation_instruction}"
            "All experimental results MUST be presented in LaTeX tables or inline prose. "
            "Raw metric path formats like 'method/env/step/metric: value' are FORBIDDEN "
            "in the paper text. Convert all data to clean, formatted presentation.\n"
            "The paper MUST fit within 10 pages (excluding references and appendix). "
            "Aim for 8-9 pages of main content. Be concise.\n"
            "FIGURE RULES: When referencing figures, use ONLY \\ref{fig:label} cross-references. "
            "NEVER add bold standalone paragraphs like '**Figure 1.**' after figure environments. "
            "Do NOT add \\clearpage before or after figures/tables unless absolutely necessary.\n"
            "TABLE RULES: Tables MUST use standard LaTeX tabular syntax with bare braces: "
            "\\begin{tabular}{lcc}, NOT \\begin{tabular}\\{lcc\\}. "
            "NEVER use '--' as placeholder values in table cells. "
            "If a metric is unavailable, write 'N/A' or omit the row entirely.\n"
            "Outline:\n{outline}"
        ),
        "max_tokens": 16384,
    },
    "peer_review": {
        "system": "You are a balanced conference reviewer.",
        "user": (
            "Simulate peer review from at least 3 reviewer perspectives.\n"
            "Output markdown with Reviewer A (methodology expert), "
            "Reviewer B (domain expert), and Reviewer C (statistics/rigor expert), "
            "each including strengths, weaknesses, and actionable revisions.\n\n"
            "Check specifically:\n"
            "1. TOPIC ALIGNMENT: Does the paper stay on topic ({topic})? "
            "Flag any sections where the paper drifts to unrelated topics or "
            "presents environment issues as contributions.\n"
            "2. CLAIM-EVIDENCE ALIGNMENT: For EACH claim in the title, abstract, "
            "and conclusion, verify there is a specific metric/table/figure in "
            "the Results section supporting it. Flag unsupported claims.\n"
            "3. STATISTICAL VALIDITY: Are confidence intervals or error bars "
            "reported? Is n>1 (multiple seeds)? Are significance tests appropriate?\n"
            "4. COMPLETENESS: Does the paper have all required sections with "
            "sufficient depth? A NeurIPS paper body should be 5,000-6,500 words.\n"
            "5. REPRODUCIBILITY: Are hyperparameters, random seeds, compute "
            "resources, and dataset details fully specified?\n"
            "6. WRITING QUALITY: Is the paper written in flowing prose or bullet lists? "
            "Flag any bullet-point lists in Method/Results/Discussion. Check for "
            "excessive hedging ('we do not claim'). Verify title is <= 14 words.\n"
            "7. FIGURES: Does the paper include at least 2 figures? Zero figures = desk reject.\n"
            "8. CITATION DISTRIBUTION: Are citations only in Intro/Related Work? "
            "Method, Experiments, and Discussion MUST also cite relevant papers.\n\n"
            "Paper draft:\n{draft}\n\n"
            "Experiment evidence for verification:\n{experiment_evidence}"
        ),
        "max_tokens": 8192,
    },
    "paper_revision": {
        "system": (
            "You are a paper revision expert.\n\n"
            "TITLE AND ABSTRACT ALIGNMENT (CRITICAL):\n"
            "- After reviewing experimental evidence, UPDATE the title if results "
            "do not support the original claim.\n"
            "- If the proposed method does NOT beat baselines, use a title like "
            "'An Empirical Study of...', 'When X Falls Short: ...', or "
            "'Investigating ... : Negative Results and Insights'.\n"
            "- Rewrite the abstract to accurately reflect what was FOUND, not "
            "what was hoped. The abstract must match actual numbers.\n"
            "- The conclusion MUST match actual results — no aspirational claims.\n\n"
            "IMPORTANT WRITING RULES:\n"
            "- Do NOT add disclaimers like 'due to computational constraints' "
            "or 'this analysis was not conducted'. If a limitation exists, "
            "mention it ONCE in the Limitations section only.\n"
            "- Focus 80% of the paper on what was DONE and what was FOUND.\n"
            "- Do NOT add hedging language that was not in the original draft.\n"
            "- Keep Limitations to 200-400 words with 3-5 concise points.\n"
            "- Ensure every comparison claim cites a p-value or states that "
            "the difference is not statistically significant.\n"
        ),
        "user": (
            "{academic_style_guide}\n"
            "{narrative_writing_rules}\n"
            "{anti_hedging_rules}\n"
            "{anti_repetition_rules}\n"
            "Revise the paper draft to address all review comments.\n"
            "Return revised markdown only.\n\n"
            "CRITICAL REVISION RULES:\n"
            "- Transform any remaining bullet-point lists in the body into flowing "
            "prose paragraphs. The only allowed lists are in the Introduction's contribution "
            "paragraph and the Limitations section.\n"
            "- The title MUST be <= 14 words with a catchy method name.\n"
            "- MANDATORY: The revised paper MUST contain at least 2 markdown image references\n"
            "  (![Caption](charts/...)). If the draft has zero figures, ADD them in the Results\n"
            "  section using the chart files. A paper with zero figures will be desk-rejected.\n"
            "- Consolidate ALL hedging/caveats into Limitations section only.\n"
            "- The final paper body MUST be <= 6,500 words (standard 9-page conference limit).\n"
            "  If the current draft exceeds this, compress by removing redundant restatements.\n"
            "- If the paper exceeds 10 pages, aggressively cut redundant content, "
            "merge similar sections, and tighten prose. Target 8-9 pages of main content.\n"
            "- Do NOT add '**Figure N.**' bold paragraphs after figure environments — "
            "use only \\ref{fig:label} cross-references. Do NOT add \\clearpage "
            "before figures or tables.\n"
            "- NEVER use '--' placeholder values in tables. Replace with actual values or 'N/A'.\n"
            "- CITATION FORMAT (CRITICAL): All citations MUST remain in [cite_key] bracket "
            "format exactly as they appear in the draft, e.g. [smith2024transformer]. "
            "Do NOT convert them to author-year format like [Smith et al., 2024] or "
            "(Smith et al., 2024). The downstream LaTeX converter relies on the "
            "[cite_key] format to generate \\cite{{}} commands. Changing the format "
            "will break all references in the final PDF.\n"
            "- CITATION KEYS (CRITICAL): Do NOT invent or add new citation keys that "
            "are not already present in the draft. If you want to reference additional "
            "prior work, describe it in prose WITHOUT a citation bracket. Every "
            "[cite_key] you write MUST already exist in the bibliography. Adding "
            "hallucinated keys like [smith2020method] creates broken [?] references "
            "in the final PDF.\n"
            "{writing_structure}\n"
            "{topic_constraint}"
            "Draft:\n{draft}\n\nReviews:\n{reviews}"
        ),
        "max_tokens": 16384,
    },
    # ── Phase H: Finalization ────────────────────────────────────────────
    "quality_gate": {
        "system": "You are a final quality gate evaluator.",
        "user": (
            "Evaluate revised paper quality and return JSON.\n"
            "Schema: {score_1_to_10:number, verdict:string, strengths:[...], "
            "weaknesses:[...], required_actions:[...]}.\n"
            "Threshold: {quality_threshold}\n"
            "Paper:\n{revised}"
        ),
        "json_mode": True,
    },
    "knowledge_archive": {
        "system": "You produce reproducibility-focused research retrospectives.",
        "user": (
            "{preamble}\n\n"
            "Write retrospective archive markdown with lessons, "
            "reproducibility notes, and future work.\n"
            "Decision:\n{decision}\n\nAnalysis:\n{analysis}\n\n"
            "Revised paper:\n{revised}"
        ),
        "max_tokens": 8192,
    },
    "export_publish": {
        "system": "You are a publication formatting editor.",
        "user": (
            "Format revised paper into clean final markdown for publication "
            "export.\n"
            "Preserve content quality and readability.\n"
            "CITATION FORMAT (CRITICAL): All citations MUST remain in [cite_key] bracket "
            "format, e.g. [smith2024transformer]. Do NOT convert to author-year "
            "format like [Smith et al., 2024]. The [cite_key] format is required "
            "for downstream LaTeX \\cite{{}} generation.\n"
            "Input paper:\n{revised}"
        ),
        "max_tokens": 16384,
    },
}
