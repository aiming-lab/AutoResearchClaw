"""Acquirer Agent — generates data loading code and download scripts.

Produces three outputs consumed by the code generation stage:
1. Data loading snippets (``get_datasets()`` function)
2. Baseline method snippets (model instantiation code)
3. ``setup.py`` additions for dataset downloading
"""

from __future__ import annotations

import logging
from typing import Any

from researchclaw.agents.base import AgentStepResult, BaseAgent

logger = logging.getLogger(__name__)


class AcquirerAgent(BaseAgent):
    """Generates data loading, baseline, and download code."""

    name = "acquirer"

    def _generate_data_loader(
        self,
        benchmarks: list[dict[str, Any]],
        topic: str,
    ) -> str:
        """Ask LLM to generate a robust data loading function."""
        bench_specs = []
        for b in benchmarks:
            spec = (
                f"- {b.get('name', 'Unknown')} (tier {b.get('tier', '?')}, "
                f"role: {b.get('role', 'secondary')})\n"
                f"  API: {b.get('api', 'N/A')}\n"
                f"  Metrics: {b.get('metrics', [])}\n"
                f"  Note: {b.get('note', '')}"
            )
            bench_specs.append(spec)

        system = (
            "You are an expert ML engineer. Generate a Python function that loads "
            "and prepares datasets for an ML experiment.\n\n"
            "REQUIREMENTS:\n"
            "- Function signature: def get_datasets(data_root='/workspace/data') -> dict\n"
            "- Returns dict with keys: 'train', 'val', 'test' (each a Dataset or DataLoader)\n"
            "- Include appropriate transforms (normalization, augmentation for training)\n"
            "- Handle both torchvision and HuggingFace datasets APIs\n"
            "- Include proper train/val/test splits\n"
            "- Add error handling with informative messages\n"
            "- For pre-cached datasets (tier 1), use download=False\n"
            "- For downloadable datasets (tier 2), use download=True in setup.py\n"
            "- Include a DATA_CONFIG dict with dataset metadata (num_classes, input_shape, etc.)\n\n"
            "Return ONLY the Python code, no explanation."
        )
        user = (
            f"Research Topic: {topic}\n\n"
            f"Datasets to load:\n" + "\n".join(bench_specs) + "\n\n"
            "Generate the data loading code."
        )
        return self._chat(system, user, max_tokens=4096, temperature=0.2)

    def _generate_baseline_code(
        self,
        baselines: list[dict[str, Any]],
        benchmarks: list[dict[str, Any]],
        topic: str,
    ) -> str:
        """Ask LLM to generate baseline method instantiation code."""
        base_specs = []
        for bl in baselines:
            spec = (
                f"- {bl.get('name', 'Unknown')}\n"
                f"  Source: {bl.get('source', 'N/A')}\n"
                f"  Paper: {bl.get('paper', 'N/A')}"
            )
            base_specs.append(spec)

        primary_bench = next(
            (b for b in benchmarks if b.get("role") == "primary"),
            benchmarks[0] if benchmarks else {},
        )

        system = (
            "You are an expert ML engineer. Generate Python code that instantiates "
            "baseline methods for comparison in an ML experiment.\n\n"
            "REQUIREMENTS:\n"
            "- Function signature: def get_baselines(num_classes, device='cuda') -> dict\n"
            "- Returns dict mapping method_name -> model (nn.Module)\n"
            "- Each model must be ready for training (correct output dimensions)\n"
            "- Use pretrained weights where available (for feature extractors)\n"
            "- Adapt final layer to match num_classes of the target dataset\n"
            "- Include a BASELINES_CONFIG dict with metadata (param_count, paper, etc.)\n"
            "- Handle missing optional packages gracefully\n\n"
            "Return ONLY the Python code, no explanation."
        )
        user = (
            f"Research Topic: {topic}\n"
            f"Primary Dataset: {primary_bench.get('name', 'N/A')} "
            f"({primary_bench.get('classes', '?')} classes)\n\n"
            f"Baseline Methods:\n" + "\n".join(base_specs) + "\n\n"
            "Generate the baseline instantiation code."
        )
        return self._chat(system, user, max_tokens=4096, temperature=0.2)

    def _generate_setup_script(
        self,
        benchmarks: list[dict[str, Any]],
        required_pip: list[str],
    ) -> str:
        """Generate setup.py content for dataset downloading."""
        # Tier 2 datasets need download scripts
        tier2 = [b for b in benchmarks if b.get("tier", 1) >= 2]

        if not tier2 and not required_pip:
            return ""

        lines = [
            '"""Setup script for dataset downloading and environment preparation.',
            '',
            'This script runs during Phase 1 (setup) of the Docker sandbox,',
            'when network access is available. It downloads datasets and installs',
            'any additional dependencies.',
            '"""',
            '',
            'import os',
            'import sys',
            '',
            'DATA_ROOT = "/workspace/data"',
            'HF_CACHE = os.path.join(DATA_ROOT, "hf")',
            '',
            '',
            'def download_datasets():',
            '    """Download all required datasets."""',
            '    os.makedirs(DATA_ROOT, exist_ok=True)',
            '    os.makedirs(HF_CACHE, exist_ok=True)',
            '',
        ]

        for b in tier2:
            api = b.get("api", "")
            name = b.get("name", "unknown")
            if "torchvision" in api:
                # Convert download=False to download=True for setup
                dl_api = api.replace("download=False", "download=True")
                lines.extend([
                    f'    # Download {name}',
                    '    try:',
                    f'        import torchvision',
                    f'        {dl_api}',
                    f'        print(f"Downloaded {name}")',
                    f'    except Exception as e:',
                    f'        print(f"Warning: Failed to download {name}: {{e}}")',
                    '',
                ])
            elif "datasets.load_dataset" in api or "load_dataset" in api:
                # Rewrite qualified `datasets.load_dataset(...)` to
                # `load_dataset(...)` so it matches the `from datasets import`
                _dl_api = api.replace("datasets.load_dataset", "load_dataset")
                lines.extend([
                    f'    # Download {name}',
                    '    try:',
                    f'        from datasets import load_dataset',
                    f'        {_dl_api}',
                    f'        print(f"Downloaded {name}")',
                    f'    except Exception as e:',
                    f'        print(f"Warning: Failed to download {name}: {{e}}")',
                    '',
                ])
            elif "PygNodePropPredDataset" in api or "PygGraphPropPredDataset" in api:
                lines.extend([
                    f'    # Download {name}',
                    '    try:',
                    f'        from ogb.nodeproppred import PygNodePropPredDataset' if 'Node' in api
                    else f'        from ogb.graphproppred import PygGraphPropPredDataset',
                    f'        {api}',
                    f'        print(f"Downloaded {name}")',
                    f'    except Exception as e:',
                    f'        print(f"Warning: Failed to download {name}: {{e}}")',
                    '',
                ])

        lines.extend([
            '',
            'if __name__ == "__main__":',
            '    download_datasets()',
            '    print("Setup complete.")',
        ])

        return "\n".join(lines)

    def _generate_requirements(self, required_pip: list[str]) -> str:
        """Generate requirements.txt content for additional packages."""
        if not required_pip:
            return ""
        # Filter out packages that are already in the Docker image
        builtin = {
            "torch", "torchvision", "torchaudio", "numpy", "scipy",
            "sklearn", "scikit-learn", "pandas", "matplotlib", "seaborn",
            "tqdm", "gymnasium", "networkx", "timm", "einops",
            "torchmetrics", "transformers", "datasets", "accelerate",
            "peft", "trl", "bitsandbytes", "tokenizers", "safetensors",
            "h5py", "tensorboard", "pillow", "pyyaml", "kornia",
            "albumentations",
        }
        extra = [p for p in required_pip if p.lower() not in builtin]
        return "\n".join(extra) if extra else ""

    # -- Code cleanup ------------------------------------------------------

    @staticmethod
    def _strip_fences(code: str) -> str:
        """Remove markdown code fences if present."""
        code = code.strip()
        if code.startswith("```"):
            # Remove opening fence
            first_nl = code.index("\n") if "\n" in code else len(code)
            code = code[first_nl + 1:]
        if code.endswith("```"):
            code = code[:-3].rstrip()
        return code

    # -- Main entry point --------------------------------------------------

    def execute(self, context: dict[str, Any]) -> AgentStepResult:
        """Generate data loading, baseline, and download code.

        Context keys:
            topic (str): Research topic
            selection (dict): Output from SelectorAgent
        """
        topic = context.get("topic", "")
        selection = context.get("selection", {})

        benchmarks = selection.get("selected_benchmarks", [])
        baselines = selection.get("selected_baselines", [])
        required_pip = selection.get("required_pip", [])

        if not benchmarks:
            return self._make_result(False, error="No benchmarks selected")

        # 1. Generate data loading code
        self.logger.info("Generating data loading code for %d datasets", len(benchmarks))
        data_loader_code = self._strip_fences(
            self._generate_data_loader(benchmarks, topic)
        )

        # 2. Generate baseline code
        baseline_code = ""
        if baselines:
            self.logger.info("Generating baseline code for %d methods", len(baselines))
            baseline_code = self._strip_fences(
                self._generate_baseline_code(baselines, benchmarks, topic)
            )

        # 3. Generate setup.py
        setup_code = self._generate_setup_script(benchmarks, required_pip)

        # 4. Generate requirements.txt
        requirements = self._generate_requirements(required_pip)

        result = {
            "data_loader_code": data_loader_code,
            "baseline_code": baseline_code,
            "setup_code": setup_code,
            "requirements": requirements,
            "benchmark_names": [b.get("name", "Unknown") for b in benchmarks],
            "baseline_names": [bl.get("name", "Unknown") for bl in baselines],
        }

        self.logger.info("Acquirer complete: %d code artifacts generated",
                         sum(1 for v in result.values() if v))

        return self._make_result(True, data=result)
