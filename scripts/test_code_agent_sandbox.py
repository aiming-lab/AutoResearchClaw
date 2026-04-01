#!/usr/bin/env python3
"""Test CodeAgent with Docker sandbox exec-fix loop.

Generates code with Phase 1-4 (architecture, exec-fix, review),
runs in Docker sandbox, verifies the exec-fix loop catches and fixes errors.

Usage:
    python scripts/test_code_agent_sandbox.py [--model gpt-5.1] [--test-id 1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from researchclaw.config import DockerSandboxConfig, ExperimentConfig
from researchclaw.experiment.docker_sandbox import DockerSandbox
from researchclaw.llm.client import LLMClient, LLMConfig
from researchclaw.pipeline.code_agent import CodeAgent, CodeAgentConfig
from researchclaw.prompts import PromptManager

# ---------------------------------------------------------------------------
# Test case (simple — should run quickly in sandbox)
# ---------------------------------------------------------------------------

TEST_CASES = {
    1: {
        "name": "ViT on CIFAR-10 (sandbox)",
        "topic": (
            "Comparing Vision Transformer (ViT) variants for image classification: "
            "investigate how patch size and number of attention heads affect "
            "classification accuracy on CIFAR-10"
        ),
        "exp_plan": """
objectives:
  - Compare ViT-Tiny variants with different patch sizes (4, 16)
  - Evaluate multi-head self-attention with different head counts (4, 8)
datasets:
  - name: CIFAR-10
    source: torchvision.datasets.CIFAR10
    train_size: 50000
    test_size: 10000
baselines:
  - name: SimpleViT-P16
    description: Standard ViT with patch_size=16, 4 heads, learnable pos encoding
proposed_methods:
  - name: SmallPatch-ViT
    implementation_spec:
      class_name: SmallPatchViT
      key_methods: [forward, _create_patches, _attention]
      differentiator: Uses patch_size=4 for finer-grained spatial features
  - name: ManyHead-ViT
    implementation_spec:
      class_name: ManyHeadViT
      key_methods: [forward, _multi_head_attention]
      differentiator: Uses 8 attention heads instead of 4
ablations:
  - name: SinusoidalPos-ViT
    description: Replace learnable positional encoding with sinusoidal
metrics:
  - accuracy (higher is better)
  - training_loss
compute_budget:
  time_limit_sec: 120
  epochs: 3
""",
        "metric": "accuracy",
    },
    2: {
        "name": "OOD Detection (sandbox)",
        "topic": (
            "Detecting distribution shift using uncertainty estimation: "
            "comparing Monte Carlo Dropout and Deep Ensembles "
            "for out-of-distribution detection on corrupted CIFAR-10"
        ),
        "exp_plan": """
objectives:
  - Implement 2 uncertainty estimation methods for OOD detection
  - Evaluate on CIFAR-10 vs Gaussian noise corruption as OOD
  - Compare AUROC for separating in-distribution from OOD samples
datasets:
  - name: CIFAR-10
    source: torchvision.datasets.CIFAR10
    role: in-distribution
  - name: CIFAR-10-C
    source: Generated via Gaussian noise corruption
    role: out-of-distribution
baselines:
  - name: MCDropout
    description: Monte Carlo Dropout with 20 forward passes
    implementation_spec:
      class_name: MCDropoutDetector
      key_methods: [predict_with_uncertainty, _mc_forward, compute_auroc]
proposed_methods:
  - name: DeepEnsemble
    implementation_spec:
      class_name: DeepEnsembleDetector
      key_methods: [train_ensemble, predict_with_uncertainty]
      differentiator: Trains 3 independent models, uses prediction disagreement
ablations:
  - name: MCDropout-5passes
    description: MC Dropout with only 5 forward passes
metrics:
  - auroc (higher is better)
compute_budget:
  time_limit_sec: 120
  epochs: 3
""",
        "metric": "auroc",
    },
}


def make_sandbox_factory(docker_cfg: DockerSandboxConfig):
    """Return a factory function that creates DockerSandbox instances."""
    def factory(exp_config, workdir: Path):
        return DockerSandbox(docker_cfg, workdir)
    return factory


def main():
    parser = argparse.ArgumentParser(description="Test CodeAgent with Docker sandbox")
    parser.add_argument("--model", default="gpt-5.1", help="Model to use")
    parser.add_argument("--test-id", type=int, default=1, help="Test case ID")
    parser.add_argument("--output-dir", default="test_outputs_sandbox", help="Output dir")
    parser.add_argument("--exec-fix-iters", type=int, default=3, help="Max exec-fix iterations")
    parser.add_argument("--timeout", type=int, default=180, help="Sandbox timeout (sec)")
    args = parser.parse_args()

    # Setup LLM
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url or not api_key:
        print("ERROR: Set OPENAI_BASE_URL and OPENAI_API_KEY")
        sys.exit(1)

    llm_config = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        primary_model=args.model,
        fallback_models=[],
        max_tokens=16384,
        temperature=0.7,
        timeout_sec=300,
    )
    llm = LLMClient(llm_config)

    print(f"Testing LLM connectivity ({args.model})... ", end="", flush=True)
    ok, msg = llm.preflight()
    if not ok:
        print(f"FAILED: {msg}")
        sys.exit(1)
    print("OK")

    # Docker sandbox setup
    docker_cfg = DockerSandboxConfig(
        image="researchclaw/experiment:latest",
        gpu_enabled=True,
        memory_limit_mb=16384,
        network_policy="setup_only",
    )

    if not DockerSandbox.check_docker_available():
        print("ERROR: Docker not available")
        sys.exit(1)
    if not DockerSandbox.ensure_image(docker_cfg.image):
        print(f"ERROR: Docker image {docker_cfg.image} not found")
        sys.exit(1)
    print(f"Docker sandbox ready: {docker_cfg.image}")

    # Select test case
    tc = TEST_CASES.get(args.test_id)
    if not tc:
        print(f"ERROR: Unknown test ID {args.test_id}")
        sys.exit(1)

    pm = PromptManager()
    output_dir = Path(args.output_dir)
    stage_dir = output_dir / f"test_{args.test_id}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # CodeAgent with sandbox enabled
    config = CodeAgentConfig(
        architecture_planning=True,
        exec_fix_max_iterations=args.exec_fix_iters,
        exec_fix_timeout_sec=args.timeout,
        tree_search_enabled=False,
        review_max_rounds=2,
    )

    sandbox_factory = make_sandbox_factory(docker_cfg)

    agent = CodeAgent(
        llm=llm,
        prompts=pm,
        config=config,
        stage_dir=stage_dir,
        sandbox_factory=sandbox_factory,
    )

    print(f"\n{'='*60}")
    print(f"Test {args.test_id}: {tc['name']}")
    print(f"  exec_fix_max_iterations: {args.exec_fix_iters}")
    print(f"  sandbox_timeout: {args.timeout}s")
    print(f"{'='*60}")

    t0 = time.time()
    result = agent.generate(
        topic=tc["topic"],
        exp_plan=tc["exp_plan"],
        metric=tc["metric"],
        pkg_hint=(
            "\nAVAILABLE PACKAGES (docker mode): Python stdlib, numpy, "
            "torch, torchvision, sklearn, scipy, pandas, matplotlib, "
            "tqdm, timm, einops, torchmetrics, gymnasium, networkx.\n"
            "GPU: NVIDIA RTX 6000 Ada (49GB VRAM). "
            "Use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` "
            "for tensor operations.\n"
            "DATA PATH: CIFAR-10 is pre-cached at /opt/datasets/cifar-10-batches-py/. "
            "Use `torchvision.datasets.CIFAR10(root='/opt/datasets', download=False)`.\n"
        ),
        max_tokens=16384,
    )
    elapsed = time.time() - t0

    # Report
    print(f"\n--- Generation Report ---")
    print(f"Time: {elapsed:.1f}s")
    print(f"LLM calls: {result.total_llm_calls}")
    print(f"Sandbox runs: {result.total_sandbox_runs}")
    print(f"Review rounds: {result.review_rounds}")
    print(f"Best score: {result.best_score}")

    # Write files
    for fname, code in result.files.items():
        fpath = stage_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(code, encoding="utf-8")
        lines = len(code.split("\n"))
        print(f"  {fname}: {lines} lines")

    # Write arch spec
    if result.architecture_spec:
        (stage_dir / "architecture_spec.yaml").write_text(
            result.architecture_spec, encoding="utf-8"
        )

    # Write validation log
    (stage_dir / "validation_log.json").write_text(
        json.dumps({
            "log": result.validation_log,
            "total_llm_calls": result.total_llm_calls,
            "total_sandbox_runs": result.total_sandbox_runs,
            "review_rounds": result.review_rounds,
            "best_score": result.best_score,
            "elapsed_sec": round(elapsed, 1),
        }, indent=2),
        encoding="utf-8",
    )

    # Final sandbox run for end-to-end verification
    print(f"\n--- Final sandbox verification ---")
    workdir = stage_dir / "_final_run"
    workdir.mkdir(parents=True, exist_ok=True)
    sandbox = DockerSandbox(docker_cfg, workdir)
    final_result = sandbox.run_project(
        stage_dir, entry_point="main.py", timeout_sec=args.timeout,
    )
    print(f"Return code: {final_result.returncode}")
    print(f"Elapsed: {final_result.elapsed_sec:.1f}s")
    print(f"Timed out: {final_result.timed_out}")
    if final_result.metrics:
        print(f"Metrics: {json.dumps(dict(final_result.metrics), indent=2)}")
    if final_result.returncode != 0:
        print(f"STDERR (last 500):\n{final_result.stderr[-500:]}")
    else:
        print("SUCCESS: Code runs to completion in Docker sandbox!")
        stdout_lines = final_result.stdout.strip().split("\n")
        print(f"STDOUT (last 10 lines):")
        for line in stdout_lines[-10:]:
            print(f"  {line}")

    # Save final run results
    (stage_dir / "final_run_result.json").write_text(
        json.dumps({
            "returncode": final_result.returncode,
            "elapsed_sec": final_result.elapsed_sec,
            "timed_out": final_result.timed_out,
            "metrics": dict(final_result.metrics) if final_result.metrics else {},
            "stdout_tail": "\n".join(stdout_lines[-20:]) if final_result.returncode == 0 else "",
            "stderr_tail": final_result.stderr[-1000:] if final_result.returncode != 0 else "",
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
