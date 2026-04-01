#!/usr/bin/env python3
"""End-to-end integration test for OpenCode Beast Mode.

Simulates Pipeline stages 1-9 artifacts, then invokes Beast Mode
to generate experiment code via OpenCode CLI.

Usage:
    python scripts/test_beast_mode_e2e.py
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from researchclaw.pipeline.opencode_bridge import (
    OpenCodeBridge,
    count_historical_failures,
    score_complexity,
)

# ============================================================
# Simulated Pipeline Artifacts
# ============================================================

TOPIC = (
    "Adaptive Mixtures of Local Experts for Image Classification: "
    "Dynamic Gating with Load-Balanced Sparse Routing on CIFAR-10"
)

# Simulated Stage 9 output: exp_plan.yaml content
EXP_PLAN = textwrap.dedent("""\
    topic: >
      Adaptive Mixtures of Local Experts for Image Classification:
      Dynamic Gating with Load-Balanced Sparse Routing on CIFAR-10

    objectives:
      - Investigate whether sparse Mixture-of-Experts (MoE) routing improves
        accuracy over dense baselines under a fixed parameter budget
      - Compare top-k routing vs soft routing vs hash-based routing
      - Ablate the load-balancing auxiliary loss

    datasets:
      - CIFAR-10 (pre-cached at /opt/datasets/cifar10)

    baselines:
      - name: dense_resnet18
        description: Standard ResNet-18 with all parameters active
        implementation_spec:
          class_name: DenseResNet18Trainer
          key_hyperparameters:
            batch_size: 128
            learning_rate: 0.1
            epochs: 20
            weight_decay: 5e-4

      - name: dense_wider_resnet
        description: Wider ResNet with ~same FLOPs as MoE model
        implementation_spec:
          class_name: DenseWiderResNetTrainer
          key_hyperparameters:
            batch_size: 128
            learning_rate: 0.1
            epochs: 20

    proposed_methods:
      - name: topk_sparse_moe
        description: >
          Sparse MoE with top-2 gating. Each MoE layer has 4 expert MLPs,
          a gating network selects top-2 per token. Load-balancing loss
          ensures even expert utilization.
        implementation_spec:
          class_name: TopKSparseMoETrainer
          algorithm_steps:
            - Build backbone CNN (first 3 ResNet blocks)
            - Replace final block with MoE layer (4 experts, top-2 gating)
            - Gating network: linear projection → softmax → top-k selection
            - Load-balance loss: CV of expert load across batch
            - Total loss = CE + lambda_lb * load_balance_loss
          key_hyperparameters:
            batch_size: 128
            learning_rate: 0.05
            epochs: 20
            num_experts: 4
            top_k: 2
            lambda_lb: 0.01

      - name: soft_routing_moe
        description: >
          Soft MoE where all experts contribute with learned weights
          (no hard top-k). Softer gradient flow but higher compute.
        implementation_spec:
          class_name: SoftRoutingMoETrainer
          key_hyperparameters:
            batch_size: 128
            learning_rate: 0.05
            epochs: 20
            num_experts: 4

    ablations:
      - name: topk_moe_no_load_balance
        description: TopK MoE without load-balancing loss (lambda_lb=0)
        what_is_removed: Load-balancing auxiliary loss
        expected_effect: Expert collapse — one expert dominates, accuracy drops
        how_it_differs:
          - Set lambda_lb = 0
          - Everything else identical to topk_sparse_moe

      - name: topk_moe_single_expert
        description: TopK MoE with top_k=1 (only one expert per sample)
        what_is_removed: Multi-expert routing (reduced to single expert)
        expected_effect: Reduced model capacity per sample, likely lower accuracy
        how_it_differs:
          - Set top_k = 1 instead of 2
          - Keep load-balancing loss active

    metrics:
      primary_metric:
        name: test_accuracy
        direction: maximize
        description: Classification accuracy on CIFAR-10 test set
      secondary_metrics:
        - name: expert_utilization_cv
          description: Coefficient of variation of expert usage (lower = more balanced)
        - name: training_time_sec
          description: Wall-clock training time

    compute_budget:
      effective_time_seconds: 240
      estimated_seconds_per_run: 40
      seeds_per_condition: 3
      total_conditions: 6
      notes:
        - Use small models (< 5M params) to fit within budget
        - Use 20 epochs max
        - Early stopping if no improvement for 5 epochs
""")

PKG_HINT = textwrap.dedent("""\
    AVAILABLE PACKAGES (docker mode): Python stdlib, numpy, torch, sklearn, scipy, pandas,
    torchvision, torchaudio, matplotlib, seaborn, scipy, tqdm, transformers, datasets,
    timm, einops, torchmetrics, and additional pip-installable packages via requirements.txt.
    GPU: NVIDIA RTX 6000 Ada (cuda). You MAY use PyTorch with GPU acceleration.
    Use `device = torch.device('cuda')` for tensor operations.

    ## Compute Budget Constraint
    - Total execution time limit: 240 seconds
    - Design experiments that complete within this budget
    - Implement a time guard: stop gracefully at 80% of budget (192 seconds)
""")

EXTRA_GUIDANCE = textwrap.dedent("""\
    ## Dataset Guidance
    CIFAR-10 is pre-cached at /opt/datasets/cifar10.
    Use: torchvision.datasets.CIFAR10(root='/opt/datasets/cifar10', download=False)

    ## Multi-Seed Enforcement
    Run each condition with seeds [0, 1, 2]. Report mean ± std for all metrics.

    ## Hyperparameter Reporting
    Print all hyperparameters at the start of each condition run.
""")


def main() -> None:
    print("=" * 70)
    print("OpenCode Beast Mode — End-to-End Integration Test")
    print("=" * 70)

    # Step 1: Complexity scoring
    print("\n[Step 1] Complexity scoring...")
    cplx = score_complexity(
        exp_plan=EXP_PLAN,
        topic=TOPIC,
        historical_failures=0,
        threshold=0.4,  # Lower threshold to ensure trigger for this test
    )
    print(f"  Score:  {cplx.score:.4f}")
    print(f"  Signals: {json.dumps(cplx.signals, indent=4)}")
    print(f"  Recommendation: {cplx.recommendation}")
    print(f"  Reason: {cplx.reason}")

    if cplx.recommendation != "beast_mode":
        print("\n  [!] Score below threshold. Forcing beast mode for test purposes.\n")

    # Step 2: Check OpenCode availability
    print("\n[Step 2] Checking OpenCode availability...")
    available = OpenCodeBridge.check_available()
    if not available:
        print("  [FATAL] OpenCode CLI not installed. Cannot proceed.")
        sys.exit(1)
    print("  OpenCode CLI: OK")

    # Step 3: Create test workspace and invoke
    print("\n[Step 3] Invoking OpenCode beast mode...")
    test_dir = PROJECT_ROOT / "test_outputs_beast_mode"
    test_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = test_dir / f"stage-10_{int(time.time())}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Write complexity analysis
    (stage_dir / "complexity_analysis.json").write_text(
        json.dumps({
            "score": cplx.score,
            "signals": cplx.signals,
            "recommendation": cplx.recommendation,
            "reason": cplx.reason,
        }, indent=2),
        encoding="utf-8",
    )

    # NOTE: Azure AI Services endpoints don't support OpenCode's Responses API.
    # The bridge auto-detects Azure and falls back to Anthropic provider.
    bridge = OpenCodeBridge(
        model="anthropic/claude-sonnet-4-6",  # Direct Anthropic model
        llm_base_url="https://huaxi-mlg4x1rk-eastus2.services.ai.azure.com/openai/v1",
        api_key_env="AZURE_OPENAI_API_KEY",
        llm_provider="azure",
        timeout_sec=300,
        max_retries=1,
        workspace_cleanup=False,  # Keep workspace for inspection
    )

    t0 = time.time()
    result = bridge.generate(
        stage_dir=stage_dir,
        topic=TOPIC,
        exp_plan=EXP_PLAN,
        metric="test_accuracy",
        pkg_hint=PKG_HINT,
        extra_guidance=EXTRA_GUIDANCE,
        time_budget_sec=240,
    )
    elapsed = time.time() - t0

    # Step 4: Evaluate results
    print(f"\n[Step 4] Results (elapsed: {elapsed:.1f}s)")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error or 'None'}")
    print(f"  Files: {list(result.files.keys())}")
    print(f"  OpenCode elapsed: {result.elapsed_sec:.1f}s")

    if not result.success:
        print(f"\n  [FAILED] Beast mode failed: {result.error}")
        print(f"  Log (last 1000 chars):\n{result.opencode_log[-1000:]}")
        # Write log for debugging
        (stage_dir / "opencode_log.txt").write_text(
            result.opencode_log, encoding="utf-8",
        )
        (stage_dir / "beast_mode_log.json").write_text(
            json.dumps({
                "success": False,
                "error": result.error,
                "elapsed_sec": result.elapsed_sec,
            }, indent=2),
            encoding="utf-8",
        )
        sys.exit(1)

    # Write generated files
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for fname, code in result.files.items():
        fpath = exp_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(code, encoding="utf-8")
    print(f"\n  Files written to: {exp_dir}")

    # Write beast mode log
    (stage_dir / "beast_mode_log.json").write_text(
        json.dumps({
            "success": True,
            "elapsed_sec": result.elapsed_sec,
            "files": list(result.files.keys()),
        }, indent=2),
        encoding="utf-8",
    )

    # Step 5: Quality evaluation
    print("\n[Step 5] Quality evaluation...")
    checks = {
        "main.py exists": "main.py" in result.files,
        "main.py is non-empty": len(result.files.get("main.py", "")) > 100,
        "Has metric print": "test_accuracy" in result.files.get("main.py", ""),
        "Has seed loop": "seed" in result.files.get("main.py", "").lower(),
        "Has CIFAR-10": "cifar" in result.files.get("main.py", "").lower(),
        "Has torch import": "import torch" in result.files.get("main.py", ""),
        "No argparse": "argparse" not in result.files.get("main.py", ""),
        "Has multiple conditions": any(
            kw in result.files.get("main.py", "").lower()
            for kw in ["baseline", "dense", "moe", "expert", "condition"]
        ),
        "Has time guard": any(
            kw in result.files.get("main.py", "")
            for kw in ["time.time", "time.monotonic", "time_budget", "time_limit"]
        ),
    }

    all_pass = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check_name}")

    # Count lines of code
    total_loc = sum(len(code.splitlines()) for code in result.files.values())
    py_files = [f for f in result.files if f.endswith(".py")]
    print(f"\n  Total files: {len(result.files)}")
    print(f"  Python files: {len(py_files)}")
    print(f"  Total lines of code: {total_loc}")

    # Try AST parsing main.py
    import ast
    try:
        ast.parse(result.files["main.py"])
        print("  [PASS] main.py AST parse: valid Python")
    except SyntaxError as e:
        print(f"  [FAIL] main.py AST parse error: {e}")
        all_pass = False

    # Print first 50 lines of main.py for manual inspection
    main_lines = result.files.get("main.py", "").splitlines()
    print(f"\n  --- main.py preview (first 50 of {len(main_lines)} lines) ---")
    for i, line in enumerate(main_lines[:50], 1):
        print(f"  {i:4d} | {line}")
    if len(main_lines) > 50:
        print(f"  ... ({len(main_lines) - 50} more lines)")

    # Final verdict
    print("\n" + "=" * 70)
    pass_count = sum(1 for v in checks.values() if v)
    total = len(checks)
    if all_pass:
        print(f"VERDICT: ALL CHECKS PASSED ({pass_count}/{total})")
    else:
        print(f"VERDICT: {pass_count}/{total} checks passed")
    print(f"Stage dir: {stage_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
