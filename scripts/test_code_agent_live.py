#!/usr/bin/env python3
"""Live test of CodeAgent with real LLM — evaluates code generation quality.

This script directly invokes the CodeAgent with real experiment plans
and evaluates the quality of generated code. No full pipeline needed.

Usage:
    python scripts/test_code_agent_live.py [--model gpt-4.1] [--test-id 1]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from researchclaw.llm.client import LLMClient, LLMConfig
from researchclaw.pipeline.code_agent import CodeAgent, CodeAgentConfig
from researchclaw.prompts import PromptManager

# ---------------------------------------------------------------------------
# Test cases — progressively harder experiment scenarios
# ---------------------------------------------------------------------------

TEST_CASES = {
    1: {
        "name": "Vision Transformer on CIFAR-10",
        "topic": (
            "Comparing Vision Transformer (ViT) variants for image classification: "
            "investigate how patch size, number of attention heads, and positional "
            "encoding strategies affect classification accuracy on CIFAR-10"
        ),
        "exp_plan": """
objectives:
  - Compare ViT-Tiny variants with different patch sizes (4, 8, 16)
  - Evaluate multi-head self-attention with different head counts (2, 4, 8)
  - Compare learnable vs sinusoidal positional encodings
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
  time_limit_sec: 300
  epochs: 10
""",
        "metric": "accuracy",
        "min_files": 2,
        "min_classes": 3,
        "required_imports": ["torch", "torchvision"],
    },
    2: {
        "name": "Distribution Shift Detection via Uncertainty",
        "topic": (
            "Detecting distribution shift in deployed ML models using "
            "uncertainty estimation: comparing Monte Carlo Dropout, "
            "Deep Ensembles, and Spectral-Normalized Neural GP (SNGP) "
            "for out-of-distribution detection on corrupted CIFAR-10"
        ),
        "exp_plan": """
objectives:
  - Implement 3 uncertainty estimation methods for OOD detection
  - Evaluate on CIFAR-10 vs CIFAR-10-C (corrupted) as OOD
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
    description: Monte Carlo Dropout with 30 forward passes, mean+std of softmax
    implementation_spec:
      class_name: MCDropoutDetector
      key_methods: [predict_with_uncertainty, _mc_forward, compute_auroc]
      differentiator: Standard MC Dropout baseline
proposed_methods:
  - name: DeepEnsemble
    implementation_spec:
      class_name: DeepEnsembleDetector
      key_methods: [train_ensemble, predict_with_uncertainty, _member_forward]
      differentiator: Trains 3 independent models, uses prediction disagreement
  - name: SNGP
    implementation_spec:
      class_name: SNGPDetector
      key_methods: [forward, _spectral_norm_layer, _gp_output_layer]
      differentiator: Spectral normalization + GP output layer for distance-aware uncertainty
ablations:
  - name: MCDropout-10passes
    description: MC Dropout with only 10 forward passes (reduced compute)
metrics:
  - auroc (higher is better)
  - ece (expected calibration error, lower is better)
compute_budget:
  time_limit_sec: 300
  epochs: 5
""",
        "metric": "auroc",
        "min_files": 2,
        "min_classes": 4,
        "required_imports": ["torch", "numpy"],
    },
    3: {
        "name": "Meta-Learning Few-Shot with MAML",
        "topic": (
            "Few-shot learning with gradient-based meta-learning: comparing "
            "MAML, Reptile, and Prototypical Networks on Omniglot-style "
            "synthetic tasks with 5-way 1-shot and 5-way 5-shot settings"
        ),
        "exp_plan": """
objectives:
  - Implement 3 few-shot learning algorithms from scratch
  - Evaluate on synthetic few-shot tasks (5-way, 1-shot and 5-shot)
  - Compare accuracy and convergence speed
datasets:
  - name: SyntheticFewShot
    source: Generated in-code (random linear classification tasks)
    n_classes: 20
    samples_per_class: 20
baselines:
  - name: ProtoNet
    description: Prototypical Networks — learn embedding, classify by nearest class prototype
    implementation_spec:
      class_name: PrototypicalNetwork
      key_methods: [embed, compute_prototypes, classify, meta_train_step]
      differentiator: Non-gradient meta-learning baseline using metric space
proposed_methods:
  - name: MAML
    implementation_spec:
      class_name: MAMLLearner
      key_methods: [inner_loop, outer_loop, meta_train_step, adapt]
      differentiator: Second-order gradient-based meta-learning with inner loop adaptation
  - name: Reptile
    implementation_spec:
      class_name: ReptileLearner
      key_methods: [inner_loop, meta_update, meta_train_step]
      differentiator: First-order approximation — SGD on tasks, move toward task-optimal weights
ablations:
  - name: MAML-FirstOrder
    description: MAML with first-order approximation (no second derivatives)
metrics:
  - accuracy (higher is better)
  - meta_train_loss
compute_budget:
  time_limit_sec: 300
  meta_epochs: 200
  inner_steps: 5
  inner_lr: 0.01
""",
        "metric": "accuracy",
        "min_files": 2,
        "min_classes": 3,
        "required_imports": ["torch"],
    },
}


# ---------------------------------------------------------------------------
# Code quality analysis
# ---------------------------------------------------------------------------


def analyze_code_quality(files: dict[str, str], test_case: dict) -> dict:
    """Analyze the quality of generated code."""
    report = {
        "test_name": test_case["name"],
        "num_files": len(files),
        "file_names": list(files.keys()),
        "total_lines": 0,
        "effective_lines": 0,
        "classes_found": [],
        "functions_found": [],
        "imports_found": [],
        "issues": [],
        "scores": {},
    }

    all_code = ""
    for fname, code in files.items():
        all_code += code + "\n"
        lines = code.split("\n")
        report["total_lines"] += len(lines)
        effective = [
            l for l in lines
            if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("import") and not l.strip().startswith("from")
        ]
        report["effective_lines"] += len(effective)

        # AST analysis
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    method_lines = sum(
                        n.end_lineno - n.lineno + 1
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.end_lineno
                    )
                    report["classes_found"].append({
                        "name": node.name,
                        "file": fname,
                        "methods": methods,
                        "method_count": len(methods),
                        "total_method_lines": method_lines,
                    })
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    report["functions_found"].append({
                        "name": node.name,
                        "file": fname,
                        "lines": (node.end_lineno or node.lineno) - node.lineno + 1,
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            report["imports_found"].append(alias.name.split(".")[0])
                    else:
                        if node.module:
                            report["imports_found"].append(node.module.split(".")[0])
        except SyntaxError as e:
            report["issues"].append(f"SyntaxError in {fname}: {e}")

    report["imports_found"] = sorted(set(report["imports_found"]))

    # Scoring
    # 1. File count (target: min_files)
    file_score = min(10, (len(files) / test_case["min_files"]) * 10)
    report["scores"]["file_structure"] = round(file_score, 1)

    # 2. Class count (target: min_classes)
    class_score = min(10, (len(report["classes_found"]) / test_case["min_classes"]) * 10)
    report["scores"]["class_coverage"] = round(class_score, 1)

    # 3. Code depth (effective lines)
    depth_score = min(10, report["effective_lines"] / 30)  # 300 lines = 10
    report["scores"]["code_depth"] = round(depth_score, 1)

    # 4. Method richness (average methods per class)
    if report["classes_found"]:
        avg_methods = sum(c["method_count"] for c in report["classes_found"]) / len(report["classes_found"])
        method_score = min(10, avg_methods / 0.5)  # 5 methods/class = 10
        report["scores"]["method_richness"] = round(method_score, 1)
    else:
        report["scores"]["method_richness"] = 0

    # 5. Import coverage
    required = set(test_case.get("required_imports", []))
    found = set(report["imports_found"])
    if required:
        import_score = len(required & found) / len(required) * 10
    else:
        import_score = 10
    report["scores"]["import_coverage"] = round(import_score, 1)

    # 6. Syntax validity
    syntax_score = 10 if not any("SyntaxError" in i for i in report["issues"]) else 0
    report["scores"]["syntax_valid"] = syntax_score

    # Overall score
    scores = report["scores"]
    report["overall_score"] = round(
        sum(scores.values()) / len(scores), 1
    )

    # Quality checks
    if len(files) < test_case["min_files"]:
        report["issues"].append(
            f"Too few files: {len(files)} < {test_case['min_files']}"
        )
    if len(report["classes_found"]) < test_case["min_classes"]:
        report["issues"].append(
            f"Too few classes: {len(report['classes_found'])} < {test_case['min_classes']}"
        )
    for cls in report["classes_found"]:
        if cls["total_method_lines"] < 10:
            report["issues"].append(
                f"Class {cls['name']} has only {cls['total_method_lines']} method lines (too thin)"
            )

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Live test CodeAgent quality")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use")
    parser.add_argument("--test-id", type=int, default=0, help="Test case ID (0=all)")
    parser.add_argument("--no-sandbox", action="store_true", help="Skip sandbox exec-fix")
    parser.add_argument("--tree-search", action="store_true", help="Enable tree search")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory")
    args = parser.parse_args()

    # Setup LLM client
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url or not api_key:
        print("ERROR: Set OPENAI_BASE_URL and OPENAI_API_KEY environment variables")
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

    # Quick connectivity test
    print(f"Testing LLM connectivity ({args.model})... ", end="", flush=True)
    ok, msg = llm.preflight()
    if not ok:
        print(f"FAILED: {msg}")
        sys.exit(1)
    print("OK")

    pm = PromptManager()

    # Select test cases
    if args.test_id > 0:
        if args.test_id not in TEST_CASES:
            print(f"ERROR: Unknown test ID {args.test_id}. Available: {list(TEST_CASES.keys())}")
            sys.exit(1)
        cases = {args.test_id: TEST_CASES[args.test_id]}
    else:
        cases = TEST_CASES

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []

    for test_id, tc in cases.items():
        print(f"\n{'='*60}")
        print(f"Test {test_id}: {tc['name']}")
        print(f"{'='*60}")

        stage_dir = output_dir / f"test_{test_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        config = CodeAgentConfig(
            architecture_planning=True,
            exec_fix_max_iterations=0 if args.no_sandbox else 3,
            tree_search_enabled=args.tree_search,
            review_max_rounds=2,
        )

        agent = CodeAgent(
            llm=llm,
            prompts=pm,
            config=config,
            stage_dir=stage_dir,
            sandbox_factory=None,  # No sandbox for quick test
        )

        t0 = time.time()
        result = agent.generate(
            topic=tc["topic"],
            exp_plan=tc["exp_plan"],
            metric=tc["metric"],
            pkg_hint=(
                "\nAVAILABLE PACKAGES (docker mode): Python stdlib, numpy, "
                "torch, torchvision, sklearn, scipy, pandas, matplotlib.\n"
                "GPU: NVIDIA RTX 6000 Ada (49GB VRAM). "
                "Use `device = torch.device('cuda')` for tensor operations.\n"
            ),
            max_tokens=16384,
        )
        elapsed = time.time() - t0

        print(f"\nGeneration time: {elapsed:.1f}s")
        print(f"LLM calls: {result.total_llm_calls}")
        print(f"Review rounds: {result.review_rounds}")
        print(f"Architecture spec: {len(result.architecture_spec)} chars")

        # Write generated files
        for fname, code in result.files.items():
            fpath = stage_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")
            lines = len(code.split("\n"))
            print(f"  {fname}: {lines} lines")

        # Write architecture spec
        if result.architecture_spec:
            (stage_dir / "architecture_spec.yaml").write_text(
                result.architecture_spec, encoding="utf-8"
            )

        # Analyze quality
        report = analyze_code_quality(result.files, tc)
        report["generation_time_sec"] = round(elapsed, 1)
        report["llm_calls"] = result.total_llm_calls
        report["review_rounds"] = result.review_rounds
        report["architecture_spec_chars"] = len(result.architecture_spec)

        # Print report
        print(f"\n--- Quality Report ---")
        print(f"Files: {report['num_files']}")
        print(f"Total lines: {report['total_lines']}")
        print(f"Effective lines: {report['effective_lines']}")
        print(f"Classes: {len(report['classes_found'])}")
        for cls in report["classes_found"]:
            print(f"  - {cls['name']} ({cls['method_count']} methods, {cls['total_method_lines']} lines)")
        print(f"Imports: {', '.join(report['imports_found'])}")
        print(f"\nScores:")
        for k, v in report["scores"].items():
            print(f"  {k}: {v}/10")
        print(f"  OVERALL: {report['overall_score']}/10")
        if report["issues"]:
            print(f"\nIssues:")
            for issue in report["issues"]:
                print(f"  - {issue}")

        # Save report
        (stage_dir / "quality_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        all_reports.append(report)

    # Summary
    if len(all_reports) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in all_reports:
            print(f"  {r['test_name']}: {r['overall_score']}/10 "
                  f"({r['effective_lines']} lines, {len(r['classes_found'])} classes)")
        avg = sum(r["overall_score"] for r in all_reports) / len(all_reports)
        print(f"\n  Average: {avg:.1f}/10")

    # Save all reports
    (output_dir / "all_reports.json").write_text(
        json.dumps(all_reports, indent=2), encoding="utf-8"
    )
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
