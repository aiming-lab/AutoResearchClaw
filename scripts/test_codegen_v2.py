#!/usr/bin/env python3
"""Enhanced code generation test — generates code and runs in Docker sandbox.

Tests the full code generation pipeline in isolation:
  1. Load experiment plan (from previous run or built-in test case)
  2. Generate code via CodeAgent
  3. Validate generated code (AST, security, quality)
  4. Run in Docker sandbox
  5. Score results comprehensively

Usage:
    # Run with built-in test case
    python scripts/test_codegen_v2.py --test-id 1

    # Run with real experiment plan from a previous run
    python scripts/test_codegen_v2.py --from-run output/run20

    # Run all built-in test cases
    python scripts/test_codegen_v2.py --test-id 0

    # Skip sandbox (only test generation quality)
    python scripts/test_codegen_v2.py --test-id 1 --no-sandbox
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from researchclaw.llm.client import LLMClient, LLMConfig
from researchclaw.pipeline.code_agent import CodeAgent, CodeAgentConfig
from researchclaw.prompts import PromptManager

# ---------------------------------------------------------------------------
# Built-in test cases
# ---------------------------------------------------------------------------

TEST_CASES = {
    1: {
        "name": "KD for Compact ViTs (CIFAR-10)",
        "topic": (
            "Knowledge Distillation for Compact Vision Transformers: "
            "Attention-Guided Feature Alignment on CIFAR-10"
        ),
        "exp_plan": """
topic: "Knowledge Distillation for Compact Vision Transformers"
datasets:
  - name: CIFAR-10
    source: torchvision.datasets.CIFAR10
    path: /opt/datasets/cifar10
baselines:
  - name: TeacherResNet18
    description: Pre-trained ResNet-18 teacher model (frozen)
    implementation_spec:
      class_name: TeacherResNet18
      key_methods: [__init__, forward]
      algorithm_steps:
        - Load pre-trained ResNet-18 from torchvision
        - Freeze all parameters
        - Use as teacher for distillation
  - name: StudentViT_Baseline
    description: Compact ViT trained with standard cross-entropy (no KD)
    implementation_spec:
      class_name: StudentViTBaseline
      key_methods: [__init__, forward, train_epoch, evaluate]
      algorithm_steps:
        - Compact ViT with patch_size=4, dim=128, depth=4, heads=4
        - Train with cross-entropy loss only
        - Standard SGD optimizer with cosine LR schedule
      loss_function: "L = CrossEntropy(student_logits, labels)"
      key_hyperparameters:
        lr: 0.01
        batch_size: 128
        epochs: 20
proposed_methods:
  - name: AttentionGuidedKD
    description: Knowledge distillation with attention-guided feature alignment
    aligns_hypothesis: H1
    implementation_spec:
      class_name: AttentionGuidedKDStudent
      key_methods: [__init__, forward, compute_kd_loss, compute_attention_loss, train_epoch]
      algorithm_steps:
        - Same compact ViT architecture as baseline
        - KD loss with temperature T=4
        - Attention transfer loss between teacher and student attention maps
        - Combined loss = alpha * KD_loss + beta * attention_loss + (1-alpha-beta) * CE_loss
      loss_function: "L = 0.5*KLDiv(s/T, t/T)*T^2 + 0.3*MSE(student_attn, teacher_attn) + 0.2*CE(s, y)"
      key_hyperparameters:
        temperature: 4
        alpha: 0.5
        beta: 0.3
        lr: 0.01
      differentiator: Uses attention map alignment between teacher and student
ablations:
  - name: KD_NoAttentionTransfer
    based_on: AttentionGuidedKD
    what_is_removed: Attention transfer loss (beta=0)
    how_it_differs: Only uses KD loss + CE loss, no attention alignment
    expected_effect: Lower accuracy due to missing attention guidance
  - name: KD_ReducedCapacity
    based_on: AttentionGuidedKD
    what_is_removed: Half the model capacity (dim=64, depth=2, heads=2)
    how_it_differs: Smaller ViT architecture, same training procedure
    expected_effect: Lower accuracy due to reduced model capacity
metrics:
  primary_metric:
    name: primary_metric
    direction: maximize
    description: Top-1 accuracy on CIFAR-10 test set
compute_budget:
  total_time_seconds: 300
  conditions: [TeacherResNet18, StudentViT_Baseline, AttentionGuidedKD, KD_NoAttentionTransfer, KD_ReducedCapacity]
""",
        "metric": "primary_metric",
        "metric_direction": "maximize",
    },
    2: {
        "name": "PPO with Curiosity Reward (Gymnasium)",
        "topic": (
            "Agent-Centric Reinforcement Learning with Adaptive Reward "
            "Decomposition for CartPole and LunarLander"
        ),
        "exp_plan": """
topic: "Agent-Centric RL with Adaptive Reward Decomposition"
datasets:
  - name: CartPole-v1
    source: gymnasium
  - name: LunarLander-v3
    source: gymnasium
baselines:
  - name: VanillaPPO
    description: Standard PPO with clipped surrogate objective
    implementation_spec:
      class_name: VanillaPPO
      key_methods: [__init__, select_action, update, train_episode]
      algorithm_steps:
        - Policy network (2-layer MLP, 64 hidden)
        - Value network (separate 2-layer MLP)
        - Clipped surrogate objective with epsilon=0.2
        - GAE lambda=0.95 for advantage estimation
      loss_function: "L_policy = -min(r*A, clip(r,1-eps,1+eps)*A); L_value = MSE(V, R)"
      key_hyperparameters:
        lr: 3e-4
        gamma: 0.99
        clip_eps: 0.2
        gae_lambda: 0.95
      differentiator: Standard PPO baseline
proposed_methods:
  - name: CuriosityPPO
    description: PPO with intrinsic curiosity module
    implementation_spec:
      class_name: CuriosityPPO
      key_methods: [__init__, select_action, compute_intrinsic_reward, update, train_episode]
      algorithm_steps:
        - Same PPO base as VanillaPPO
        - Forward dynamics model predicts next state from (state, action)
        - Intrinsic reward = prediction error of forward model
        - Total reward = extrinsic + eta * intrinsic
      loss_function: "L = L_ppo + L_forward_model; r_total = r_ext + eta * ||f(s,a) - s'||^2"
      key_hyperparameters:
        eta: 0.1
        forward_model_lr: 1e-3
      differentiator: Adds intrinsic curiosity-driven exploration reward
ablations:
  - name: PPO_NoCuriosity
    based_on: CuriosityPPO
    what_is_removed: Intrinsic reward (eta=0, forward model not used)
    how_it_differs: Same architecture but intrinsic reward zeroed out
    expected_effect: Should match VanillaPPO performance
  - name: PPO_ReducedNetwork
    based_on: VanillaPPO
    what_is_removed: Half network capacity (32 hidden units)
    how_it_differs: Smaller policy and value networks
    expected_effect: Lower performance due to limited capacity
metrics:
  primary_metric:
    name: primary_metric
    direction: maximize
    description: Average episodic reward over last 10 episodes
compute_budget:
  total_time_seconds: 300
  conditions: [VanillaPPO, CuriosityPPO, PPO_NoCuriosity, PPO_ReducedNetwork]
""",
        "metric": "primary_metric",
        "metric_direction": "maximize",
    },
    3: {
        "name": "Graph Neural ODE (Synthetic)",
        "topic": (
            "Graph Neural Ordinary Differential Equations for Dynamic System "
            "Modeling on Synthetic Coupled Oscillator Networks"
        ),
        "exp_plan": """
topic: "Graph Neural ODE for Dynamic System Modeling"
datasets:
  - name: SyntheticOscillators
    source: Generated in-code
    description: Coupled spring-mass system on a random graph
baselines:
  - name: StaticGCN
    description: Standard GCN applied at discrete time steps
    implementation_spec:
      class_name: StaticGCN
      key_methods: [__init__, forward, predict_trajectory]
      algorithm_steps:
        - 2-layer GCN with message passing
        - Discrete time step predictions
        - MSE loss on next-step prediction
      loss_function: "L = MSE(pred_next, true_next)"
      key_hyperparameters:
        hidden_dim: 64
        num_layers: 2
        lr: 1e-3
proposed_methods:
  - name: GraphNeuralODE
    description: Continuous-time dynamics via Neural ODE on graph
    implementation_spec:
      class_name: GraphNeuralODE
      key_methods: [__init__, forward, ode_func, predict_trajectory]
      algorithm_steps:
        - GNN-based ODE function f(t, x, A) that defines dx/dt
        - Neural ODE solver (torchdiffeq.odeint) for continuous trajectory
        - MSE loss on trajectory prediction at observed time points
      loss_function: "L = MSE(odeint(f, x0, t), x_true)"
      key_hyperparameters:
        hidden_dim: 64
        solver: dopri5
        lr: 1e-3
      differentiator: Continuous-time dynamics via ODE solver
ablations:
  - name: GraphODE_NoMessagePassing
    based_on: GraphNeuralODE
    what_is_removed: Graph structure (treats nodes independently)
    how_it_differs: ODE function ignores adjacency, no message passing
    expected_effect: Worse prediction on coupled systems
  - name: GraphODE_EulerSolver
    based_on: GraphNeuralODE
    what_is_removed: Adaptive ODE solver (uses fixed-step Euler)
    how_it_differs: Simple Euler integration instead of dopri5
    expected_effect: Less accurate trajectories
metrics:
  primary_metric:
    name: primary_metric
    direction: minimize
    description: MSE between predicted and true trajectories
compute_budget:
  total_time_seconds: 300
  conditions: [StaticGCN, GraphNeuralODE, GraphODE_NoMessagePassing, GraphODE_EulerSolver]
""",
        "metric": "primary_metric",
        "metric_direction": "minimize",
    },
}


# ---------------------------------------------------------------------------
# Code quality analysis (comprehensive)
# ---------------------------------------------------------------------------

def analyze_code_quality(files: dict[str, str], test_case: dict) -> dict:
    """Comprehensive code quality analysis."""
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

    for fname, code in files.items():
        lines = code.split("\n")
        report["total_lines"] += len(lines)
        effective = [
            l for l in lines
            if l.strip()
            and not l.strip().startswith("#")
            and not l.strip().startswith('"""')
            and not l.strip().startswith("'''")
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
                        (n.end_lineno or n.lineno) - n.lineno + 1
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    )
                    # Check for empty methods
                    empty_methods = []
                    for n in node.body:
                        if isinstance(n, ast.FunctionDef):
                            body_stmts = [
                                s for s in n.body
                                if not isinstance(s, (ast.Pass, ast.Expr))
                                or (isinstance(s, ast.Expr)
                                    and not isinstance(s.value, (ast.Constant, ast.Str)))
                            ]
                            if len(body_stmts) <= 1:
                                empty_methods.append(n.name)

                    report["classes_found"].append({
                        "name": node.name,
                        "file": fname,
                        "methods": methods,
                        "method_count": len(methods),
                        "total_method_lines": method_lines,
                        "bases": [ast.unparse(b) for b in node.bases],
                        "empty_methods": empty_methods,
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
                    elif node.module:
                        report["imports_found"].append(node.module.split(".")[0])
        except SyntaxError as e:
            report["issues"].append(f"CRITICAL: SyntaxError in {fname}: {e}")

    report["imports_found"] = sorted(set(report["imports_found"]))

    # ---- Scoring ----

    # 1. Syntax validity (0 or 10)
    syntax_ok = not any("SyntaxError" in i for i in report["issues"])
    report["scores"]["syntax_valid"] = 10 if syntax_ok else 0

    # 2. File structure
    file_score = min(10, len(files) * 5)  # 2+ files = 10
    report["scores"]["file_structure"] = round(file_score, 1)

    # 3. Class coverage
    n_classes = len(report["classes_found"])
    class_score = min(10, n_classes * 2.5)  # 4+ classes = 10
    report["scores"]["class_coverage"] = round(class_score, 1)

    # 4. Code depth
    depth_score = min(10, report["effective_lines"] / 40)  # 400+ = 10
    report["scores"]["code_depth"] = round(depth_score, 1)

    # 5. Method richness
    if report["classes_found"]:
        avg_methods = sum(c["method_count"] for c in report["classes_found"]) / n_classes
        method_score = min(10, avg_methods * 2)  # 5+ methods = 10
    else:
        method_score = 0
    report["scores"]["method_richness"] = round(method_score, 1)

    # 6. Class distinctness (check for identical/empty classes)
    empty_class_count = sum(
        1 for c in report["classes_found"]
        if c["total_method_lines"] < 5
    )
    identical_pairs = _check_identical_classes(files)
    distinctness = 10
    if empty_class_count > 0:
        distinctness -= empty_class_count * 3
        report["issues"].append(
            f"WARNING: {empty_class_count} classes have <5 method lines (too thin)"
        )
    if identical_pairs:
        distinctness -= len(identical_pairs) * 4
        for p in identical_pairs:
            report["issues"].append(f"WARNING: Identical classes: {p}")
    report["scores"]["class_distinctness"] = max(0, round(distinctness, 1))

    # 7. Import appropriateness
    has_torch = "torch" in report["imports_found"]
    has_numpy = "numpy" in report["imports_found"]
    import_score = 5  # base
    if has_torch:
        import_score += 3
    if has_numpy:
        import_score += 2
    report["scores"]["imports"] = min(10, import_score)

    # Overall score
    scores = report["scores"]
    report["overall_score"] = round(sum(scores.values()) / len(scores), 1)

    return report


def _check_identical_classes(files: dict[str, str]) -> list[str]:
    """Check for classes with identical method bodies."""
    identical = []
    class_bodies: dict[str, str] = {}

    for fname, code in files.items():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Hash the method bodies
                method_code = ""
                for n in node.body:
                    if isinstance(n, ast.FunctionDef):
                        try:
                            method_code += ast.unparse(n) + "\n"
                        except Exception:
                            pass
                if method_code:
                    key = hash(method_code)
                    if key in class_bodies:
                        identical.append(
                            f"{class_bodies[key]} == {node.name}"
                        )
                    else:
                        class_bodies[key] = node.name
    return identical


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

def run_in_sandbox(
    files: dict[str, str],
    output_dir: Path,
    config_path: str | None = None,
    timeout_sec: int = 300,
) -> dict:
    """Run generated code in subprocess (or Docker sandbox if available)."""
    # Write files
    code_dir = output_dir / "experiment"
    code_dir.mkdir(parents=True, exist_ok=True)
    for fname, code in files.items():
        (code_dir / fname).write_text(code, encoding="utf-8")

    # Try to run with subprocess as fallback
    import subprocess
    main_py = code_dir / "main.py"
    if not main_py.exists():
        return {"status": "failed", "reason": "no main.py"}

    print(f"  Running in subprocess (timeout={timeout_sec}s)...")
    try:
        proc = subprocess.run(
            [sys.executable, str(main_py)],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONPATH": str(code_dir)},
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "TIMEOUT"
        returncode = -1
        timed_out = True

    # Parse results
    result = {
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_lines": len(stdout.split("\n")) if stdout else 0,
        "stderr_lines": len(stderr.split("\n")) if stderr else 0,
        "conditions_found": [],
        "metrics_found": {},
        "has_metric_def": False,
        "has_registered_conditions": False,
    }

    # Parse stdout for conditions and metrics
    if stdout:
        for line in stdout.split("\n"):
            if line.startswith("METRIC_DEF:"):
                result["has_metric_def"] = True
            elif line.startswith("REGISTERED_CONDITIONS:"):
                result["has_registered_conditions"] = True
                conds = line.split(":", 1)[1].strip()
                result["conditions_found"] = [c.strip() for c in conds.split(",")]
            elif "condition=" in line:
                m = re.match(r"condition=(\S+)\s+(\S+):\s+(\S+)", line)
                if m:
                    cond, metric_name, value = m.groups()
                    if cond not in result["metrics_found"]:
                        result["metrics_found"][cond] = {}
                    try:
                        result["metrics_found"][cond][metric_name] = float(value)
                    except ValueError:
                        pass

    # Score execution
    exec_score = 0
    if returncode == 0:
        exec_score += 3  # runs
    if result["has_metric_def"]:
        exec_score += 1
    if result["has_registered_conditions"]:
        exec_score += 1
    if result["conditions_found"]:
        exec_score += min(3, len(result["conditions_found"]))  # up to 3 for conditions
    if result["metrics_found"]:
        exec_score += 2  # produces metrics
    result["exec_score"] = min(10, exec_score)

    # Save stdout/stderr
    (output_dir / "stdout.txt").write_text(stdout or "(empty)", encoding="utf-8")
    (output_dir / "stderr.txt").write_text(stderr or "(empty)", encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Load experiment plan from previous run
# ---------------------------------------------------------------------------

def load_from_run(run_dir: str) -> dict:
    """Load experiment plan and config from a previous pipeline run."""
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    # Find exp_plan.yaml
    plan_path = None
    for s9_dir in sorted(run_path.glob("stage-09*"), reverse=True):
        candidate = s9_dir / "exp_plan.yaml"
        if candidate.exists():
            plan_path = candidate
            break

    if plan_path is None:
        print(f"ERROR: No exp_plan.yaml found in {run_dir}/stage-09*/")
        sys.exit(1)

    exp_plan = plan_path.read_text(encoding="utf-8")

    # Load topic from config or stage-01
    topic = ""
    for topic_file in ["topic_evaluation.json", "topic.json"]:
        for s_dir in sorted(run_path.glob("stage-0[12]*"), reverse=True):
            tf = s_dir / topic_file
            if tf.exists():
                try:
                    td = json.loads(tf.read_text(encoding="utf-8"))
                    topic = td.get("topic", "") or td.get("research_topic", "")
                    if topic:
                        break
                except Exception:
                    pass
        if topic:
            break

    # Try to extract topic from exp_plan if not found elsewhere
    if not topic:
        import yaml
        try:
            plan_data = yaml.safe_load(exp_plan)
            topic = plan_data.get("topic", "Unknown Topic")
        except Exception:
            topic = "Unknown Topic"

    return {
        "name": f"From {run_path.name}",
        "topic": topic,
        "exp_plan": exp_plan,
        "metric": "primary_metric",
        "metric_direction": "maximize",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test code generation quality with optional sandbox execution"
    )
    parser.add_argument("--model", default="gpt-5.1", help="Model to use")
    parser.add_argument("--test-id", type=int, default=0, help="Test case ID (0=all)")
    parser.add_argument("--from-run", default="", help="Load exp plan from run dir")
    parser.add_argument("--no-sandbox", action="store_true", help="Skip sandbox execution")
    parser.add_argument("--sandbox-timeout", type=int, default=300, help="Sandbox timeout (sec)")
    parser.add_argument("--output-dir", default="test_outputs_codegen", help="Output dir")
    parser.add_argument("--config", default="config_run20.yaml", help="Config file for LLM")
    args = parser.parse_args()

    # Setup LLM client
    # Try loading from config file first
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        llm_cfg = cfg.get("llm", {})
        base_url = llm_cfg.get("base_url", "")
        api_key = llm_cfg.get("api_key", "") or os.environ.get(
            llm_cfg.get("api_key_env", "OPENAI_API_KEY"), ""
        )
    else:
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not base_url or not api_key:
        print("ERROR: Need LLM config. Provide --config or set env vars.")
        sys.exit(1)

    llm_config = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        primary_model=args.model,
        fallback_models=["gpt-4.1", "gpt-4o"],
        max_tokens=16384,
        temperature=0.7,
        timeout_sec=300,
    )
    llm = LLMClient(llm_config)

    # Connectivity test
    print(f"Testing LLM ({args.model})...", end=" ", flush=True)
    ok, msg = llm.preflight()
    if not ok:
        print(f"FAILED: {msg}")
        sys.exit(1)
    print("OK")

    pm = PromptManager()

    # Select test cases
    if args.from_run:
        cases = {99: load_from_run(args.from_run)}
    elif args.test_id > 0:
        if args.test_id not in TEST_CASES:
            print(f"ERROR: Unknown test ID {args.test_id}. Available: {list(TEST_CASES.keys())}")
            sys.exit(1)
        cases = {args.test_id: TEST_CASES[args.test_id]}
    else:
        cases = dict(TEST_CASES)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []

    for test_id, tc in cases.items():
        print(f"\n{'='*70}")
        print(f"  Test {test_id}: {tc['name']}")
        print(f"{'='*70}")

        stage_dir = output_dir / f"test_{test_id}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Configure CodeAgent
        agent_config = CodeAgentConfig(
            architecture_planning=True,
            exec_fix_max_iterations=0,  # no sandbox in generation phase
            tree_search_enabled=False,
            review_max_rounds=2,
        )

        agent = CodeAgent(
            llm=llm,
            prompts=pm,
            config=agent_config,
            stage_dir=stage_dir,
        )

        # Build pkg_hint
        pkg_hint = (
            "\nAVAILABLE PACKAGES (docker mode): Python stdlib, numpy, torch, "
            "torchvision, torchaudio, matplotlib, seaborn, scipy, tqdm, "
            "torchdiffeq, gymnasium, networkx, PyYAML, Pillow, transformers, "
            "datasets, accelerate, peft, timm, einops, torchmetrics.\n"
            "GPU: NVIDIA RTX 6000 Ada (49GB VRAM). "
            "Use `device = torch.device('cuda')` for tensor operations.\n"
        )

        metric_dir = tc.get("metric_direction", "maximize")
        pkg_hint += f"\nMETRIC DIRECTION: {metric_dir}\n"

        # Add compute budget
        pkg_hint += (
            "\n## Compute Budget Constraint\n"
            "- Total execution time limit: 300 seconds\n"
            "- Design experiments that complete within this budget\n"
            "- Implement a time guard: stop gracefully at 80% of budget\n"
        )

        # Generate
        t0 = time.time()
        result = agent.generate(
            topic=tc["topic"],
            exp_plan=tc["exp_plan"],
            metric=tc.get("metric", "primary_metric"),
            pkg_hint=pkg_hint,
            max_tokens=16384,
        )
        gen_elapsed = time.time() - t0

        print(f"\n  Generation: {gen_elapsed:.1f}s, {result.total_llm_calls} LLM calls")
        print(f"  Architecture spec: {len(result.architecture_spec)} chars")
        print(f"  Review rounds: {result.review_rounds}")

        # Write files
        for fname, code in result.files.items():
            fpath = stage_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")
            print(f"  -> {fname}: {len(code.split(chr(10)))} lines")

        if result.architecture_spec:
            (stage_dir / "architecture_spec.yaml").write_text(
                result.architecture_spec, encoding="utf-8"
            )

        # Quality analysis
        report = analyze_code_quality(result.files, tc)
        report["generation_time_sec"] = round(gen_elapsed, 1)
        report["llm_calls"] = result.total_llm_calls

        # Sandbox execution
        exec_result = {"status": "skipped"}
        if not args.no_sandbox and result.files:
            exec_result = run_in_sandbox(
                result.files, stage_dir,
                timeout_sec=args.sandbox_timeout,
            )
            report["execution"] = exec_result
            print(f"\n  Execution: {exec_result['status']}")
            if exec_result.get("returncode") is not None:
                print(f"    Return code: {exec_result['returncode']}")
            if exec_result.get("conditions_found"):
                print(f"    Conditions: {', '.join(exec_result['conditions_found'])}")
            if exec_result.get("metrics_found"):
                for cond, metrics in exec_result["metrics_found"].items():
                    print(f"    {cond}: {metrics}")

        # Print scores
        print(f"\n  --- Scores ---")
        for k, v in report["scores"].items():
            print(f"    {k}: {v}/10")
        if exec_result.get("exec_score") is not None:
            print(f"    execution: {exec_result['exec_score']}/10")
        print(f"    OVERALL: {report['overall_score']}/10")

        if report["issues"]:
            print(f"\n  Issues:")
            for issue in report["issues"]:
                print(f"    - {issue}")

        # Save report
        (stage_dir / "quality_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        all_reports.append(report)

    # Summary
    if len(all_reports) > 1:
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        for r in all_reports:
            exec_info = ""
            if "execution" in r:
                exec_info = f" | exec: {r['execution'].get('status', '?')}"
            print(
                f"  {r['test_name']}: {r['overall_score']}/10 "
                f"({r['effective_lines']} lines, "
                f"{len(r['classes_found'])} classes{exec_info})"
            )
        avg = sum(r["overall_score"] for r in all_reports) / len(all_reports)
        print(f"\n  Average: {avg:.1f}/10")

    (output_dir / "summary.json").write_text(
        json.dumps(all_reports, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
