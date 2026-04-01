# AutoResearchClaw: Self-Iterating Experiment Optimization — Showcase

> Figure: `docs/figures/iteration_improvement_showcase.png` / `.pdf`

---

## Overview

This figure demonstrates AutoResearchClaw's core capability: **autonomous self-iteration of experimental methods**. Starting from an initial experiment design, the pipeline automatically:

1. Runs the experiment in a sandboxed environment
2. Analyzes the results and identifies weaknesses
3. Proposes algorithmic improvements via LLM reasoning
4. Implements code modifications and re-runs
5. Retains the best-performing version, discards regressions

Below we describe two representative cases from actual pipeline runs.

---

## Case A: Continual Meta-Learning for Few-Shot Adaptation

**Research Topic:** Designing meta-learning algorithms that adapt to non-stationary task distributions, where the underlying data distribution shifts over time.

**Metric:** Post-adaptation query error on held-out tasks (lower = better). Converted to accuracy (%) in the figure.

### Iteration Progression

| Round | Accuracy | What the Pipeline Did |
|-------|----------|----------------------|
| **Baseline** | **25.9%** | Initial experiment code with 6 standard conditions (random search, Bayesian optimization, PPO, etc.). Basic meta-learning framework without domain-specific adaptations. |
| **Iter 1** | **81.2%** (+55.3 pts) | **Major architectural redesign.** The pipeline identified that the baseline methods were generic RL algorithms ill-suited for meta-learning. It autonomously: (1) Replaced generic methods with domain-specific ones: `replay_meta`, `context_gated_replay`, `online_meta_sgd`, `adaptive_lr_meta`; (2) Implemented a two-layer neural encoder with MAML-style inner-loop adaptation; (3) Added context-gated experience replay that modulates replay intensity based on context similarity; (4) Introduced per-parameter meta-SGD learning rates. |
| **Iter 2** | **77.5%** (-3.7 pts) | **Failed experiment — automatically detected and recovered.** The pipeline attempted to simplify the architecture by replacing the deep encoder with a prototype network. This reduced model expressiveness and degraded performance. The pipeline automatically detected the regression and retained the Iter 1 code as the best version. |
| **Iter 3** | **93.4%** (+15.9 pts) | **Architecture refinement with regularization.** Learning from both the success of Iter 1 and the failure of Iter 2, the pipeline: (1) Adopted a linear classifier with proper gradient-based inner-loop adaptation (simpler than Iter 1's deep encoder but more expressive than Iter 2's prototypes); (2) Added L2 anchor regularization to prevent catastrophic forgetting during adaptation; (3) Implemented cosine similarity-based context gating (more robust than prototype-distance gating); (4) Increased seed count from 24 to 28 for more robust statistics; (5) Added new comparison conditions: `prototype_regularized_meta`, `drift_aware_meta`. |
| **Iter 4** | **93.4%** (converged) | Minor hyperparameter adjustments. Pipeline recognized convergence and stopped. |

**Key Insight:** The pipeline demonstrated the ability to **recover from a failed approach** (Iter 2's prototype networks) by synthesizing lessons from both successful (Iter 1) and failed (Iter 2) attempts to arrive at a superior solution (Iter 3).

---

## Case B: RLHF with Curriculum-Based Reward Shaping for LLM Alignment

**Research Topic:** Improving LLM alignment through reinforcement learning from human feedback, with a curriculum-based approach that gradually increases task difficulty.

**Metric:** 1 − alignment_error (higher = better). Represents how well the trained policy aligns with human preferences.

### Iteration Progression

| Round | Alignment | What the Pipeline Did |
|-------|-----------|----------------------|
| **Baseline** | **35.6%** | Vanilla PPO policy with linear reward function. Direct preference feedback from environment oracle. No learned reward model, no curriculum scheduling. |
| **Iter 1** | **35.6%** (no change) | Minor code modifications that did not affect performance. Pipeline correctly identified no improvement and continued iterating. |
| **Iter 2** | **61.6%** (+26.0 pts) | **Core algorithmic innovation.** The pipeline introduced three key components: (1) **Learned preference reward model** — a logistic regression model trained on preference pairs: P(prefer chosen \| feature delta), updated online with Adam optimizer; (2) **Reward mixing schedule** — gradually increases reliance on the learned reward model from 10% to 80% over training (coefficient ramp); (3) **Curriculum power shaping** — nonlinear difficulty progression (power=1.4) that gives the agent more time on easier problems before advancing. |
| **Iter 3** | **63.0%** (+1.4 pts) | **Multi-signal evaluation.** Added: (1) **Rank-normalized multi-action evaluation** — samples up to 4 actions per state and evaluates preference feedback for each, converting to rank-based scores in [-1, +1]; (2) **Direct reward regression head** — a second regression-based reward predictor using ridge regression, blended with the classification head; (3) **Policy EMA** — exponential moving average of policy parameters (decay=0.92) with anchor regularization for training stability. |
| **Iter 4** | **66.6%** (+3.6 pts) | **Confidence-aware reward integration.** Added: (1) **Confidence-gated reward** — measures learned reward model accuracy, then uses softmax entropy to modulate how much the reward signal influences actions; (2) **Mini-batch reward model updates** — trains on 3 randomly sampled past preference pairs per step (not just current); (3) **Margin bonus** — early-curriculum episodes receive extra reward shaping from preference margins (coef=0.18 × (1−level) × tanh(margin)). |

**Key Insight:** The pipeline demonstrated **incremental technical sophistication** — each iteration built upon the previous one by adding a specific, well-motivated technique. The progression from vanilla PPO → learned reward model → multi-signal evaluation → confidence gating mirrors how a human researcher would iteratively refine an RLHF system.

---

## What This Demonstrates

1. **Autonomous Problem Diagnosis:** The pipeline identifies *why* performance is limited (e.g., "generic RL methods are unsuitable for meta-learning") and proposes targeted solutions.

2. **Failure Recovery:** When an iteration produces worse results (Case A, Iter 2), the pipeline automatically detects the regression, retains the previous best version, and learns from the failure to produce a better solution in the next iteration.

3. **Progressive Refinement:** Rather than making random changes, the pipeline demonstrates cumulative improvement — each iteration builds on insights from previous ones (Case B: reward model → rank normalization → confidence gating).

4. **Domain-Appropriate Innovation:** The pipeline generates methods that are appropriate for the specific research domain (context-gated replay for meta-learning, preference reward models for RLHF), not just generic hyperparameter tuning.

5. **Convergence Detection:** The pipeline automatically recognizes when further iterations are unlikely to yield improvement and terminates, avoiding wasted computation.

---

## Data Sources

- Case A: `artifacts/rc-20260314-132748-0ec2c9/stage-13_v2/refinement_log.json`
- Case B: `artifacts/rc-20260314-132748-91c516/stage-13/refinement_log.json`
- Figure script: `scripts/plot_iteration_showcase.py`
