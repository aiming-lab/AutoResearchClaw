---
name: experimental-design
description: Best practices for designing reproducible ML experiments. Use when planning ablations, baselines, or controlled experiments.
metadata:
  category: experiment
  trigger-keywords: "experiment,ablation,baseline,control,hypothesis,reproducib"
  applicable-stages: "9,10,12"
  priority: "2"
  version: "1.0"
  author: researchclaw
  references: "Bouthillier et al., Accounting for Variance in ML Benchmarks, MLSys 2021"
---

## Experimental Design Best Practice
1. ALWAYS include meaningful baselines (not just random):
   - At least one classical method baseline
   - At least one recent SOTA method baseline
   - A simple-but-strong baseline (e.g., linear probe, k-NN)
2. Use MULTIPLE random seeds (minimum 3, ideally 5)
3. Report mean +/- std across seeds
4. Design ablations that isolate EACH key component:
   - Remove one component at a time
   - Each ablation must be meaningfully different from baseline
5. Control variables: change only ONE thing per comparison
6. Use standard splits (train/val/test) — never test on training data
7. Report wall-clock time and memory usage alongside accuracy
