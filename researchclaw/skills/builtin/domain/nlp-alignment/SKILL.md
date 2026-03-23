---
name: nlp-alignment
description: Best practices for LLM alignment techniques including RLHF, DPO, and instruction tuning. Use when working on alignment or safety.
metadata:
  category: domain
  trigger-keywords: "alignment,rlhf,dpo,reward model,preference,instruction tuning,safety"
  applicable-stages: "9,10"
  priority: "4"
  version: "1.0"
  author: researchclaw
  references: "Ouyang et al., Training language models to follow instructions, NeurIPS 2022; Rafailov et al., DPO, NeurIPS 2023"
---

## LLM Alignment Best Practice
Methods:
- RLHF: Train reward model → PPO fine-tuning (complex but powerful)
- DPO: Direct preference optimization (simpler, no reward model needed)
- GRPO: Group relative policy optimization
- SFT: Supervised fine-tuning as alignment baseline

Training recipe:
- Start with SFT on high-quality instruction data
- DPO: lr=5e-7, beta=0.1, batch_size=64
- PPO: lr=1e-6, clip=0.2, KL coeff=0.02
- Use reference model for KL penalty
- Evaluate on safety benchmarks (TruthfulQA, BBQ, etc.)

Common pitfalls:
- Reward hacking: model finds shortcuts to high reward
- Mode collapse: model generates repetitive outputs
- Catastrophic forgetting: loses general capabilities
