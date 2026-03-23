---
name: nlp-pretraining
description: Best practices for language model pretraining and fine-tuning. Use when generating or reviewing NLP training code.
metadata:
  category: domain
  trigger-keywords: "language model,pretraining,fine-tuning,bert,gpt,llm,transformer,nlp,text"
  applicable-stages: "9,10"
  priority: "3"
  version: "1.0"
  author: researchclaw
  references: "Devlin et al., BERT, NAACL 2019; Hu et al., LoRA, ICLR 2022"
---

## NLP Pretraining/Fine-tuning Best Practice
Fine-tuning recipe:
- Use pre-trained checkpoints (HuggingFace hub)
- AdamW optimizer, lr=2e-5 to 5e-5
- Linear warmup (6% of total steps) + linear decay
- Batch size: 16-32 (use gradient accumulation for larger effective batch)
- 3-5 epochs for classification, 1-2 for generation
- Weight decay: 0.01

Parameter-efficient methods:
- LoRA: r=8-64, alpha=16-128, apply to q/v projections
- Prefix tuning: 10-20 prefix tokens
- Adapters: bottleneck dimension 64-256

Evaluation:
- Classification: accuracy, F1 (macro for imbalanced)
- Generation: perplexity, BLEU/ROUGE, human evaluation
- Use multiple seeds and report mean +/- std
